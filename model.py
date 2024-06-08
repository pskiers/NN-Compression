# Most of the code from https://github.com/Qualcomm-AI-research/FFNet

import torch
from torch import nn
from torch.nn import functional as F
from scipy import ndimage
import numpy as np
import torchvision.models as models
import pytorch_lightning as pl


def dense_kernel_initializer(tensor):
    _, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    init_range = 1.0 / np.sqrt(fan_out)

    return nn.init.uniform_(tensor, a=-init_range, b=init_range)


def model_weight_initializer(m):
    """
    Usage:
        model = Model()
        model.apply(weight_init)
    """
    if isinstance(m, nn.Conv2d):
        # Yes, this non-fancy init is on purpose,
        # and seems to work better in practice for segmentation
        if hasattr(m, "weight"):
            nn.init.normal_(m.weight, std=0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0001)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.Linear):
        dense_kernel_initializer(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)


# The modules here currently assume that there are always 4 branches.
# It would need to be adapted in order to support a variable number of branches

# TODO : Pass BN momentum through config
BN_MOMENTUM = 0.1
gpu_up_kwargs = {"mode": "bilinear", "align_corners": True}
mobile_up_kwargs = {"mode": "nearest"}
relu_inplace = True

# TODO : Replace functional interpolate operations with upsample modules


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_chan,
        out_chan,
        ks=3,
        stride=1,
        padding=1,
        activation=nn.ReLU,
        *args,
        **kwargs,
    ):
        super(ConvBNReLU, self).__init__()
        layers = [
            nn.Conv2d(
                in_chan,
                out_chan,
                kernel_size=ks,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_chan, momentum=BN_MOMENTUM),
        ]
        if activation:
            layers.append(activation(inplace=relu_inplace))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class AdapterConv(nn.Module):
    def __init__(self, in_channels=[256, 512, 1024, 2048], out_channels=[64, 128, 256, 512]):
        super(AdapterConv, self).__init__()
        assert len(in_channels) == len(out_channels), "Number of input and output branches should match"
        self.adapter_conv = nn.ModuleList()

        for k in range(len(in_channels)):
            self.adapter_conv.append(
                ConvBNReLU(in_channels[k], out_channels[k], ks=1, stride=1, padding=0),
            )

    def forward(self, x):
        out = []
        for k in range(len(self.adapter_conv)):
            out.append(self.adapter_conv[k](x[k]))
        return out


class UpsampleCat(nn.Module):
    def __init__(self, upsample_kwargs=gpu_up_kwargs):
        super(UpsampleCat, self).__init__()
        self._up_kwargs = upsample_kwargs

    def forward(self, x):
        """Upsample and concatenate feature maps."""
        assert isinstance(x, list) or isinstance(x, tuple)
        # print(self._up_kwargs)
        x0 = x[0]
        _, _, H, W = x0.size()
        for i in range(1, len(x)):
            x0 = torch.cat([x0, F.interpolate(x[i], (H, W), **self._up_kwargs)], dim=1)
        return x0


class UpBranch(nn.Module):
    def __init__(
        self,
        in_channels=[64, 128, 256, 512],
        out_channels=[128, 128, 128, 128],
        upsample_kwargs=gpu_up_kwargs,
    ):
        super(UpBranch, self).__init__()

        self._up_kwargs = upsample_kwargs

        self.fam_32_sm = ConvBNReLU(in_channels[3], out_channels[3], ks=3, stride=1, padding=1)
        self.fam_32_up = ConvBNReLU(in_channels[3], in_channels[2], ks=1, stride=1, padding=0)
        self.fam_16_sm = ConvBNReLU(in_channels[2], out_channels[2], ks=3, stride=1, padding=1)
        self.fam_16_up = ConvBNReLU(in_channels[2], in_channels[1], ks=1, stride=1, padding=0)
        self.fam_8_sm = ConvBNReLU(in_channels[1], out_channels[1], ks=3, stride=1, padding=1)
        self.fam_8_up = ConvBNReLU(in_channels[1], in_channels[0], ks=1, stride=1, padding=0)
        self.fam_4 = ConvBNReLU(in_channels[0], out_channels[0], ks=3, stride=1, padding=1)

        self.high_level_ch = sum(out_channels)
        self.out_channels = out_channels

    def forward(self, x):

        feat4, feat8, feat16, feat32 = x

        smfeat_32 = self.fam_32_sm(feat32)
        upfeat_32 = self.fam_32_up(feat32)

        _, _, H, W = feat16.size()
        x = F.interpolate(upfeat_32, (H, W), **self._up_kwargs) + feat16
        smfeat_16 = self.fam_16_sm(x)
        upfeat_16 = self.fam_16_up(x)

        _, _, H, W = feat8.size()
        x = F.interpolate(upfeat_16, (H, W), **self._up_kwargs) + feat8
        smfeat_8 = self.fam_8_sm(x)
        upfeat_8 = self.fam_8_up(x)

        _, _, H, W = feat4.size()
        smfeat_4 = self.fam_4(F.interpolate(upfeat_8, (H, W), **self._up_kwargs) + feat4)

        return smfeat_4, smfeat_8, smfeat_16, smfeat_32


class FFNetUpHead(nn.Module):
    def __init__(
        self,
        in_chans,
        use_adapter_conv=True,
        head_type="A_mobile",
        task="segmentation_A",
        num_classes=19,
        base_chans=[64, 128, 256, 512],
        dropout_rate=None,  # Only used for classification
        *args,
        **kwargs,
    ):
        super(FFNetUpHead, self).__init__()
        layers = []
        # base_chans = [64, 128, 128, 128]
        if head_type.startswith("A"):
            base_chans = [64, 128, 256, 512]
        elif head_type.startswith("B"):
            base_chans = [64, 128, 128, 256]
        elif head_type.startswith("C"):
            base_chans = [128, 128, 128, 128]

        if use_adapter_conv:
            layers.append(AdapterConv(in_chans, base_chans))
            in_chans = base_chans[:]

        if head_type == "A":
            layers.append(UpBranch(in_chans))
        elif head_type == "A_mobile":
            layers.append(UpBranch(in_chans, upsample_kwargs=mobile_up_kwargs))
        elif head_type == "B":
            layers.append(UpBranch(in_chans, [96, 96, 64, 32]))
        elif head_type == "B_mobile":
            layers.append(UpBranch(in_chans, [96, 96, 64, 32], upsample_kwargs=mobile_up_kwargs))
        elif head_type == "C":
            layers.append(UpBranch(in_chans, [128, 16, 16, 16]))
        elif head_type == "C_mobile":
            layers.append(UpBranch(in_chans, [128, 16, 16, 16], upsample_kwargs=mobile_up_kwargs))
        else:
            raise ValueError(f"Unknown FFNetUpHead type {head_type}")

        self.num_features = layers[-1].high_level_ch
        self.num_multi_scale_features = layers[-1].out_channels

        if task.startswith("segmentation"):
            if "mobile" in head_type:
                layers.append(UpsampleCat(mobile_up_kwargs))
            else:
                layers.append(UpsampleCat(gpu_up_kwargs))

            # Gets single scale input
            if "_C" in task:
                mid_feat = 128
                layers.append(
                    SegmentationHead_NoSigmoid_1x1(
                        self.num_features,
                        mid_feat,
                        num_outputs=num_classes,
                    )
                )
            elif "_B" in task:
                mid_feat = 256
                layers.append(
                    SegmentationHead_NoSigmoid_3x3(
                        self.num_features,
                        mid_feat,
                        num_outputs=num_classes,
                    )
                )
            elif "_A" in task:
                mid_feat = 512
                layers.append(
                    SegmentationHead_NoSigmoid_1x1(
                        self.num_features,
                        mid_feat,
                        num_outputs=num_classes,
                    )
                )
            else:
                raise ValueError(f"Unknown Segmentation Head {task}")

        elif task == "classification":
            # Gets multi scale input
            layers.append(
                ClassificationHead(
                    self.num_multi_scale_features,
                    [128, 256, 512, 1024],
                    num_outputs=num_classes,
                    dropout_rate=dropout_rate,
                )
            )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SimpleBottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(SimpleBottleneckBlock, self).__init__()
        bn_mom = 0.1
        bn_eps = 1e-5

        self.downsample = None
        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * self.expansion, momentum=bn_mom),
            )

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=bn_mom)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=bn_mom)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ClassificationHead(nn.Module):
    def __init__(
        self,
        pre_head_channels,
        head_channels=[128, 256, 512, 1024],
        num_outputs=1,
        dropout_rate=None,
    ):
        super(ClassificationHead, self).__init__()

        self.dropout_rate = dropout_rate
        bn_mom = 0.1
        bn_eps = 1e-5
        head_block_type = SimpleBottleneckBlock
        head_expansion = 4

        expansion_layers = []
        for i, pre_head_channel in enumerate(pre_head_channels):
            expansion_layer = head_block_type(
                pre_head_channel,
                int(head_channels[i] / head_expansion),
            )
            expansion_layers.append(expansion_layer)
        self.expansion_layers = nn.ModuleList(expansion_layers)

        # downsampling modules
        downsampling_layers = []
        for i in range(len(pre_head_channels) - 1):
            input_channels = head_channels[i]
            output_channels = head_channels[i + 1]

            downsampling_layer = nn.Sequential(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.BatchNorm2d(output_channels, momentum=bn_mom),
                nn.ReLU(),
            )

            downsampling_layers.append(downsampling_layer)
        self.downsampling_layers = nn.ModuleList(downsampling_layers)

        self.final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=head_channels[-1],
                out_channels=2048,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(2048, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(
            2048,
            num_outputs,
        )

    def forward(self, x):

        next_x = self.expansion_layers[0](x[0])
        for i in range(len(self.downsampling_layers)):
            next_x = self.expansion_layers[i + 1](x[i + 1]) + self.downsampling_layers[i](next_x)
        x = next_x

        x = self.final_layer(x)
        x = self.adaptive_avg_pool(x).squeeze()

        if self.dropout_rate:
            x = torch.nn.functional.dropout(x, p=self._model_config.dropout_rate, training=self.training)

        x = self.classifier(x)
        return x


class SegmentationHead_NoSigmoid_3x3(nn.Module):
    def __init__(self, backbone_channels, mid_channels=256, kernel_size=3, num_outputs=1):
        super(SegmentationHead_NoSigmoid_3x3, self).__init__()
        last_inp_channels = backbone_channels
        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=mid_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm2d(mid_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace),
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=num_outputs,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
            ),
        )

    def forward(self, x):
        x = self.last_layer(x)
        return x


class SegmentationHead_NoSigmoid_1x1(nn.Module):
    def __init__(
        self, backbone_channels, mid_channels=512, kernel_size=3, num_outputs=1, out_width=256, out_height=256
    ):
        super(SegmentationHead_NoSigmoid_1x1, self).__init__()
        last_inp_channels = backbone_channels
        self.width = out_width
        self.height = out_height
        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=mid_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm2d(mid_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace),
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=num_outputs,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

    def forward(self, x):
        x = self.last_layer(x)
        return x


class GaussianConv2D(nn.Module):
    """
    Gaussian smoothing + downsampling, applied independently per channel
    THIS IS NOT MEANT FOR USE ON MOBILE. MIGHT BE HORRIBLY SLOW
    """

    def __init__(self, channels, kernel_size, sigma, stride=1):
        super(GaussianConv2D, self).__init__()
        assert isinstance(kernel_size, int), "Specify kernel size as int. Both dimensions will get the same kernel size"
        assert isinstance(sigma, float), "Specify sigma as float. Anisotropic gaussian"

        kernel = torch.zeros(kernel_size, kernel_size)
        mean_loc = int((kernel_size - 1) / 2)  # Because 0 indexed
        kernel[mean_loc, mean_loc] = 1
        kernel = torch.from_numpy(ndimage.gaussian_filter(kernel.numpy(), sigma=sigma))

        # Make a dwise conv out of the kernel
        # Weights of shape out_channels, in_channels/groups, k, k
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        kernel = kernel.repeat(channels, 1, 1, 1)

        self.conv = F.conv2d
        # Register the kernel buffer instead of as a parameter, so that the training doesn't
        # happily update it
        self.register_buffer("weight", kernel)
        self.channels = channels
        self.stride = stride

    def forward(self, input):
        return self.conv(input, weight=self.weight, groups=self.channels, stride=self.stride)


def dice_loss(preds, targests):
    eps = 0.00001
    return 1 - (2 * (preds * targests).sum() + eps) / (torch.pow(preds, 2).sum() + torch.pow(targests, 2).sum() + eps)


# Main model class
class SemanticSegmentationModel(pl.LightningModule):
    def __init__(
        self,
        ffnet_head_type="A",
        num_classes=35,
        task="segmentation_A",
        use_adapter_convs=True,
        dropout_rate=None,
        **kwargs,
    ):
        super(SemanticSegmentationModel, self).__init__()
        self.encoder = models.efficientnet_b0()
        self.use_adapter_convs = use_adapter_convs
        self.ffnet_head_type = ffnet_head_type
        self.task = task
        self.head = FFNetUpHead(
            in_chans=[24, 40, 80, 112],
            use_adapter_conv=use_adapter_convs,
            head_type=ffnet_head_type,
            num_classes=num_classes,
            task=task,
            dropout_rate=dropout_rate,
        )

    def forward(self, x):
        out = [x]
        for layer in self.encoder.features[:6]:
            out.append(layer(out[-1]))
        features = [out[3], out[4], out[5], out[6]]
        return self.head(features)

    def init_model(self, pretrained_path=None, strict_loading=True, backbone_only=False):
        print(f"Initializing {self.model_name} weights")
        self.apply(model_weight_initializer)
        if pretrained_path:
            pretrained_dict = torch.load(pretrained_path, map_location={"cuda:0": "cpu"})
            if backbone_only:
                backbone_dict = {}
                for k, v in pretrained_dict.items():
                    if k.startswith("backbone_model"):
                        backbone_dict[k] = v
                self.load_state_dict(backbone_dict, strict=strict_loading)
            else:
                self.load_state_dict(pretrained_dict, strict=strict_loading)

    def _step(self, batch):
        x, y = batch
        batch_size, num_classes, width, height = y.shape
        out = self(x)  # the output is (35x64x64) so we need to iterpolate it to (35x256x256)
        out = torch.nn.functional.interpolate(
            out, size=(width, height), mode="bilinear", align_corners=True
        )  # in the FFNet repo this is exactly how they trasform the output.
        out = F.softmax(out, dim=1)
        loss = dice_loss(out, y)

        # pixel accuracy
        y_idx = y.argmax(dim=1)
        out_idx = out.argmax(dim=1)
        y_equal_pred = out_idx == y_idx
        pix_accuracy = y_equal_pred.sum() / (batch_size * width * height)

        # mean accuracy
        class_accuracies = []
        for cls in range(num_classes):
            mask = y_idx == cls
            if mask.sum() > 0:
                class_accuracies.append((y_equal_pred * mask).sum() / mask.sum())
        mean_accuracy = sum(class_accuracies) / len(class_accuracies)

        # IoU
        ious = []
        for cls in range(num_classes):
            pred_mask = out_idx == cls
            target_mask = y_idx == cls

            intersection = (pred_mask & target_mask).sum(dim=(1, 2))
            union = (pred_mask | target_mask).sum(dim=(1, 2))

            union = torch.where(union == 0, torch.tensor(1.0).to(union.device), union)

            iou = intersection.float() / union.float()
            ious.append(iou)
        iou = torch.stack(ious).mean().item()

        metrics = {
            "pixel_accuracy": pix_accuracy, 
            "mean_accuracy": mean_accuracy,
            "iou": iou,
        }
        return loss, metrics

    def training_step(self, batch, batch_idx):
        loss, metrics = self._step(batch)

        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/pixel_accuracy", metrics["pixel_accuracy"], prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/mean_accuracy", metrics["mean_accuracy"], prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/iou", metrics["iou"], prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics = self._step(batch)

        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("val/pixel_accuracy", metrics["pixel_accuracy"], prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("val/mean_accuracy", metrics["mean_accuracy"], prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("val/iou", metrics["iou"], prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
