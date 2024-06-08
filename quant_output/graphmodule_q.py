
import torch
import torch.nn
import torch.functional
import torch.nn.functional
import torch.quantization
import torch.nn.quantized


class QGraphModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.fake_quant_0 = torch.quantization.QuantStub()
        self.encoder_features_0_0 = torch.nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.encoder_features_0_2 = torch.nn.SiLU(inplace=True)
        self.encoder_features_1_0_block_0_0 = torch.nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32)
        self.encoder_features_1_0_block_0_2 = torch.nn.SiLU(inplace=True)
        self.encoder_features_1_0_block_1_avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.encoder_features_1_0_block_1_fc1 = torch.nn.Conv2d(32, 8, kernel_size=(1, 1), stride=(1, 1))
        self.encoder_features_1_0_block_1_activation = torch.nn.SiLU(inplace=True)
        self.encoder_features_1_0_block_1_fc2 = torch.nn.Conv2d(8, 32, kernel_size=(1, 1), stride=(1, 1))
        self.encoder_features_1_0_block_1_scale_activation = torch.nn.Sigmoid()
        self.encoder_features_1_0_block_2_0 = torch.nn.Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1))
        self.encoder_features_2_0_block_0_0 = torch.nn.Conv2d(16, 96, kernel_size=(1, 1), stride=(1, 1))
        self.encoder_features_2_0_block_0_2 = torch.nn.SiLU(inplace=True)
        self.encoder_features_2_0_block_1_0 = torch.nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=96)
        self.encoder_features_2_0_block_1_2 = torch.nn.SiLU(inplace=True)
        self.encoder_features_2_0_block_2_avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.encoder_features_2_0_block_2_fc1 = torch.nn.Conv2d(96, 4, kernel_size=(1, 1), stride=(1, 1))
        self.encoder_features_2_0_block_2_activation = torch.nn.SiLU(inplace=True)
        self.encoder_features_2_0_block_2_fc2 = torch.nn.Conv2d(4, 96, kernel_size=(1, 1), stride=(1, 1))
        self.encoder_features_2_0_block_2_scale_activation = torch.nn.Sigmoid()
        self.encoder_features_2_0_block_3_0 = torch.nn.Conv2d(96, 24, kernel_size=(1, 1), stride=(1, 1))
        self.encoder_features_2_1_block_0_0 = torch.nn.Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1))
        self.encoder_features_2_1_block_0_2 = torch.nn.SiLU(inplace=True)
        self.encoder_features_2_1_block_1_0 = torch.nn.Conv2d(144, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=144)
        self.encoder_features_2_1_block_1_2 = torch.nn.SiLU(inplace=True)
        self.encoder_features_2_1_block_2_avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.encoder_features_2_1_block_2_fc1 = torch.nn.Conv2d(144, 6, kernel_size=(1, 1), stride=(1, 1))
        self.encoder_features_2_1_block_2_activation = torch.nn.SiLU(inplace=True)
        self.encoder_features_2_1_block_2_fc2 = torch.nn.Conv2d(6, 144, kernel_size=(1, 1), stride=(1, 1))
        self.encoder_features_2_1_block_2_scale_activation = torch.nn.Sigmoid()
        self.encoder_features_2_1_block_3_0 = torch.nn.Conv2d(144, 24, kernel_size=(1, 1), stride=(1, 1))
        self.encoder_features_3_0_block_0_0 = torch.nn.Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1))
        self.encoder_features_3_0_block_0_2 = torch.nn.SiLU(inplace=True)
        self.encoder_features_3_0_block_1_0 = torch.nn.Conv2d(144, 144, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=144)
        self.encoder_features_3_0_block_1_2 = torch.nn.SiLU(inplace=True)
        self.encoder_features_3_0_block_2_avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.encoder_features_3_0_block_2_fc1 = torch.nn.Conv2d(144, 6, kernel_size=(1, 1), stride=(1, 1))
        self.encoder_features_3_0_block_2_activation = torch.nn.SiLU(inplace=True)
        self.encoder_features_3_0_block_2_fc2 = torch.nn.Conv2d(6, 144, kernel_size=(1, 1), stride=(1, 1))
        self.encoder_features_3_0_block_2_scale_activation = torch.nn.Sigmoid()
        self.encoder_features_3_0_block_3_0 = torch.nn.Conv2d(144, 40, kernel_size=(1, 1), stride=(1, 1))
        self.encoder_features_3_1_block_0_0 = torch.nn.Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1))
        self.encoder_features_3_1_block_0_2 = torch.nn.SiLU(inplace=True)
        self.encoder_features_3_1_block_1_0 = torch.nn.Conv2d(240, 240, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=240)
        self.encoder_features_3_1_block_1_2 = torch.nn.SiLU(inplace=True)
        self.encoder_features_3_1_block_2_avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.encoder_features_3_1_block_2_fc1 = torch.nn.Conv2d(240, 10, kernel_size=(1, 1), stride=(1, 1))
        self.encoder_features_3_1_block_2_activation = torch.nn.SiLU(inplace=True)
        self.encoder_features_3_1_block_2_fc2 = torch.nn.Conv2d(10, 240, kernel_size=(1, 1), stride=(1, 1))
        self.encoder_features_3_1_block_2_scale_activation = torch.nn.Sigmoid()
        self.encoder_features_3_1_block_3_0 = torch.nn.Conv2d(240, 40, kernel_size=(1, 1), stride=(1, 1))
        self.encoder_features_4_0_block_0_0 = torch.nn.Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1))
        self.encoder_features_4_0_block_0_2 = torch.nn.SiLU(inplace=True)
        self.encoder_features_4_0_block_1_0 = torch.nn.Conv2d(240, 240, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=240)
        self.encoder_features_4_0_block_1_2 = torch.nn.SiLU(inplace=True)
        self.encoder_features_4_0_block_2_avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.encoder_features_4_0_block_2_fc1 = torch.nn.Conv2d(240, 10, kernel_size=(1, 1), stride=(1, 1))
        self.encoder_features_4_0_block_2_activation = torch.nn.SiLU(inplace=True)
        self.encoder_features_4_0_block_2_fc2 = torch.nn.Conv2d(10, 240, kernel_size=(1, 1), stride=(1, 1))
        self.encoder_features_4_0_block_2_scale_activation = torch.nn.Sigmoid()
        self.encoder_features_4_0_block_3_0 = torch.nn.Conv2d(240, 80, kernel_size=(1, 1), stride=(1, 1))
        self.encoder_features_4_1_block_0_0 = torch.nn.Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1))
        self.encoder_features_4_1_block_0_2 = torch.nn.SiLU(inplace=True)
        self.encoder_features_4_1_block_1_0 = torch.nn.Conv2d(480, 480, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=480)
        self.encoder_features_4_1_block_1_2 = torch.nn.SiLU(inplace=True)
        self.encoder_features_4_1_block_2_avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.encoder_features_4_1_block_2_fc1 = torch.nn.Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
        self.encoder_features_4_1_block_2_activation = torch.nn.SiLU(inplace=True)
        self.encoder_features_4_1_block_2_fc2 = torch.nn.Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
        self.encoder_features_4_1_block_2_scale_activation = torch.nn.Sigmoid()
        self.encoder_features_4_1_block_3_0 = torch.nn.Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1))
        self.encoder_features_4_2_block_0_0 = torch.nn.Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1))
        self.encoder_features_4_2_block_0_2 = torch.nn.SiLU(inplace=True)
        self.encoder_features_4_2_block_1_0 = torch.nn.Conv2d(480, 480, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=480)
        self.encoder_features_4_2_block_1_2 = torch.nn.SiLU(inplace=True)
        self.encoder_features_4_2_block_2_avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.encoder_features_4_2_block_2_fc1 = torch.nn.Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
        self.encoder_features_4_2_block_2_activation = torch.nn.SiLU(inplace=True)
        self.encoder_features_4_2_block_2_fc2 = torch.nn.Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
        self.encoder_features_4_2_block_2_scale_activation = torch.nn.Sigmoid()
        self.encoder_features_4_2_block_3_0 = torch.nn.Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1))
        self.encoder_features_5_0_block_0_0 = torch.nn.Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1))
        self.encoder_features_5_0_block_0_2 = torch.nn.SiLU(inplace=True)
        self.encoder_features_5_0_block_1_0 = torch.nn.Conv2d(480, 480, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=480)
        self.encoder_features_5_0_block_1_2 = torch.nn.SiLU(inplace=True)
        self.encoder_features_5_0_block_2_avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.encoder_features_5_0_block_2_fc1 = torch.nn.Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
        self.encoder_features_5_0_block_2_activation = torch.nn.SiLU(inplace=True)
        self.encoder_features_5_0_block_2_fc2 = torch.nn.Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
        self.encoder_features_5_0_block_2_scale_activation = torch.nn.Sigmoid()
        self.encoder_features_5_0_block_3_0 = torch.nn.Conv2d(480, 112, kernel_size=(1, 1), stride=(1, 1))
        self.encoder_features_5_1_block_0_0 = torch.nn.Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1))
        self.encoder_features_5_1_block_0_2 = torch.nn.SiLU(inplace=True)
        self.encoder_features_5_1_block_1_0 = torch.nn.Conv2d(672, 672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=672)
        self.encoder_features_5_1_block_1_2 = torch.nn.SiLU(inplace=True)
        self.encoder_features_5_1_block_2_avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.encoder_features_5_1_block_2_fc1 = torch.nn.Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))
        self.encoder_features_5_1_block_2_activation = torch.nn.SiLU(inplace=True)
        self.encoder_features_5_1_block_2_fc2 = torch.nn.Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))
        self.encoder_features_5_1_block_2_scale_activation = torch.nn.Sigmoid()
        self.encoder_features_5_1_block_3_0 = torch.nn.Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1))
        self.encoder_features_5_2_block_0_0 = torch.nn.Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1))
        self.encoder_features_5_2_block_0_2 = torch.nn.SiLU(inplace=True)
        self.encoder_features_5_2_block_1_0 = torch.nn.Conv2d(672, 672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=672)
        self.encoder_features_5_2_block_1_2 = torch.nn.SiLU(inplace=True)
        self.encoder_features_5_2_block_2_avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.encoder_features_5_2_block_2_fc1 = torch.nn.Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))
        self.encoder_features_5_2_block_2_activation = torch.nn.SiLU(inplace=True)
        self.encoder_features_5_2_block_2_fc2 = torch.nn.Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))
        self.encoder_features_5_2_block_2_scale_activation = torch.nn.Sigmoid()
        self.encoder_features_5_2_block_3_0 = torch.nn.Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1))
        self.head_layers_0_adapter_conv_0_layers_0_0 = torch.nn.Conv2d(24, 64, kernel_size=(1, 1), stride=(1, 1))
        self.head_layers_0_adapter_conv_0_layers_0_1 = torch.nn.ReLU(inplace=True)
        self.head_layers_0_adapter_conv_1_layers_0_0 = torch.nn.Conv2d(40, 128, kernel_size=(1, 1), stride=(1, 1))
        self.head_layers_0_adapter_conv_1_layers_0_1 = torch.nn.ReLU(inplace=True)
        self.head_layers_0_adapter_conv_2_layers_0_0 = torch.nn.Conv2d(80, 256, kernel_size=(1, 1), stride=(1, 1))
        self.head_layers_0_adapter_conv_2_layers_0_1 = torch.nn.ReLU(inplace=True)
        self.head_layers_0_adapter_conv_3_layers_0_0 = torch.nn.Conv2d(112, 512, kernel_size=(1, 1), stride=(1, 1))
        self.head_layers_0_adapter_conv_3_layers_0_1 = torch.nn.ReLU(inplace=True)
        self.head_layers_1_fam_32_sm_layers_0_0 = torch.nn.Conv2d(512, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.head_layers_1_fam_32_sm_layers_0_1 = torch.nn.ReLU(inplace=True)
        self.head_layers_1_fam_32_up_layers_0_0 = torch.nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
        self.head_layers_1_fam_32_up_layers_0_1 = torch.nn.ReLU(inplace=True)
        self.head_layers_1_fam_16_sm_layers_0_0 = torch.nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.head_layers_1_fam_16_sm_layers_0_1 = torch.nn.ReLU(inplace=True)
        self.head_layers_1_fam_16_up_layers_0_0 = torch.nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
        self.head_layers_1_fam_16_up_layers_0_1 = torch.nn.ReLU(inplace=True)
        self.head_layers_1_fam_8_sm_layers_0_0 = torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.head_layers_1_fam_8_sm_layers_0_1 = torch.nn.ReLU(inplace=True)
        self.head_layers_1_fam_8_up_layers_0_0 = torch.nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
        self.head_layers_1_fam_8_up_layers_0_1 = torch.nn.ReLU(inplace=True)
        self.head_layers_1_fam_4_layers_0_0 = torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.head_layers_1_fam_4_layers_0_1 = torch.nn.ReLU(inplace=True)
        self.head_layers_3_last_layer_0_0 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.head_layers_3_last_layer_0_1 = torch.nn.ReLU(inplace=True)
        self.head_layers_3_last_layer_3 = torch.nn.Conv2d(512, 35, kernel_size=(1, 1), stride=(1, 1))
        self.fake_dequant_0 = torch.quantization.DeQuantStub()
        self.float_functional_simple_0 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_1 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_2 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_3 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_4 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_5 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_6 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_7 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_8 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_9 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_10 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_11 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_12 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_13 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_14 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_15 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_16 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_17 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_18 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_19 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_20 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_21 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_22 = torch.ao.nn.quantized.FloatFunctional()

    def forward(self, input_0_f):
        fake_quant_0 = self.fake_quant_0(input_0_f)
        input_0_f = None
        encoder_features_0_0 = self.encoder_features_0_0(fake_quant_0)
        fake_quant_0 = None
        encoder_features_0_2 = self.encoder_features_0_2(encoder_features_0_0)
        encoder_features_0_0 = None
        encoder_features_1_0_block_0_0 = self.encoder_features_1_0_block_0_0(encoder_features_0_2)
        encoder_features_0_2 = None
        encoder_features_1_0_block_0_2 = self.encoder_features_1_0_block_0_2(encoder_features_1_0_block_0_0)
        encoder_features_1_0_block_0_0 = None
        encoder_features_1_0_block_1_avgpool = self.encoder_features_1_0_block_1_avgpool(encoder_features_1_0_block_0_2)
        encoder_features_1_0_block_1_fc1 = self.encoder_features_1_0_block_1_fc1(encoder_features_1_0_block_1_avgpool)
        encoder_features_1_0_block_1_avgpool = None
        encoder_features_1_0_block_1_activation = self.encoder_features_1_0_block_1_activation(encoder_features_1_0_block_1_fc1)
        encoder_features_1_0_block_1_fc1 = None
        encoder_features_1_0_block_1_fc2 = self.encoder_features_1_0_block_1_fc2(encoder_features_1_0_block_1_activation)
        encoder_features_1_0_block_1_activation = None
        encoder_features_1_0_block_1_scale_activation = self.encoder_features_1_0_block_1_scale_activation(encoder_features_1_0_block_1_fc2)
        encoder_features_1_0_block_1_fc2 = None
        fuse_expand_as_0_f = encoder_features_1_0_block_1_scale_activation.expand_as(encoder_features_1_0_block_0_2)
        encoder_features_1_0_block_1_scale_activation = None
        mul_0_f = self.float_functional_simple_0.mul(fuse_expand_as_0_f, encoder_features_1_0_block_0_2)
        fuse_expand_as_0_f = None
        encoder_features_1_0_block_0_2 = None
        encoder_features_1_0_block_2_0 = self.encoder_features_1_0_block_2_0(mul_0_f)
        mul_0_f = None
        encoder_features_2_0_block_0_0 = self.encoder_features_2_0_block_0_0(encoder_features_1_0_block_2_0)
        encoder_features_1_0_block_2_0 = None
        encoder_features_2_0_block_0_2 = self.encoder_features_2_0_block_0_2(encoder_features_2_0_block_0_0)
        encoder_features_2_0_block_0_0 = None
        encoder_features_2_0_block_1_0 = self.encoder_features_2_0_block_1_0(encoder_features_2_0_block_0_2)
        encoder_features_2_0_block_0_2 = None
        encoder_features_2_0_block_1_2 = self.encoder_features_2_0_block_1_2(encoder_features_2_0_block_1_0)
        encoder_features_2_0_block_1_0 = None
        encoder_features_2_0_block_2_avgpool = self.encoder_features_2_0_block_2_avgpool(encoder_features_2_0_block_1_2)
        encoder_features_2_0_block_2_fc1 = self.encoder_features_2_0_block_2_fc1(encoder_features_2_0_block_2_avgpool)
        encoder_features_2_0_block_2_avgpool = None
        encoder_features_2_0_block_2_activation = self.encoder_features_2_0_block_2_activation(encoder_features_2_0_block_2_fc1)
        encoder_features_2_0_block_2_fc1 = None
        encoder_features_2_0_block_2_fc2 = self.encoder_features_2_0_block_2_fc2(encoder_features_2_0_block_2_activation)
        encoder_features_2_0_block_2_activation = None
        encoder_features_2_0_block_2_scale_activation = self.encoder_features_2_0_block_2_scale_activation(encoder_features_2_0_block_2_fc2)
        encoder_features_2_0_block_2_fc2 = None
        fuse_expand_as_1_f = encoder_features_2_0_block_2_scale_activation.expand_as(encoder_features_2_0_block_1_2)
        encoder_features_2_0_block_2_scale_activation = None
        mul_1_f = self.float_functional_simple_1.mul(fuse_expand_as_1_f, encoder_features_2_0_block_1_2)
        fuse_expand_as_1_f = None
        encoder_features_2_0_block_1_2 = None
        encoder_features_2_0_block_3_0 = self.encoder_features_2_0_block_3_0(mul_1_f)
        mul_1_f = None
        encoder_features_2_1_block_0_0 = self.encoder_features_2_1_block_0_0(encoder_features_2_0_block_3_0)
        encoder_features_2_1_block_0_2 = self.encoder_features_2_1_block_0_2(encoder_features_2_1_block_0_0)
        encoder_features_2_1_block_0_0 = None
        encoder_features_2_1_block_1_0 = self.encoder_features_2_1_block_1_0(encoder_features_2_1_block_0_2)
        encoder_features_2_1_block_0_2 = None
        encoder_features_2_1_block_1_2 = self.encoder_features_2_1_block_1_2(encoder_features_2_1_block_1_0)
        encoder_features_2_1_block_1_0 = None
        encoder_features_2_1_block_2_avgpool = self.encoder_features_2_1_block_2_avgpool(encoder_features_2_1_block_1_2)
        encoder_features_2_1_block_2_fc1 = self.encoder_features_2_1_block_2_fc1(encoder_features_2_1_block_2_avgpool)
        encoder_features_2_1_block_2_avgpool = None
        encoder_features_2_1_block_2_activation = self.encoder_features_2_1_block_2_activation(encoder_features_2_1_block_2_fc1)
        encoder_features_2_1_block_2_fc1 = None
        encoder_features_2_1_block_2_fc2 = self.encoder_features_2_1_block_2_fc2(encoder_features_2_1_block_2_activation)
        encoder_features_2_1_block_2_activation = None
        encoder_features_2_1_block_2_scale_activation = self.encoder_features_2_1_block_2_scale_activation(encoder_features_2_1_block_2_fc2)
        encoder_features_2_1_block_2_fc2 = None
        fuse_expand_as_2_f = encoder_features_2_1_block_2_scale_activation.expand_as(encoder_features_2_1_block_1_2)
        encoder_features_2_1_block_2_scale_activation = None
        mul_2_f = self.float_functional_simple_2.mul(fuse_expand_as_2_f, encoder_features_2_1_block_1_2)
        fuse_expand_as_2_f = None
        encoder_features_2_1_block_1_2 = None
        encoder_features_2_1_block_3_0 = self.encoder_features_2_1_block_3_0(mul_2_f)
        mul_2_f = None
        add_0_f = self.float_functional_simple_3.add(encoder_features_2_1_block_3_0, encoder_features_2_0_block_3_0)
        encoder_features_2_1_block_3_0 = None
        encoder_features_2_0_block_3_0 = None
        encoder_features_3_0_block_0_0 = self.encoder_features_3_0_block_0_0(add_0_f)
        encoder_features_3_0_block_0_2 = self.encoder_features_3_0_block_0_2(encoder_features_3_0_block_0_0)
        encoder_features_3_0_block_0_0 = None
        encoder_features_3_0_block_1_0 = self.encoder_features_3_0_block_1_0(encoder_features_3_0_block_0_2)
        encoder_features_3_0_block_0_2 = None
        encoder_features_3_0_block_1_2 = self.encoder_features_3_0_block_1_2(encoder_features_3_0_block_1_0)
        encoder_features_3_0_block_1_0 = None
        encoder_features_3_0_block_2_avgpool = self.encoder_features_3_0_block_2_avgpool(encoder_features_3_0_block_1_2)
        encoder_features_3_0_block_2_fc1 = self.encoder_features_3_0_block_2_fc1(encoder_features_3_0_block_2_avgpool)
        encoder_features_3_0_block_2_avgpool = None
        encoder_features_3_0_block_2_activation = self.encoder_features_3_0_block_2_activation(encoder_features_3_0_block_2_fc1)
        encoder_features_3_0_block_2_fc1 = None
        encoder_features_3_0_block_2_fc2 = self.encoder_features_3_0_block_2_fc2(encoder_features_3_0_block_2_activation)
        encoder_features_3_0_block_2_activation = None
        encoder_features_3_0_block_2_scale_activation = self.encoder_features_3_0_block_2_scale_activation(encoder_features_3_0_block_2_fc2)
        encoder_features_3_0_block_2_fc2 = None
        fuse_expand_as_3_f = encoder_features_3_0_block_2_scale_activation.expand_as(encoder_features_3_0_block_1_2)
        encoder_features_3_0_block_2_scale_activation = None
        mul_3_f = self.float_functional_simple_4.mul(fuse_expand_as_3_f, encoder_features_3_0_block_1_2)
        fuse_expand_as_3_f = None
        encoder_features_3_0_block_1_2 = None
        encoder_features_3_0_block_3_0 = self.encoder_features_3_0_block_3_0(mul_3_f)
        mul_3_f = None
        encoder_features_3_1_block_0_0 = self.encoder_features_3_1_block_0_0(encoder_features_3_0_block_3_0)
        encoder_features_3_1_block_0_2 = self.encoder_features_3_1_block_0_2(encoder_features_3_1_block_0_0)
        encoder_features_3_1_block_0_0 = None
        encoder_features_3_1_block_1_0 = self.encoder_features_3_1_block_1_0(encoder_features_3_1_block_0_2)
        encoder_features_3_1_block_0_2 = None
        encoder_features_3_1_block_1_2 = self.encoder_features_3_1_block_1_2(encoder_features_3_1_block_1_0)
        encoder_features_3_1_block_1_0 = None
        encoder_features_3_1_block_2_avgpool = self.encoder_features_3_1_block_2_avgpool(encoder_features_3_1_block_1_2)
        encoder_features_3_1_block_2_fc1 = self.encoder_features_3_1_block_2_fc1(encoder_features_3_1_block_2_avgpool)
        encoder_features_3_1_block_2_avgpool = None
        encoder_features_3_1_block_2_activation = self.encoder_features_3_1_block_2_activation(encoder_features_3_1_block_2_fc1)
        encoder_features_3_1_block_2_fc1 = None
        encoder_features_3_1_block_2_fc2 = self.encoder_features_3_1_block_2_fc2(encoder_features_3_1_block_2_activation)
        encoder_features_3_1_block_2_activation = None
        encoder_features_3_1_block_2_scale_activation = self.encoder_features_3_1_block_2_scale_activation(encoder_features_3_1_block_2_fc2)
        encoder_features_3_1_block_2_fc2 = None
        fuse_expand_as_4_f = encoder_features_3_1_block_2_scale_activation.expand_as(encoder_features_3_1_block_1_2)
        encoder_features_3_1_block_2_scale_activation = None
        mul_4_f = self.float_functional_simple_5.mul(fuse_expand_as_4_f, encoder_features_3_1_block_1_2)
        fuse_expand_as_4_f = None
        encoder_features_3_1_block_1_2 = None
        encoder_features_3_1_block_3_0 = self.encoder_features_3_1_block_3_0(mul_4_f)
        mul_4_f = None
        add_1_f = self.float_functional_simple_6.add(encoder_features_3_1_block_3_0, encoder_features_3_0_block_3_0)
        encoder_features_3_1_block_3_0 = None
        encoder_features_3_0_block_3_0 = None
        encoder_features_4_0_block_0_0 = self.encoder_features_4_0_block_0_0(add_1_f)
        encoder_features_4_0_block_0_2 = self.encoder_features_4_0_block_0_2(encoder_features_4_0_block_0_0)
        encoder_features_4_0_block_0_0 = None
        encoder_features_4_0_block_1_0 = self.encoder_features_4_0_block_1_0(encoder_features_4_0_block_0_2)
        encoder_features_4_0_block_0_2 = None
        encoder_features_4_0_block_1_2 = self.encoder_features_4_0_block_1_2(encoder_features_4_0_block_1_0)
        encoder_features_4_0_block_1_0 = None
        encoder_features_4_0_block_2_avgpool = self.encoder_features_4_0_block_2_avgpool(encoder_features_4_0_block_1_2)
        encoder_features_4_0_block_2_fc1 = self.encoder_features_4_0_block_2_fc1(encoder_features_4_0_block_2_avgpool)
        encoder_features_4_0_block_2_avgpool = None
        encoder_features_4_0_block_2_activation = self.encoder_features_4_0_block_2_activation(encoder_features_4_0_block_2_fc1)
        encoder_features_4_0_block_2_fc1 = None
        encoder_features_4_0_block_2_fc2 = self.encoder_features_4_0_block_2_fc2(encoder_features_4_0_block_2_activation)
        encoder_features_4_0_block_2_activation = None
        encoder_features_4_0_block_2_scale_activation = self.encoder_features_4_0_block_2_scale_activation(encoder_features_4_0_block_2_fc2)
        encoder_features_4_0_block_2_fc2 = None
        fuse_expand_as_5_f = encoder_features_4_0_block_2_scale_activation.expand_as(encoder_features_4_0_block_1_2)
        encoder_features_4_0_block_2_scale_activation = None
        mul_5_f = self.float_functional_simple_7.mul(fuse_expand_as_5_f, encoder_features_4_0_block_1_2)
        fuse_expand_as_5_f = None
        encoder_features_4_0_block_1_2 = None
        encoder_features_4_0_block_3_0 = self.encoder_features_4_0_block_3_0(mul_5_f)
        mul_5_f = None
        encoder_features_4_1_block_0_0 = self.encoder_features_4_1_block_0_0(encoder_features_4_0_block_3_0)
        encoder_features_4_1_block_0_2 = self.encoder_features_4_1_block_0_2(encoder_features_4_1_block_0_0)
        encoder_features_4_1_block_0_0 = None
        encoder_features_4_1_block_1_0 = self.encoder_features_4_1_block_1_0(encoder_features_4_1_block_0_2)
        encoder_features_4_1_block_0_2 = None
        encoder_features_4_1_block_1_2 = self.encoder_features_4_1_block_1_2(encoder_features_4_1_block_1_0)
        encoder_features_4_1_block_1_0 = None
        encoder_features_4_1_block_2_avgpool = self.encoder_features_4_1_block_2_avgpool(encoder_features_4_1_block_1_2)
        encoder_features_4_1_block_2_fc1 = self.encoder_features_4_1_block_2_fc1(encoder_features_4_1_block_2_avgpool)
        encoder_features_4_1_block_2_avgpool = None
        encoder_features_4_1_block_2_activation = self.encoder_features_4_1_block_2_activation(encoder_features_4_1_block_2_fc1)
        encoder_features_4_1_block_2_fc1 = None
        encoder_features_4_1_block_2_fc2 = self.encoder_features_4_1_block_2_fc2(encoder_features_4_1_block_2_activation)
        encoder_features_4_1_block_2_activation = None
        encoder_features_4_1_block_2_scale_activation = self.encoder_features_4_1_block_2_scale_activation(encoder_features_4_1_block_2_fc2)
        encoder_features_4_1_block_2_fc2 = None
        fuse_expand_as_6_f = encoder_features_4_1_block_2_scale_activation.expand_as(encoder_features_4_1_block_1_2)
        encoder_features_4_1_block_2_scale_activation = None
        mul_6_f = self.float_functional_simple_8.mul(fuse_expand_as_6_f, encoder_features_4_1_block_1_2)
        fuse_expand_as_6_f = None
        encoder_features_4_1_block_1_2 = None
        encoder_features_4_1_block_3_0 = self.encoder_features_4_1_block_3_0(mul_6_f)
        mul_6_f = None
        add_2_f = self.float_functional_simple_9.add(encoder_features_4_1_block_3_0, encoder_features_4_0_block_3_0)
        encoder_features_4_1_block_3_0 = None
        encoder_features_4_0_block_3_0 = None
        encoder_features_4_2_block_0_0 = self.encoder_features_4_2_block_0_0(add_2_f)
        encoder_features_4_2_block_0_2 = self.encoder_features_4_2_block_0_2(encoder_features_4_2_block_0_0)
        encoder_features_4_2_block_0_0 = None
        encoder_features_4_2_block_1_0 = self.encoder_features_4_2_block_1_0(encoder_features_4_2_block_0_2)
        encoder_features_4_2_block_0_2 = None
        encoder_features_4_2_block_1_2 = self.encoder_features_4_2_block_1_2(encoder_features_4_2_block_1_0)
        encoder_features_4_2_block_1_0 = None
        encoder_features_4_2_block_2_avgpool = self.encoder_features_4_2_block_2_avgpool(encoder_features_4_2_block_1_2)
        encoder_features_4_2_block_2_fc1 = self.encoder_features_4_2_block_2_fc1(encoder_features_4_2_block_2_avgpool)
        encoder_features_4_2_block_2_avgpool = None
        encoder_features_4_2_block_2_activation = self.encoder_features_4_2_block_2_activation(encoder_features_4_2_block_2_fc1)
        encoder_features_4_2_block_2_fc1 = None
        encoder_features_4_2_block_2_fc2 = self.encoder_features_4_2_block_2_fc2(encoder_features_4_2_block_2_activation)
        encoder_features_4_2_block_2_activation = None
        encoder_features_4_2_block_2_scale_activation = self.encoder_features_4_2_block_2_scale_activation(encoder_features_4_2_block_2_fc2)
        encoder_features_4_2_block_2_fc2 = None
        fuse_expand_as_7_f = encoder_features_4_2_block_2_scale_activation.expand_as(encoder_features_4_2_block_1_2)
        encoder_features_4_2_block_2_scale_activation = None
        mul_7_f = self.float_functional_simple_10.mul(fuse_expand_as_7_f, encoder_features_4_2_block_1_2)
        fuse_expand_as_7_f = None
        encoder_features_4_2_block_1_2 = None
        encoder_features_4_2_block_3_0 = self.encoder_features_4_2_block_3_0(mul_7_f)
        mul_7_f = None
        add_3_f = self.float_functional_simple_11.add(encoder_features_4_2_block_3_0, add_2_f)
        encoder_features_4_2_block_3_0 = None
        add_2_f = None
        encoder_features_5_0_block_0_0 = self.encoder_features_5_0_block_0_0(add_3_f)
        encoder_features_5_0_block_0_2 = self.encoder_features_5_0_block_0_2(encoder_features_5_0_block_0_0)
        encoder_features_5_0_block_0_0 = None
        encoder_features_5_0_block_1_0 = self.encoder_features_5_0_block_1_0(encoder_features_5_0_block_0_2)
        encoder_features_5_0_block_0_2 = None
        encoder_features_5_0_block_1_2 = self.encoder_features_5_0_block_1_2(encoder_features_5_0_block_1_0)
        encoder_features_5_0_block_1_0 = None
        encoder_features_5_0_block_2_avgpool = self.encoder_features_5_0_block_2_avgpool(encoder_features_5_0_block_1_2)
        encoder_features_5_0_block_2_fc1 = self.encoder_features_5_0_block_2_fc1(encoder_features_5_0_block_2_avgpool)
        encoder_features_5_0_block_2_avgpool = None
        encoder_features_5_0_block_2_activation = self.encoder_features_5_0_block_2_activation(encoder_features_5_0_block_2_fc1)
        encoder_features_5_0_block_2_fc1 = None
        encoder_features_5_0_block_2_fc2 = self.encoder_features_5_0_block_2_fc2(encoder_features_5_0_block_2_activation)
        encoder_features_5_0_block_2_activation = None
        encoder_features_5_0_block_2_scale_activation = self.encoder_features_5_0_block_2_scale_activation(encoder_features_5_0_block_2_fc2)
        encoder_features_5_0_block_2_fc2 = None
        fuse_expand_as_8_f = encoder_features_5_0_block_2_scale_activation.expand_as(encoder_features_5_0_block_1_2)
        encoder_features_5_0_block_2_scale_activation = None
        mul_8_f = self.float_functional_simple_12.mul(fuse_expand_as_8_f, encoder_features_5_0_block_1_2)
        fuse_expand_as_8_f = None
        encoder_features_5_0_block_1_2 = None
        encoder_features_5_0_block_3_0 = self.encoder_features_5_0_block_3_0(mul_8_f)
        mul_8_f = None
        encoder_features_5_1_block_0_0 = self.encoder_features_5_1_block_0_0(encoder_features_5_0_block_3_0)
        encoder_features_5_1_block_0_2 = self.encoder_features_5_1_block_0_2(encoder_features_5_1_block_0_0)
        encoder_features_5_1_block_0_0 = None
        encoder_features_5_1_block_1_0 = self.encoder_features_5_1_block_1_0(encoder_features_5_1_block_0_2)
        encoder_features_5_1_block_0_2 = None
        encoder_features_5_1_block_1_2 = self.encoder_features_5_1_block_1_2(encoder_features_5_1_block_1_0)
        encoder_features_5_1_block_1_0 = None
        encoder_features_5_1_block_2_avgpool = self.encoder_features_5_1_block_2_avgpool(encoder_features_5_1_block_1_2)
        encoder_features_5_1_block_2_fc1 = self.encoder_features_5_1_block_2_fc1(encoder_features_5_1_block_2_avgpool)
        encoder_features_5_1_block_2_avgpool = None
        encoder_features_5_1_block_2_activation = self.encoder_features_5_1_block_2_activation(encoder_features_5_1_block_2_fc1)
        encoder_features_5_1_block_2_fc1 = None
        encoder_features_5_1_block_2_fc2 = self.encoder_features_5_1_block_2_fc2(encoder_features_5_1_block_2_activation)
        encoder_features_5_1_block_2_activation = None
        encoder_features_5_1_block_2_scale_activation = self.encoder_features_5_1_block_2_scale_activation(encoder_features_5_1_block_2_fc2)
        encoder_features_5_1_block_2_fc2 = None
        fuse_expand_as_9_f = encoder_features_5_1_block_2_scale_activation.expand_as(encoder_features_5_1_block_1_2)
        encoder_features_5_1_block_2_scale_activation = None
        mul_9_f = self.float_functional_simple_13.mul(fuse_expand_as_9_f, encoder_features_5_1_block_1_2)
        fuse_expand_as_9_f = None
        encoder_features_5_1_block_1_2 = None
        encoder_features_5_1_block_3_0 = self.encoder_features_5_1_block_3_0(mul_9_f)
        mul_9_f = None
        add_4_f = self.float_functional_simple_14.add(encoder_features_5_1_block_3_0, encoder_features_5_0_block_3_0)
        encoder_features_5_1_block_3_0 = None
        encoder_features_5_0_block_3_0 = None
        encoder_features_5_2_block_0_0 = self.encoder_features_5_2_block_0_0(add_4_f)
        encoder_features_5_2_block_0_2 = self.encoder_features_5_2_block_0_2(encoder_features_5_2_block_0_0)
        encoder_features_5_2_block_0_0 = None
        encoder_features_5_2_block_1_0 = self.encoder_features_5_2_block_1_0(encoder_features_5_2_block_0_2)
        encoder_features_5_2_block_0_2 = None
        encoder_features_5_2_block_1_2 = self.encoder_features_5_2_block_1_2(encoder_features_5_2_block_1_0)
        encoder_features_5_2_block_1_0 = None
        encoder_features_5_2_block_2_avgpool = self.encoder_features_5_2_block_2_avgpool(encoder_features_5_2_block_1_2)
        encoder_features_5_2_block_2_fc1 = self.encoder_features_5_2_block_2_fc1(encoder_features_5_2_block_2_avgpool)
        encoder_features_5_2_block_2_avgpool = None
        encoder_features_5_2_block_2_activation = self.encoder_features_5_2_block_2_activation(encoder_features_5_2_block_2_fc1)
        encoder_features_5_2_block_2_fc1 = None
        encoder_features_5_2_block_2_fc2 = self.encoder_features_5_2_block_2_fc2(encoder_features_5_2_block_2_activation)
        encoder_features_5_2_block_2_activation = None
        encoder_features_5_2_block_2_scale_activation = self.encoder_features_5_2_block_2_scale_activation(encoder_features_5_2_block_2_fc2)
        encoder_features_5_2_block_2_fc2 = None
        fuse_expand_as_10_f = encoder_features_5_2_block_2_scale_activation.expand_as(encoder_features_5_2_block_1_2)
        encoder_features_5_2_block_2_scale_activation = None
        mul_10_f = self.float_functional_simple_15.mul(fuse_expand_as_10_f, encoder_features_5_2_block_1_2)
        fuse_expand_as_10_f = None
        encoder_features_5_2_block_1_2 = None
        encoder_features_5_2_block_3_0 = self.encoder_features_5_2_block_3_0(mul_10_f)
        mul_10_f = None
        add_5_f = self.float_functional_simple_16.add(encoder_features_5_2_block_3_0, add_4_f)
        encoder_features_5_2_block_3_0 = None
        add_4_f = None
        head_layers_0_adapter_conv_0_layers_0_0 = self.head_layers_0_adapter_conv_0_layers_0_0(add_0_f)
        add_0_f = None
        head_layers_0_adapter_conv_0_layers_0_1 = self.head_layers_0_adapter_conv_0_layers_0_1(head_layers_0_adapter_conv_0_layers_0_0)
        head_layers_0_adapter_conv_0_layers_0_0 = None
        head_layers_0_adapter_conv_1_layers_0_0 = self.head_layers_0_adapter_conv_1_layers_0_0(add_1_f)
        add_1_f = None
        head_layers_0_adapter_conv_1_layers_0_1 = self.head_layers_0_adapter_conv_1_layers_0_1(head_layers_0_adapter_conv_1_layers_0_0)
        head_layers_0_adapter_conv_1_layers_0_0 = None
        head_layers_0_adapter_conv_2_layers_0_0 = self.head_layers_0_adapter_conv_2_layers_0_0(add_3_f)
        add_3_f = None
        head_layers_0_adapter_conv_2_layers_0_1 = self.head_layers_0_adapter_conv_2_layers_0_1(head_layers_0_adapter_conv_2_layers_0_0)
        head_layers_0_adapter_conv_2_layers_0_0 = None
        head_layers_0_adapter_conv_3_layers_0_0 = self.head_layers_0_adapter_conv_3_layers_0_0(add_5_f)
        add_5_f = None
        head_layers_0_adapter_conv_3_layers_0_1 = self.head_layers_0_adapter_conv_3_layers_0_1(head_layers_0_adapter_conv_3_layers_0_0)
        head_layers_0_adapter_conv_3_layers_0_0 = None
        head_layers_1_fam_32_sm_layers_0_0 = self.head_layers_1_fam_32_sm_layers_0_0(head_layers_0_adapter_conv_3_layers_0_1)
        head_layers_1_fam_32_sm_layers_0_1 = self.head_layers_1_fam_32_sm_layers_0_1(head_layers_1_fam_32_sm_layers_0_0)
        head_layers_1_fam_32_sm_layers_0_0 = None
        head_layers_1_fam_32_up_layers_0_0 = self.head_layers_1_fam_32_up_layers_0_0(head_layers_0_adapter_conv_3_layers_0_1)
        head_layers_0_adapter_conv_3_layers_0_1 = None
        head_layers_1_fam_32_up_layers_0_1 = self.head_layers_1_fam_32_up_layers_0_1(head_layers_1_fam_32_up_layers_0_0)
        head_layers_1_fam_32_up_layers_0_0 = None
        size_0_f = head_layers_0_adapter_conv_2_layers_0_1.size()
        interpolate_0_f = torch.nn.functional.interpolate(head_layers_1_fam_32_up_layers_0_1, size=[size_0_f[2], size_0_f[3]], scale_factor=None, mode='bilinear', align_corners=True, recompute_scale_factor=None, antialias=False)
        head_layers_1_fam_32_up_layers_0_1 = None
        size_0_f = None
        add_6_f = self.float_functional_simple_17.add(interpolate_0_f, head_layers_0_adapter_conv_2_layers_0_1)
        interpolate_0_f = None
        head_layers_0_adapter_conv_2_layers_0_1 = None
        head_layers_1_fam_16_sm_layers_0_0 = self.head_layers_1_fam_16_sm_layers_0_0(add_6_f)
        head_layers_1_fam_16_sm_layers_0_1 = self.head_layers_1_fam_16_sm_layers_0_1(head_layers_1_fam_16_sm_layers_0_0)
        head_layers_1_fam_16_sm_layers_0_0 = None
        head_layers_1_fam_16_up_layers_0_0 = self.head_layers_1_fam_16_up_layers_0_0(add_6_f)
        add_6_f = None
        head_layers_1_fam_16_up_layers_0_1 = self.head_layers_1_fam_16_up_layers_0_1(head_layers_1_fam_16_up_layers_0_0)
        head_layers_1_fam_16_up_layers_0_0 = None
        size_1_f = head_layers_0_adapter_conv_1_layers_0_1.size()
        interpolate_1_f = torch.nn.functional.interpolate(head_layers_1_fam_16_up_layers_0_1, size=[size_1_f[2], size_1_f[3]], scale_factor=None, mode='bilinear', align_corners=True, recompute_scale_factor=None, antialias=False)
        head_layers_1_fam_16_up_layers_0_1 = None
        size_1_f = None
        add_7_f = self.float_functional_simple_18.add(interpolate_1_f, head_layers_0_adapter_conv_1_layers_0_1)
        interpolate_1_f = None
        head_layers_0_adapter_conv_1_layers_0_1 = None
        head_layers_1_fam_8_sm_layers_0_0 = self.head_layers_1_fam_8_sm_layers_0_0(add_7_f)
        head_layers_1_fam_8_sm_layers_0_1 = self.head_layers_1_fam_8_sm_layers_0_1(head_layers_1_fam_8_sm_layers_0_0)
        head_layers_1_fam_8_sm_layers_0_0 = None
        head_layers_1_fam_8_up_layers_0_0 = self.head_layers_1_fam_8_up_layers_0_0(add_7_f)
        add_7_f = None
        head_layers_1_fam_8_up_layers_0_1 = self.head_layers_1_fam_8_up_layers_0_1(head_layers_1_fam_8_up_layers_0_0)
        head_layers_1_fam_8_up_layers_0_0 = None
        size_2_f = head_layers_0_adapter_conv_0_layers_0_1.size()
        interpolate_2_f = torch.nn.functional.interpolate(head_layers_1_fam_8_up_layers_0_1, size=[size_2_f[2], size_2_f[3]], scale_factor=None, mode='bilinear', align_corners=True, recompute_scale_factor=None, antialias=False)
        head_layers_1_fam_8_up_layers_0_1 = None
        size_2_f = None
        add_8_f = self.float_functional_simple_19.add(interpolate_2_f, head_layers_0_adapter_conv_0_layers_0_1)
        interpolate_2_f = None
        head_layers_0_adapter_conv_0_layers_0_1 = None
        head_layers_1_fam_4_layers_0_0 = self.head_layers_1_fam_4_layers_0_0(add_8_f)
        add_8_f = None
        head_layers_1_fam_4_layers_0_1 = self.head_layers_1_fam_4_layers_0_1(head_layers_1_fam_4_layers_0_0)
        head_layers_1_fam_4_layers_0_0 = None
        size_3_f = head_layers_1_fam_4_layers_0_1.size()
        interpolate_3_f = torch.nn.functional.interpolate(head_layers_1_fam_8_sm_layers_0_1, size=[size_3_f[2], size_3_f[3]], scale_factor=None, mode='bilinear', align_corners=True, recompute_scale_factor=None, antialias=False)
        head_layers_1_fam_8_sm_layers_0_1 = None
        cat_0_f = self.float_functional_simple_20.cat([head_layers_1_fam_4_layers_0_1, interpolate_3_f], dim=1)
        head_layers_1_fam_4_layers_0_1 = None
        interpolate_3_f = None
        interpolate_4_f = torch.nn.functional.interpolate(head_layers_1_fam_16_sm_layers_0_1, size=[size_3_f[2], size_3_f[3]], scale_factor=None, mode='bilinear', align_corners=True, recompute_scale_factor=None, antialias=False)
        head_layers_1_fam_16_sm_layers_0_1 = None
        cat_1_f = self.float_functional_simple_21.cat([cat_0_f, interpolate_4_f], dim=1)
        cat_0_f = None
        interpolate_4_f = None
        interpolate_5_f = torch.nn.functional.interpolate(head_layers_1_fam_32_sm_layers_0_1, size=[size_3_f[2], size_3_f[3]], scale_factor=None, mode='bilinear', align_corners=True, recompute_scale_factor=None, antialias=False)
        head_layers_1_fam_32_sm_layers_0_1 = None
        size_3_f = None
        cat_2_f = self.float_functional_simple_22.cat([cat_1_f, interpolate_5_f], dim=1)
        cat_1_f = None
        interpolate_5_f = None
        head_layers_3_last_layer_0_0 = self.head_layers_3_last_layer_0_0(cat_2_f)
        cat_2_f = None
        head_layers_3_last_layer_0_1 = self.head_layers_3_last_layer_0_1(head_layers_3_last_layer_0_0)
        head_layers_3_last_layer_0_0 = None
        head_layers_3_last_layer_3 = self.head_layers_3_last_layer_3(head_layers_3_last_layer_0_1)
        head_layers_3_last_layer_0_1 = None
        fake_dequant_0 = self.fake_dequant_0(head_layers_3_last_layer_3)
        head_layers_3_last_layer_3 = None
        return fake_dequant_0


if __name__ == "__main__":
    model = QGraphModule()
    model.load_state_dict(torch.load('quant_output/graphmodule_q.pth'))

    model.eval()
    model.cpu()

    dummy_input_0 = torch.ones((1, 3, 256, 256), dtype=torch.float32)

    output = model(dummy_input_0)
    print(output)

