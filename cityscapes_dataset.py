from PIL import Image
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import os
from typing import Callable
from tqdm import tqdm
from collections import namedtuple


mapping_20 = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 1,
    8: 8,
    9: 1,
    10: 1,
    11: 2,
    12: 2,
    13: 2,
    14: 2,
    15: 2,
    16: 2,
    17: 3,
    18: 3,
    19: 3,
    20: 3,
    21: 4,
    22: 4,
    23: 5,
    24: 6,
    25: 6,
    26: 7,
    27: 7,
    28: 7,
    29: 7,
    30: 7,
    31: 7,
    32: 7,
    33: 7,
    34: 7,
}

Label = namedtuple(
    "Label", ["name", "id", "trainId", "category", "categoryId", "hasInstances", "ignoreInEval", "color"]
)
labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label("unlabeled", 0, 255, "void", 0, False, True, (0, 0, 0)),
    Label("ego vehicle", 1, 255, "void", 0, False, True, (0, 0, 0)),
    Label("rectification border", 2, 255, "void", 0, False, True, (0, 0, 0)),
    Label("out of roi", 3, 255, "void", 0, False, True, (0, 0, 0)),
    Label("static", 4, 255, "void", 0, False, True, (0, 0, 0)),
    Label("dynamic", 5, 255, "void", 0, False, True, (111, 74, 0)),
    Label("ground", 6, 255, "void", 0, False, True, (81, 0, 81)),
    Label("road", 7, 0, "ground", 1, False, False, (128, 64, 128)),
    Label("sidewalk", 8, 1, "ground", 8, False, False, (244, 35, 232)),
    Label("parking", 9, 255, "ground", 1, False, True, (250, 170, 160)),
    Label("rail track", 10, 255, "ground", 1, False, True, (230, 150, 140)),
    Label("building", 11, 2, "construction", 2, False, False, (70, 70, 70)),
    Label("wall", 12, 3, "construction", 2, False, False, (102, 102, 156)),
    Label("fence", 13, 4, "construction", 2, False, False, (190, 153, 153)),
    Label("guard rail", 14, 255, "construction", 2, False, True, (180, 165, 180)),
    Label("bridge", 15, 255, "construction", 2, False, True, (150, 100, 100)),
    Label("tunnel", 16, 255, "construction", 2, False, True, (150, 120, 90)),
    Label("pole", 17, 5, "object", 3, False, False, (153, 153, 153)),
    Label("polegroup", 18, 255, "object", 3, False, True, (153, 153, 153)),
    Label("traffic light", 19, 6, "object", 3, False, False, (250, 170, 30)),
    Label("traffic sign", 20, 7, "object", 3, False, False, (220, 220, 0)),
    Label("vegetation", 21, 8, "nature", 4, False, False, (107, 142, 35)),
    Label("terrain", 22, 9, "nature", 4, False, False, (152, 251, 152)),
    Label("sky", 23, 10, "sky", 5, False, False, (70, 130, 180)),
    Label("person", 24, 11, "human", 6, True, False, (220, 20, 60)),
    Label("rider", 25, 12, "human", 6, True, False, (255, 0, 0)),
    Label("car", 26, 13, "vehicle", 7, True, False, (0, 0, 142)),
    Label("truck", 27, 14, "vehicle", 7, True, False, (0, 0, 70)),
    Label("bus", 28, 15, "vehicle", 7, True, False, (0, 60, 100)),
    Label("caravan", 29, 255, "vehicle", 7, True, True, (0, 0, 90)),
    Label("trailer", 30, 255, "vehicle", 7, True, True, (0, 0, 110)),
    Label("train", 31, 16, "vehicle", 7, True, False, (0, 80, 100)),
    Label("motorcycle", 32, 17, "vehicle", 7, True, False, (0, 0, 230)),
    Label("bicycle", 33, 18, "vehicle", 7, True, False, (119, 11, 32)),
    Label("license plate", 34, 19, "vehicle", 7, False, True, (0, 0, 142)),
]


id2color = {label.id: np.asarray(label.color) for label in labels}


def find_closest_labels_vectorized(mask, mapping):  # 'mapping' is a RGB color tuple to categorical number dictionary

    closest_distance = np.full([mask.shape[0], mask.shape[1]], 10000)
    closest_category = np.full([mask.shape[0], mask.shape[1]], None)

    for id, color in mapping.items():  # iterate over every color mapping
        dist = np.sqrt(np.linalg.norm(mask - color.reshape([1, 1, -1]), axis=-1))
        is_closer = closest_distance > dist
        closest_distance = np.where(is_closer, dist, closest_distance)
        closest_category = np.where(is_closer, id, closest_category)

    return closest_category


class CitySegDataset(data.Dataset):
    preprocessed_imgs_prefix = "prep_imgs_"
    preprocessed_masks_prefix = "prep_segs_"
    width = 512
    height = 256

    def __init__(
        self,
        root: str,
        split: str = "train",
        transforms_img: Callable | None = None,
        transforms_seg: Callable | None = None,
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.transforms_img = transforms_img
        self.transforms_seg = transforms_seg
        self.path_to_prep_imgs = os.path.join(self.root, self.preprocessed_imgs_prefix + self.split)
        self.path_to_prep_seg_masks = os.path.join(self.root, self.preprocessed_masks_prefix + self.split)
        if (
            not os.path.exists(self.path_to_prep_imgs)
            or len(os.listdir(self.path_to_prep_imgs)) == 0
            or not os.path.exists(self.path_to_prep_seg_masks)
            or len(os.listdir(self.path_to_prep_seg_masks)) == 0
        ):
            self.preprocess_ds()

        img_paths = os.listdir(self.path_to_prep_imgs)
        self.images = {int(path.split("_")[1][:-4]): os.path.join(self.path_to_prep_imgs, path) for path in img_paths}
        seg_paths = os.listdir(self.path_to_prep_seg_masks)
        self.segmentations = {
            int(path.split("_")[1]): os.path.join(self.path_to_prep_seg_masks, path) for path in seg_paths
        }

    def preprocess_ds(self):
        path_to_imgs = os.path.join(self.root, self.split)
        if not os.path.exists(self.path_to_prep_imgs):
            os.makedirs(self.path_to_prep_imgs)
        if not os.path.exists(self.path_to_prep_seg_masks):
            os.makedirs(self.path_to_prep_seg_masks)
        for i, name in enumerate(tqdm(os.listdir(path_to_imgs), desc="Preprocesing dataset ...")):
            img = Image.open(os.path.join(path_to_imgs, name))
            left_box = (0, 0, self.width // 2, self.height)
            right_box = (self.width // 2, 0, self.width, self.height)

            seg_mask_path = os.path.join(self.path_to_prep_seg_masks, "seg_" + str(i))
            seg_mask = find_closest_labels_vectorized(np.array(img.crop(right_box)), id2color).astype(int)
            seg_mask = F.one_hot(torch.from_numpy(seg_mask), num_classes=len(mapping_20)).permute(2, 0, 1).float()
            torch.save(obj=seg_mask, f=seg_mask_path)

            img_path = os.path.join(self.path_to_prep_imgs, "img_" + str(i) + ".jpg")
            img.crop(left_box).save(fp=img_path)

    def __getitem__(self, index):
        img = Image.open(self.images[index])
        if self.transforms_img is not None:
            img = self.transforms_img(img)
        seg_mask = torch.load(self.segmentations[index])
        if self.transforms_seg is not None:
            seg_mask = self.transforms_seg(seg_mask)
        return img, seg_mask

    def __len__(self):
        return len(self.images)
