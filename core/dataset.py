"""
COCO-format dataset + augmentation pipeline for single-class person detection.
================================================================================
Works directly with a Roboflow COCO export:

    <root>/
      train/ _annotations.coco.json  +  *.jpg
      valid/ _annotations.coco.json  +  *.jpg
      test/  _annotations.coco.json  +  *.jpg

Roboflow adds a dummy super-category (id 0). We collapse EVERY foreground
annotation to label 1 ("person"), matching torchvision's convention that
label 0 == background.
"""

import os

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from torchvision import tv_tensors
from torchvision.transforms import v2
from pycocotools.coco import COCO


PERSON_LABEL = 1  # foreground; 0 is reserved for background by torchvision


class CocoPersonDataset(Dataset):
    """Loads images + boxes from a COCO json, returns (image, target) pairs."""

    def __init__(self, img_dir, ann_file, transforms=None, keep_empty=True):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.transforms = transforms
        self.keep_empty = keep_empty

        ids = sorted(self.coco.imgs.keys())
        if not keep_empty:
            ids = [i for i in ids if len(self.coco.getAnnIds(imgIds=i)) > 0]
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def _load_target(self, img_id, canvas_size):
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        boxes, labels, areas, iscrowd = [], [], [], []
        for a in anns:
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(PERSON_LABEL)          # collapse all -> person
            areas.append(a.get("area", w * h))
            iscrowd.append(a.get("iscrowd", 0))

        if boxes:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)

        return {
            "boxes": tv_tensors.BoundingBoxes(
                boxes_t, format="XYXY", canvas_size=canvas_size),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
        }

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        file_name = self.coco.loadImgs(img_id)[0]["file_name"]
        path = os.path.join(self.img_dir, file_name)

        img = tv_tensors.Image(read_image(path, ImageReadMode.RGB))
        target = self._load_target(img_id, canvas_size=img.shape[-2:])

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target


def get_transforms(train: bool):
    """
    Detection augmentation via torchvision.transforms.v2 (box-aware).

    We do NOT normalize/resize here — every torchvision detector normalizes and
    resizes internally (GeneralizedRCNNTransform). We only convert to float
    [0,1] and, for training, apply geometric + photometric augmentation.
    """
    tfms = []
    if train:
        tfms += [
            v2.RandomPhotometricDistort(p=0.5),
            v2.RandomZoomOut(fill={tv_tensors.Image: (123, 117, 104), "others": 0}, p=0.3),
            v2.RandomIoUCrop(),
            v2.RandomHorizontalFlip(p=0.5),
        ]
    tfms += [
        v2.ToDtype(torch.float32, scale=True),
        v2.SanitizeBoundingBoxes(),   # drop degenerate boxes after geometry ops
    ]
    return v2.Compose(tfms)


def collate_fn(batch):
    """Detection batches keep variable-sized images as plain tuples."""
    return tuple(zip(*batch))
