#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring
import os
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch
import torchvision
from PIL import Image


class MyDataset(torchvision.datasets.vision.VisionDataset):
    def __init__(
            self,
            root: str,
            name: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
    ) -> None:
        super(MyDataset, self).__init__(root, transforms, transform, target_transform)
        images_dir = Path(root) / 'images' / name
        labels_dir = Path(root) / 'labels' / name
        self.images = [n for n in images_dir.iterdir()]
        self.labels = []
        for image in self.images:
            base, _ = os.path.splitext(os.path.basename(image))
            label = labels_dir / f'{base}.txt'
            self.labels.append(label if label.exists() else None)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        img = Image.open(self.images[idx]).convert('RGB')
        # mode = torchvision.io.image.ImageReadMode.RGB
        # img = torchvision.io.read_image(self.images[idx], mode)

        label_file = self.labels[idx]
        if label_file is not None:  # found
            with open(label_file, 'r') as f:
                labels = [x.split() for x in f.read().strip().splitlines()]
                labels = np.array(labels, dtype=np.float32)
        else:  # missing
            labels = np.zeros((0, 5), dtype=np.float32)
        # print(f'{idx}, {label_file}, {labels.shape}')

        boxes = []
        classes = []
        for label in labels:
            x, y, w, h = label[1:]
            boxes.append([
                (x - w / 2) * img.width,
                (y - h / 2) * img.height,
                (x + w / 2) * img.width,
                (y + h / 2) * img.height])
            classes.append(label[0])

        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(classes, dtype=torch.int64)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self) -> int:
        return len(self.images)
