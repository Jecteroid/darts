import cv2 as cv
import numpy as np

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
from torch.utils.data import Dataset


class Wheat(Dataset):

  def __init__(self, dataframe, image_dir, transforms=None):
    super().__init__()
    self.image_ids = dataframe['image_id'].unique()
    self.df = dataframe
    self.image_dir = image_dir
    self.transforms = transforms

  def __getitem__(self, idx: int):
    image_id = self.image_ids[idx]
    records = self.df[self.df['image_id'] == image_id]

    image = cv.imread(f'{self.image_dir}/{image_id}.jpg', cv.IMREAD_COLOR)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.0

    boxes = records[['x', 'y', 'w', 'h']].values

    area = boxes[:, 2] * boxes[:, 3]
    area = torch.as_tensor(area, dtype=torch.float32)

    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

    # there is only one class
    labels = torch.ones((records.shape[0],), dtype=torch.int64)
    # suppose all instances are not crowd
    iscrowd = torch.zeros((records.shape[0],), dtype=torch.uint8)

    target = {}
    target['boxes'] = boxes
    target['labels'] = labels
    target['image_id'] = torch.tensor([idx])
    target['area'] = area
    target['iscrowd'] = iscrowd

    if self.transforms:
      sample = {
        'image': image,
        'bboxes': target['boxes'],
        'labels': labels,
      }
      sample = self.transforms(**sample)
      image = sample['image']
      target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)

    return image, target, image_id

  def __len__(self) -> int:
    return len(self.image_ids)

  # albumentations
  #  https://github.com/albumentations-team/albumentations

  @staticmethod
  def get_train_transform():
    return A.Compose([
      A.Flip(0.5),
      ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

  @staticmethod
  def get_valid_transform():
    return A.Compose([
      ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})