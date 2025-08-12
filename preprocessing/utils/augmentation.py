import albumentations as A
import numpy as np
from albumentations import ToTensorV2
from global_config import (
    MEAN_IMAGENET,
    STD_IMAGENET,
    SIZE_IMAGENET,
)

train_transform = A.Compose([
    A.RandomResizedCrop(
        size=SIZE_IMAGENET,
        scale=(0.75, 1.0),
        ratio=(0.75, 1.0)
    ),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=180, p=0.5),
    # A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.75),
    A.Normalize(
        mean=MEAN_IMAGENET,
        std=STD_IMAGENET,
    ),
    ToTensorV2(),
])
val_transform = A.Compose([
    A.Resize(*SIZE_IMAGENET),
    A.Normalize(
        mean=MEAN_IMAGENET,
        std=STD_IMAGENET,
    ),
    ToTensorV2(),
])
