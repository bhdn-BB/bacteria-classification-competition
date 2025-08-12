import os
from albumentations import ToTensorV2
from global_config import NUM_CLASSES, COLORS_CHANNEL
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from preprocessing.utils.balanced_color_channels import balance_color_channel


class ProteinDataset(Dataset):

    def __init__(
            self,
            df,
            image_dir,
            name_target_col,
            transform=None,
            # balance_channel=None,
    ):
        self.df = df
        self.name_target_col = name_target_col
        self.image_dir = image_dir
        self.transform = transform
        # self.balance_channel = balance_channel

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row['Id']
        labels = list(map(int, row[self.name_target_col].split()))
        #labels -> one hot encoding
        label_one_hot = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        label_one_hot[labels] = 1.0

        channels = []
        for color in COLORS_CHANNEL:
            current_path = os.path.join(self.image_dir, f'{image_id}_{color}.png')
            img = cv2.imread(current_path, cv2.IMREAD_GRAYSCALE)
            channels.append(img)

        image = np.stack(channels, axis=-1).astype(np.float32)
        # if self.balance_channel:
        #     image = balance_color_channel(image, target_channel=self.balance_channel)
        if self.transform:
            image = self.transform(image=image)['image']
        else:
            image = ToTensorV2()(image=image)["image"]
        return image, label_one_hot