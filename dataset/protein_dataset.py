import os
from global_config import NUM_CLASSES
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class ProteinDataset(Dataset):

    def __init__(self, df, image_dir, transform, name_target_col):
        self.df = df
        self.name_target_col = name_target_col
        self.image_dir = image_dir
        self.transform = transform

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
        for color in ['red', 'green', 'blue', 'yellow']:
            current_path = os.path.join(self.image_dir, f'{image_id}_{color}.png')
            img = cv2.imread(current_path, cv2.IMREAD_GRAYSCALE)
            channels.append(img)

        image = np.stack(channels, axis=-1).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image)             #convert to TensorV2
            image = augmented['image']
        else:
            image = torch.tensor(np.transpose(image, (2, 0, 1)), dtype=torch.float32)
        return image, label_one_hot