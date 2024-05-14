import os

import torch.utils.data as data
from data.transform import Custom_data_transforms

import pandas as pd
from torchvision.io import read_image


# Fasion MNIST (sample)
class CustomDataset(data.Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

# DEFAULT
# class CustomDataset(data.Dataset):
#     def __init__(self, root_path: str, mode: str = 'train', transform=None):
#         self.mode = mode
#         self.transform = transform if transform is not None else Custom_data_transforms
#
#         self.x = []
#         self.y = []
#
#     def __len__(self):
#         return len(self.x)
#
#     def __getitem__(self, item):
#         x = self.x[item]
#         y = self.y[item]
#
#         return x, y

