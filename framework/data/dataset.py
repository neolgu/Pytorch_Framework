import torch.utils.data as data
from data.transform import BaseModel_data_transforms
import os


class CustomDataset(data.Dataset):
    def __init__(self, root_path, mode='train', transform=None):
        self.mode = mode
        self.transform = transform

        # bla bla
        self.x = []
        self.y = []

    def __len__(self):  # data len
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]

        return x, y
