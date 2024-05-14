import torch
from torch.utils.data import DataLoader
from data.transform import Custom_data_transforms
from data.dataset import CustomDataset


def prepare_dataloader(data_path, batch_size, mode="train", transform=None):
    if transform is None:
        transform = Custom_data_transforms
    data = CustomDataset(data_path, mode, transform)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    return data_loader

