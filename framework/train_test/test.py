import os

import torch
import torch.nn as nn
import torch.utils.data as data
from argparse import ArgumentParser
import tqdm
import torchvision
from train_test.utils import get_config
from model.BasicModel import CustomNetwork
from data.transform import BaseModel_data_transforms
from data.dataset import CustomDataset

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='config/config.yaml',
                    help="testing configuration")


class Tester:
    def __init__(self, config, model_path=None):
        self.config = get_config(config)
        self.use_cuda = self.config['cuda']
        self.device_ids = self.config['gpu_ids']
        self.batch_size = self.config["batch_size"]
        self.model_path = model_path

        self.net = self.model_selection(self.config["model_name"], self.config["num_classes"])

        if self.use_cuda:
            self.net.to(self.device_ids[0])

    def model_selection(self, model_name, num_classes):
        if model_name == 'basemodel':
            return CustomNetwork()
        else:
            raise NotImplementedError(model_name)

    def test(self):
        print("Cuda: ", torch.cuda.is_available())
        print("Device id: ", self.device_ids[0])

        test_data = CustomDataset(self.config["train_data_path"], "test", BaseModel_data_transforms)
        test_dataset_size = len(test_data)
        test_dataloader = data.DataLoader(test_data, batch_size=self.batch_size, shuffle=True)

        self.net.load_state_dict(torch.load(self.model_path))
        self.net = nn.DataParallel(self.net)

        corrects = 0
        acc = 0.0

        self.net.eval()
        with torch.no_grad():
            for (images, labels) in tqdm.tqdm(test_dataloader):
                if self.use_cuda:
                    images = images.to(self.device_ids[0])
                    labels = labels.to(self.device_ids[0])
                outputs = self.net(images).squeeze(1)
                _, preds = torch.max(outputs.data, 1)
                corrects += torch.sum(preds == labels.data).to(torch.float32)
                # print('Iter Acc {:.4f}'.format(torch.sum(preds == labels.data).to(torch.float32)/self.batch_size))
            acc = corrects / test_dataset_size
            print('Test Acc: {:.4f}'.format(acc))
