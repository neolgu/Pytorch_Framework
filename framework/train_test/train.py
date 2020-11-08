import os

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.optim import lr_scheduler
from argparse import ArgumentParser
import tqdm
import torchvision
from train_test.utils import get_config
from model.BasicModel import CustomNetwork
from data.transform import BaseModel_data_transforms
from data.dataset import CustomDataset

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='config/config.yaml',
                    help="training configuration")


class Trainer:
    def __init__(self, config, model_path=None):
        self.config = get_config(config)
        self.use_cuda = self.config['cuda']
        self.device_ids = self.config['gpu_ids']
        self.batch_size = self.config["batch_size"]
        self.model_path = model_path

        self.net = self.model_selection(self.config["model_name"], self.config["num_classes"])
        # optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config['lr'],
                                          betas=(self.config['beta1'], self.config['beta2']), eps=1e-08)
        # loss
        self.criterion = nn.CrossEntropyLoss()
        # scheduler
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

        if self.use_cuda:
            self.net.to(self.device_ids[0])

    def model_selection(self, model_name, num_classes):
        if model_name == 'basemodel':
            return CustomNetwork()
        else:
            raise NotImplementedError(model_name)

    def train(self, continue_train=False):
        print("Cuda: ", torch.cuda.is_available())
        print("Device id: ", self.device_ids[0])

        if not os.path.exists(self.config["save_path"]):
            os.mkdir(self.config["save_path"])

        train_data = CustomDataset(self.config["train_data_path"], "train", BaseModel_data_transforms)
        # val_data = CustomDataset(self.config["train_data_path"], "val", BaseModel_data_transforms)

        train_dataset_size = len(train_data)
        # val_dataset_size = len(val_data)

        train_dataloader = data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        # val_dataloader = data.DataLoader(val_data, batch_size=self.batch_size, shuffle=True)

        if continue_train:
            self.net.load_state_dict(torch.load(self.model_path))

        self.net = nn.DataParallel(self.net)

        for epoch in range(self.config["epoch"]):
            print('Epoch {}/{}'.format(epoch + 1, self.config["epoch"]))
            print('-' * 10)

            self.net.train()

            train_loss = 0.0
            train_corrects = 0.0

            iteration = 0
            for (images, labels) in tqdm.tqdm(train_dataloader):
                if self.use_cuda:
                    images = images.to(self.device_ids[0])
                    labels = labels.to(self.device_ids[0])
                self.optimizer.zero_grad()
                outputs = self.net(images).squeeze(1)
                _, preds = torch.max(outputs.data, 1)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                iter_loss = loss.data.item()
                train_loss += iter_loss
                iter_corrects = torch.sum(preds == labels.data).to(torch.float32)
                train_corrects += iter_corrects
                iteration += 1
                if not (iteration % self.config['print_iter']):
                    print('iteration {} train loss: {:.4f} Acc: {:.4f}'.format(iteration, iter_loss / self.batch_size,
                                                                               iter_corrects / self.batch_size))

            epoch_loss = train_loss / train_dataset_size
            epoch_acc = train_corrects / train_dataset_size
            print('epoch train loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

            self.scheduler.step()

            torch.save(self.net.module.state_dict(), os.path.join(self.config["save_path"], "{}.tar".format(epoch)))
            print("saved at {}".format(os.path.join(self.config["save_path"], "{}.tar".format(epoch))))

            # validation part
            """
            val_loss = 0.0
            val_corrects = 0.0

            best_acc = 0.0

            self.net.eval()
            with torch.no_grad():
                for (images, labels) in tqdm.tqdm(val_dataloader):
                    if self.use_cuda:
                        images = images.to(self.device_ids[0])
                        labels = labels.to(self.device_ids[0])
                    outputs = self.net(images).squeeze(1)
                    _, preds = torch.max(outputs.data, 1)
                    val_corrects += torch.sum(preds == labels.data).to(torch.float32)
                epoch_acc = val_corrects / val_dataset_size
                print('epoch val Acc: {:.4f}'.format(epoch_acc))
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = self.net.state_dict()
            """
        # print("Best val Acc: {:.4f}".format(best_acc))
        # self.net.load_state_dict(best_model_wts)
        #
        # torch.save(self.net.module.state_dict(), os.path.join(self.config["save_path"], "best_acc.tar"))
