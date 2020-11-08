from train_test.train import Trainer
from train_test.test import Tester

import torchvision
import torchvision.transforms as transforms

if __name__ == '__main__':
    trainer = Trainer("config/config.yaml")
    trainer.train()
    # tester = Tester("config/config.yaml", "checkpoint/train/1.tar")
    # tester.test()
