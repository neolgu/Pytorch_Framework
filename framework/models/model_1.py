import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseCNN(nn.Module):
    def __init__(self, input_channels: int = 1, output_size: int = 10):
        super(BaseCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(in_features=64 * 6 * 6, out_features=600)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=output_size)

    def forward(self, x: torch.Tensor):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


if __name__ == "__main__":
    net = BaseCNN(1, 10)

    x = torch.zeros((2, 3, 32, 32))
    print(x.shape)
    print(net.forward(x).shape)
