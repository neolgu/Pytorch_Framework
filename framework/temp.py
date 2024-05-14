import torchvision

data = torchvision.datasets.FashionMNIST(
    root="./dataset",
    train=True,
    download=True
)

data = torchvision.datasets.FashionMNIST(
    root="./dataset",
    train=False,
    download=True
)