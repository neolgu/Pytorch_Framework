import torch
from argparse import ArgumentParser
from tqdm import tqdm

from .utils import prepare_dataloader
from models import get_model

def test_model(model, test_dataloader, device):
    model.eval()

    test_corrects = 0.0

    for images, labels in tqdm(test_dataloader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images).squeeze(1)
        _, preds = torch.max(outputs.data, 1)

        test_corrects += torch.sum(preds == labels.data)

    acc = test_corrects / len(test_dataloader.dataset)
    print('Test Acc: {:.4f}'.format(acc))


def test(config):
    device = torch.device(config["device"])
    test_dataloader = prepare_dataloader(config["data_path"], config["batch_size"], mode="test", transform=None)

    model = get_model(config["model_name"]).to(device)
    model.load_state_dict(torch.load(config["model_path"]))

    test_model(model, test_dataloader, device)


if __name__ == "__main__":
    test()