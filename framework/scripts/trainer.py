import os
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from .utils import prepare_dataloader
from models import get_model


def train_model(model, train_dataloader, criterion, optimizer, scheduler, device, epoch, save_path):
    model.train()

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for e in range(epoch):
        train_loss = 0.0
        train_corrects = 0.0

        for images, labels in tqdm(train_dataloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images).squeeze(1)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_corrects += torch.sum(preds == labels.data)

        epoch_loss = train_loss / len(train_dataloader.dataset)
        epoch_acc = train_corrects / len(train_dataloader.dataset)

        print(f'Epoch Loss: {epoch_loss:.4f} | Epoch Acc: {epoch_acc:.4f}')

        scheduler.step()

        # Save model checkpoint
        checkpoint_path = os.path.join(save_path, f"{e}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved model checkpoint at {checkpoint_path}")


def train(config):
    device = torch.device(config["device"])
    train_dataloader = prepare_dataloader(config["data_path"], config["batch_size"], mode="train", transform=None)

    model = get_model(config["model_name"]).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['lr'],
                           betas=(config['beta1'], config['beta2']), eps=1e-08)

    # loss
    criterion = torch.nn.CrossEntropyLoss()

    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    train_model(model, train_dataloader, criterion, optimizer, scheduler, device, config["epoch"], config["save_path"])


if __name__ == "__main__":
    train()