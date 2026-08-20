import copy
import os
import time

import numpy as np
import torch
import torch.optim as optim
import torchvision.utils

from torch import nn
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms, models

from radio_fits_folder import FitsImageFolder
from tools.image_tool import imshow

DATA_ROOT = "../data/ROGUE_datasets"

transforms = transforms.Compose([
    transforms.ToTensor()])

radio_dat = FitsImageFolder(root=DATA_ROOT, transform=transforms)
data_indices = list(range(len(radio_dat)))
np.random.shuffle(data_indices)

validSetSize = int(np.floor(0.2 * len(data_indices)))
trainSetSize = len(data_indices) - validSetSize

trainSet, validSet = random_split(radio_dat, [trainSetSize, validSetSize])

dataset = {"train": trainSet, "validation": validSet}

print('Train data set:', len(dataset['train']))
print('Valid data set:', len(dataset['validation']))

trainLoader = DataLoader(trainSet, batch_size=8, shuffle=True)
validLoader = DataLoader(validSet, batch_size=8, shuffle=False)

dataloader = {"train": trainLoader, "validation": validLoader}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")


def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    model.to(device)

    total_step = len(dataloader["train"])
    print(f"total_step: {total_step}")

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('=' * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "validation"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for i, (inputs, labels) in enumerate(dataloader[phase], 0):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, dim=1)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / len(dataset[phase])
            epoch_acc = running_corrects.double() / len(dataset[phase])

            print('{} Loss: {:.6f} Acc: {:.6f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == "validation" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def build_model():
    my_model = models.resnet18(pretrained=True)

    for param in my_model.parameters():
        param.requires_grad = False

    my_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=3)
    nn.init.xavier_uniform_(my_model.conv1.weight)
    num_features = my_model.fc.in_features
    my_model.fc = nn.Linear(num_features, len(radio_dat.classes))

    return my_model


if __name__ == '__main__':
    model = build_model()

    criterion = nn.CrossEntropyLoss()
    SGD_optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 5 epochs
    lr_scheduler = optim.lr_scheduler.StepLR(SGD_optimizer, step_size=5, gamma=0.1)

    trained_model = train_model(model, criterion, SGD_optimizer, lr_scheduler)
