import copy
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

# import from local project
from FitsImageFolder import FitsImageFolder
from mynetwork import CelestialClassficationNet
from optical_dataset import OptDataSet
from utils import append_new_line
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

print("torch version: ", torch.__version__)
batch_size = 8
# src_root_path = '/mnt/DataDisk/Duncan/sources'
# src_root_path = '/mnt/DataDisk/Duncan/sources_for_Sean'
# src_root_path = '/Volumes/Expansion/VLASS/sources_for_Sean'  # 原始 macOS 硬盘路径（已不可用）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
src_root_path = os.path.join(PROJECT_ROOT, "source_cutouts", "cdfs", "ps")
print(src_root_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

print("Reading catalogue...")

# Data augmentation and normalization for training
# Just normalization for validation
train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=(-90, 90))

])
validation_transforms = transforms.Compose([
    transforms.ToTensor()
])

# 加载图像数据集，并在加载的时候对图像施加变换
whole_dataset = FitsImageFolder(root=src_root_path)
trainData = OptDataSet(whole_dataset, transform=train_transforms)
validationData = OptDataSet(whole_dataset, validation_transforms)
testData = OptDataSet(whole_dataset, validation_transforms)

# Create the index splits for training, validation and test
train_size = 0.8
num_train = len(whole_dataset)
indices = list(range(num_train))
split = int(np.floor(train_size * num_train))
split2 = int(np.floor((train_size + (1 - train_size) / 2) * num_train))
np.random.shuffle(indices)
train_idx, valid_idx, test_idx = indices[:split], indices[split:split2], indices[split2:]

trainset = Subset(trainData, indices=train_idx)
validset = Subset(validationData, indices=valid_idx)
testset = Subset(testData, indices=test_idx)

dataset = {'train': trainset, 'val': validset, 'test': testset}

training_loader = DataLoader(
    trainset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,
)

validation_loader = DataLoader(
    validset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=8,
)

test_loader = DataLoader(
    testset,
    batch_size=1,
    shuffle=False
)

dataloaders = {'train': training_loader, 'val': validation_loader, 'test': test_loader}

print("Train set size: ", len(trainset))
print("Validation set size: ", len(validset))
print("Test set size: ", len(testset))

class_names = whole_dataset.classes
print(class_names)

training_epoch_loss = []
validation_epoch_loss = []
training_epoch_accuracy = []
validation_epoch_accuracy = []

note_files = "data/ps_source_testing_notes.csv"



def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    model.to(device)

    total_step = len(training_loader)
    print("total_step:", total_step)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            for i, (inputs, labels, wise_asinh_mag, ps_coords) in enumerate(dataloaders[phase]):
                inputs, labels, wise_asinh_mag = inputs.to(device), labels.to(device), wise_asinh_mag.to(device)  # inputs: [8, 5, 240, 240], labels:[8, 1], wise_asinh_mag:[8, 2]
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    image = inputs.float()
                    wise_asinh_mag = wise_asinh_mag.float()
                    outputs = model(image, wise_asinh_mag)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataset[phase])
            epoch_acc = running_corrects.double() / len(dataset[phase])

            print('{} Loss: {:.6f} Acc: {:.6f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'train':
                training_epoch_loss.append(epoch_loss)
                training_epoch_accuracy.append(epoch_acc)
            else:
                validation_epoch_loss.append(epoch_loss)
                validation_epoch_accuracy.append(epoch_acc)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        with open(note_files, 'a') as file:
            file.write(
                f"{training_epoch_loss[-1]}, {validation_epoch_loss[-1]}, {training_epoch_accuracy[-1]}, {validation_epoch_accuracy[-1]}\n")

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # Save the best model weights
    PATH = "data/trained_model/PS_classification_model_wts.pt"
    torch.save(best_model_wts, PATH)
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def test_accuracy(net):
    confusion_matrix = torch.zeros(3, 3)
    correct = 0
    total = 0
    net.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        for i, (inputs, labels, wise_asinh_mag, ps_coords) in enumerate(test_loader):
            inputs, labels, wise_asinh_mag = inputs.to(device), labels.to(device), wise_asinh_mag.to(device)
            image = inputs.float()
            wise_asinh_mag = wise_asinh_mag.float()

            ps_ra = ps_coords[0].item()
            ps_dec = ps_coords[1].item()

            outputs = net(image, wise_asinh_mag)
            ps_source_class_probs = F.softmax(outputs, dim=1)

            ps_source_class_probs = ps_source_class_probs.tolist()[0]
            GALAXY_prob = ps_source_class_probs[0]
            QSO_prob = ps_source_class_probs[1]
            STAR_prob = ps_source_class_probs[2]

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

            pred = predicted.item()
            lab = labels.item()

            append_new_line(note_files,
                            f"{ps_ra}, {ps_dec}, {GALAXY_prob}, {QSO_prob}, {STAR_prob}, {lab}, {pred}")

    print("confusion_matrix: ", confusion_matrix)
    print(confusion_matrix.diag() / confusion_matrix.sum(1))
    print("accuracy over all: ", correct / total)
    return correct / total


def set_ps_model(model_path="crossmatch/models/opt_classification_model_wts.pt"):
    ps_source_classification_model = CelestialClassficationNet()
    ps_source_classification_model.load_state_dict(torch.load(model_path))
    ps_source_classification_model.eval()
    ps_source_classification_model.to(device)
    return ps_source_classification_model


if __name__ == '__main__':
    model_ft = set_ps_model(model_path="data/trained_model/PS_classification_model_wts.pt")
    # model = CelestialClassficationNet().to(device)
    #
    # criterion = nn.CrossEntropyLoss()
    #
    # # Observe that all parameters are being optimized
    # optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # # Decay LR by a factor of 0.1 every 7 epochs
    # exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    #
    # model_ft = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)
    with open(note_files, "w+") as f:
        f.write("ra, dec, galaxy_prob, qso_prob, star_prob, label, predicted")

    test_accuracy(model_ft)
