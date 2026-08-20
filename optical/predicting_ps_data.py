import copy
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from mynetwork import CelestialClassficationNet
from dataset_to_predict import PredictingDataset
from utils import append_new_line

print("torch version: ", torch.__version__)
batch_size = 8
OPT_SOURCE_CLASSIFICATION_MODEL_PATH = "../data/trained_model/PS_classification_model_wts.pt"
ps_positive_samples_csv="../data/catalogs/preprocessed_cat/PS_p_RGZ_samples.csv",
ps_negative_samples_csv="../data/catalogs/preprocessed_cat/PS_n_samples_RGZ_all.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

# Data augmentation and normalization for training
# Just normalization for validation
transforms = transforms.Compose([
    transforms.ToTensor()
])

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

dataset = PredictingDataset(opt_root_dir=os.path.join(PROJECT_ROOT, "source_cutouts", "cdfs", "ps"),
                  ps_positive_samples_csv="../data/catalogs/preprocessed_cat/PS_p_RGZ_samples.csv",
                  ps_negative_samples_csv="../data/catalogs/preprocessed_cat/PS_n_samples_RGZ_all.csv",
                  img_transforms=transforms)

dataloader = DataLoader(dataset)

def set_ps_model():
    ps_source_classification_model = CelestialClassficationNet()
    ps_source_classification_model.load_state_dict(torch.load(OPT_SOURCE_CLASSIFICATION_MODEL_PATH))
    ps_source_classification_model.eval()
    ps_source_classification_model.to(device)
    return ps_source_classification_model

model = set_ps_model()

def test_accuracy(net):
    confusion_matrix = torch.zeros(3, 3)
    correct = 0
    total = 0
    net.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        for i, (inputs, labels, wise_asinh_mag, ps_coords) in enumerate(dataloader):
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


test_accuracy(model)


