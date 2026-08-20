import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CelestialClassficationNet(nn.Module):
    def __init__(self):
        super(CelestialClassficationNet, self).__init__()
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn.conv1 = nn.Conv2d(5, 64, kernel_size=(7, 7), stride=(2, 2), padding=3)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 20)
        self.fc1 = nn.Linear(20 + 2, 8)
        self.fc2 = nn.Linear(8, 3)

    def forward(self, image, WISE_mag_info):  # , WISE_mag_info
        x1 = self.cnn(image)  # 1 x 20
        x2 = WISE_mag_info

        x = torch.cat((x1, x2), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    # def forward(self, image):  # , WISE_mag_info
    #     x = self.cnn(image)  # 1 x 20
    #     x = F.relu(self.fc1(x))
    #     x = self.fc2(x)
    #     return x

# model = CelestialClassficationNet()
# torchsummary.summary(model, (5, 240, 240))