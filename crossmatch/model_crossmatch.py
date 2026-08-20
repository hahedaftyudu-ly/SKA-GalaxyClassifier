import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RadioOpticalCrossmatchModel(nn.Module):
    def __init__(self, pos_dims=3):
        """
        pos_dims: number of position features.
        - pos_dims=3: fc1 input = 20+10+3+3 = 36 (RGZ_all_negative, RGZ_ROGUE series)
        - pos_dims=2: fc1 input = 20+10+3+2 = 35 (crossmatch_model_wts, RGZ1_1, RGZ3_1)
        """
        super(RadioOpticalCrossmatchModel, self).__init__()
        self.pos_dims = pos_dims

        self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)
        self.cnn.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=3)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 20)

        self.cnn_opt = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)
        self.cnn_opt.conv1 = nn.Conv2d(5, 64, kernel_size=(7, 7), stride=(2, 2), padding=3)
        self.cnn_opt.fc = nn.Linear(self.cnn_opt.fc.in_features, 10)

        # fc1 = cnn(20) + cnn_opt(10) + ps_probs(3) + pos_dims
        fc1_in = 20 + 10 + 3 + pos_dims
        self.fc1 = nn.Linear(fc1_in, 8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, radio_img_data, ps_bg_imgcube, ps_source_class_probs, ps_cutouts_pos_info, centroid, geodesic):
        x1 = self.cnn(radio_img_data)
        x2 = self.cnn_opt(ps_bg_imgcube)

        # Extract position features based on pos_dims
        if self.pos_dims == 3:
            # Take first component from each pair
            pos_feats = torch.cat([
                ps_cutouts_pos_info[:, 0:1],
                centroid[:, 0:1],
                geodesic[:, 0:1]
            ], dim=1)
        elif self.pos_dims == 2:
            pos_feats = torch.cat([
                centroid[:, 0:1],
                geodesic[:, 0:1]
            ], dim=1)
        else:
            pos_feats = torch.cat([
                ps_cutouts_pos_info[:, 0:1],
                centroid[:, 0:1],
                geodesic[:, 0:1]
            ], dim=1)[:, :self.pos_dims]

        x = torch.cat((x1, x2, ps_source_class_probs, pos_feats), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    @staticmethod
    def detect_pos_dims(checkpoint_path):
        """Detect position dims from a checkpoint without loading the full model"""
        sd = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        fc1_shape = sd['fc1.weight'].shape  # [8, N]
        fc1_in = fc1_shape[1]
        pos_dims = fc1_in - 20 - 10 - 3  # 35 => 2, 36 => 3
        return pos_dims
