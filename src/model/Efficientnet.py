import timm
from torch import nn
import torch

class Efficientnet(nn.Module):
    def __init__(self, image_size=256, num_classes=11, extra_features=11):
        super().__init__()
        self.backbone1 = timm.create_model('tf_efficientnetv2_m.in21k_ft_in1k', pretrained=False, num_classes=num_classes)
        self.fc1 = nn.Linear(11 + 7, num_classes)
        # self.fc1 = nn.Linear(11 + 7, num_classes)
        # self.alpha = nn.Parameter(torch.tensor(0.5))  # Learnable weight for combining features

    def forward(self, x, x2, metadata1, metadata2):
        out1 = self.backbone1(x)  # [B, 11]
        out = torch.cat([out1, metadata1], dim=1)  # [B, 11 + 7]
        out = self.fc1(out)  # [B, 11]
        return out
    