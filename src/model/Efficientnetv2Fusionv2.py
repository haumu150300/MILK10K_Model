import timm
from torch import nn
import torch

class Efficientnetv2Fusionv2(nn.Module):
    def __init__(self, image_size=256, num_classes=11, extra_features=11):
        super().__init__()
        self.backbone1 = timm.create_model('efficientnetv2_rw_s.ra2_in1k', in_chans=3, pretrained=True, num_classes=num_classes)
        self.backbone2 = timm.create_model('efficientnetv2_rw_s.ra2_in1k', in_chans=3, pretrained=True, num_classes=num_classes)
        self.classifier1 = nn.Linear(11 + 7, num_classes)
        self.classifier2 = nn.Linear(11 + 7, num_classes)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Learnable weight for fusion

    def forward(self, x, metadata1, x2, metadata2):
        out1 = self.backbone1(x)  # [B, 11]
        out2 = self.backbone2(x2)  # [B, 11]
        
        # Concatenate features + metadata
        x1 = torch.cat([out1, metadata1], dim=1).to(out1.device)  # [B, C + metadata_dim]
        x2 = torch.cat([out2, metadata2], dim=1).to(out2.device)  # [B, C + metadata_dim]
        out1 = self.classifier1(x1)
        out2 = self.classifier2(x2)
        
        alpha = torch.sigmoid(self.alpha)
        out = alpha * out1 + (1 - alpha) * out2  # Weighted fusion
        return out
    