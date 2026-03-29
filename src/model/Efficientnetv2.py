import timm
from torch import nn
import torch

class Efficientnet(nn.Module):
    def __init__(self, image_size=256, num_classes=11, extra_features=11):
        super().__init__()
        self.backbone = timm.create_model('efficientnetv2_rw_s.ra2_in1k', in_chans=3, pretrained=True, num_classes=num_classes)
        self.classifier = nn.Linear(11 + 7, num_classes)

    def forward(self, x, metadata1):
        out1 = self.backbone(x)  # [B, 11]
        # Concatenate features + metadata
        x = torch.cat([out1, metadata1], dim=1).to(out1.device)  # [B, C + metadata_dim]
        out = self.classifier(x)
        return out
    