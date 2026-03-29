import timm
from torch import nn
import torch

class Convnext(nn.Module):
    def __init__(self, image_size=256, num_classes=11, extra_features=11):
        super().__init__()
        self.backbone = timm.create_model('convnext_small.in12k_ft_in1k_384', pretrained=True, num_classes=num_classes)
        self.classifier = nn.Linear(11 + 7, num_classes)

    def forward(self, x, metadata):
        out = self.backbone(x)  # [B, 11]
        # Concatenate features + metadata
        x = torch.cat([out, metadata], dim=1).to(out.device)  # [B, C + metadata_dim]
        out = self.classifier(x)
        return out
    

