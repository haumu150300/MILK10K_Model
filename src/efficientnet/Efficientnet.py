import timm
from torch import nn

class Efficientnet(nn.Module):
    def __init__(self, image_size=256, num_classes=11):
        super().__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)