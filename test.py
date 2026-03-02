from config import Config
import os
import pandas as pd
import torch
from src.model.MyModel import MyCNN
import torchvision.transforms as transforms
from PIL import Image
import tqdm 
from src.unet.Unet import UNet

model = UNet(n_channels=3, n_classes=10, image_size=256)
x = torch.randn(1, 3, 256, 256)
output = model(x)


