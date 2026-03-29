import timm

from config import Config
import os
import pandas as pd
import torch
from src.model.MyModel import MyCNN
import torchvision.transforms as transforms
from PIL import Image
import tqdm
import numpy as np


x = True

print('x value:', x)

print('not x: ', not x)