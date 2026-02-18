import torch
import torch.nn as nn



class MyCNN(nn.Module):
    def __init__(self, image_size=256):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * (image_size // 4) * (image_size // 4), 128)
        self.fc2 = nn.Linear(128, 11)  # Assuming 11 classes
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)


    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)