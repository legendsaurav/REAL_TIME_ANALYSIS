import torch
import torch.nn as nn
from config import GRID_SIZE, BBOXES_PER_CELL, NUM_CLASSES

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
    def forward(self, x):
        return self.leaky(self.bn(self.conv(x)))

class YoloModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(3, 32, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            ConvBlock(32, 64, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            ConvBlock(64, 128, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            ConvBlock(128, 256, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            ConvBlock(256, 512, 3, 1, 1),
            nn.MaxPool2d(2, 2),
        )
        self.head = nn.Sequential(
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, BBOXES_PER_CELL * 5 + NUM_CLASSES, 1, 1, 0)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        x = x.permute(0, 2, 3, 1)  # [batch, grid, grid, features]
        return x
