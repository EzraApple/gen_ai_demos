import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.ins = in_channels
        self.outs = out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, image):
        return self.block(image)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_sample = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels)
        )

    def forward(self, image):
        return self.down_sample(image)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_sample = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, image, residual):
        upped = self.up_sample(image)
        combine = torch.cat([upped, residual], dim=1)
        return self.conv(combine)


class UNet(nn.Module):
    def __init__(self, image_channels, hidden_channels, n_classes):
        super().__init__()
        self.input = ConvBlock(image_channels, hidden_channels)
        self.down = nn.ModuleList([
            Down(hidden_channels, hidden_channels * 2),
            Down(hidden_channels * 2, hidden_channels * 4),
            Down(hidden_channels * 4, hidden_channels * 8),
            Down(hidden_channels * 8, hidden_channels * 16)
        ])
        self.up = nn.ModuleList([
            Up(hidden_channels * 16, hidden_channels * 8),
            Up(hidden_channels * 8, hidden_channels * 4),
            Up(hidden_channels * 4, hidden_channels * 2),
            Up(hidden_channels * 2, hidden_channels)
        ])
        self.output = nn.Conv2d(hidden_channels, n_classes, kernel_size=1)

    def forward(self, image):
        residuals = []
        image = self.input(image)
        for down in self.down:
            residuals.insert(0, image)
            image = down(image)
        for i, up in enumerate(self.up):
            image = up(image, residuals[i])
        segmentation = self.output(image)
        return segmentation
