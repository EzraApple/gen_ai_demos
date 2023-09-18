import torch
from torch import nn, optim
import matplotlib.pyplot as plt


class Generator(nn.Module):
    """
    noise_channels x 1 x 1 --> 3 x 32 x 32
    """
    def __init__(self, noise_channels, image_channels, features):
        super().__init__()
        self.generate = nn.Sequential(
            nn.ConvTranspose2d(noise_channels, features, 4, 2, 0),
            nn.ReLU(),
            nn.ConvTranspose2d(features, features, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(features, features, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(features, image_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.generate(x)


class Discriminator(nn.Module):
    """
    3 x 32 x 32 --> 1 x 1 x 1
    """
    def __init__(self, image_channels, features):
        super().__init__()
        self.discriminate = nn.Sequential(
            nn.Conv2d(image_channels, features, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(features, features, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(features, features, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(features, features, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(features, 1, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.discriminate(x)
