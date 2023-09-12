import torch
from torch import nn, optim
import matplotlib.pyplot as plt


class Generator(nn.Module):
    def __init__(self, noise_channels, image_channels, features):
        super().__init__()
        # N x noise_channels x 1 x 1 --> N x 3 x 64 x 64
        self.generate = nn.Sequential(
            nn.ConvTranspose2d(noise_channels, features, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(features, 2 * features, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(2 * features, 4 * features, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(4 * features, 8 * features, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8 * features, image_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.generate(x)


class Discriminator(nn.Module):
    def __init__(self, image_channels, features):
        super().__init__()
        # N x 3 x W x H --> N x 1 x 1 x 1
        self.discriminate = nn.Sequential(
            nn.Conv2d(image_channels, 2*features, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(2*features, 4*features, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(4*features, 8*features, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8*features, 16*features, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16*features, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.discriminate(x)


class Trainer(nn.Module):
    def __init__(self,
                 noise_channels,
                 img_channels,
                 discriminator,
                 generator,
                 features,
                 loader,
                 criterion,
                 device="cpu",
                 ):
        self.device = device

        self.disc = discriminator(img_channels, features).to(device)
        self.gen = generator(noise_channels, img_channels, features).to(device)
        self.loader = loader
        self.criterion = criterion

        self.gen_losses = []
        self.disc_losses = []

    def train(self, epochs):
        pass

    def display_losses(self):
        pass

    def save_models(self):
        pass

    def load_models(self):
        pass

    def generate(self):
        pass


