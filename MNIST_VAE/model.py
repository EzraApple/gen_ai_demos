import torch
from torch import nn


class VAE(nn.module):
    def __init__(self, input_dim, hidden_dim, latent_dim, device):
        super().__init__()

        self.device = device
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )

        self.mean = nn.Linear(latent_dim, 2)
        self.variance = nn.Linear(latent_dim, 2)

        self.decoder = nn.Sequential(
            nn.Linear(2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, image):
        latent = self.encode(image)
        mean, variance = self.mean(latent), self.variance(latent)
        return mean, variance

    def decode(self, sample):
        return self.decoder(sample)

    def reparametrize(self, mean, variance):
        noise = torch.rand_like(variance).to(self.device)
        sample = mean + noise * variance
        return sample

    def forward(self, image):
        mean, variance = self.encode(image)
        sample = self.reparametrize(mean, variance)
        return self.decode(sample), mean, variance
