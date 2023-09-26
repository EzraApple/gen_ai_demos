import torch
from torch import nn


class VAE(nn.Module):
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
        self.log_variance = nn.Linear(latent_dim, 2)

        self.decoder = nn.Sequential(
            nn.Linear(2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, image):
        latent = self.encoder(image)
        mean, log_variance = self.mean(latent), self.log_variance(latent)
        return mean, log_variance

    def decode(self, sample):
        return self.decoder(sample)

    def reparametrize(self, mean, variance):
        noise = torch.rand_like(variance).to(self.device)
        sample = mean + noise * variance
        return sample

    def forward(self, image):
        mean, log_variance = self.encode(image)
        sample = self.reparametrize(mean, (0.5 * log_variance).exp())
        return self.decode(sample), mean, log_variance
