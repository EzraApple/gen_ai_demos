import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, noise_dim, image_dim, features):
        super().__init__()
        self.generate = nn.Sequential(
            nn.Linear(noise_dim, features),
            nn.ReLU(),
            nn.Linear(features, 2 * features),
            nn.ReLU(),
            nn.Linear(2 * features, 4 * features),
            nn.ReLU(),
            nn.Linear(4 * features, image_dim),
            nn.Sigmoid(),
        )

    def forward(self, noise):
        return self.generate(noise)


class Critic(nn.Module):
    def __init__(self, image_dim, features):
        super().__init__()
        self.discriminate = nn.Sequential(
            nn.Linear(image_dim, 4 * features),
            nn.ReLU(),
            nn.Linear(4 * features, 2 * features),
            nn.ReLU(),
            nn.Linear(2 * features, features),
            nn.ReLU(),
            nn.Linear(features, 1),
            nn.ReLU(),
        )

    def forward(self, image):
        return self.discriminate(image)

    def gradient_penalty(self, real, fake):
        # mix real and fake, and get criticism
        batch_size = real.shape[0]
        proportions = torch.rand((batch_size, 1, 1)).expand_as(real)
        mixed = proportions * real + (1 - proportions) * fake
        criticisms = self.forward(mixed)

        # get gradients
        gradients = torch.autograd.grad(
            inputs=mixed,
            outputs=criticisms,
            grad_outputs=torch.ones_like(criticisms),
            create_graph=True,
            retain_graph=True
        )

        # flatten so one row per image
        gradients = gradients.view(len(gradients), -1)

        # norm and penalty
        norm = gradients.norm(2, dim=1)
        penalty = torch.mean(torch.square(norm - 1))

        return penalty
