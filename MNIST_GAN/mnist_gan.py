import torch
from torch import nn, optim
import matplotlib.pyplot as plt


class Generator(nn.Module):
    def __init__(self, noise_dim, image_dim):
        super().__init__()
        self.generate = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, image_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.generate(x)


class Discriminator(nn.Module):
    def __init__(self, image_dim):
        super().__init__()
        self.discriminate = nn.Sequential(
            nn.Linear(image_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.discriminate(x)


class Trainer:
    def __init__(self,
                 noise_dim,
                 img_dim,
                 discriminator,
                 generator,
                 loader,
                 criterion,
                 device="cpu",
                 ):

        self.noise_dim = noise_dim
        self.img_dim = img_dim
        self.device = device

        self.disc = discriminator(img_dim).to(device)
        self.gen = generator(noise_dim, img_dim).to(device)
        self.loader = loader
        self.criterion = criterion

        self.gen_losses = []
        self.disc_losses = []

    def train(self, epochs):
        device = self.device
        optim_gen = optim.Adam(self.gen.parameters(), lr=3e-4)
        optim_disc = optim.Adam(self.disc.parameters(), lr=3e-4)

        for epoch in range(epochs):
            g_loss = 0
            d_loss = 0
            for img, _ in self.loader:
                img = img.view(-1, 784).to(device)
                batch_size = img.shape[0]

                # Discriminator training - train with real and generated images
                noise = torch.randn(self.noise_dim).to(device)
                fake = self.gen(noise)

                disc_real = self.disc(img).view(-1)
                loss_d_real = self.criterion(disc_real, torch.ones_like(disc_real))

                disc_fake = self.disc(fake).view(-1)
                loss_d_fake = self.criterion(disc_fake, torch.zeros_like(disc_fake))

                loss_d = (loss_d_real + loss_d_fake) / 2
                self.disc.zero_grad()
                loss_d.backward(retain_graph=True)
                optim_disc.step()

                # Generator training - get discriminated against
                output = self.disc(fake).view(-1)
                loss_g = self.criterion(output, torch.ones_like(output))
                self.gen.zero_grad()
                loss_g.backward()
                optim_gen.step()

                g_loss += loss_g
                d_loss += loss_d

            print(f"Epoch {epoch + 1} | Generator Loss {g_loss/len(self.loader)} | Discriminator Loss {d_loss / len(self.loader)}")
            self.gen_losses.append(g_loss.detach()/len(self.loader))
            self.disc_losses.append(d_loss.detach() / len(self.loader))

    def display_losses(self):
        epochs = range(len(self.gen_losses))
        plt.plot(epochs, self.gen_losses, color="blue", label="Generator Losses")
        plt.plot(epochs, self.disc_losses, color="green", label="Discriminator Losses")
        plt.title("Model Loss vs. Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")

    def save_models(self):
        torch.save(self.gen.state_dict(), "../models/model_state_dicts/mnist_gan/gen.pt")
        torch.save(self.disc.state_dict(), "../models/model_state_dicts/mnist_gan/disc.pt")

    def load_models(self):
        self.gen.load_state_dict(torch.load("../models/model_state_dicts/mnist_gan/gen.pt"))
        self.disc.load_state_dict(torch.load("../models/model_state_dicts/mnist_gan/disc.pt"))

    def generate(self):
        self.gen.eval()
        imgs = []
        for i in range(10):
            noise = torch.randn((1, self.noise_dim)).to(self.device)
            img = self.gen(noise).detach().cpu().numpy().reshape(28, 28)
            imgs.append(img)
        fig, axes = plt.subplots(2, 5)
        fig.set_dpi(150)
        for i, ax in enumerate(axes.flatten()):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(imgs[i], cmap="gray")

