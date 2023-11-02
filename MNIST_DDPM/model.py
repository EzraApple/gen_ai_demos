import torch
from torch import nn
import matplotlib.pyplot as plt


class Block(nn.Module):

    def __init__(self, shape, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(shape),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.SiLU()
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):

    def __init__(self, hidden_channels, num_steps, time_embedding_dim):
        super().__init__()
        self.time_embedding = nn.Embedding(num_steps, time_embedding_dim)
        self.time_embedding.weight.data = self._sine_embedding(num_steps, time_embedding_dim)
        self.time_embedding.requires_grad_(False)

        # down stage
        self.time1 = self._make_te(time_embedding_dim, 1)
        self.block1 = nn.Sequential(
            Block((1, 28, 28), 1, hidden_channels),
            Block((hidden_channels, 28, 28), hidden_channels, hidden_channels)
        )
        self.down1 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1)

        self.time2 = self._make_te(time_embedding_dim, hidden_channels)
        self.block2 = nn.Sequential(
            Block((hidden_channels, 14, 14), hidden_channels, 2 * hidden_channels),
            Block((2 * hidden_channels, 14, 14), 2 * hidden_channels, 2 * hidden_channels)
        )
        self.down2 = nn.Conv2d(2 * hidden_channels, 2 * hidden_channels, kernel_size=4, stride=2, padding=1)

        self.time3 = self._make_te(time_embedding_dim, 2 * hidden_channels)
        self.block3 = nn.Sequential(
            Block((2 * hidden_channels, 7, 7), 2 * hidden_channels, 4 * hidden_channels),
            Block((4 * hidden_channels, 7, 7), 4 * hidden_channels, 4 * hidden_channels)
        )
        self.down3 = nn.Conv2d(4 * hidden_channels, 4 * hidden_channels, kernel_size=4, stride=2, padding=1)

        # bottleneck
        self.time_middle = self._make_te(time_embedding_dim, 4 * hidden_channels)
        self.block_middle = nn.Sequential(
            Block((4 * hidden_channels, 3, 3), 4 * hidden_channels, 2 * hidden_channels),
            Block((2 * hidden_channels, 3, 3), 2 * hidden_channels, 2 * hidden_channels),
            Block((2 * hidden_channels, 3, 3), 2 * hidden_channels, 4 * hidden_channels)
        )

        # up stage
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(4 * hidden_channels, 4 * hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(4 * hidden_channels, 4 * hidden_channels, kernel_size=2, stride=1)
        )
        self.time4 = self._make_te(time_embedding_dim, 8 * hidden_channels)
        self.block4 = nn.Sequential(
            Block((8 * hidden_channels, 7, 7), 8 * hidden_channels, 4 * hidden_channels),
            Block((4 * hidden_channels, 7, 7), 4 * hidden_channels, 2 * hidden_channels)
        )

        self.up2 = nn.ConvTranspose2d(2 * hidden_channels, 2 * hidden_channels, kernel_size=4, stride=2, padding=1)
        self.time5 = self._make_te(time_embedding_dim, 4 * hidden_channels)
        self.block5 = nn.Sequential(
            Block((4 * hidden_channels, 14, 14), 4 * hidden_channels, 2 * hidden_channels),
            Block((2 * hidden_channels, 14, 14), 2 * hidden_channels, hidden_channels),
        )

        self.up3 = nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1)
        self.time6 = self._make_te(time_embedding_dim, 2 * hidden_channels)
        self.block6 = nn.Sequential(
            Block((2 * hidden_channels, 28, 28), 2 * hidden_channels, 2 * hidden_channels),
            Block((2 * hidden_channels, 28, 28), 2 * hidden_channels, hidden_channels),
        )

        self.out = nn.Sequential(
            Block((hidden_channels, 28, 28), hidden_channels, hidden_channels),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x, t):
        t = self.time_embedding(t)
        batch_size = x.shape[0]

        img1 = self.block1(x + self.time1(t).reshape(batch_size, -1, 1, 1))
        img2 = self.block2(self.down1(img1) + self.time2(t).reshape(batch_size, -1, 1, 1))
        img3 = self.block3(self.down2(img2) + self.time3(t).reshape(batch_size, -1, 1, 1))

        bottleneck = self.block_middle(self.down3(img3) + self.time_middle(t).reshape(batch_size, -1, 1, 1))

        comb1 = torch.cat((img3, self.up1(bottleneck)), dim=1)
        img4 = self.block4(comb1 + self.time4(t).reshape(batch_size, -1, 1, 1))

        comb2 = torch.cat((img2, self.up2(img4)), dim=1)
        img5 = self.block5(comb2 + self.time5(t).reshape(batch_size, -1, 1, 1))

        comb3 = torch.cat((img1, self.up3(img5)), dim=1)
        img6 = self.block6(comb3 + self.time6(t).reshape(batch_size, -1, 1, 1))

        out = self.out(img6)

        return out

    @staticmethod
    def _sine_embedding(n_steps, time_embedding):
        embedding = torch.zeros(n_steps, time_embedding)
        wk = torch.tensor([1 / 10_000 ** (2 * j / time_embedding) for j in range(time_embedding)])
        wk = wk.reshape((1, time_embedding))
        t = torch.arange(n_steps).reshape((n_steps, 1))
        embedding[:, ::2] = torch.sin(t * wk[:, ::2])
        embedding[:, 1::2] = torch.cos(t * wk[:, ::2])
        return embedding

    @staticmethod
    def _make_te(time_embedding_dim, out_dimension):
        return nn.Sequential(
            nn.Linear(time_embedding_dim, out_dimension),
            nn.SiLU(),
            nn.Linear(out_dimension, out_dimension)
        )


class DDPM(nn.Module):
    def __init__(self, unet, num_steps, min_beta=1e-4, max_beta=2e-2, device='cpu'):
        super().__init__()
        self.model = unet.to(device)
        self.num_steps = num_steps
        self.betas = torch.linspace(min_beta, max_beta, num_steps).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]).to(device)
        self.device = device

    def forward(self, image, timestep, noise=None):
        batch = image.shape[0]
        alpha_bar = self.alpha_bars[timestep]

        if noise is None:
            noise = torch.randn_like(image).to(self.device)

        weighted_image = alpha_bar.sqrt().reshape(batch, 1, 1, 1) * image
        weighted_noise = (1 - alpha_bar).sqrt().reshape(batch, 1, 1, 1) * noise

        return weighted_image + weighted_noise

    def reverse(self, image, timestep):
        return self.model(image, timestep)

    def generate_images(self, grid_shape):
        l, w = grid_shape
        samples = l * w

        with torch.no_grad():

            # Starting from random noise
            x = torch.randn(samples, 1, 28, 28).to(self.device)

            for idx, t in enumerate(list(range(self.num_steps))[::-1]):
                # Estimating noise to be removed
                time_tensor = (torch.ones(samples, 1) * t).to(self.device).long()
                eta_theta = self.reverse(x, time_tensor)

                alpha_t = self.alphas[t]
                alpha_t_bar = self.alpha_bars[t]

                # Partially denoising the image
                x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

                if t > 0:
                    z = torch.randn(samples, 1, 28, 28).to(self.device)

                    # Option 1: sigma_t squared = beta_t
                    beta_t = self.betas[t]
                    sigma_t = beta_t.sqrt()

                    # Option 2: sigma_t squared = beta_tilda_t
                    # prev_alpha_t_bar = ddpm.alpha_bars[t-1] if t > 0 else ddpm.alphas[0]
                    # beta_tilda_t = ((1 - prev_alpha_t_bar)/(1 - alpha_t_bar)) * beta_t
                    # sigma_t = beta_tilda_t.sqrt()

                    # Adding some more noise like in Langevin Dynamics fashion
                    x = x + sigma_t * z

        fig, axes = plt.subplots(l, w)
        fig.set_dpi(200)
        for i, ax in enumerate(axes.flatten()):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(x[i].cpu().squeeze(), cmap="gray")

    def demo_forward(self, image, percents=(0.25, 0.5, 0.75, 1)):
        timesteps = torch.tensor([int(percent * self.num_steps) - 1 for percent in percents])
        image = image.to(self.device)
        images = [image] + [self.forward(image, t) for t in timesteps]

        fig, axes = plt.subplots(1, len(images))
        fig.set_dpi(200)
        for i, ax in enumerate(axes):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(images[i].cpu().squeeze(), cmap="gray")
