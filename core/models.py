import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearAE(nn.Module):
    def __init__(self, latent_dim=3):
        super().__init__()
        self.enc = nn.Linear(784, latent_dim, bias=False)
        self.dec = nn.Linear(latent_dim, 784, bias=False)
    def forward(self, x):
        z = self.enc(x.view(x.size(0), -1))
        return self.dec(z), z

class DeepAE(nn.Module):
    def __init__(self, latent_dim=3, non_linear=True):
        super().__init__()
        act = nn.ReLU() if non_linear else nn.Identity()
        self.enc = nn.Sequential(nn.Linear(784, 256), act, nn.Linear(256, 64), act, nn.Linear(64, latent_dim))
        output_act = nn.Sigmoid() if non_linear else nn.Identity()
        self.dec = nn.Sequential(nn.Linear(latent_dim, 64), act, nn.Linear(64, 256), act, nn.Linear(256, 784), output_act)
    def forward(self, x):
        z = self.enc(x.view(x.size(0), -1))
        return self.dec(z), z

class VAE(nn.Module):
    def __init__(self, latent_dim=3):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(784, 400), nn.ReLU())
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 400), nn.ReLU(), nn.Linear(400, 784), nn.Sigmoid())

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        h = self.encoder(x.view(-1, 784))
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

class ConvAE(nn.Module):
    def __init__(self, latent_dim=3):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(32*7*7, latent_dim)
        )
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 32*7*7), nn.ReLU(),
            nn.Unflatten(1, (32, 7, 7)),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid()
        )
    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z