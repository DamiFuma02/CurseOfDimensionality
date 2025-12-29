import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearAE(nn.Module):
    def __init__(self,data_dim=784, latent_dim=3):
        super().__init__()
        self.enc = nn.Linear(data_dim, latent_dim, bias=False)
        self.dec = nn.Linear(latent_dim, data_dim, bias=False)
    def forward(self, x):
        z = self.enc(x.view(x.size(0), -1))
        return self.dec(z), z

class DeepAE(nn.Module):
    def __init__(self,data_dim=784, latent_dim=3, non_linear=True):
        super().__init__()
        act = nn.ReLU() if non_linear else nn.Identity()
        self.enc = nn.Sequential(nn.Linear(data_dim, 512), act,nn.Linear(512, 256), act,nn.Linear(256, 128), act,nn.Linear(128, latent_dim))
        output_act = nn.Sigmoid() if non_linear else nn.Identity()
        self.dec = nn.Sequential(nn.Linear(latent_dim, 64), act, nn.Linear(64, 256), act, nn.Linear(256, data_dim), output_act)
    def forward(self, x):
        z = self.enc(x.view(x.size(0), -1))
        return self.dec(z), z

class VAE(nn.Module):
    def __init__(self,data_dim=784, latent_dim=3):
        super().__init__()
        self._data_dim = data_dim
        self.encoder = nn.Sequential(nn.Linear(self._data_dim, 512), nn.ReLU())
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 512), nn.ReLU(), nn.Linear(512, self._data_dim), nn.Sigmoid())

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        h = self.encoder(x.view(-1, self._data_dim))
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

class ConvAE(nn.Module):
    def __init__(self,data_dim=784, latent_dim=3):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(), # Result: 128x4x4
            nn.Flatten(), nn.Linear(128*4*4, latent_dim)
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


class TransformerAE(nn.Module):
    def __init__(self,data_dim=784, latent_dim=3, patch_size=4, embed_dim=128, num_heads=8, num_layers=2):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (28 // patch_size) ** 2  # Per MNIST: (28/4)^2 = 49 patches

        self.patch_embed = nn.Linear(patch_size * patch_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.to_latent = nn.Linear(embed_dim * num_patches, latent_dim)

        self.from_latent = nn.Linear(latent_dim, embed_dim * num_patches)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim * num_patches, 512),
            nn.ReLU(),
            nn.Linear(512, data_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: [B, 1, 28, 28]
        p = self.patch_size
        patches = x.unfold(2, p, p).unfold(3, p, p).reshape(x.shape[0], -1, p * p)

        # Embedding + Position
        x = self.patch_embed(patches) + self.pos_embed

        # Attention
        x = self.transformer_encoder(x)

        # Latent
        z = self.to_latent(x.reshape(x.shape[0], -1))

        # Reconstruction
        rec = self.decoder(self.from_latent(z))
        return rec.view(-1, 1, 28, 28), z