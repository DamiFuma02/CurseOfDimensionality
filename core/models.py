import torch
import torch.nn as nn


class LinearAE(nn.Module):
    def __init__(self,data_dim=784, latent_dim=3):
        super().__init__()
        self.enc = nn.Linear(data_dim, latent_dim, bias=False)
        self.dec = nn.Linear(latent_dim, data_dim, bias=False)
    def forward(self, x):
        z = self.enc(x.view(x.size(0), -1))
        out = self.dec(z).view(-1, 1, 28, 28)
        return out, z

class DeepAE(nn.Module):
    def __init__(self,data_dim=784, latent_dim=3, non_linear=True):
        super().__init__()
        act = nn.LeakyReLU(0.2) if non_linear else nn.Identity()
        def block(in_f, out_f):
            layers = [nn.Linear(in_f, out_f)]
            if non_linear: layers.append(nn.BatchNorm1d(out_f))
            layers.append(act)
            return nn.Sequential(*layers)

        self.enc = nn.Sequential(
            block(data_dim, 512), block(512, 256), block(256, 128),
            nn.Linear(128, latent_dim)
        )
        output_act = nn.Sigmoid() if non_linear else nn.Identity()
        self.dec = nn.Sequential(
            block(latent_dim, 128), block(128, 256), block(256, 512),
            nn.Linear(512, data_dim), output_act
        )
    def forward(self, x):
        z = self.enc(x.view(x.size(0), -1))
        out = self.dec(z).view(-1, 1, 28, 28)
        return out, z

class VAE(nn.Module):
    def __init__(self, data_dim=784, latent_dim=3):
        super().__init__()
        self._data_dim = data_dim
        act = nn.LeakyReLU(0.2) # Uniformato a DeepAE
        self.encoder = nn.Sequential(
            nn.Linear(data_dim, 512), nn.BatchNorm1d(512), act,
            nn.Linear(512, 256), nn.BatchNorm1d(256), act,
            nn.Linear(256, 128), nn.BatchNorm1d(128), act
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.BatchNorm1d(128), act,
            nn.Linear(128, 256), nn.BatchNorm1d(256), act,
            nn.Linear(256, 512), nn.BatchNorm1d(512), act,
            nn.Linear(512, data_dim), nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        h = self.encoder(x.view(-1, self._data_dim))
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        out = self.decoder(z).view(-1, 1, 28, 28)
        return out, mu, logvar

class ConvAE(nn.Module):
    def __init__(self, latent_dim=3):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Flatten(), nn.Linear(128*4*4, latent_dim)
        )
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 128*4*4), nn.LeakyReLU(0.2),
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=0), nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid()
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
        # dropout=0 per confronto equo sulla capacità di ricostruzione
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True, dropout=0)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.to_latent = nn.Linear(embed_dim * num_patches, latent_dim)

        self.from_latent = nn.Linear(latent_dim, embed_dim * num_patches)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim * num_patches, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, data_dim), nn.Sigmoid()
        )

    def forward(self, x):
        p = self.patch_size
        patches = x.unfold(2, p, p).unfold(3, p, p).reshape(x.shape[0], -1, p * p)
        x_emb = self.patch_embed(patches) + self.pos_embed
        x_trans = self.transformer_encoder(x_emb)
        z = self.to_latent(x_trans.reshape(x_trans.shape[0], -1))
        rec = self.decoder(self.from_latent(z))
        return rec.view(-1, 1, 28, 28), z