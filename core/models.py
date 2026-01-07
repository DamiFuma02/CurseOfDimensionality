import torch
import torch.nn as nn


def fc_block(in_f, out_f, non_linear=True, use_bn=True, last_layer=False, output_act=None):
    """
    Crea un blocco Fully Connected standard.
    - non_linear: se False, restituisce solo nn.Linear (per modelli lineari)
    - use_bn: se aggiungere BatchNorm1d
    - last_layer: se è l'ultimo layer dello stadio (es. la proiezione latente), spesso non vuole BN/Act
    - output_act: attivazione specifica (es. Sigmoid) per l'output finale del decoder
    """
    layers = [nn.Linear(in_f, out_f, bias=not use_bn)]

    if non_linear and not last_layer:
        if use_bn:
            layers.append(nn.BatchNorm1d(out_f))
        layers.append(nn.LeakyReLU(0.2))

    if output_act is not None:
        layers.append(output_act)

    return nn.Sequential(*layers)

class LinearAE(nn.Module):
    def __init__(self,data_dim=784, latent_dim=3):
        super().__init__()
        self.enc = fc_block(data_dim, latent_dim, non_linear=False)
        self.dec = fc_block(latent_dim, data_dim, non_linear=False)
    def forward(self, x):
        z = self.enc(x.view(x.size(0), -1))
        out = self.dec(z).view(-1, 1, 28, 28)
        return out, z

class DeepAE(nn.Module):
    def __init__(self, data_dim=784, latent_dim=3, non_linear=True):
        super().__init__()
        # Encoder
        self.enc = nn.Sequential(
            fc_block(data_dim, 512, non_linear),
            fc_block(512, 256, non_linear),
            fc_block(256, 128, non_linear),
            fc_block(128, latent_dim, non_linear, last_layer=True)
        )
        # Decoder
        out_act = nn.Sigmoid() if non_linear else None
        self.dec = nn.Sequential(
            fc_block(latent_dim, 128, non_linear),
            fc_block(128, 256, non_linear),
            fc_block(256, 512, non_linear),
            fc_block(512, data_dim, non_linear, last_layer=True, output_act=out_act)
        )
    def forward(self, x):
        z = self.enc(x.view(x.size(0), -1))
        out = self.dec(z).view(-1, 1, 28, 28)
        return out, z

class VAE(nn.Module):
    def __init__(self, data_dim=784, latent_dim=3):
        super().__init__()
        act = nn.LeakyReLU(0.2) # Uniformato a DeepAE
        self.encoder = nn.Sequential(
            fc_block(data_dim, 512),
            fc_block(512, 256),
            fc_block(256, 128)
        )
        self.fc_mu = fc_block(128, latent_dim, last_layer=True)
        self.fc_logvar = fc_block(128, latent_dim, last_layer=True)
        self.decoder = nn.Sequential(
            fc_block(latent_dim, 128),
            fc_block(128, 256),
            fc_block(256, 512),
            fc_block(512, data_dim, last_layer=True, output_act=nn.Sigmoid())
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        h = self.encoder(x.view(x.size(0), -1))
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        out = self.decoder(z).view(-1, 1, 28, 28)
        return out, mu, logvar

class ConvAE(nn.Module):
    def __init__(self, latent_dim=3):
        super().__init__()
        # O = floor((I+2P-KS)/S)+1
        # input (B,1,28,28) batch of B images with 1 channel and 28 x 28 pixels
        self.conv_enc = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2), # (B,32,14,14)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2), # (B,64,7,7)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2), # (B,128,4,4)
        )
        self.fc_enc = nn.Sequential(
            nn.Flatten(),  # (B,128*4*4)
            fc_block(128*4*4, 128), # only 1 layer to reduce the number of parameters
            fc_block(128, latent_dim, last_layer=True)  # match the latent layer dimensionality with other AEs
        )
        self.fc_dec = nn.Sequential(
            fc_block(latent_dim, 128),
            fc_block(128, 128*4*4, last_layer=True),
            nn.Unflatten(1, (128, 4, 4)),
        )
        # (B,128,4,4)
        self.conv_dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0), nn.BatchNorm2d(64), nn.LeakyReLU(0.2), # (B,64,7,7)
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2), # (B,32,14,14)
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1), nn.Sigmoid() # (B,1,28,28)
        )

    def forward(self, x):
        z = self.fc_enc(self.conv_enc(x))
        out = self.conv_dec(self.fc_dec(z))
        return out, z


class TransformerAE(nn.Module):
    def __init__(self,data_dim=784, latent_dim=3, patch_size=4, embed_dim=128, num_heads=8, num_layers=2):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (28 // patch_size) ** 2  # Per MNIST: (28/4)^2 = 49 patches

        self.patch_embed = nn.Linear(patch_size * patch_size, embed_dim)

        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True, dropout=0,
            dim_feedforward=512, activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

        self.to_latent = nn.Linear(embed_dim * num_patches, latent_dim)

        self.from_latent = nn.Linear(latent_dim, 512)
        self.decoder = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, data_dim),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.decoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w = x.shape
        p = self.patch_size

        patches = x.unfold(2, p, p).unfold(3, p, p).reshape(b, -1, p * p)

        x = self.patch_embed(patches) + self.pos_embed

        x = self.transformer_encoder(x)
        x = self.norm(x)  # LayerNorm fondamentale

        z = self.to_latent(x.reshape(b, -1))

        rec = self.decoder(self.from_latent(z))

        return rec.view(-1, 1, 28, 28), z