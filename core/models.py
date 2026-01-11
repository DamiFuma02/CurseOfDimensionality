import copy
import torch
import torch.nn as nn

from core.constants import IMG_H, IMG_W, LATENT_SPACE_DIM, IMG_CHANNELS


def fc_block(in_f, out_f, non_linear=True, use_bn=True, last_layer=False, output_act=None):
    """
    Creates a standard Fully Connected block.
    - non_linear: if False, returns only nn.Linear (for linear models).
    - use_bn: whether to add BatchNorm1d.
    - last_layer: if this is the last layer of the stage (e.g., the latent projection); often, Batch Normalization (BN) or Activation is not required.
    - output_act: specific activation (e.g., Sigmoid) for the final output of the decoder.
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
    def __init__(self,data_dim=IMG_H*IMG_W, latent_dim=LATENT_SPACE_DIM):
        super().__init__()
        self.enc = fc_block(data_dim, latent_dim, non_linear=False)
        self.dec = fc_block(latent_dim, data_dim, non_linear=False)
    def forward(self, x):
        z = self.enc(x.view(x.size(0), -1))
        out = self.dec(z).view(-1, IMG_CHANNELS, IMG_H, IMG_W)
        return out, z

class DeepAE(nn.Module):
    def __init__(self, data_dim=IMG_H*IMG_W, latent_dim=LATENT_SPACE_DIM, non_linear=True):
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
        out = self.dec(z).view(-1, IMG_CHANNELS, IMG_H, IMG_W)
        return out, z

class VAE(nn.Module):
    def __init__(self, data_dim=IMG_H*IMG_W, latent_dim=LATENT_SPACE_DIM):
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
        out = self.decoder(z).view(-1, 1, IMG_H, IMG_W)
        return out, mu, logvar

class ConvAE(nn.Module):
    def __init__(self, latent_dim=LATENT_SPACE_DIM):
        super().__init__()
        # O = floor((I+2P-KS)/S)+1
        # input (B,IMG_CHANNELS,IMG_H,IMG_W) batch of B images with IMG_CHANNELS channel and IMG_H x IMG_W pixels
        self.conv_enc = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(IMG_CHANNELS, 32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2), # (B,32,14,14)
            ),
            nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2), # (B,64,7,7)
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=7), nn.BatchNorm2d(128), nn.LeakyReLU(0.2), # (B,128,1,1)
            )
        ])
        self.fc_enc = nn.Sequential(
            nn.Flatten(),  # (B,128*1*1)
            fc_block(128, latent_dim, last_layer=True)  # match the latent layer dimensionality with other AEs
        )
        self.fc_dec = nn.Sequential(
            fc_block(latent_dim, 128),
            nn.Unflatten(1, (128, 1, 1)),
        )
        # (B,128,1,1)
        self.conv_dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=7), nn.BatchNorm2d(64), nn.LeakyReLU(0.2), # (B,64,7,7)
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2), # (B,32,14,14)
            nn.ConvTranspose2d(32, IMG_CHANNELS, kernel_size=3, stride=2, padding=1, output_padding=1), nn.Sigmoid() # (B,IMG_CHANNELS,IMG_H,IMG_W)
        )

    def forward(self, x):
        enc_features = []
        latent_feature = x
        for layer in self.conv_enc:
            latent_feature = layer(latent_feature)
            enc_features.append(latent_feature)
        z = self.fc_enc(latent_feature)
        out = z
        for layer in self.fc_dec:
            out = layer(out)
        out = self.conv_dec(out)
        return out, z, enc_features


class TransformerAE(nn.Module):
    def __init__(self, img_width=IMG_W, img_height=IMG_H, patch_size=4, in_chans=IMG_CHANNELS, latent_dim=LATENT_SPACE_DIM, embed_dim=128, num_heads=8, depth=4):
        super().__init__()
        self.patch_size = patch_size
        self.hh = img_height // patch_size
        self.ww = img_width // patch_size
        self.num_patches = self.hh * self.ww

        # 1. Patch Embedding (Using Conv2d is more efficient than unfold)
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)

        # 2. ENCODER
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True,
            dim_feedforward=embed_dim*4, activation='gelu', norm_first=True
        )
        self.encoder = nn.ModuleList([copy.deepcopy(enc_layer) for _ in range(depth)])

        # 3. LATENT BOTTLENECK
        self.to_latent = nn.Linear(embed_dim * self.num_patches, latent_dim)
        self.from_latent = nn.Linear(latent_dim, embed_dim * self.num_patches)

        # 4. DECODER (Symmetric Transformer)
        dec_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True,
            dim_feedforward=embed_dim*4, activation='gelu', norm_first=True
        )
        self.decoder = nn.ModuleList([copy.deepcopy(dec_layer) for _ in range(depth)])

        # 5. RECONSTRUCTION HEAD
        self.reconstruct = nn.Linear(embed_dim, patch_size * patch_size * in_chans)

    def forward(self, x):
        b, c, h, w = x.shape

        # Patchify
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed

        # Encode + Collect features for CKA
        enc_features = []
        for block in self.encoder:
            x = block(x)
            enc_features.append(x)

        # Bottleneck
        z = self.to_latent(x.reshape(b, -1))

        # Decode
        x = self.from_latent(z).reshape(b, self.num_patches, -1)
        for block in self.decoder:
            x = block(x)

        # Reconstruct pixels
        x = self.reconstruct(x)  # (B, 49, 16)

        # Reshape patches back to image
        p = self.patch_size
        x = x.view(b, h // p, w // p, p, p)
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        out = torch.sigmoid(x.view(b, c, h, w))

        return out, z, enc_features

class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        return self.classifier(x)