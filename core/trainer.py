import torch
import torch.nn as nn
import torch.optim as optim

class ModelTrainer:
    def __init__(self, device):
        self.device = device

    def train_standard(self, model, loader, epochs=10, lr=1e-3):
        model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        for epoch in range(epochs):
            for imgs, _ in loader:
                imgs = imgs.to(self.device)
                optimizer.zero_grad()
                recon, _ = model(imgs)
                loss = criterion(recon, imgs.view(imgs.size(0), -1) if recon.dim()==2 else imgs)
                loss.backward()
                optimizer.step()
        return model

    def train_vae(self, model, loader, epochs=10, lr=1e-3, beta=1.0):
        model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        for epoch in range(epochs):
            for imgs, _ in loader:
                imgs = imgs.to(self.device)
                optimizer.zero_grad()
                recon, mu, logvar = model(imgs)
                mse = nn.functional.mse_loss(recon, imgs.view(imgs.size(0), -1), reduction='sum')
                kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = (mse + beta * kld) / imgs.size(0)
                loss.backward()
                optimizer.step()
        return model