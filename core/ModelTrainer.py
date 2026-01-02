import torch
import torch.nn as nn
import torch.optim as optim

class ModelTrainer:
    def __init__(self, device):
        self.device = device

    def _get_val_loss(self, model, val_loader, criterion, is_vae=False, beta=1.0):
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, _ in val_loader:
                imgs = imgs.to(self.device)
                num_pixels = imgs.numel() / imgs.size(0)  # es. 784 per MNIST
                if is_vae:
                    recon, mu, logvar = model(imgs)
                    target = imgs.view(imgs.size(0), -1)
                    mse = nn.functional.mse_loss(recon, target, reduction='mean')
                    kld = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
                    kld_normalized = kld / num_pixels
                    loss = mse + beta * kld_normalized
                else:
                    recon, _ = model(imgs)
                    target = imgs.view(imgs.size(0), -1) if recon.dim() == 2 else imgs
                    loss = criterion(recon, target)
                val_loss += loss.item()
        return val_loss / len(val_loader)

    def train_standard(self, model, train_loader, val_loader, epochs=10, lr=1e-3):
        model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        history = {'train_loss': [], 'val_loss': []}
        for epoch in range(epochs):
            train_loss = 0.0
            model.train()
            for imgs, _ in train_loader:
                imgs = imgs.to(self.device)
                optimizer.zero_grad()
                recon, _ = model(imgs)
                loss = criterion(recon, imgs.view(imgs.size(0), -1) if recon.dim()==2 else imgs)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            avg_train = train_loss / len(train_loader)
            avg_val = self._get_val_loss(model, val_loader, criterion)
            history['train_loss'].append(avg_train)
            history['val_loss'].append(avg_val)
        return model, history

    def train_vae(self, model, train_loader, val_loader, epochs=10, lr=1e-3, beta=1.0):
        model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        history = {'train_loss': [], 'val_loss': []}
        for epoch in range(epochs):
            train_loss = 0.0
            model.train()
            for imgs, _ in train_loader:
                imgs = imgs.to(self.device)
                num_pixels = imgs.numel() / imgs.size(0)
                optimizer.zero_grad()
                recon, mu, logvar = model(imgs)
                mse = nn.functional.mse_loss(recon, imgs.view(imgs.size(0), -1), reduction='mean')
                kld = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
                kld_normalized = kld / num_pixels

                loss = mse + beta * kld_normalized
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            avg_train = train_loss / len(train_loader)
            avg_val = self._get_val_loss(model, val_loader, None, is_vae=True, beta=beta)

            history['train_loss'].append(avg_train)
            history['val_loss'].append(avg_val)
        return model, history