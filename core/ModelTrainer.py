import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ModelTrainer:
    def __init__(self, device):
        self.device = device

    def _get_loss(self, model, imgs, criterion, is_vae=False, beta=1.0):
        imgs = imgs.to(self.device)
        if is_vae:
            recon, mu, logvar = model(imgs)
            mse = F.mse_loss(recon, imgs, reduction='mean')
            # KLD normalizzata per dimensione immagine per bilanciare i pesi della loss
            kld = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            return mse + beta * (kld / 784)
        else:
            recon, _ = model(imgs)
            return criterion(recon, imgs)

    def train(self, model, train_loader, val_loader, epochs=10, lr=1e-3, is_vae=False, beta=1.0):
        model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for imgs, _ in train_loader:
                optimizer.zero_grad()
                loss = self._get_loss(model, imgs, criterion, is_vae, beta)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for imgs, _ in val_loader:
                    val_loss += self._get_loss(model, imgs, criterion, is_vae, beta).item()

            history['train_loss'].append(train_loss / len(train_loader))
            history['val_loss'].append(val_loss / len(val_loader))
        return model, history