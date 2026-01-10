import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from core.models import Classifier


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
            recon, *_ = model(imgs)
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

    def train_classifier(self, ae_model, train_loader, val_loader, latent_dim, epochs=10, lr=1e-3, is_vae=False):
        is_pca = not isinstance(ae_model, nn.Module)
        if not is_pca:
            ae_model.eval()  # Freeze the Autoencoder
        classifier = Classifier(latent_dim,10).to(self.device)
        optimizer = optim.Adam(classifier.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

        for epoch in range(epochs):
            # --- Training Phase ---
            classifier.train()
            train_loss, correct, total = 0, 0, 0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                with torch.no_grad():
                    if is_pca:
                        x_flat = imgs.view(imgs.size(0), -1).cpu().numpy()
                        z = torch.from_numpy(ae_model.transform(x_flat)).float().to(self.device)
                    elif is_vae:
                        z = ae_model.fc_mu(ae_model.conv_enc(imgs.view(imgs.size(0), -1)))
                    else:
                        # works for both (recon, z) and (recon, z, features)
                        _, z, *_ =  ae_model(imgs)

                optimizer.zero_grad()
                outputs = classifier(z.detach())
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            history['train_loss'].append(train_loss / len(train_loader))
            history['train_acc'].append(100. * correct / total)

            # --- Validation Phase ---
            classifier.eval()
            val_loss, correct, total = 0, 0, 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    if is_pca:
                        x_flat = imgs.view(imgs.size(0), -1).cpu().numpy()
                        z = torch.from_numpy(ae_model.transform(x_flat)).float().to(self.device)
                    elif is_vae:
                        z = ae_model.fc_mu(ae_model.conv_enc(imgs.view(imgs.size(0), -1)))
                    else:
                        # works for both (recon, z) and (recon, z, features)
                        _, z, *_ = ae_model(imgs)
                    outputs = classifier(z)
                    val_loss += criterion(outputs, labels).item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

            history['val_loss'].append(val_loss / len(val_loader))
            history['val_acc'].append(100. * correct / total)

        return classifier, history