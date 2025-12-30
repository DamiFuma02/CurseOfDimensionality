import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np


class DataManager:
    def __init__(self, batch_size=128, limit_samples=2000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        # Configurazione Semantica FashionMNIST
        self.classes_orig = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                             'Boot']
        self.semantic_order = [0, 6, 2, 4, 3, 1, 8, 5, 7, 9]
        self.semantic_names = [self.classes_orig[i] for i in self.semantic_order]
        self.remap_dict = {orig: sem for sem, orig in enumerate(self.semantic_order)}

        transform = transforms.Compose([transforms.ToTensor()])
        self.train_set = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        self.test_set = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

        # Subset per valutazione
        indices = np.random.choice(len(self.test_set), limit_samples, replace=False)
        self.eval_loader = DataLoader(Subset(self.test_set, indices), batch_size=limit_samples, shuffle=False)

    def get_loaders(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True), self.eval_loader

    def remap_labels(self, labels):
        return np.array([self.remap_dict[int(l)] for l in labels])