import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split


class Dataset:
    def __init__(self, start, end, size, device, val_ratio=0.2, batch_size=64):
        self.start = start
        self.end = end
        self.size = size
        self.device = device
        self.val_ratio = val_ratio
        self.batch_size = batch_size

    def generate_data(self):
        x = np.linspace(self.start, self.end, self.size, dtype=np.float32).reshape(-1, 1)
        y = self._function(x)
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device)
        return x_tensor, y_tensor

    def get_loaders(self):
        x, y = self.generate_data()
        dataset = TensorDataset(x, y)
        val_size = int(len(dataset) * self.val_ratio)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        return x, y, train_loader, val_loader

    @staticmethod
    def _function(x):
        return np.sin(x) + 0.3 * x**2
