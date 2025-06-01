import numpy as np
import torch


class Dataset:
    def __init__(self, start, end, size, device):
        self.start = start
        self.end = end
        self.size = size
        self.device = device

    def generate_data(self):
        x = np.linspace(self.start, self.end, self.size, dtype=np.float32).reshape(-1, 1)
        y = self._function(x)
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device)
        return x_tensor, y_tensor

    @staticmethod
    def _function(x):
        return np.sin(x) + 0.3 + x**2
