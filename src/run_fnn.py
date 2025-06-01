from src.data.dataset import Dataset
from src.model.fnn import FNN
from src.model.train import Trainer
from src.utils.plot import Plotter

import torch
import torch.nn as nn
import torch.optim as optim


class RunFNN:
    def __init__(self, start, end, size, batch_size, hidden_units, epochs, lr, device, val_ratio=0.2):
        self.device = device
        self.data = Dataset(start, end, size, device, val_ratio=val_ratio, batch_size=batch_size)
        self.hidden_units = hidden_units
        self.epochs = epochs
        self.lr = lr

    def build_model(self):
        return FNN(input_dim=1, hidden_units=self.hidden_units, output_dim=1).to(self.device)

    def train_and_evaluate(self):
        x, y, train_loader, val_loader = self.data.get_loaders()
        model = self.build_model()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        trainer = Trainer(model, criterion, optimizer, train_loader, self.epochs, self.device)
        train_losses, val_losses = trainer.train_with_validation(val_loader)
        return model, x, y, train_losses, val_losses

    def plot_results(self, model, x, y, train_losses, val_losses):
        model.eval()
        with torch.no_grad():
            predictions = model(x).cpu().numpy()
        Plotter(
            x.cpu().numpy(),
            y.cpu().numpy(),
            predictions,
            title="FNN Regression",
            losses=train_losses,
            val_losses=val_losses,
        ).plot()

    def run(self):
        model, x, y, train_losses, val_losses = self.train_and_evaluate()
        self.plot_results(model, x, y, train_losses, val_losses)
