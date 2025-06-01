from src.data.dataset import Dataset
from src.model.fnn import FNN
from src.model.train import Trainer
from src.utils.plot import Plotter

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim


class RunFNN:
    def __init__(self, start, end, size, batch_size, hidden_units, epochs, lr, device):
        self.device = device
        self.data = Dataset(start, end, size, device)
        self.batch_size = batch_size
        self.hidden_units = hidden_units
        self.epochs = epochs
        self.lr = lr

    def prepare_data(self):
        x, y = self.data.generate_data()
        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return x, y, dataloader

    def build_model(self):
        return FNN(input_dim=1, hidden_units=self.hidden_units, output_dim=1)

    def run(self):
        x, y, dataloader = self.prepare_data()
        model = self.build_model()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), self.lr)
        trainer = Trainer(model, criterion, optimizer)
        trainer.train(dataloader, self.epochs, self.device)
        model.eval()
        with torch.no_grad():
            predictions = model(x).cpu().numpy()
        plotter = Plotter(x.cpu().numpy(), y.cpu().numpy(), predictions, title="FNN Regression")
        plotter.plot()
