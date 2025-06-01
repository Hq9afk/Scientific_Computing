import torch.nn as nn


class FNN(nn.Module):
    def __init__(self, input_dim, hidden_units, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_dim),
        )

    def forward(self, x):
        return self.net(x)
