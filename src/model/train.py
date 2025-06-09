import torch


class Trainer:
    def __init__(self, model, criterion, optimizer, dataloader, epochs, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.epochs = epochs
        self.device = device

    def train(self):
        self.model.to(self.device)
        losses = []
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            for inputs, targets in self.dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / len(self.dataloader.dataset)
            losses.append(epoch_loss)
        return losses

    def train_with_validation(self, val_loader):
        self.model.to(self.device)
        train_losses = []
        val_losses = []
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            for inputs, targets in self.dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / len(self.dataloader.dataset)
            train_losses.append(epoch_loss)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_inputs, val_targets in val_loader:
                    val_inputs, val_targets = (
                        val_inputs.to(self.device),
                        val_targets.to(self.device),
                    )
                    val_outputs = self.model(val_inputs)
                    loss = self.criterion(val_outputs, val_targets)
                    val_loss += loss.item() * val_inputs.size(0)
            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)
            if (epoch + 1) % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/{self.epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

        return train_losses, val_losses
