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
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{self.epochs}, Loss: {epoch_loss:.4f}")
