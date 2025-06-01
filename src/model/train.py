class Trainer:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self, dataloader, epochs, device):
        self.model.to(device)
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / len(dataloader.dataset)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {epoch_loss:.4f}")
