# Update Plotter to accept val_losses and plot both
import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, x, y, predicted, title, losses=None, val_losses=None):
        self.x = x
        self.y = y
        self.predicted = predicted
        self.title = title
        self.losses = losses
        self.val_losses = val_losses

    def plot(self):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.x, self.y, label="True f(x)")
        plt.plot(self.x, self.predicted, label="Predicted", linestyle="--")
        plt.legend()
        plt.title(self.title)
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.grid(True)

        if self.losses is not None:
            plt.subplot(1, 2, 2)
            plt.plot(self.losses, label="Train Loss")
            if self.val_losses is not None:
                plt.plot(self.val_losses, label="Val Loss")
            plt.title("Training/Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.grid(True)
            plt.legend()

        plt.tight_layout()
        plt.show()
