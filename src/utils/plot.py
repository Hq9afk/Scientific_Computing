import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, x, y, predicted, title):
        self.x = x
        self.y = y
        self.predicted = predicted
        self.title = title

    def plot(self):
        plt.plot(self.x, self.y, label="True f(x)")
        plt.plot(self.x, self.predicted, label="Predicted", linestyle="--")
        plt.legend()
        plt.title(self.title)
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.grid(True)
        plt.show()
