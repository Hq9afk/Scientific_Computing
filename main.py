from src.run_fnn import RunFNN as run
import torch


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    pipeline = run(
        start=-10,  # start point of dataset
        end=10,  # end point of dataset
        size=1000,  # size of dataset
        batch_size=64,
        hidden_units=64,
        epochs=1000,
        lr=0.001,
        device=device,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
