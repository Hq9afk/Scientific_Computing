import torch
from src.run_fnn import RunFNN
from src.utils.metrics import compute_metrics

def predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = RunFNN(
        start=-10,
        end=10,
        size=1000,
        batch_size=64,
        hidden_units=64,
        epochs=1000,
        lr=0.001,
        device=device
    )

    # Train model & get data
    model, x, y_true, _, _ = predictor.train_and_evaluate()

    # Predict
    model.eval()
    with torch.no_grad():
        y_pred = model(x).cpu()

    # Tính các chỉ số sai số
    metrics = compute_metrics(y_true.cpu().numpy(), y_pred.numpy())

    print("\n📊 Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    return y_true.cpu(), y_pred

if __name__ == "__main__":
    predict()
