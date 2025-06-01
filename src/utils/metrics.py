from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def compute_metrics(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    return {"MAE": mae, "MSE": mse, "RMSE": rmse}
