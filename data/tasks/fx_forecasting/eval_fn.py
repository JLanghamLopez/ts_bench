import numpy as np


def rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    y_pred, y_true: [N, horizon, D] or any broadcastable shape.
    """
    return float(np.sqrt(((y_pred - y_true) ** 2).mean()))


def mae(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.abs(y_pred - y_true).mean())


def mape(y_pred: np.ndarray, y_true: np.ndarray, eps: float = 1e-8) -> float:
    return float((np.abs(y_pred - y_true) / (np.abs(y_true) + eps)).mean())


def evaluate_forecast(y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    """
    Convenience wrapper that returns all metrics at once.
    """
    assert (
        y_pred.shape == y_true.shape
    ), f"Prediction shape {y_pred.shape} != ground truth shape {y_true.shape}"
    return {
        "rmse": rmse(y_pred, y_true),
        "mae": mae(y_pred, y_true),
        "mape": mape(y_pred, y_true),
    }
