import torch


def rmse(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    y_pred, y_true: [N, horizon, D] or any broadcastable shape.
    """
    return torch.sqrt(((y_pred - y_true) ** 2).mean()).item()


def mae(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    return (y_pred - y_true).abs().mean().item()


def mape(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-8) -> float:
    return ((y_pred - y_true).abs() / (y_true.abs() + eps)).mean().item()


def evaluate_forecast(y_pred: torch.Tensor, y_true: torch.Tensor) -> dict[str, float]:
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
