from typing import Optional, Tuple

import numpy as np

from ts_bench.task_bank import TaskType


def _ensure_ndarray(obj):
    """
    Ensure object is a numpy ndarray with float32 dtype.
    """
    if isinstance(obj, np.ndarray):
        if obj.dtype != np.float32:
            return obj.astype(np.float32)
        return obj
    else:
        raise ValueError(f"Cannot convert object of type {type(obj)} to numpy array")


# Input validation
def validate_inputs(
    task_type: TaskType,
    pred: np.ndarray,
    gt: np.ndarray,
) -> Tuple[bool, Optional[str]]:
    try:
        if pred is None:
            raise ValueError("Prediction tensor is None.")
        if gt is None:
            raise ValueError("Ground truth tensor is None.")

        # rank check
        if pred.ndim != gt.ndim:
            raise ValueError(
                f"Prediction rank {pred.ndim} != ground truth rank {gt.ndim}."
            )

        # both must be 3D: [N, T or H, D]
        if pred.ndim != 3:
            raise ValueError(
                f"Tensors must be 3D [N, T_or_H, D], got pred.shape={pred.shape}."
            )

        # Dtype check
        if pred.dtype != np.float32:
            raise ValueError(f"Prediction dtype must be float32, got {pred.dtype}.")
        if gt.dtype != np.float32:
            raise ValueError(f"Ground truth dtype must be float32, got {gt.dtype}.")

        # NaN / Inf check
        if np.isnan(pred).any():
            raise ValueError("Predictions contain NaN values.")
        if np.isinf(pred).any():
            raise ValueError("Predictions contain Inf values.")

        if np.isnan(gt).any():
            raise ValueError("Ground truth contains NaN values.")

        # Shape match per task type
        if task_type == TaskType.TIME_SERIES_FORECASTING:
            # pred: [N, horizon, D]
            # gt  : [N, horizon, D]
            if pred.shape != gt.shape:
                raise ValueError(
                    f"Forecasting shape mismatch: pred {pred.shape} != gt {gt.shape}"
                )

        elif task_type == TaskType.TIME_SERIES_GENERATION:
            # pred: [N, T, D]
            # gt  : [N, T, D]
            if pred.shape != gt.shape:
                raise ValueError(
                    f"Generation shape mismatch: pred {pred.shape} != gt {gt.shape}"
                )

        else:
            raise ValueError(f"Unsupported task type {task_type}")

        return True, None

    except Exception as e:
        return False, str(e)
