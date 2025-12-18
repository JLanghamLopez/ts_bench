import os
import pickle
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from data.task_bank import TaskType


def check_response(response: str) -> None:
    """
    Check that response is a valid path to a .npy prediction file.
    """
    if not isinstance(response, str):
        raise TypeError(f"Response must be a string path, got {type(response)}")

    if not os.path.isfile(response):
        raise FileNotFoundError(f"Prediction file does not exist: {response}")

    # check if it is .npy file
    if not response.lower().endswith(".npy"):
        raise ValueError(f"Predictions must be saved as a .npy file, got: {response}")


def _ensure_tensor(obj):
    if isinstance(obj, torch.Tensor):
        return obj
    elif isinstance(obj, np.ndarray):
        return torch.tensor(obj, dtype=torch.float32)
    elif isinstance(obj, pd.DataFrame):
        return torch.tensor(obj.values, dtype=torch.float32)
    else:
        raise ValueError(f"Cannot convert object of type {type(obj)} to torch.Tensor")


def load_predictions(path: Path) -> torch.Tensor:
    # already checked, may be deleted
    if path.suffix.lower() != ".npy":
        raise ValueError(f"Prediction file must be .npy, got {path}")

    arr = np.load(path)

    if arr.dtype != np.float32:
        raise TypeError(f"Prediction array must be float32, got {arr.dtype}")

    return torch.from_numpy(arr)


def load_ground_truth(path: Path) -> torch.Tensor:
    suffix = path.suffix.lower()

    if suffix == ".pkl":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return _ensure_tensor(obj)

    elif suffix == ".npy":
        arr = np.load(path)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)
        return torch.from_numpy(arr)

    else:
        raise ValueError(f"Unsupported ground-truth file extension: {suffix}")


# Input validation
def validate_inputs(
    task_type: TaskType,
    pred: torch.Tensor,
    gt: torch.Tensor,
) -> Tuple[bool, Optional[str]]:
    try:
        if pred is None:
            raise ValueError("Prediction tensor is None.")
        if gt is None:
            raise ValueError("Ground truth tensor is None.")

        # rank check
        if pred.dim() != gt.dim():
            raise ValueError(
                f"Prediction rank {pred.dim()} != ground truth rank {gt.dim()}."
            )

        # both must be 3D: [N, T or H, D]
        if pred.dim() != 3:
            raise ValueError(
                f"Tensors must be 3D [N, T_or_H, D], got pred.shape={pred.shape}."
            )

        # Dtype check
        if pred.dtype != torch.float32:
            raise ValueError(f"Prediction dtype must be float32, got {pred.dtype}.")
        if gt.dtype != torch.float32:
            raise ValueError(f"Ground truth dtype must be float32, got {gt.dtype}.")

        # NaN / Inf check
        if torch.isnan(pred).any():
            raise ValueError("Predictions contain NaN values.")
        if torch.isinf(pred).any():
            raise ValueError("Predictions contain Inf values.")

        if torch.isnan(gt).any():
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

        # Device check: Placeholder

        return True, None

    except Exception as e:
        return False, str(e)


# Evaluation dispatcher with batch support
def run_eval(
    eval_fn: Callable[[torch.Tensor, torch.Tensor], Dict[str, float]],
    pred_tensor: torch.Tensor,
    gt_tensor: torch.Tensor,
    batch_size: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate forecasting or generation tasks.

    If batch_size is None:
         evaluate on entire test set at once.

    If batch_size is integer:
         evaluate in mini-batches and average metrics.
    """

    # whole-test evaluation (default)
    if batch_size is None:
        return eval_fn(pred_tensor, gt_tensor)

    # mini-batch evaluation
    N = pred_tensor.shape[0]
    outputs = []

    for i in range(0, N, batch_size):
        p_batch = pred_tensor[i : i + batch_size]
        g_batch = gt_tensor[i : i + batch_size]

        outputs.append(eval_fn(p_batch, g_batch))

    # aggregate batch metrics
    final_metrics = {}
    for key in outputs[0].keys():
        final_metrics[key] = float(np.mean([o[key] for o in outputs]))

    return final_metrics
