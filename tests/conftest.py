"""
Shared pytest fixtures for torch-to-numpy migration tests.
"""

import numpy as np
import pytest


@pytest.fixture
def seed():
    """Fixed random seed for reproducibility."""
    return 42


@pytest.fixture
def sample_forecasting_data(seed):
    """
    Generate deterministic sample data for forecasting tests.
    Shape: [N, horizon, D] = [100, 24, 5]
    """
    np.random.seed(seed)
    pred = np.random.randn(100, 24, 5).astype(np.float32)
    gt = np.random.randn(100, 24, 5).astype(np.float32)
    return pred, gt


@pytest.fixture
def sample_generation_data(seed):
    """
    Generate deterministic sample data for generation tests.
    Shape: [B, T, D] = [50, 64, 4]
    """
    np.random.seed(seed)
    x_fake = np.random.randn(50, 64, 4).astype(np.float32)
    x_real = np.random.randn(50, 64, 4).astype(np.float32)
    return x_fake, x_real


@pytest.fixture
def sample_3d_tensor(seed):
    """
    Generate a single 3D tensor for ACF tests.
    Shape: [B, T, D] = [32, 48, 3]
    """
    np.random.seed(seed)
    return np.random.randn(32, 48, 3).astype(np.float32)


@pytest.fixture
def small_generation_data(seed):
    """
    Smaller data for faster histogram tests.
    Shape: [B, T, D] = [20, 16, 2]
    """
    np.random.seed(seed + 1)  # Different seed for variety
    x_fake = np.random.randn(20, 16, 2).astype(np.float32)
    x_real = np.random.randn(20, 16, 2).astype(np.float32)
    return x_fake, x_real

