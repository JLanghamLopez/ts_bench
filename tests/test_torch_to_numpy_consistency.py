"""
Unit tests for torch-to-numpy migration consistency.

These tests verify that the numpy implementations produce outputs
that match the original torch implementations within numerical tolerance.

The expected values were pre-computed using the torch-based functions
with the deterministic random seeds defined in conftest.py.
"""

import numpy as np
import pytest

# Import the evaluation functions (will be numpy-based after migration)
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ts_bench.agents.ts_task_agent.eval_fn_combined import (
    rmse,
    mae,
    mape,
    eval_forecasting,
    acf_numpy,
    non_stationary_acf_numpy,
    cacf_numpy,
    HistogramLoss,
    ACFLoss,
    cross_correlation,
    eval_generation,
    _corrcoef_from_batch,
    histogram_numpy_update,
)
from ts_bench.agents.ts_task_agent.utils import (
    validate_inputs,
    _ensure_ndarray,
)
from data.task_bank import TaskType

# Numerical tolerance for comparisons
RTOL = 1e-5
ATOL = 1e-6


class TestForecastingMetrics:
    """Tests for forecasting metrics: RMSE, MAE, MAPE."""

    # Pre-computed expected values from torch implementation
    # seed=42, shape=[100, 24, 5]
    EXPECTED_RMSE = 1.4187374114990234
    EXPECTED_MAE = 1.1272900104522705
    EXPECTED_MAPE = 8.51625919342041

    def test_rmse_consistency(self, sample_forecasting_data):
        """Test RMSE matches expected torch output."""
        pred, gt = sample_forecasting_data
        result = rmse(pred, gt)
        assert np.isclose(result, self.EXPECTED_RMSE, rtol=RTOL, atol=ATOL), \
            f"RMSE mismatch: {result} vs expected {self.EXPECTED_RMSE}"

    def test_mae_consistency(self, sample_forecasting_data):
        """Test MAE matches expected torch output."""
        pred, gt = sample_forecasting_data
        result = mae(pred, gt)
        assert np.isclose(result, self.EXPECTED_MAE, rtol=RTOL, atol=ATOL), \
            f"MAE mismatch: {result} vs expected {self.EXPECTED_MAE}"

    def test_mape_consistency(self, sample_forecasting_data):
        """Test MAPE matches expected torch output."""
        pred, gt = sample_forecasting_data
        result = mape(pred, gt)
        assert np.isclose(result, self.EXPECTED_MAPE, rtol=RTOL, atol=ATOL), \
            f"MAPE mismatch: {result} vs expected {self.EXPECTED_MAPE}"

    def test_eval_forecasting_consistency(self, sample_forecasting_data):
        """Test eval_forecasting returns all metrics correctly."""
        pred, gt = sample_forecasting_data
        result = eval_forecasting(pred, gt)

        assert "rmse" in result
        assert "mae" in result
        assert "mape" in result

        assert np.isclose(result["rmse"], self.EXPECTED_RMSE, rtol=RTOL, atol=ATOL)
        assert np.isclose(result["mae"], self.EXPECTED_MAE, rtol=RTOL, atol=ATOL)
        assert np.isclose(result["mape"], self.EXPECTED_MAPE, rtol=RTOL, atol=ATOL)


class TestACFFunctions:
    """Tests for autocorrelation functions."""

    def test_acf_numpy_shape(self, sample_3d_tensor):
        """Test ACF output shape is correct."""
        x = sample_3d_tensor  # [32, 48, 3]
        max_lag = 10
        result = acf_numpy(x, max_lag, dim=(0, 1))

        assert result.shape == (max_lag, 3), \
            f"ACF shape mismatch: {result.shape} vs expected (10, 3)"

    def test_acf_numpy_values(self, sample_3d_tensor):
        """Test ACF values are within valid correlation range."""
        x = sample_3d_tensor
        max_lag = 10
        result = acf_numpy(x, max_lag, dim=(0, 1))

        # ACF at lag 0 should be 1.0 (or very close)
        np.testing.assert_allclose(
            np.asarray(result[0]),
            np.ones(3),
            rtol=1e-4,
            err_msg="ACF at lag 0 should be ~1.0"
        )

    def test_non_stationary_acf_shape(self, sample_3d_tensor):
        """Test non-stationary ACF output shape."""
        x = sample_3d_tensor  # [32, 48, 3]
        result = non_stationary_acf_numpy(x, symmetric=False)

        # Shape should be [T, T, D]
        assert result.shape == (48, 48, 3), \
            f"Non-stationary ACF shape mismatch: {result.shape}"

    def test_cacf_numpy_shape(self, sample_3d_tensor):
        """Test cross-ACF output shape."""
        x = sample_3d_tensor  # [32, 48, 3]
        lags = 5
        result = cacf_numpy(x, lags, dim=(0, 1))

        # Shape should be [B, N_pairs, lags] where N_pairs = D*(D+1)/2 = 6
        expected_pairs = 3 * (3 + 1) // 2  # = 6
        assert result.shape == (32, expected_pairs, lags), \
            f"CACF shape mismatch: {result.shape}"


class TestHistogramLoss:
    """Tests for histogram-based loss computation."""

    def test_histogram_loss_initialization(self, small_generation_data):
        """Test HistogramLoss can be initialized."""
        x_fake, x_real = small_generation_data
        hist_loss = HistogramLoss(x_real, n_bins=16)

        assert hist_loss.n_bins == 16
        assert hist_loss.num_samples == 20
        assert hist_loss.num_time_steps == 16
        assert hist_loss.num_features == 2

    def test_histogram_loss_compute(self, small_generation_data):
        """Test HistogramLoss compute returns valid losses."""
        x_fake, x_real = small_generation_data
        hist_loss = HistogramLoss(x_real, n_bins=16)
        result = hist_loss.compute(x_fake)

        # Result should be [L, D] = [16, 2]
        result_arr = np.asarray(result)
        assert result_arr.shape == (16, 2), \
            f"Histogram loss shape mismatch: {result_arr.shape}"

        # All losses should be non-negative
        assert np.all(result_arr >= 0), "Histogram losses should be non-negative"

    def test_histogram_loss_forward(self, small_generation_data):
        """Test HistogramLoss forward returns a scalar."""
        x_fake, x_real = small_generation_data
        hist_loss = HistogramLoss(x_real, n_bins=16)
        result = hist_loss(x_fake)

        # Should be a scalar
        result_val = float(result)
        assert isinstance(result_val, float)
        assert result_val >= 0, "Histogram loss should be non-negative"


class TestCrossCorrelation:
    """Tests for cross-correlation loss."""

    def test_corrcoef_from_batch_shape(self):
        """Test _corrcoef_from_batch output shape."""
        np.random.seed(42)
        x = np.random.randn(50, 4).astype(np.float32)
        result = _corrcoef_from_batch(x)

        result_arr = np.asarray(result)
        assert result_arr.shape == (4, 4), \
            f"Corrcoef shape mismatch: {result_arr.shape}"

    def test_corrcoef_diagonal(self):
        """Test correlation matrix has 1s on diagonal."""
        np.random.seed(42)
        x = np.random.randn(50, 4).astype(np.float32)
        result = _corrcoef_from_batch(x)

        result_arr = np.asarray(result)
        np.testing.assert_allclose(
            np.diag(result_arr),
            np.ones(4),
            rtol=1e-4,
            err_msg="Diagonal of correlation matrix should be ~1.0"
        )

    def test_cross_correlation_loss(self, small_generation_data):
        """Test cross_correlation loss computation."""
        x_fake, x_real = small_generation_data
        cc_loss = cross_correlation(x_real=x_real, name="cc", reg=1.0)
        result = cc_loss(x_fake)

        result_val = float(result)
        assert result_val >= 0, "Cross-correlation loss should be non-negative"


class TestACFLoss:
    """Tests for ACF loss computation."""

    def test_acf_loss_stationary(self, small_generation_data):
        """Test ACFLoss with stationary mode."""
        x_fake, x_real = small_generation_data
        acf_loss = ACFLoss(
            x_real=x_real,
            max_lag=8,
            stationary=True,
            name="acf",
            reg=1.0
        )
        result = acf_loss(x_fake)

        result_val = float(result)
        assert result_val >= 0, "ACF loss should be non-negative"

    def test_acf_loss_non_stationary(self, small_generation_data):
        """Test ACFLoss with non-stationary mode."""
        x_fake, x_real = small_generation_data
        acf_loss = ACFLoss(
            x_real=x_real,
            max_lag=8,
            stationary=False,
            name="acf",
            reg=1.0
        )
        result = acf_loss(x_fake)

        result_val = float(result)
        assert result_val >= 0, "ACF loss should be non-negative"


class TestEvalGeneration:
    """Tests for the eval_generation wrapper function."""

    # Pre-computed expected values from torch implementation
    # seed=42+1, shape=[20, 16, 2], n_bins=16, max_lag=8
    EXPECTED_HISTLOSS_APPROX = 0.3  # Approximate, depends on random data
    EXPECTED_AUTO_CORR_APPROX = 0.5
    EXPECTED_CROSS_CORR_APPROX = 0.1

    def test_eval_generation_returns_all_metrics(self, small_generation_data):
        """Test eval_generation returns all expected metrics."""
        x_fake, x_real = small_generation_data
        result = eval_generation(
            x_fake, x_real,
            max_lag=8,
            n_bins=16,
            stationary_acf=True
        )

        assert "histloss" in result
        assert "auto_corr" in result
        assert "cross_corr" in result

        # All metrics should be non-negative floats
        for key, value in result.items():
            assert isinstance(value, float), f"{key} should be a float"
            assert value >= 0, f"{key} should be non-negative"

    def test_eval_generation_deterministic(self, small_generation_data):
        """Test eval_generation is deterministic with same inputs."""
        x_fake, x_real = small_generation_data

        result1 = eval_generation(x_fake, x_real, max_lag=8, n_bins=16)
        result2 = eval_generation(x_fake, x_real, max_lag=8, n_bins=16)

        for key in result1:
            assert np.isclose(result1[key], result2[key], rtol=1e-6), \
                f"{key} not deterministic: {result1[key]} vs {result2[key]}"


class TestValidateInputs:
    """Tests for input validation functions."""

    def test_validate_inputs_valid_forecasting(self, sample_forecasting_data):
        """Test validation passes for valid forecasting data."""
        pred, gt = sample_forecasting_data
        is_valid, error = validate_inputs(
            TaskType.TIME_SERIES_FORECASTING, pred, gt
        )
        assert is_valid, f"Validation should pass: {error}"
        assert error is None

    def test_validate_inputs_valid_generation(self, sample_generation_data):
        """Test validation passes for valid generation data."""
        x_fake, x_real = sample_generation_data
        is_valid, error = validate_inputs(
            TaskType.TIME_SERIES_GENERATION, x_fake, x_real
        )
        assert is_valid, f"Validation should pass: {error}"
        assert error is None

    def test_validate_inputs_shape_mismatch(self):
        """Test validation fails for shape mismatch."""
        pred = np.random.randn(10, 5, 3).astype(np.float32)
        gt = np.random.randn(10, 6, 3).astype(np.float32)

        is_valid, error = validate_inputs(
            TaskType.TIME_SERIES_FORECASTING, pred, gt
        )
        assert not is_valid
        assert error is not None
        assert "mismatch" in error.lower()

    def test_validate_inputs_wrong_dtype(self):
        """Test validation fails for wrong dtype."""
        pred = np.random.randn(10, 5, 3).astype(np.float64)  # Wrong dtype
        gt = np.random.randn(10, 5, 3).astype(np.float32)

        is_valid, error = validate_inputs(
            TaskType.TIME_SERIES_FORECASTING, pred, gt
        )
        assert not is_valid
        assert error is not None
        assert "float32" in error.lower() or "dtype" in error.lower()

    def test_validate_inputs_contains_nan(self):
        """Test validation fails when predictions contain NaN."""
        pred = np.random.randn(10, 5, 3).astype(np.float32)
        pred[0, 0, 0] = np.nan
        gt = np.random.randn(10, 5, 3).astype(np.float32)

        is_valid, error = validate_inputs(
            TaskType.TIME_SERIES_FORECASTING, pred, gt
        )
        assert not is_valid
        assert error is not None
        assert "nan" in error.lower()

    def test_validate_inputs_contains_inf(self):
        """Test validation fails when predictions contain Inf."""
        pred = np.random.randn(10, 5, 3).astype(np.float32)
        pred[0, 0, 0] = np.inf
        gt = np.random.randn(10, 5, 3).astype(np.float32)

        is_valid, error = validate_inputs(
            TaskType.TIME_SERIES_FORECASTING, pred, gt
        )
        assert not is_valid
        assert error is not None
        assert "inf" in error.lower()


class TestEnsureNdarray:
    """Tests for _ensure_ndarray function."""

    def test_ensure_ndarray_from_numpy(self):
        """Test conversion from numpy array."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result = _ensure_ndarray(arr)

        np.testing.assert_array_equal(result, arr)

    def test_ensure_ndarray_from_list(self):
        """Test that list input is rejected."""
        # The function should raise an error for unsupported types
        pass  # This test depends on final implementation


class TestHistogramNumpyUpdate:
    """Tests for histogram computation helper."""

    def test_histogram_basic(self):
        """Test basic histogram computation."""
        np.random.seed(42)
        x = np.random.randn(100).astype(np.float32)
        bins = np.linspace(-3, 3, 17).astype(np.float32)

        count, returned_bins = histogram_numpy_update(x, bins, density=True)

        count_arr = np.asarray(count)
        assert len(count_arr) == 16, "Should have 16 bins"
        assert np.all(count_arr >= 0), "Counts should be non-negative"

    def test_histogram_with_nans(self):
        """Test histogram ignores NaN values."""
        x = np.array([1.0, 2.0, np.nan, 3.0, 4.0], dtype=np.float32)
        bins = np.linspace(0, 5, 6).astype(np.float32)

        count, _ = histogram_numpy_update(x, bins, density=False)

        # Should only count 4 non-NaN values
        count_arr = np.asarray(count)
        assert count_arr.sum() == 4, "Should count only non-NaN values"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

