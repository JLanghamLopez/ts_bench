import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ----------------- Time-series-generation evaluation functions -------------------
def acf_numpy(x: np.ndarray, max_lag: int, dim: Tuple[int, ...] = (0, 1)) -> np.ndarray:
    """
    Computes (possibly multivariate) autocorrelation function along time dimension.

    Args:
        x: Array of shape [B, S, D].
        max_lag: Number of lags to compute.
        dim: Dimensions over which to average (default: (0, 1) over batch and time).

    Returns:
        Array of shape [max_lag, D] if dim == (0, 1),
        or concatenated along last dimension otherwise.
    """
    acf_list: List[np.ndarray] = []

    # Center
    x = x - x.mean(axis=(0, 1), keepdims=False)
    var = np.var(x, ddof=0, axis=(0, 1))  # [D]

    for i in range(max_lag):
        if i > 0:
            # Correct slicing over all three dims
            y = x[:, i:, :] * x[:, :-i, :]
        else:
            y = x**2

        acf_i = np.mean(y, axis=dim) / var
        acf_list.append(acf_i)

    if dim == (0, 1):
        return np.stack(acf_list, axis=0)
    else:
        return np.concatenate(acf_list, axis=1)


def non_stationary_acf_numpy(
    X: np.ndarray,
    symmetric: bool = False,
) -> np.ndarray:
    """
    Compute correlation matrix between any two time points of the time series.

    Args:
        X: Array [B, T, D].
        symmetric: If True, fill both upper and lower triangles (symmetric matrix).
                   If False, only upper triangle [t <= tau] is filled, lower remains zero.

    Returns:
        Correlation array of shape [T, T, D] where entry (t_i, t_j, d)
        is the correlation between the d-th coordinate of X_{t_i} and X_{t_j}.
    """
    B, T, D = X.shape
    correlations = np.zeros((T, T, D), dtype=X.dtype)

    eps = 1e-8
    for t in range(T):
        x_t = X[:, t, :]  # [B, D]
        norm_t = np.linalg.norm(x_t, axis=0)  # [D]
        for tau in range(t, T):
            x_tau = X[:, tau, :]  # [B, D]
            norm_tau = np.linalg.norm(x_tau, axis=0)  # [D]

            denom = norm_t * norm_tau
            denom = np.maximum(denom, eps)

            numerator = np.sum(x_t * x_tau, axis=0)  # [D]
            correlation = numerator / denom  # [D]

            correlations[t, tau, :] = correlation
            if symmetric:
                correlations[tau, t, :] = correlation

    return correlations


def cacf_numpy(x: np.ndarray, lags: int, dim: Tuple[int, ...] = (0, 1)) -> np.ndarray:
    """
    Computes the cross-correlation between feature dimension pairs over time.

    Args:
        x: Array [B, T, D].
        lags: Number of lags to compute.
        dim: Unused for now but kept for interface symmetry.

    Returns:
        Array of shape [B, N_pairs, lags],
        where N_pairs = number of lower-triangular (including diagonal) pairs of features.
    """

    def get_lower_triangular_indices(n: int):
        indices = np.tril_indices(n, k=0)
        return [list(indices[0]), list(indices[1])]

    # x: [B, T, D]
    pair_indices = get_lower_triangular_indices(x.shape[2])  # 2 x N_pairs

    # Normalize per (batch, time, feature)
    x_mean = x.mean(axis=(0, 1), keepdims=True)
    x_std = x.std(axis=(0, 1), keepdims=True)
    x_std = np.maximum(x_std, 1e-8)  # Avoid division by zero
    x = (x - x_mean) / x_std

    x_l = x[..., pair_indices[0]]  # [B, T, N_pairs]
    x_r = x[..., pair_indices[1]]  # [B, T, N_pairs]

    cacf_list: List[np.ndarray] = []
    for i in range(lags):
        if i > 0:
            y = x_l[:, i:, :] * x_r[:, :-i, :]
        else:
            y = x_l * x_r
        cacf_i = np.mean(y, axis=1)  # [B, N_pairs]
        cacf_list.append(cacf_i)

    cacf = np.stack(cacf_list, axis=2)  # [B, N_pairs, lags]
    return cacf


class Loss:
    """Base class for loss computation (replaces nn.Module)."""

    def __init__(
        self,
        name: str,
        reg: float = 1.0,
        transform=lambda x: x,
        threshold: float = 10.0,
        backward: bool = False,
        norm_foo=lambda x: x,
    ):
        self.name = name
        self.reg = reg
        self.transform = transform
        self.threshold = threshold
        self.backward = backward
        self.norm_foo = norm_foo
        self.loss_componentwise: Optional[np.ndarray] = None

    def __call__(self, x_fake: np.ndarray) -> float:
        self.loss_componentwise = self.compute(x_fake)
        return float(self.reg * self.loss_componentwise.mean())

    def compute(self, x_fake: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @property
    def success(self) -> bool:
        if self.loss_componentwise is None:
            return False
        return bool(np.all(self.loss_componentwise <= self.threshold))


def acf_diff(x: np.ndarray) -> np.ndarray:
    # x: e.g. [lags, D] or [T, T, D]; sum over first axis
    return np.sqrt((x**2).sum(axis=0))


def cc_diff(x: np.ndarray) -> np.ndarray:
    return np.abs(x).sum(axis=0)


class ACFLoss(Loss):
    def __init__(
        self,
        x_real: np.ndarray,
        max_lag: int = 64,
        stationary: bool = True,
        **kwargs,
    ):
        super().__init__(norm_foo=acf_diff, **kwargs)
        self.max_lag = min(max_lag, x_real.shape[1])
        self.stationary = stationary

        if stationary:
            self.acf_real = acf_numpy(
                self.transform(x_real),
                self.max_lag,
                dim=(0, 1),
            )  # [max_lag, D]
        else:
            self.acf_real = non_stationary_acf_numpy(
                self.transform(x_real),
                symmetric=False,
            )  # [T, T, D]

    def compute(self, x_fake: np.ndarray) -> np.ndarray:
        if self.stationary:
            acf_fake = acf_numpy(
                self.transform(x_fake),
                self.max_lag,
                dim=(0, 1),
            )
        else:
            acf_fake = non_stationary_acf_numpy(
                self.transform(x_fake),
                symmetric=False,
            )

        diff = acf_fake - self.acf_real
        return self.norm_foo(diff)


def _corrcoef_from_batch(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Compute feature-wise correlation matrix for x.

    Args:
        x: Array [N, D], N samples, D features.

    Returns:
        Corr matrix [D, D].
    """
    if x.ndim != 2:
        raise ValueError(f"Input to _corrcoef_from_batch must be 2D, got {x.shape}")

    N = x.shape[0]
    if N <= 1:
        # Degenerate case: return identity
        return np.eye(x.shape[1], dtype=x.dtype)

    # Center
    x_centered = x - x.mean(axis=0, keepdims=True)

    # Covariance
    cov = x_centered.T @ x_centered / (N - 1)  # [D, D]

    # Standard deviations
    std = x_centered.std(axis=0, ddof=1)
    std = np.maximum(std, eps)  # [D]
    denom = std[:, None] * std[None, :]  # [D, D]

    corr = cov / denom
    return corr


class cross_correlation(Loss):
    def __init__(self, x_real: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        self.x_real = x_real

    def compute(self, x_fake: np.ndarray) -> np.ndarray:
        """
        Compute difference between real and fake feature-wise correlation matrices.

        x_* expected shape: [B, T, D].
        We first average over time to get [B, D], then compute corrcoef over batch.
        """
        # Mean over time: [B, D]
        fake_mean = x_fake.mean(axis=1)
        real_mean = self.x_real.mean(axis=1)

        fake_corr = _corrcoef_from_batch(fake_mean)
        real_corr = _corrcoef_from_batch(real_mean)

        return np.abs(fake_corr - real_corr)


def histogram_numpy_update(
    x: np.ndarray,
    bins: np.ndarray,
    density: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the histogram of an array using provided bins, ignoring NaNs.

    Args:
        x: Input array (1D).
        bins: Precomputed bin edges [n_bins + 1].
        density: Whether to normalize the histogram.

    Returns:
        count: [n_bins], counts or densities.
        bins: Bin edges [n_bins + 1].
    """
    # Remove NaNs
    x = x[~np.isnan(x)]
    if x.size == 0:
        logger.warning(
            "There are only NaNs in the input array for histogram computation."
        )
        return np.zeros(len(bins) - 1, dtype=np.float32), bins

    n_bins = len(bins) - 1
    count, _ = np.histogram(x, bins=bins)
    count = count.astype(np.float32)

    if density:
        count = count / x.size * n_bins
    return count, bins


class HistogramLoss:
    """Histogram-based loss computation (replaces nn.Module)."""

    @staticmethod
    def precompute_histograms(
        x: np.ndarray,
        n_bins: int,
    ) -> Tuple[
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
    ]:
        """
        Precompute per-time, per-feature histograms of real data.

        Args:
            x: Array [N, L, D].
            n_bins: Number of bins.

        Returns:
            densities: List[L] of arrays [D, n_bins]
            center_bin_locs: List[L] of arrays [D, n_bins]
            bin_widths: List[L] of arrays [D]
            bin_edges: List[L] of arrays [D, n_bins + 1]
            sample_sizes: List[L] of arrays [D]
        """
        N, L, D = x.shape
        densities: List[np.ndarray] = []
        center_bin_locs: List[np.ndarray] = []
        bin_widths: List[np.ndarray] = []
        bin_edges: List[np.ndarray] = []
        sample_sizes: List[np.ndarray] = []

        for t in range(L):
            per_time_densities: List[np.ndarray] = []
            per_time_center_locs: List[np.ndarray] = []
            per_time_widths: List[np.ndarray] = []
            per_time_bins: List[np.ndarray] = []
            per_time_sample_sizes: List[np.ndarray] = []

            for d in range(D):
                x_ti = x[:, t, d].reshape(-1)
                x_ti = x_ti[~np.isnan(x_ti)]

                if x_ti.size == 0:
                    per_time_densities.append(np.zeros(n_bins, dtype=np.float32))
                    per_time_center_locs.append(np.zeros(n_bins, dtype=np.float32))
                    per_time_widths.append(np.array(1.0, dtype=np.float32))
                    per_time_bins.append(np.zeros(n_bins + 1, dtype=np.float32))
                    per_time_sample_sizes.append(np.array(0, dtype=np.float32))
                    continue

                min_val = float(x_ti.min())
                max_val = float(x_ti.max())

                # Handle degenerate or single-sample cases
                if x_ti.size > 1 and abs(max_val - min_val) < 1e-10:
                    max_val += 1e-5
                    min_val -= 1e-5
                    logger.warning(
                        "All values are the same for a time and feature. "
                        "Adding a small perturbation to the range. "
                        "The loss might not be as representative as desired."
                    )
                elif x_ti.size == 1:
                    max_val += 1e-5
                    min_val -= 1e-5

                bins_arr = np.linspace(min_val, max_val, n_bins + 1, dtype=np.float32)
                density_arr, bins_arr = histogram_numpy_update(
                    x_ti, bins_arr, density=True
                )

                per_time_densities.append(density_arr)
                bin_width = bins_arr[1] - bins_arr[0]
                center_bin_loc = 0.5 * (bins_arr[1:] + bins_arr[:-1])

                per_time_center_locs.append(center_bin_loc)
                per_time_widths.append(bin_width)
                per_time_bins.append(bins_arr)
                per_time_sample_sizes.append(np.array(x_ti.size, dtype=np.float32))

            densities.append(np.stack(per_time_densities, axis=0))  # [D, n_bins]
            center_bin_locs.append(
                np.stack(per_time_center_locs, axis=0)
            )  # [D, n_bins]
            bin_widths.append(np.stack(per_time_widths, axis=0))  # [D]
            bin_edges.append(np.stack(per_time_bins, axis=0))  # [D, n_bins+1]
            sample_sizes.append(np.stack(per_time_sample_sizes, axis=0))  # [D]

        return densities, center_bin_locs, bin_widths, bin_edges, sample_sizes

    def __init__(self, x_real: np.ndarray, n_bins: int):
        """
        Initializes the HistogramLoss with the real data distribution.

        Args:
            x_real: Real data array of shape (N, L, D).
            n_bins: Number of bins for the histograms.
        """
        self.n_bins = n_bins
        self.num_samples, self.num_time_steps, self.num_features = x_real.shape

        logger.debug(
            f"Initializing HistogramLoss with {self.num_samples} samples, "
            f"{self.num_time_steps} time steps, and {self.num_features} features."
        )

        (
            densities,
            center_locs,
            bin_widths,
            bin_edges,
            sample_sizes,
        ) = self.precompute_histograms(
            x_real,
            n_bins,
        )

        # Store as regular lists of numpy arrays
        self.densities = densities
        self.center_bin_locs = center_locs
        self.bin_widths = bin_widths
        self.bin_edges = bin_edges
        self.sample_sizes = np.stack(sample_sizes, axis=0)  # [L, D]

    def compute(self, x_fake: np.ndarray) -> np.ndarray:
        """
        Computes the histogram loss between real and fake data distributions.

        Notes:
            We noticed issues in the case of the comparison of densities ala Dirac measure.
            Use with caution in that case.

        Args:
            x_fake: Fake data array of shape (N, L, D).

        Returns:
            all_losses: Array [L, D], loss per time per feature.
        """
        assert (
            x_fake.shape[2] == self.num_features
        ), f"Expected {self.num_features} features in x_fake, but got {x_fake.shape[2]}."
        assert (
            x_fake.shape[1] == self.num_time_steps
        ), f"Expected {self.num_time_steps} time steps in x_fake, but got {x_fake.shape[1]}."

        all_losses: List[np.ndarray] = []

        for t in range(self.num_time_steps):
            per_time_losses: List[float] = []

            for d in range(self.num_features):
                loc: np.ndarray = self.center_bin_locs[t][d]  # [n_bins]
                # If center locs are all zero, treat as empty / invalid
                if np.all(np.abs(loc) < 1e-16):
                    per_time_losses.append(2.0)
                    continue

                x_ti = x_fake[:, t, d].reshape(-1)
                x_ti = x_ti[~np.isnan(x_ti)].reshape(-1)

                if x_ti.size == 0:
                    per_time_losses.append(2.0)
                    continue

                # Compute fake histogram using precomputed bin edges
                edges = self.bin_edges[t][d]  # [n_bins+1]
                density_fake, _ = np.histogram(
                    x_ti,
                    bins=edges,
                )
                density_fake = density_fake.astype(np.float32) / x_ti.size * self.n_bins

                abs_metric = float(np.abs(density_fake - self.densities[t][d]).mean())

                num_samples_oob = np.sum((x_ti < edges[0]) | (x_ti > edges[-1]))
                out_of_bound_error = float(num_samples_oob / x_ti.size)

                per_time_losses.append(abs_metric + out_of_bound_error)

            all_losses.append(np.array(per_time_losses, dtype=np.float32))

        if not all_losses:
            logger.error(
                "All time steps and features contain NaNs or empty data, yielding no valid losses."
            )
            return np.array(1.0, dtype=np.float32)

        all_losses_array = np.stack(all_losses, axis=0)  # [L, D]
        return all_losses_array / 2.0

    def _weigh_by_sample_size(self, loss: np.ndarray) -> float:
        assert (
            loss.shape == self.sample_sizes.shape
        ), f"Loss and sample sizes should have the same shape, but got {loss.shape} and {self.sample_sizes.shape}."
        weights = self.sample_sizes / self.sample_sizes.sum()
        return float((loss * weights).sum())

    def __call__(
        self, x_fake: np.ndarray, ignore_features: Optional[List[int]] = None
    ) -> float:
        """
        Weighted scalar histogram loss, optionally ignoring some feature indices.
        """
        try:
            base_loss = self.compute(x_fake)  # [L, D]

            if ignore_features is None or (
                hasattr(ignore_features, "__len__") and len(ignore_features) == 0
            ):
                return self._weigh_by_sample_size(base_loss)

            ignore_indices = np.array(ignore_features, dtype=np.int64)
            mask = np.ones(self.num_features, dtype=bool)
            mask[ignore_indices] = False

            # Need to adjust sample_sizes for masked features
            masked_sample_sizes = self.sample_sizes[:, mask]
            masked_loss = base_loss[:, mask]
            weights = masked_sample_sizes / masked_sample_sizes.sum()
            return float((masked_loss * weights).sum())
        except Exception as e:
            logger.error(
                f"Error in the computation of the HistogramLoss. "
                f"Return maximal error. Details: {e}"
            )
            return 1.0


def eval_generation(
    x_fake: np.ndarray,
    x_real: np.ndarray,
    max_lag: int = 64,
    n_bins: int = 32,
    stationary_acf: bool = True,
) -> Dict[str, float]:
    """
    Convenience wrapper that evaluates time-series generation quality via:

        1. Histogram-based marginal distribution loss
        2. Autocorrelation loss (ACF)
        3. Cross-correlation loss (feature-wise)

    Args:
        x_fake: Generated series [B, T, D].
        x_real: Real series [B, T, D].
        max_lag: Max lag for ACF.
        n_bins: Number of bins for histogram.
        stationary_acf: Whether to use stationary ACF.

    Returns:
        Dictionary of scalar metrics:
            {
                "histloss": float,
                "auto_corr": float,
                "cross_corr": float,
            }
    """
    # Histogram loss
    hist_loss_module = HistogramLoss(x_real, n_bins=n_bins)
    histogram_value = hist_loss_module(x_fake)

    # ACF loss
    acf_loss_module = ACFLoss(
        x_real=x_real,
        max_lag=max_lag,
        stationary=stationary_acf,
        name="acf",
        reg=1.0,
    )
    acf_value = acf_loss_module(x_fake)

    # Cross-correlation loss (uses Loss.forward -> mean over matrix)
    cc_module = cross_correlation(
        x_real=x_real,
        name="crosscorr",
        reg=1.0,
    )
    cross_corr_value = cc_module(x_fake)

    return {
        "histloss": float(histogram_value),
        "auto_corr": float(acf_value),
        "cross_corr": float(cross_corr_value),
    }
