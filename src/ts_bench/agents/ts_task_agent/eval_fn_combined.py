import logging
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

logger = logging.getLogger(__name__)

# ----------------- Time-series-forecasting evaluation functions -------------------


def rmse(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    y_pred, y_true: [N, horizon, D] or any broadcastable shape.
    """
    return torch.sqrt(((y_pred - y_true) ** 2).mean()).item()


def mae(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    return (y_pred - y_true).abs().mean().item()


def mape(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-8) -> float:
    return ((y_pred - y_true).abs() / (y_true.abs() + eps)).mean().item()


def eval_forecasting(y_pred: torch.Tensor, y_true: torch.Tensor) -> dict[str, float]:
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


# ----------------- Time-series-generation evaluation functions -------------------
def acf_torch(
    x: torch.Tensor, max_lag: int, dim: Tuple[int, ...] = (0, 1)
) -> torch.Tensor:
    """
    Computes (possibly multivariate) autocorrelation function along time dimension.

    Args:
        x: Tensor of shape [B, S, D].
        max_lag: Number of lags to compute.
        dim: Dimensions over which to average (default: (0, 1) over batch and time).

    Returns:
        Tensor of shape [max_lag, D] if dim == (0, 1),
        or concatenated along last dimension otherwise.
    """
    acf_list: List[torch.Tensor] = []

    # Center
    x = x - x.mean(dim=(0, 1), keepdim=False)
    var = torch.var(x, unbiased=False, dim=(0, 1))  # [D]

    for i in range(max_lag):
        if i > 0:
            # Correct slicing over all three dims
            y = x[:, i:, :] * x[:, :-i, :]
        else:
            y = x.pow(2)

        acf_i = torch.mean(y, dim=dim) / var
        acf_list.append(acf_i)

    if dim == (0, 1):
        return torch.stack(acf_list, dim=0)
    else:
        return torch.cat(acf_list, dim=1)


def non_stationary_acf_torch(
    X: torch.Tensor,
    symmetric: bool = False,
) -> torch.Tensor:
    """
    Compute correlation matrix between any two time points of the time series.

    Args:
        X: Tensor [B, T, D].
        symmetric: If True, fill both upper and lower triangles (symmetric matrix).
                   If False, only upper triangle [t <= tau] is filled, lower remains zero.

    Returns:
        Correlation tensor of shape [T, T, D] where entry (t_i, t_j, d)
        is the correlation between the d-th coordinate of X_{t_i} and X_{t_j}.
    """
    B, T, D = X.shape
    device = X.device
    correlations = torch.zeros(T, T, D, device=device)

    eps = 1e-8
    for t in range(T):
        x_t = X[:, t, :]  # [B, D]
        norm_t = torch.norm(x_t, dim=0)  # [D]
        for tau in range(t, T):
            x_tau = X[:, tau, :]  # [B, D]
            norm_tau = torch.norm(x_tau, dim=0)  # [D]

            denom = norm_t * norm_tau
            denom = denom.clamp_min(eps)

            numerator = torch.sum(x_t * x_tau, dim=0)  # [D]
            correlation = numerator / denom  # [D]

            correlations[t, tau, :] = correlation
            if symmetric:
                correlations[tau, t, :] = correlation

    return correlations


def cacf_torch(
    x: torch.Tensor, lags: int, dim: Tuple[int, ...] = (0, 1)
) -> torch.Tensor:
    """
    Computes the cross-correlation between feature dimension pairs over time.

    Args:
        x: Tensor [B, T, D].
        lags: Number of lags to compute.
        dim: Unused for now but kept for interface symmetry.

    Returns:
        Tensor of shape [B, N_pairs, lags],
        where N_pairs = number of lower-triangular (including diagonal) pairs of features.
    """

    def get_lower_triangular_indices(n: int):
        return [list(idx) for idx in torch.tril_indices(n, n)]

    # x: [B, T, D]
    pair_indices = get_lower_triangular_indices(x.shape[2])  # 2 x N_pairs

    # Normalize per (batch, time, feature)
    x = (x - x.mean(dim=(0, 1), keepdim=True)) / x.std(dim=(0, 1), keepdim=True)

    x_l = x[..., pair_indices[0]]  # [B, T, N_pairs]
    x_r = x[..., pair_indices[1]]  # [B, T, N_pairs]

    cacf_list: List[torch.Tensor] = []
    for i in range(lags):
        if i > 0:
            y = x_l[:, i:, :] * x_r[:, :-i, :]
        else:
            y = x_l * x_r
        cacf_i = torch.mean(y, dim=1)  # [B, N_pairs]
        cacf_list.append(cacf_i)

    cacf = torch.stack(cacf_list, dim=2)  # [B, N_pairs, lags]
    return cacf


class Loss(nn.Module):
    def __init__(
        self,
        name: str,
        reg: float = 1.0,
        transform=lambda x: x,
        threshold: float = 10.0,
        backward: bool = False,
        norm_foo=lambda x: x,
    ):
        super().__init__()
        self.name = name
        self.reg = reg
        self.transform = transform
        self.threshold = threshold
        self.backward = backward
        self.norm_foo = norm_foo
        self.loss_componentwise: Optional[torch.Tensor] = None

    def forward(self, x_fake: torch.Tensor) -> torch.Tensor:
        self.loss_componentwise = self.compute(x_fake)
        return self.reg * self.loss_componentwise.mean()

    def compute(self, x_fake: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @property
    def success(self) -> bool:
        if self.loss_componentwise is None:
            return False
        return torch.all(self.loss_componentwise <= self.threshold)


def acf_diff(x: torch.Tensor) -> torch.Tensor:
    # x: e.g. [lags, D] or [T, T, D]; sum over first axis
    return torch.sqrt(torch.pow(x, 2).sum(dim=0))


def cc_diff(x: torch.Tensor) -> torch.Tensor:
    return torch.abs(x).sum(dim=0)


class ACFLoss(Loss):
    def __init__(
        self,
        x_real: torch.Tensor,
        max_lag: int = 64,
        stationary: bool = True,
        **kwargs,
    ):
        super().__init__(norm_foo=acf_diff, **kwargs)
        self.max_lag = min(max_lag, x_real.shape[1])
        self.stationary = stationary

        if stationary:
            self.acf_real = acf_torch(
                self.transform(x_real),
                self.max_lag,
                dim=(0, 1),
            )  # [max_lag, D]
        else:
            self.acf_real = non_stationary_acf_torch(
                self.transform(x_real),
                symmetric=False,
            )  # [T, T, D]

    def compute(self, x_fake: torch.Tensor) -> torch.Tensor:
        if self.stationary:
            acf_fake = acf_torch(
                self.transform(x_fake),
                self.max_lag,
                dim=(0, 1),
            )
        else:
            acf_fake = non_stationary_acf_torch(
                self.transform(x_fake),
                symmetric=False,
            ).to(x_fake.device)

        diff = acf_fake - self.acf_real.to(x_fake.device)
        return self.norm_foo(diff)


def _corrcoef_from_batch(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute feature-wise correlation matrix for x.

    Args:
        x: Tensor [N, D], N samples, D features.

    Returns:
        Corr matrix [D, D].
    """
    if x.dim() != 2:
        raise ValueError(f"Input to _corrcoef_from_batch must be 2D, got {x.shape}")

    N = x.shape[0]
    if N <= 1:
        # Degenerate case: return identity
        return torch.eye(x.shape[1], device=x.device)

    # Center
    x_centered = x - x.mean(dim=0, keepdim=True)

    # Covariance
    cov = x_centered.T @ x_centered / (N - 1)  # [D, D]

    # Standard deviations
    std = x_centered.std(dim=0, unbiased=True).clamp_min(eps)  # [D]
    denom = std[:, None] * std[None, :]  # [D, D]

    corr = cov / denom
    return corr


class cross_correlation(Loss):
    def __init__(self, x_real: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        self.x_real = x_real

    def compute(self, x_fake: torch.Tensor) -> torch.Tensor:
        """
        Compute difference between real and fake feature-wise correlation matrices.

        x_* expected shape: [B, T, D].
        We first average over time to get [B, D], then compute corrcoef over batch.
        """
        # Mean over time: [B, D]
        fake_mean = x_fake.mean(dim=1)
        real_mean = self.x_real.mean(dim=1)

        fake_corr = _corrcoef_from_batch(fake_mean)
        real_corr = _corrcoef_from_batch(real_mean.to(fake_corr.device))

        return torch.abs(fake_corr - real_corr)


def histogram_torch_update(
    x: torch.Tensor,
    bins: torch.Tensor,
    density: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the histogram of a tensor using provided bins, ignoring NaNs.

    Args:
        x: Input tensor (1D).
        bins: Precomputed bin edges [n_bins + 1].
        density: Whether to normalize the histogram.

    Returns:
        count: [n_bins], counts or densities.
        bins: Bin edges [n_bins + 1].
    """
    # Remove NaNs
    x = x[~torch.isnan(x)]
    if x.numel() == 0:
        logger.warning(
            "There are only NaNs in the input tensor for histogram computation."
        )
        return torch.zeros(len(bins) - 1, device=x.device), bins

    n_bins = len(bins) - 1
    count = torch.histc(
        x,
        bins=n_bins,
        min=bins[0].item(),
        max=bins[-1].item(),
    )
    if density:
        count = count / x.numel() * n_bins
    return count, bins


class HistogramLoss(nn.Module):
    @staticmethod
    def precompute_histograms(
        x: torch.Tensor,
        n_bins: int,
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
    ]:
        """
        Precompute per-time, per-feature histograms of real data.

        Args:
            x: Tensor [N, L, D].
            n_bins: Number of bins.

        Returns:
            densities: List[L] of tensors [D, n_bins]
            center_bin_locs: List[L] of tensors [D, n_bins]
            bin_widths: List[L] of tensors [D]
            bin_edges: List[L] of tensors [D, n_bins + 1]
            sample_sizes: List[L] of tensors [D]
        """
        N, L, D = x.shape
        densities: List[torch.Tensor] = []
        center_bin_locs: List[torch.Tensor] = []
        bin_widths: List[torch.Tensor] = []
        bin_edges: List[torch.Tensor] = []
        sample_sizes: List[torch.Tensor] = []

        for t in range(L):
            per_time_densities: List[torch.Tensor] = []
            per_time_center_locs: List[torch.Tensor] = []
            per_time_widths: List[torch.Tensor] = []
            per_time_bins: List[torch.Tensor] = []
            per_time_sample_sizes: List[torch.Tensor] = []

            for d in range(D):
                x_ti = x[:, t, d].reshape(-1)
                x_ti = x_ti[~torch.isnan(x_ti)]

                if x_ti.numel() == 0:
                    per_time_densities.append(torch.zeros(n_bins, device=x.device))
                    per_time_center_locs.append(torch.zeros(n_bins, device=x.device))
                    per_time_widths.append(torch.tensor(1.0, device=x.device))
                    per_time_bins.append(torch.zeros(n_bins + 1, device=x.device))
                    per_time_sample_sizes.append(torch.tensor(0, device=x.device))
                    continue

                min_val = x_ti.min().item()
                max_val = x_ti.max().item()

                # Handle degenerate or single-sample cases
                if x_ti.numel() > 1 and abs(max_val - min_val) < 1e-10:
                    max_val += 1e-5
                    min_val -= 1e-5
                    logger.warning(
                        "All values are the same for a time and feature. "
                        "Adding a small perturbation to the range. "
                        "The loss might not be as representative as desired."
                    )
                elif x_ti.numel() == 1:
                    max_val += 1e-5
                    min_val -= 1e-5

                bins = torch.linspace(min_val, max_val, n_bins + 1, device=x.device)
                density, bins = histogram_torch_update(x_ti, bins, density=True)

                per_time_densities.append(density)
                bin_width = bins[1] - bins[0]
                center_bin_loc = 0.5 * (bins[1:] + bins[:-1])

                per_time_center_locs.append(center_bin_loc)
                per_time_widths.append(bin_width)
                per_time_bins.append(bins)
                per_time_sample_sizes.append(
                    torch.tensor(x_ti.numel(), device=x.device)
                )

            densities.append(torch.stack(per_time_densities, dim=0))  # [D, n_bins]
            center_bin_locs.append(
                torch.stack(per_time_center_locs, dim=0)
            )  # [D, n_bins]
            bin_widths.append(torch.stack(per_time_widths, dim=0))  # [D]
            bin_edges.append(torch.stack(per_time_bins, dim=0))  # [D, n_bins+1]
            sample_sizes.append(torch.stack(per_time_sample_sizes, dim=0))  # [D]

        return densities, center_bin_locs, bin_widths, bin_edges, sample_sizes

    def __init__(self, x_real: torch.Tensor, n_bins: int):
        """
        Initializes the HistogramLoss with the real data distribution.

        Args:
            x_real: Real data tensor of shape (N, L, D).
            n_bins: Number of bins for the histograms.
        """
        super().__init__()

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

        # Store as non-trainable parameters for safe device handling
        self.densities = nn.ParameterList(
            [nn.Parameter(density, requires_grad=False) for density in densities]
        )
        self.center_bin_locs = nn.ParameterList(
            [nn.Parameter(loc, requires_grad=False) for loc in center_locs]
        )
        self.bin_widths = nn.ParameterList(
            [nn.Parameter(width, requires_grad=False) for width in bin_widths]
        )
        self.bin_edges = nn.ParameterList(
            [nn.Parameter(bins, requires_grad=False) for bins in bin_edges]
        )
        self.sample_sizes = nn.Parameter(
            torch.stack(sample_sizes, dim=0), requires_grad=False
        )  # [L, D]

    def compute(self, x_fake: torch.Tensor) -> torch.Tensor:
        """
        Computes the histogram loss between real and fake data distributions.

        Notes:
            We noticed issues in the case of the comparison of densities ala Dirac measure.
            Use with caution in that case.

        Args:
            x_fake: Fake data tensor of shape (N, L, D).

        Returns:
            all_losses: Tensor [L, D], loss per time per feature.
        """
        if x_fake.device != self.densities[0].device:
            logger.warning(
                f"x_fake is on device {x_fake.device}, "
                f"but densities are on {self.densities[0].device}. "
                "Moving x_fake to the correct device."
            )
            x_fake = x_fake.to(self.densities[0].device)

        assert (
            x_fake.shape[2] == self.num_features
        ), f"Expected {self.num_features} features in x_fake, but got {x_fake.shape[2]}."
        assert (
            x_fake.shape[1] == self.num_time_steps
        ), f"Expected {self.num_time_steps} time steps in x_fake, but got {x_fake.shape[1]}."

        all_losses: List[torch.Tensor] = []

        for t in range(self.num_time_steps):
            per_time_losses: List[torch.Tensor] = []

            for d in range(self.num_features):
                loc: torch.Tensor = self.center_bin_locs[t][d]  # [n_bins]
                # If center locs are all zero, treat as empty / invalid
                if (loc.abs() < 1e-16).all():
                    per_time_losses.append(torch.tensor(2.0, device=x_fake.device))
                    continue

                x_ti = x_fake[:, t, d].reshape(-1)
                x_ti = x_ti[~torch.isnan(x_ti)].reshape(-1)

                if x_ti.numel() == 0:
                    per_time_losses.append(torch.tensor(2.0, device=x_fake.device))
                    continue

                # Compute fake histogram using precomputed bin edges
                edges = self.bin_edges[t][d]  # [n_bins+1]
                density_fake = torch.histc(
                    x_ti,
                    bins=self.n_bins,
                    min=edges[0].item(),
                    max=edges[-1].item(),
                )
                density_fake = density_fake / x_ti.numel() * self.n_bins

                abs_metric = torch.abs(density_fake - self.densities[t][d]).mean()

                num_samples_oob = torch.sum((x_ti < edges[0]) | (x_ti > edges[-1]))
                out_of_bound_error = num_samples_oob / x_ti.numel()

                per_time_losses.append(abs_metric + out_of_bound_error)

            all_losses.append(torch.stack(per_time_losses, dim=0))

        if not all_losses:
            logger.error(
                "All time steps and features contain NaNs or empty data, yielding no valid losses."
            )
            return torch.tensor(1.0, device=x_fake.device)

        all_losses_tensor = torch.stack(all_losses, dim=0)  # [L, D]
        return all_losses_tensor / 2.0

    def _weigh_by_sample_size(self, loss: torch.Tensor) -> torch.Tensor:
        assert (
            loss.shape == self.sample_sizes.shape
        ), f"Loss and sample sizes should have the same shape, but got {loss.shape} and {self.sample_sizes.shape}."
        weights = self.sample_sizes / self.sample_sizes.sum()
        return (loss * weights).sum()

    def forward(
        self, x_fake: torch.Tensor, ignore_features: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Weighted scalar histogram loss, optionally ignoring some feature indices.
        """
        try:
            base_loss = self.compute(x_fake)  # [L, D]

            if ignore_features is None or (
                hasattr(ignore_features, "__len__") and len(ignore_features) == 0
            ):
                return self._weigh_by_sample_size(base_loss)

            ignore_indices = torch.tensor(
                ignore_features, dtype=torch.long, device=base_loss.device
            )
            mask = torch.ones(
                self.num_features, dtype=torch.bool, device=base_loss.device
            )
            mask[ignore_indices] = False

            masked_loss = self._weigh_by_sample_size(base_loss[:, mask])
            return masked_loss
        except Exception as e:
            logger.error(
                f"Error in the computation of the HistogramLoss. "
                f"Return maximal error. Details: {e}"
            )
            return torch.tensor(1.0, device=x_fake.device)


def eval_generation(
    x_fake: torch.Tensor,
    x_real: torch.Tensor,
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
    histogram_value = hist_loss_module(x_fake).item()

    # ACF loss
    acf_loss_module = ACFLoss(
        x_real=x_real,
        max_lag=max_lag,
        stationary=stationary_acf,
        name="acf",
        reg=1.0,
    )
    acf_value = acf_loss_module(x_fake).item()

    # Cross-correlation loss (uses Loss.forward -> mean over matrix)
    cc_module = cross_correlation(
        x_real=x_real,
        name="crosscorr",
        reg=1.0,
    )
    cross_corr_value = cc_module(x_fake).item()

    return {
        "histloss": float(histogram_value),
        "auto_corr": float(acf_value),
        "cross_corr": float(cross_corr_value),
    }
