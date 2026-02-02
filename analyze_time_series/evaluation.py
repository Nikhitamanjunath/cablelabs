"""
Evaluation: rolling CV, MAE, RMSE, CRPS, prediction-interval coverage and width.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs(y_true[mask] - y_pred[mask]))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not np.any(mask):
        return np.nan
    return np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))


def pi_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Fraction of actuals that fall inside [lower, upper]."""
    mask = ~(
        np.isnan(y_true)
        | np.isnan(lower)
        | np.isnan(upper)
    )
    if not np.any(mask):
        return np.nan
    inside = np.sum((y_true[mask] >= lower[mask]) & (y_true[mask] <= upper[mask]))
    return inside / np.sum(mask)


def pi_mean_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """Mean width of prediction intervals."""
    mask = ~(np.isnan(lower) | np.isnan(upper))
    if not np.any(mask):
        return np.nan
    return np.mean(upper[mask] - lower[mask])


def crps_gaussian(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> float:
    """
    CRPS for Gaussian predictive distribution (closed form).
    sigma must be > 0 where we evaluate.
    """
    mask = ~(np.isnan(y_true) | np.isnan(mu) | np.isnan(sigma) | (sigma <= 0))
    if not np.any(mask):
        return np.nan
    y = y_true[mask]
    m = mu[mask]
    s = sigma[mask]
    z = (y - m) / s
    # CRPS = s * (z * (2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi))
    from scipy import stats
    crps = s * (
        z * (2 * stats.norm.cdf(z) - 1)
        + 2 * stats.norm.pdf(z)
        - 1 / np.sqrt(np.pi)
    )
    return np.mean(crps)


def crps_from_quantiles(
    y_true: np.ndarray,
    quantiles: np.ndarray,
    levels: np.ndarray,
) -> float:
    """
    Approximate CRPS from a set of quantile forecasts.
    quantiles: (n_obs, n_quantiles); levels: (n_quantiles,) in (0, 1).
    """
    mask = ~np.isnan(y_true)
    if not np.any(mask):
        return np.nan
    y = y_true[mask]
    q = quantiles[mask]
    if q.ndim == 1:
        q = q.reshape(-1, 1)
    n = len(y)
    levels = np.asarray(levels).ravel()
    if levels.size != q.shape[1]:
        return np.nan
    # Empirical CRPS: sum over quantile levels of (level - 1(y <= q)) * (q - y)
    crps = 0.0
    for j, level in enumerate(levels):
        qj = q[:, j]
        crps += np.mean((level - (y <= qj).astype(float)) * (qj - y))
    return crps


def rolling_cv_metrics(
    series: pd.Series,
    train_size: int,
    test_size: int,
    forecast_fn,
    forecast_kwargs: Optional[Dict] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Rolling window CV: train on last train_size points, forecast test_size steps.
    Returns (actuals, point_forecasts, lower, upper) for the test window.
    forecast_fn(series, horizon=test_size, **forecast_kwargs) -> (point, lower, upper).
    """
    if forecast_kwargs is None:
        forecast_kwargs = {}
    n = len(series)
    if n < train_size + test_size:
        return (
            np.full(test_size, np.nan),
            np.full(test_size, np.nan),
            np.full(test_size, np.nan),
            np.full(test_size, np.nan),
        )
    train = series.iloc[-train_size - test_size : -test_size]
    test_actual = series.iloc[-test_size:].values
    point, lower, upper = forecast_fn(train, horizon=test_size, **forecast_kwargs)
    if point is None or len(point) != test_size:
        point = np.full(test_size, np.nan)
    if lower is None:
        lower = np.full(test_size, np.nan)
    if upper is None:
        upper = np.full(test_size, np.nan)
    return test_actual, point, lower, upper
