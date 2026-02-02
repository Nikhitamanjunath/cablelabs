"""
Baseline forecasts per PREDICTION_MODEL_SUGGESTIONS.md §6.

- Naive: next value = last observed
- Historic average: next value = mean of all observed so far
- Window average: next value = mean of last w values
- Seasonal naive: next value = value at same season last cycle (e.g. same day last week)
- Exponential smoothing: strong weight on most recent values (aligned with analyze prediction)
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def forecast_exponential_smoothing(
    series: pd.Series,
    horizon: int = 1,
    alpha: Optional[float] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Exponential smoothing: forecast = alpha * last + (1-alpha) * level.
    Strong weight on recent values. If alpha is None, use last-3 weights [0.05, 0.15, 0.80].
    PIs from empirical 1-step errors if enough history.
    """
    if len(series) < 1:
        return np.full(horizon, np.nan), None, None
    vals = series.values
    if len(vals) == 1:
        point = np.full(horizon, float(vals[-1]))
        return point, None, None
    if alpha is not None:
        level = float(vals[0])
        for y in vals[1:]:
            level = alpha * float(y) + (1 - alpha) * level
        point = np.full(horizon, level)
    else:
        if len(vals) >= 3:
            weights = np.array([0.05, 0.15, 0.80])
            point = np.full(horizon, float(np.sum(vals[-3:] * weights)))
        elif len(vals) == 2:
            point = np.full(horizon, 0.92 * float(vals[-1]) + 0.08 * float(vals[-2]))
        else:
            point = np.full(horizon, float(vals[-1]))
    if len(series) >= 3:
        errors = series.diff().dropna()
        errors = errors[~np.isnan(errors)]
        if len(errors) >= 2:
            lo = np.percentile(errors.values, 2.5)
            hi = np.percentile(errors.values, 97.5)
            lower = np.full(horizon, point[0] + lo)
            upper = np.full(horizon, point[0] + hi)
            return point, lower, upper
    return point, None, None


def forecast_naive(
    series: pd.Series,
    horizon: int = 1,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Naive: hat{y}_{t+1} = y_t.
    Returns (point_forecast, lower_95, upper_95). PIs from empirical residuals if enough history.
    """
    if len(series) < 1:
        return np.full(horizon, np.nan), None, None
    last = series.iloc[-1]
    point = np.full(horizon, last)
    # Empirical PI from past 1-step errors if we have enough data
    if len(series) >= 3:
        errors = series.diff().dropna()
        errors = errors[~np.isnan(errors)]
        if len(errors) >= 2:
            lo = np.percentile(errors, 2.5)
            hi = np.percentile(errors, 97.5)
            lower = np.full(horizon, last + lo)
            upper = np.full(horizon, last + hi)
            return point, lower, upper
    return point, None, None


def forecast_historic_avg(
    series: pd.Series,
    horizon: int = 1,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Historic average: hat{y}_{t+1} = (1/t) sum_{i=1}^t y_i.
    PIs from empirical std of series.
    """
    if len(series) < 1:
        return np.full(horizon, np.nan), None, None
    mu = np.nanmean(series.values)
    point = np.full(horizon, mu)
    if len(series) >= 3:
        std = np.nanstd(series.values)
        if std > 0:
            lower = np.full(horizon, mu - 1.96 * std)
            upper = np.full(horizon, mu + 1.96 * std)
            return point, lower, upper
    return point, None, None


def forecast_window_avg(
    series: pd.Series,
    horizon: int = 1,
    window: int = 7,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Window average: hat{y}_{t+1} = mean of last w values.
    PIs from empirical std of recent values.
    """
    if len(series) < 1:
        return np.full(horizon, np.nan), None, None
    w = min(window, len(series))
    recent = series.iloc[-w:].values
    mu = np.nanmean(recent)
    point = np.full(horizon, mu)
    if w >= 3:
        std = np.nanstd(recent)
        if std > 0:
            lower = np.full(horizon, mu - 1.96 * std)
            upper = np.full(horizon, mu + 1.96 * std)
            return point, lower, upper
    return point, None, None


def forecast_seasonal_naive(
    series: pd.Series,
    horizon: int = 1,
    period: int = 7,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Seasonal naive: for daily data with weekly seasonality, hat{y}_{t+1} = y_{t-6} (same day last week).
    period=7 for weekly.
    PIs from empirical residuals of 1-step seasonal naive errors.
    """
    if len(series) < period:
        return np.full(horizon, np.nan), None, None
    point_list = []
    for h in range(horizon):
        idx = len(series) - period + (h % period)
        if idx >= 0:
            point_list.append(series.iloc[idx])
        else:
            point_list.append(np.nan)
    point = np.array(point_list)
    # Empirical PI from past seasonal-naive errors
    if len(series) >= period + 2:
        errors = []
        for i in range(period, len(series)):
            pred = series.iloc[i - period]
            actual = series.iloc[i]
            if not (np.isnan(pred) or np.isnan(actual)):
                errors.append(actual - pred)
        if len(errors) >= 2:
            errors = np.array(errors)
            lo = np.percentile(errors, 2.5)
            hi = np.percentile(errors, 97.5)
            lower = point + lo
            upper = point + hi
            return point, lower, upper
    return point, None, None
