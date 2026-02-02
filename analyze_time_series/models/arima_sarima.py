"""
ARIMA and SARIMA wrappers using statsmodels.
Produces point forecasts and 95% prediction intervals.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.arima.model import ARIMA
    HAS_STATS = True
except ImportError:
    HAS_STATS = False


def _fit_arima(
    series: pd.Series,
    order: Tuple[int, int, int],
    seasonal_order: Optional[Tuple[int, int, int, int]] = None,
) -> Optional[object]:
    """Fit ARIMA or SARIMAX. Returns fitted model or None on failure."""
    if not HAS_STATS:
        return None
    try:
        y = series.dropna()
        if len(y) < max(order[0] + order[1] + order[2], 10):
            return None
        if seasonal_order is not None:
            s = seasonal_order[3]
            if len(y) < s * 2 + 10:
                return None
            model = SARIMAX(
                y,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
        else:
            model = ARIMA(y, order=order)
        fitted = model.fit()
        return fitted
    except Exception:
        return None


def forecast_arima(
    series: pd.Series,
    horizon: int = 1,
    order: Tuple[int, int, int] = (1, 0, 1),
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    ARIMA(p,d,q). Returns (point, lower_95, upper_95).
    Uses closed-form forecast and forecast error variance for PIs.
    """
    if not HAS_STATS or len(series) < 10:
        return np.full(horizon, np.nan), None, None
    fitted = _fit_arima(series, order)
    if fitted is None:
        return np.full(horizon, np.nan), None, None
    try:
        f = fitted.get_forecast(steps=horizon)
        point = f.predicted_mean.values
        ci = f.conf_int(alpha=0.05)
        lower = ci.iloc[:, 0].values
        upper = ci.iloc[:, 1].values
        return point, lower, upper
    except Exception:
        return np.full(horizon, np.nan), None, None


def forecast_sarima(
    series: pd.Series,
    horizon: int = 1,
    order: Tuple[int, int, int] = (1, 0, 1),
    seasonal_order: Tuple[int, int, int, int] = (1, 0, 1, 7),
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    SARIMA with seasonal period s=7 (weekly) for daily data.
    Returns (point, lower_95, upper_95).
    """
    if not HAS_STATS or len(series) < 30:
        return np.full(horizon, np.nan), None, None
    fitted = _fit_arima(series, order, seasonal_order)
    if fitted is None:
        return np.full(horizon, np.nan), None, None
    try:
        f = fitted.get_forecast(steps=horizon)
        point = f.predicted_mean.values
        ci = f.conf_int(alpha=0.05)
        lower = ci.iloc[:, 0].values
        upper = ci.iloc[:, 1].values
        return point, lower, upper
    except Exception:
        return np.full(horizon, np.nan), None, None
