"""
ETS (Error, Trend, Seasonal) wrapper using statsmodels.
Produces point forecasts and prediction intervals.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAS_STATS = True
except ImportError:
    HAS_STATS = False


def forecast_ets(
    series: pd.Series,
    horizon: int = 1,
    seasonal_periods: Optional[int] = 7,
    trend: Optional[str] = "add",
    seasonal: Optional[str] = "add",
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    ETS model. For daily data with weekly seasonality use seasonal_periods=7.
    Returns (point, lower_95, upper_95).
    """
    if not HAS_STATS or len(series) < 10:
        return np.full(horizon, np.nan), None, None
    y = series.dropna()
    if len(y) < 2 * (seasonal_periods or 1):
        return np.full(horizon, np.nan), None, None
    try:
        if seasonal_periods and seasonal:
            model = ExponentialSmoothing(
                y,
                trend=trend or None,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods,
                initialization_method="estimated",
            )
        else:
            model = ExponentialSmoothing(
                y,
                trend=trend or None,
                initialization_method="estimated",
            )
        fitted = model.fit(optimized=True, remove_bias=False)
        f = fitted.get_forecast(steps=horizon)
        point = f.predicted_mean.values
        ci = f.conf_int(alpha=0.05)
        if ci is not None and len(ci) == horizon:
            lower = ci.iloc[:, 0].values
            upper = ci.iloc[:, 1].values
            return point, lower, upper
        return point, None, None
    except Exception:
        return np.full(horizon, np.nan), None, None
