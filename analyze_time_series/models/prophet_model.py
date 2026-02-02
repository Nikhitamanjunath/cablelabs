"""
Prophet wrapper for trend + seasonality.
Produces point forecasts, prediction intervals, and decomposition (trend, seasonality).
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False


def _prepare_prophet_df(series: pd.Series) -> pd.DataFrame:
    """Prophet expects columns ds (datetime) and y."""
    df = series.reset_index()
    df.columns = ["ds", "y"]
    df["ds"] = pd.to_datetime(df["ds"])
    return df.dropna(subset=["y"])


def forecast_prophet(
    series: pd.Series,
    horizon: int = 1,
    yearly_seasonality: bool = False,
    weekly_seasonality: bool = True,
    daily_seasonality: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Prophet with daily/weekly seasonality. Returns (point, lower_95, upper_95).
    """
    if not HAS_PROPHET or len(series) < 10:
        return np.full(horizon, np.nan), None, None
    df = _prepare_prophet_df(series)
    if len(df) < 10:
        return np.full(horizon, np.nan), None, None
    try:
        m = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            interval_width=0.95,
            uncertainty_samples=100,
        )
        m.fit(df)
        last_ds = df["ds"].max()
        future = pd.DataFrame({
            "ds": pd.date_range(start=last_ds, periods=horizon + 1, freq="D")[1:],
        })
        forecast = m.predict(future)
        point = forecast["yhat"].values
        lower = forecast["yhat_lower"].values if "yhat_lower" in forecast.columns else None
        upper = forecast["yhat_upper"].values if "yhat_upper" in forecast.columns else None
        return point, lower, upper
    except Exception:
        return np.full(horizon, np.nan), None, None


def forecast_prophet_with_decomposition(
    series: pd.Series,
    horizon: int = 1,
) -> Tuple[
    np.ndarray,
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[pd.DataFrame],
]:
    """
    Prophet forecast plus decomposition (trend, weekly, etc.) for the historical series.
    Returns (point, lower, upper, decomposition_df).
    decomposition_df has columns: date, trend, seasonal_weekly (if available).
    """
    if not HAS_PROPHET or len(series) < 10:
        return np.full(horizon, np.nan), None, None, None
    df = _prepare_prophet_df(series)
    if len(df) < 10:
        return np.full(horizon, np.nan), None, None, None
    try:
        m = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False,
            interval_width=0.95,
        )
        m.fit(df)
        # Forecast
        last_ds = df["ds"].max()
        future = pd.DataFrame({
            "ds": pd.date_range(start=last_ds, periods=horizon + 1, freq="D")[1:],
        })
        forecast = m.predict(future)
        point = forecast["yhat"].values
        lower = forecast.get("yhat_lower")
        upper = forecast.get("yhat_upper")
        lower = lower.values if lower is not None else None
        upper = upper.values if upper is not None else None
        # Historical decomposition: predict on training dates to get components
        comp = m.predict(df[["ds"]])
        decomp = pd.DataFrame({
            "date": comp["ds"],
            "trend": comp["trend"].values,
            "seasonal_weekly": comp.get("weekly", pd.Series(0, index=comp.index)).values,
        })
        return point, lower, upper, decomp
    except Exception:
        return np.full(horizon, np.nan), None, None, None
