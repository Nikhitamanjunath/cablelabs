"""
Time series analysis module.

Treats each (hour, frequency_band) as a univariate daily time series.
Implements baselines, ARIMA/SARIMA, ETS, Prophet with rolling CV,
prediction intervals, and CRPS.
"""
