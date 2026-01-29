"""
Next-day prediction for analysis.

Predicts next day's values using historical data with exponential smoothing,
moving average, linear regression, or peak-detection logic. Delegates confidence
scoring to ConfidenceCalculator.
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class PredictionConfig(BaseModel):
    """Prediction configuration."""
    lookback_days: int = Field(default=6, ge=1, description="Number of previous days to use for prediction")
    prediction_method: str = Field(
        default="exponential_smoothing",
        description="Prediction method: 'linear', 'exponential_smoothing', or 'moving_average'"
    )
    # Peak detection hyperparameters
    peak_threshold_multiplier: float = Field(default=1.5, ge=0.0)
    volatility_threshold: float = Field(default=0.35, ge=0.0)
    peak_detection_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    min_lookback_for_peak: int = Field(default=4, ge=1)
    max_lookback_for_peak: int = Field(default=6, ge=1)


class Predictor:
    """
    Online learning predictor for time series data.
    Delegates confidence scoring to an injected ConfidenceCalculator.
    """

    def __init__(self, config: PredictionConfig, confidence_calculator: "ConfidenceCalculator"):
        self.config = config
        self.lookback_days = config.lookback_days
        self.prediction_method = config.prediction_method
        self._confidence = confidence_calculator

    def predict_next_day(self, historical_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Predict the next day's data using historical data.

        Returns:
            Tuple of (predicted_df, confidence_df) - both DataFrames with same structure.
        """
        freq_columns = [col for col in historical_data.columns if col not in ['date', 'hour']]
        hours = sorted(historical_data['hour'].unique())

        predictions = []
        confidences = []

        for hour in hours:
            hour_data = historical_data[historical_data['hour'] == hour]
            predicted_row = {'hour': hour}
            confidence_row = {'hour': hour}

            for freq_col in freq_columns:
                values = hour_data[freq_col].values
                valid_values = values[~np.isnan(values)]

                if len(valid_values) == 0:
                    predicted_row[freq_col] = np.nan
                    confidence_row[freq_col] = 0.0
                elif len(valid_values) == 1:
                    predicted_row[freq_col] = valid_values[0]
                    confidence_row[freq_col] = 0.3
                else:
                    predicted, confidence = self._predict_value_with_confidence(
                        valid_values, hour, freq_col
                    )
                    predicted = max(0.0, predicted)
                    predicted_row[freq_col] = predicted
                    confidence_row[freq_col] = confidence

            predictions.append(predicted_row)
            confidences.append(confidence_row)

        pred_df = pd.DataFrame(predictions)
        conf_df = pd.DataFrame(confidences)
        pred_df = pred_df.sort_values('hour', ascending=False).reset_index(drop=True)
        conf_df = conf_df.sort_values('hour', ascending=False).reset_index(drop=True)
        return pred_df, conf_df

    def _predict_value_with_confidence(
        self, values: np.ndarray, hour: Optional[int], freq_col: Optional[str]
    ) -> Tuple[float, float]:
        """Predict next value and compute confidence via ConfidenceCalculator."""
        cfg = self.config
        used_peak_detection = False

        if len(values) >= cfg.min_lookback_for_peak:
            lookback = min(cfg.max_lookback_for_peak, len(values))
            recent = values[-lookback:]
            recent_max = np.max(recent)
            recent_mean = np.mean(recent)
            recent_std = np.std(recent)

            if (
                recent_max > recent_mean * cfg.peak_threshold_multiplier
                and recent_std > recent_mean * cfg.volatility_threshold
            ):
                peak_threshold = recent_max * cfg.peak_detection_threshold
                peak_positions = [i for i, v in enumerate(values) if v >= peak_threshold]
                if peak_positions:
                    used_peak_detection = True
                    peak_idx = peak_positions[-1]
                    peak_value = values[peak_idx]
                    positions_from_end = len(values) - 1 - peak_idx
                    last_value = values[-1]

                    if positions_from_end <= 1:
                        predicted = (
                            (peak_value * 0.97 + last_value * 0.03)
                            if last_value > recent_mean * cfg.peak_threshold_multiplier
                            else peak_value * 0.97
                        )
                    elif positions_from_end <= 3:
                        predicted = peak_value * 0.94
                    elif positions_from_end <= 5:
                        predicted = peak_value * 0.90
                    else:
                        predicted = peak_value * 0.88
                    predicted = max(0.0, predicted)
                    confidence = self._confidence.score(
                        values, predicted, used_peak_detection, hour, freq_col
                    )
                    return predicted, confidence

        if self.prediction_method == "exponential_smoothing":
            predicted = self._exponential_smoothing(values)
        elif self.prediction_method == "moving_average":
            predicted = self._moving_average(values)
        elif self.prediction_method == "linear":
            predicted = self._linear_regression(values)
        else:
            predicted = self._exponential_smoothing(values)

        confidence = self._confidence.score(
            values, predicted, used_peak_detection, hour, freq_col
        )
        return predicted, confidence

    def _exponential_smoothing(self, values: np.ndarray) -> float:
        """Exponential smoothing prediction with strong emphasis on recent values."""
        if len(values) == 0:
            return np.nan
        if len(values) == 1:
            return values[0]

        lookback = min(self.config.max_lookback_for_peak, len(values))
        recent = values[-lookback:]
        max_val = np.max(recent)
        min_val = np.min(recent)
        range_val = max_val - min_val
        mean_val = np.mean(recent)
        std_val = np.std(recent)

        if range_val > 40 and std_val > mean_val * 0.8:
            max_idx_in_recent = np.argmax(recent)
            if max_idx_in_recent >= lookback // 2:
                return max_val * (0.95 if max_idx_in_recent >= lookback - 2 else 0.90)
            return max_val * 0.82

        if len(values) >= 3:
            weights = np.array([0.05, 0.15, 0.80])
            return float(np.sum(values[-3:] * weights))
        if len(values) == 2:
            return 0.92 * values[-1] + 0.08 * values[-2]
        return values[-1]

    def _moving_average(self, values: np.ndarray) -> float:
        """Moving average prediction with emphasis on recent values."""
        if len(values) == 0:
            return np.nan
        window = min(self.lookback_days, len(values))
        recent_values = values[-window:]
        weights = np.linspace(0.5, 1.0, len(recent_values))
        weights = weights / weights.sum()
        return float(np.sum(recent_values * weights))

    def _linear_regression(self, values: np.ndarray) -> float:
        """Simple linear regression prediction."""
        if len(values) < 2:
            return values[0] if len(values) == 1 else np.nan
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        next_x = len(values)
        return float(coeffs[0] * next_x + coeffs[1])
