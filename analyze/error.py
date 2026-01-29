"""
Error calculation for prediction analysis.

Provides signed error, normalized error, and aggregate statistics (mean, max, min,
zero_error_percentage) for actual vs predicted values, with configurable threshold.
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class ErrorConfig(BaseModel):
    """Error calculation configuration."""
    zero_error_threshold: float = Field(
        default=0.1,
        ge=0.0,
        description="Absolute error below this is considered 'zero error' for stats"
    )


class ErrorCalculator:
    """
    Computes signed/normalized error and aggregate error statistics.
    """

    def __init__(self, config: ErrorConfig):
        self.config = config

    def signed_error(self, actual: float, predicted: float) -> float:
        """
        Signed error: actual - predicted.
        Positive = predicted too low; negative = predicted too high.
        """
        return float(actual) - float(predicted)

    def normalized_error(self, actual: float, predicted: float) -> float:
        """
        Normalized error for accuracy: abs_error / max(actual, predicted, 1.0).
        Used for accuracy = 1.0 - normalized_error.
        """
        max_val = max(float(actual), float(predicted), 1.0)
        abs_error = abs(float(actual) - float(predicted))
        return abs_error / max_val if max_val > 0 else 0.0

    def compute_map_stats(
        self,
        predicted_df: pd.DataFrame,
        actual_df: pd.DataFrame,
        freq_columns: list,
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[np.ndarray]]:
        """
        Compute error map (signed errors per cell) and aggregate statistics.

        Returns:
            (mean_error, max_error, min_error, zero_error_percentage, error_map)
            error_map is (num_hours, num_freq_bands); stats are None if no valid cells.
        """
        num_hours = len(predicted_df)
        num_freq_bands = len(freq_columns)
        pred_sorted = predicted_df.sort_values('hour', ascending=False).reset_index(drop=True)
        actual_sorted = actual_df.sort_values('hour', ascending=False).reset_index(drop=True)

        error_map = np.zeros((num_hours, num_freq_bands))
        error_values = []
        zero_error_count = 0
        total_valid_cells = 0

        thresh = self.config.zero_error_threshold

        for row_idx in range(min(len(pred_sorted), len(actual_sorted))):
            pred_row = pred_sorted.iloc[row_idx]
            actual_row = actual_sorted.iloc[row_idx]

            for col_idx, freq_col in enumerate(freq_columns):
                pred_val = pred_row[freq_col]
                actual_val = actual_row[freq_col]

                if not np.isnan(actual_val) and not np.isnan(pred_val):
                    signed = self.signed_error(actual_val, pred_val)
                    error_map[row_idx, col_idx] = signed
                    abs_err = abs(signed)
                    error_values.append(abs_err)
                    total_valid_cells += 1
                    if abs_err < thresh:
                        zero_error_count += 1
                else:
                    error_map[row_idx, col_idx] = np.nan

        mean_error = None
        max_error = None
        min_error = None
        zero_error_percentage = None

        if error_values:
            mean_error = float(np.mean(error_values))
            non_zero_errors = [e for e in error_values if e >= thresh]
            if non_zero_errors:
                max_error = float(np.max(non_zero_errors))
                min_error = float(np.min(non_zero_errors))
            if total_valid_cells > 0:
                zero_error_percentage = (zero_error_count / total_valid_cells) * 100.0

        return mean_error, max_error, min_error, zero_error_percentage, error_map
