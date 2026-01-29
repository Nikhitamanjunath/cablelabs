"""
Confidence score calculation for prediction analysis.

Computes confidence (0.0–1.0) per prediction using data availability, variation,
trend consistency, method reliability, and historical accuracy (online learning).
"""

from typing import Optional, Tuple, Dict, List
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class ConfidenceConfig(BaseModel):
    """Confidence calculation configuration."""
    confidence_data_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    confidence_variation_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    confidence_consistency_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    confidence_method_weight: float = Field(default=0.1, ge=0.0, le=1.0)
    confidence_accuracy_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    confidence_initial: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence_min_samples: int = Field(default=3, ge=1)
    confidence_accuracy_decay: float = Field(default=0.2, ge=0.0, le=1.0)
    confidence_high_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    confidence_low_threshold: float = Field(default=0.4, ge=0.0, le=1.0)


class ConfidenceCalculator:
    """
    Calculates confidence score per prediction and maintains accuracy history
    for online learning.
    """

    def __init__(self, config: ConfidenceConfig, lookback_days: int):
        self.config = config
        self.lookback_days = lookback_days
        self.accuracy_history: Dict[Tuple[int, str], List[float]] = {}

    def update_accuracy_history(self, predicted_df: pd.DataFrame, actual_df: pd.DataFrame) -> None:
        """
        Update accuracy history based on actual vs predicted values.
        Called after predictions are made and actual data becomes available.
        """
        freq_columns = [col for col in predicted_df.columns if col not in ['date', 'hour']]
        pred_sorted = predicted_df.sort_values('hour').reset_index(drop=True)
        actual_sorted = actual_df.sort_values('hour').reset_index(drop=True)

        for row_idx in range(min(len(pred_sorted), len(actual_sorted))):
            pred_row = pred_sorted.iloc[row_idx]
            actual_row = actual_sorted.iloc[row_idx]
            hour = int(pred_row['hour'])

            for freq_col in freq_columns:
                pred_val = pred_row.get(freq_col)
                actual_val = actual_row.get(freq_col)

                if np.isnan(pred_val) or np.isnan(actual_val):
                    continue

                max_val = max(float(actual_val), float(pred_val), 1.0)
                abs_error = abs(float(actual_val) - float(pred_val))
                normalized_error = abs_error / max_val if max_val > 0 else 0.0
                accuracy = max(0.0, min(1.0, 1.0 - normalized_error))

                cell_key = (hour, freq_col)
                if cell_key not in self.accuracy_history:
                    self.accuracy_history[cell_key] = []
                self.accuracy_history[cell_key].append(accuracy)

    def score(
        self,
        values: np.ndarray,
        predicted: float,
        used_peak_detection: bool,
        hour: Optional[int] = None,
        freq_col: Optional[str] = None,
    ) -> float:
        """
        Calculate confidence score (0.0 to 1.0) for a prediction.

        Returns:
            Confidence score from 0.0 (low) to 1.0 (high).
        """
        if len(values) == 0:
            return 0.0
        if len(values) == 1:
            return 0.3

        c = self.config
        max_expected_data = self.lookback_days
        data_availability = min(1.0, len(values) / max_expected_data)
        data_score = data_availability * c.confidence_data_weight

        mean_val = np.mean(values)
        std_val = np.std(values)
        if mean_val > 0:
            cv = std_val / mean_val
            cv_score = max(0.0, 1.0 - min(1.0, cv))
            variation_score = cv_score * c.confidence_variation_weight
        else:
            variation_score = c.confidence_variation_weight * 0.5

        if len(values) >= 3:
            recent = values[-3:]
            recent_std = np.std(recent)
            recent_mean = np.mean(recent)
            if recent_mean > 0:
                recent_cv = recent_std / recent_mean
                consistency_score = max(0.0, 1.0 - min(1.0, recent_cv * 2)) * c.confidence_consistency_weight
            else:
                consistency_score = c.confidence_consistency_weight * 0.5
        else:
            consistency_score = c.confidence_consistency_weight * 0.5

        if used_peak_detection:
            method_score = c.confidence_method_weight * 0.5
        else:
            method_score = c.confidence_method_weight

        accuracy_score = 0.0
        if hour is not None and freq_col is not None:
            cell_key = (hour, freq_col)
            if cell_key in self.accuracy_history:
                accuracy_list = self.accuracy_history[cell_key]
                if len(accuracy_list) >= c.confidence_min_samples:
                    decay = c.confidence_accuracy_decay
                    weights = [(1.0 - decay) ** (len(accuracy_list) - 1 - i) for i in range(len(accuracy_list))]
                    total_weight = sum(weights)
                    if total_weight > 0:
                        weights = [w / total_weight for w in weights]
                        weighted_accuracy = sum(acc * w for acc, w in zip(accuracy_list, weights))
                        accuracy_score = weighted_accuracy * c.confidence_accuracy_weight
                    else:
                        accuracy_score = np.mean(accuracy_list) * c.confidence_accuracy_weight
                else:
                    accuracy_score = c.confidence_initial * c.confidence_accuracy_weight
            else:
                accuracy_score = c.confidence_initial * c.confidence_accuracy_weight

        total_confidence = data_score + variation_score + consistency_score + method_score + accuracy_score
        max_possible = (
            c.confidence_data_weight + c.confidence_variation_weight + c.confidence_consistency_weight
            + c.confidence_method_weight + c.confidence_accuracy_weight
        )
        if max_possible > 1.0:
            total_confidence = total_confidence / max_possible

        return max(0.0, min(1.0, total_confidence))
