"""
Anomaly detection for prediction analysis.

Detects if a predicted value is an anomaly based on historical data,
using z-score or IQR method with configurable hyperparameters.
"""

import numpy as np
from pydantic import BaseModel, Field


class AnomalyConfig(BaseModel):
    """Anomaly detection configuration."""
    anomaly_method: str = Field(
        default="z_score",
        description="Method for anomaly detection: 'z_score' or 'iqr'"
    )
    anomaly_z_score_threshold: float = Field(
        default=2.5,
        ge=0.0,
        description="Z-score threshold; values with |z-score| > this are anomalies"
    )
    anomaly_min_samples: int = Field(
        default=5,
        ge=1,
        description="Minimum number of historical samples needed for anomaly detection"
    )


class AnomalyDetector:
    """
    Detects if a predicted value is an anomaly based on historical data.
    """

    def __init__(self, config: AnomalyConfig):
        self.config = config

    def detect(self, historical_values: np.ndarray, predicted_value: float) -> bool:
        """
        Detect if a predicted value is an anomaly based on historical data.

        Args:
            historical_values: Array of historical values for this cell
            predicted_value: The predicted value to check

        Returns:
            True if anomaly, False otherwise
        """
        if len(historical_values) < self.config.anomaly_min_samples:
            return False

        valid_values = historical_values[~np.isnan(historical_values)]
        if len(valid_values) < self.config.anomaly_min_samples:
            return False

        if self.config.anomaly_method == "z_score":
            mean_val = np.mean(valid_values)
            std_val = np.std(valid_values)

            if std_val == 0:
                return abs(predicted_value - mean_val) > 0.01

            z_score = abs((predicted_value - mean_val) / std_val)
            return z_score > self.config.anomaly_z_score_threshold

        elif self.config.anomaly_method == "iqr":
            q1 = np.percentile(valid_values, 25)
            q3 = np.percentile(valid_values, 75)
            iqr = q3 - q1

            if iqr == 0:
                return abs(predicted_value - np.median(valid_values)) > 0.01

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            return predicted_value < lower_bound or predicted_value > upper_bound

        return False
