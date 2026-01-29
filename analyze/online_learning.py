"""
Online learning for prediction analysis.

Maintains state (method accuracy per cell, lookback accuracy) and persists it
across runs. Selects which prediction method and lookback to use based on
historical accuracy; records results after each day.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import random
from datetime import datetime
from pydantic import BaseModel, Field


METHOD_NAMES = ["exponential_smoothing", "moving_average", "linear", "peak_detection"]


class OnlineLearningConfig(BaseModel):
    """Online learning configuration."""
    state_file: str = Field(
        default="data/online_learning_state.json",
        description="Path to state file for persistence",
    )
    exploration_epsilon: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Probability of choosing a random method for exploration",
    )
    lookback_candidates: List[int] = Field(
        default_factory=lambda: [4, 5, 6, 7],
        description="Lookback values to try/learn; empty to disable lookback learning",
    )
    default_lookback_days: int = Field(default=6, ge=1)
    default_prediction_method: str = Field(default="exponential_smoothing")


def _cell_key(hour: int, freq_col: str) -> str:
    return f"{hour}:{freq_col}"


def _accuracy(actual: float, predicted: float) -> float:
    """Accuracy = 1 - normalized_error, clamped to [0, 1]."""
    max_val = max(float(actual), float(predicted), 1.0)
    abs_error = abs(float(actual) - float(predicted))
    normalized_error = abs_error / max_val if max_val > 0 else 0.0
    return max(0.0, min(1.0, 1.0 - normalized_error))


class OnlineLearner:
    """
    Maintains method and lookback accuracy state; selects method/lookback
    adaptively; persists state to file.
    """

    def __init__(self, config: OnlineLearningConfig):
        self.config = config
        # method_accuracy: cell_key -> method_name -> list of accuracies (most recent last)
        self._method_accuracy: Dict[str, Dict[str, List[float]]] = {}
        # lookback_accuracy: lookback_str -> list of daily mean accuracies
        self._lookback_accuracy: Dict[str, List[float]] = {}
        self._load()

    def _load(self) -> None:
        """Load state from file if present."""
        path = Path(self.config.state_file)
        if not path.exists():
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self._method_accuracy = data.get("method_accuracy", {})
            self._lookback_accuracy = data.get("lookback_accuracy", {})
        except (json.JSONDecodeError, IOError):
            pass

    def save(self) -> None:
        """Save state to file."""
        path = Path(self.config.state_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "method_accuracy": self._method_accuracy,
            "lookback_accuracy": self._lookback_accuracy,
            "meta": {"last_updated": datetime.utcnow().isoformat() + "Z", "version": 1},
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def select_method(self, hour: int, freq_col: str, available_methods: Optional[List[str]] = None) -> str:
        """
        Return the method name with best historical accuracy for this cell.
        With probability exploration_epsilon, return a random method for exploration.
        """
        methods = available_methods or METHOD_NAMES
        if random.random() < self.config.exploration_epsilon:
            return random.choice(methods)
        key = _cell_key(hour, freq_col)
        cell_data = self._method_accuracy.get(key, {})
        best_method = self.config.default_prediction_method
        best_mean = -1.0
        for method in methods:
            accs = cell_data.get(method, [])
            if not accs:
                continue
            mean_acc = sum(accs) / len(accs)
            if mean_acc > best_mean:
                best_mean = mean_acc
                best_method = method
        return best_method

    def get_lookback_days(self) -> int:
        """
        Return the lookback value that has historically given the best mean accuracy.
        If no history or lookback_candidates empty, return default_lookback_days.
        """
        candidates = self.config.lookback_candidates
        if not candidates:
            return self.config.default_lookback_days
        best_lookback = self.config.default_lookback_days
        best_mean = -1.0
        for lb in candidates:
            accs = self._lookback_accuracy.get(str(lb), [])
            if not accs:
                continue
            mean_acc = sum(accs) / len(accs)
            if mean_acc > best_mean:
                best_mean = mean_acc
                best_lookback = lb
        return best_lookback

    def record_result(
        self,
        hour: int,
        freq_col: str,
        method_used: str,
        actual: float,
        predicted: float,
    ) -> None:
        """Compute accuracy and append to this method's history for (hour, freq_col)."""
        acc = _accuracy(actual, predicted)
        key = _cell_key(hour, freq_col)
        if key not in self._method_accuracy:
            self._method_accuracy[key] = {}
        if method_used not in self._method_accuracy[key]:
            self._method_accuracy[key][method_used] = []
        self._method_accuracy[key][method_used].append(acc)

    def record_lookback_result(self, lookback_used: int, day_mean_accuracy: float) -> None:
        """Append day mean accuracy to this lookback's history."""
        key = str(lookback_used)
        if key not in self._lookback_accuracy:
            self._lookback_accuracy[key] = []
        self._lookback_accuracy[key].append(day_mean_accuracy)
