"""
Analyze module: prediction, confidence, error, and anomaly analysis.

Produces output consumable by the visualize module (same schema as before).
"""

from .analyze import main, run, load_config, AnalyzeConfig
from .prediction import Predictor, PredictionConfig
from .confidence import ConfidenceCalculator, ConfidenceConfig
from .error import ErrorCalculator, ErrorConfig
from .anomaly import AnomalyDetector, AnomalyConfig
from .image_visualizer import ImageVisualizer, ImageVisualizerConfig

__all__ = [
    "main",
    "run",
    "load_config",
    "AnalyzeConfig",
    "Predictor",
    "PredictionConfig",
    "ConfidenceCalculator",
    "ConfidenceConfig",
    "ErrorCalculator",
    "ErrorConfig",
    "AnomalyDetector",
    "AnomalyConfig",
    "ImageVisualizer",
    "ImageVisualizerConfig",
]
