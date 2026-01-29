# Analyze Module

Runs prediction, confidence, error, and anomaly analysis on transformed data and produces an output file consumable by the visualize module. Optional PNG visualization (original vs predicted, error map, confidence map) can be enabled via config.

## Overview

The analyze module is split into focused components:

- **`analyze.py`** – Main entry: loads config and transformed data, orchestrates prediction/confidence/error/anomaly, builds output DataFrame, optionally runs image visualizer, saves to parquet/csv/json.
- **`prediction.py`** – Predictor: next-day prediction using exponential smoothing, moving average, linear regression, or peak-detection logic. Own hyperparameters (lookback_days, prediction_method, peak thresholds).
- **`confidence.py`** – ConfidenceCalculator: confidence score (0.0–1.0) per prediction and accuracy history for online learning. Own hyperparameters (weights, initial, min_samples, decay, thresholds).
- **`error.py`** – ErrorCalculator: signed/normalized error and aggregate stats (mean, max, min, zero_error_percentage). Own hyperparameters (zero_error_threshold).
- **`anomaly.py`** – AnomalyDetector: z_score or IQR anomaly detection. Own hyperparameters (method, z_score_threshold, min_samples).
- **`image_visualizer.py`** – ImageVisualizer: PNG generation (original vs predicted, error map, confidence map). Uses ErrorCalculator for error map and stats.

## Components

| File | Class(es) | Role |
|------|-----------|------|
| `analyze.py` | AnalyzeConfig, main, run | Orchestration, config, output for visualize |
| `prediction.py` | Predictor, PredictionConfig | Next-day prediction |
| `confidence.py` | ConfidenceCalculator, ConfidenceConfig | Confidence score + accuracy history |
| `error.py` | ErrorCalculator, ErrorConfig | Signed/normalized error, map stats |
| `anomaly.py` | AnomalyDetector, AnomalyConfig | Anomaly detection (z_score/iqr) |
| `image_visualizer.py` | ImageVisualizer, ImageVisualizerConfig | Optional PNG visualization |

## Usage

### Basic Usage

```bash
task analyze
```

Or directly:

```bash
python analyze/analyze.py
```

Or as module (from project root):

```bash
python -m analyze.analyze
```

### Programmatic Usage

```python
from analyze import run, load_config

config = load_config("analyze/config.yaml")
run(config)
```

Or use individual classes:

```python
from analyze import Predictor, ConfidenceCalculator, ErrorCalculator, AnomalyDetector
from analyze import PredictionConfig, ConfidenceConfig, ErrorConfig, AnomalyConfig

# Build sub-configs and wire dependencies
confidence_calc = ConfidenceCalculator(ConfidenceConfig(...), lookback_days=6)
predictor = Predictor(PredictionConfig(...), confidence_calc)
error_calc = ErrorCalculator(ErrorConfig(...))
anomaly_detector = AnomalyDetector(AnomalyConfig(...))
```

## Configuration

Create `analyze/config.yaml` from `analyze/config.yaml.example`. The config uses flat keys for paths and all component hyperparameters (prediction, confidence, error, anomaly). See the example file for full options.

Key paths:

- **`transformed_data_file`** – Input from transform step (Parquet or CSV).
- **`prediction_data_file`** – Output file consumed by the visualize module (same schema: date, hour, frequency_band, predicted, confidence, actual, error, is_anomaly).
- **`output_folder`** – Where PNGs are saved when `visualize_predictions: true`.

## Output

- **Prediction data file** (e.g. `data/predictions/predictions_data.parquet`): One row per (date, hour, frequency_band) with columns date, hour, frequency_band, predicted, confidence, actual, error, is_anomaly. Consumed by the visualize module.
- **PNG images** (optional): `prediction_YYYY_MM_DD.png` in `output_folder` when `visualize_predictions: true`.

## Integration

- **transform** – Transformed data input; image visualizer uses transform for scale extraction.
- **color_scale** – Image visualizer uses ColorScale for value-to-color mapping.
- **visualize** – Reads the prediction data file produced by analyze (same path and schema as before).
