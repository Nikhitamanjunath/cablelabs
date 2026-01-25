# Predict Module

Predicts next day's frequency spectrum data using online learning and generates visualization images comparing predicted vs actual values.

## Overview

The `OnlinePredictor` class uses historical data to predict the next day's values for each hour and frequency band. The `PredictionVisualizer` class generates side-by-side comparison images showing original, predicted, and error maps.

## Components

- **`predict.py`** - Main prediction script with `OnlinePredictor` and `PredictionVisualizer` classes
- **`config.yaml`** - Configuration file for data paths, hyperparameters, and visualization options

## Usage

### Basic Usage

```bash
task predict
```

Or directly:

```bash
python predict/predict.py
```

### Programmatic Usage

```python
from predict import OnlinePredictor, PredictionVisualizer, PredictConfig, load_config
import pandas as pd

# Load config
config = load_config("predict/config.yaml")

# Initialize predictor
predictor = OnlinePredictor(config)

# Load historical data
df = pd.read_parquet("data/transformed/transformed_data.parquet")

# Predict next day
historical_data = df[df['date'].isin(['2025-01-01', '2025-01-02'])]
predicted_df = predictor.predict_next_day(historical_data)
```

## Prediction Logic

### Online Learning Approach

1. **Historical Data Window**: Uses the last N days (configurable via `lookback_days`) to predict the next day
2. **Per-Hour Prediction**: Predicts each hour (0-23) independently using historical values for that hour
3. **Per-Frequency Prediction**: Predicts each frequency band independently using historical values for that band

### Prediction Methods

The predictor supports three methods (configurable via `prediction_method`):

1. **Exponential Smoothing** (default)
   - Heavy weight on recent values: [0.05, 0.15, 0.80] for last 3 values
   - For volatile data with peaks, detects and uses peak values with decay based on recency
   - Peak detection: Identifies significant peaks (max > 1.5× mean, std > 0.35× mean)
   - Applies decay based on peak recency: 0.97 (very recent) to 0.88 (older)

2. **Moving Average**
   - Weighted moving average favoring recent values
   - Weights increase linearly from 0.5 to 1.0 for most recent
   - Window size = `lookback_days`

3. **Linear Regression**
   - Simple linear extrapolation using polynomial fitting
   - Extrapolates trend to predict next value

### Peak Detection Strategy

For volatile data, the predictor uses intelligent peak detection:
- **Peak Identification**: Finds values ≥ 80% of recent maximum
- **Recency-Based Decay**: 
  - Very recent (last 1-2 values): 97% of peak value
  - Recent (last 3-5 values): 90-94% of peak value
  - Older peaks: 88% of peak value
- **Rationale**: Peaks in spectrum data often represent actual signal events, not noise

## Visualization Logic

### Image Generation

1. **Original Image**: Extracts graph region from preprocessed image and converts colors to values, then back to colors using the same scale as predictions
2. **Predicted Image**: Converts predicted values to colors using the input scale from the original image
3. **Error Map**: Shows signed error (actual - predicted) for each cell with numeric values
4. **Color Scale**: Displays the color scale bar matching the original image's scale

### Color Mapping

- Uses the **exact same color scale** extracted from the original image
- Ensures predicted colors are directly comparable to original colors
- Never uses white/light colors - always maps to valid scale colors
- For NaN/invalid values, uses minimum color (dark blue, value 0)

### Error Calculation

- **Signed Error**: `actual - predicted` (positive = predicted too low, negative = predicted too high)
- **Zero Error Threshold**: Errors < 0.1 are considered effectively zero (accounts for rounding in color conversion)
- **Statistics**: Mean, max, min error (excluding zero errors), and percentage of zero-error cells

## Configuration

Create `predict/config.yaml` from `predict/config.yaml.example`:

```yaml
# Data paths
transformed_data_file: "data/transformed/transformed_data.parquet"
preprocessed_folder: "data/preprocessed"
output_folder: "data/predictions"

# Prediction hyperparameters
lookback_days: 6
prediction_method: "exponential_smoothing"

# Visualization
visualize_predictions: true
```

### Configuration Values

- **`transformed_data_file`**: Path to transformed data file from transform step (Parquet or CSV)
- **`preprocessed_folder`**: Path to folder containing preprocessed images (for extracting color scales)
- **`output_folder`**: Path where prediction visualization images are saved
- **`lookback_days`**: Number of previous days to use for prediction (default: 6)
- **`prediction_method`**: Method to use - `"exponential_smoothing"`, `"moving_average"`, or `"linear"` (default: `"exponential_smoothing"`)
- **`visualize_predictions`**: Enable/disable visualization image generation (default: `true`)

## Output

### Visualization Images

Each prediction generates an image file: `prediction_YYYY_MM_DD.png`

The image contains:
- **Left**: Original data (extracted from preprocessed image)
- **Center**: Predicted data (generated from predicted values)
- **Right**: Color scale bar
- **Bottom** (if actual data available): Error map showing signed errors for each cell

### Error Statistics

When actual data is available, the script prints:
- Mean error across all cells
- Max error (excluding zero errors)
- Min error (excluding zero errors)
- Percentage of cells with effectively zero error (< 0.1)

## Integration

The predict module uses:
- **`transform/transform.py`** - For extracting color scales from preprocessed images
- **`color_scale/color_scale.py`** - For color-to-number and number-to-color conversion

## Dependencies

- Pandas - DataFrame operations
- NumPy - Numerical operations
- Pillow (PIL) - Image generation
- Matplotlib - Image processing
- PyYAML - Configuration file parsing
- Pydantic - Configuration validation
- OpenCV - Image processing (via transform module)
- PyTesseract - OCR (via transform module)
