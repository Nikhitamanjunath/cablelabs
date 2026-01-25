# Predict Script

This script loads transformed data from the transform step and performs online learning to predict the next day's data. It uses the color scale from preprocessed images to generate visualization images showing actual vs predicted side by side.

## Features

- **Online Learning**: Uses historical data to predict the next day's values
- **Multiple Prediction Methods**: Supports linear regression, exponential smoothing, and moving average
- **Color Scale Integration**: Uses the actual color scale from preprocessed images to generate visualizations
- **Side-by-Side Comparison**: Shows predicted vs actual data side by side
- **Configurable Lookback**: Adjust how many previous days to use for prediction

## Setup

1. Copy the example config file:
   ```bash
   cp config.yaml.example config.yaml
   ```

2. Edit `config.yaml` to match your setup:
   - Set `transformed_data_file` to point to your transformed data file
   - Set `preprocessed_folder` to point to your preprocessed images
   - Adjust `lookback_days` to control how many previous days to use
   - Choose `prediction_method`: "linear", "exponential_smoothing", or "moving_average"

## Usage

```bash
python predict.py
```

Or using Task:
```bash
task predict
```

The script will:
1. Load the transformed data from the transform step
2. For each day (starting from `lookback_days + 1`):
   - Use the previous N days (where N = `lookback_days`) to predict the next day
   - Load the color scale from the preprocessed image for that date
   - Generate a visualization showing predicted vs actual data side by side
   - Save the visualization to the output folder

## Configuration

### Hyperparameters

- **`lookback_days`**: Number of previous days to use for prediction (default: 6)
- **`learning_rate`**: Learning rate for online learning (0.0 to 1.0, default: 0.1)
- **`prediction_method`**: Method to use for prediction:
  - `"exponential_smoothing"`: Exponential smoothing (default)
  - `"moving_average"`: Simple moving average
  - `"linear"`: Linear regression extrapolation
- **`exponential_smoothing_alpha`**: Alpha parameter for exponential smoothing (0.0 to 1.0, default: 0.3)
- **`visualize_predictions`**: Enable/disable visualization generation (default: true)

## Prediction Methods

### Exponential Smoothing
Uses exponential smoothing to predict the next value. Higher `alpha` values give more weight to recent observations.

### Moving Average
Uses a simple moving average of the last N days to predict the next value.

### Linear Regression
Uses linear regression to extrapolate the trend and predict the next value.

## Output

The script generates visualization images in the `output_folder` directory:
- `prediction_YYYY_MM_DD.png`: Side-by-side comparison of predicted vs actual data for each date

Each image shows:
- **Left side**: Predicted data
- **Right side**: Actual data (if available)
- **Right edge**: Color scale bar

## Requirements

- Requires the `transform` step to be run first to generate the transformed data file
- Requires preprocessed images to extract color scales
- All dependencies from `requirements.txt` must be installed
