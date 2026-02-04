# Interactive Visualization Module

Interactive web-based viewer for exploring prediction data with date selection and view mode toggling.

## Overview

The `visualize.py` script provides a Dash-based web interface for interactively exploring prediction data. It allows you to:
- Select any date from available predictions
- Toggle between different view modes (predicted, confidence, actual, error)
- View statistics for the selected date
- Hover over cells to see detailed values

## Components

- **`visualize.py`** - Main visualization script with Dash web server
- **`config.yaml`** - Configuration file for data path and server settings
- **`config.yaml.example`** - Example configuration file

## Usage

### Basic Usage

```bash
task visualize
```

Or directly:

```bash
python visualize/visualize.py
```

The server will start and display a URL (default: http://127.0.0.1:8050). Open this URL in your web browser to access the interactive viewer.

### Prerequisites

1. Run `task analyze` first to generate prediction data
2. The prediction data file must exist (configured in `analyze/config.yaml` as `prediction_data_file`)

## Features

### Date Selection
- Dropdown menu showing all available dates from prediction data
- Automatically selects the most recent date by default

### View Modes
Toggle between four different views using radio buttons:

1. **Predicted** - Shows predicted values with blue-to-yellow color scale
2. **Confidence** - Shows confidence scores (0.0-1.0) with green-yellow-red color scale
3. **Actual** - Shows actual values (if available) with blue-to-yellow color scale
4. **Error** - Shows signed errors (actual - predicted) with blue-white-red diverging scale

### Interactive Grid
- 24 rows (hours 23 to 0, with hour 23 at top)
- 70 columns (frequency bands from 3.1 to 3.45 GHz)
- Hover over any cell to see:
  - Hour
  - Frequency band
  - Value for the current view mode

### Statistics Panel
Displays at the bottom of the screen:
- **Mean Confidence** - Average confidence score for the date
- **Mean Error** - Average absolute error (if actual data available)
- **Max Error** - Maximum absolute error
- **Min Error** - Minimum absolute error
- **Zero Error** - Percentage of cells with effectively zero error (< 0.1)

## Configuration

Create `visualize/config.yaml` from `visualize/config.yaml.example`:

```yaml
# Path to prediction data file (created by analyze task)
prediction_data_file: "data/predictions/predictions_data.parquet"

# Port for the Dash web server
port: 8050

# Host for the Dash web server
# Use "0.0.0.0" to allow external access, "127.0.0.1" for local only
host: "127.0.0.1"
```

### Configuration Values

- **`prediction_data_file`**: Path to the prediction data file created by the `analyze` task. Must match the `prediction_data_file` value in `analyze/config.yaml`.
- **`port`**: Port number for the web server (default: 8050)
- **`host`**: Host address (default: "127.0.0.1" for local access, use "0.0.0.0" for network access)

## Data Format

The prediction data file contains one row per (date, hour, frequency_band) with columns:
- `date`: Date string (YYYY-MM-DD)
- `hour`: Hour of day (0-23)
- `frequency_band`: Frequency band string (e.g., "3.100-3.105")
- `predicted`: Predicted value (float or None)
- `confidence`: Confidence score 0.0-1.0 (float or None)
- `actual`: Actual value if available (float or None)
- `error`: Signed error (actual - predicted) if available (float or None)

## Dependencies

- Dash - Web framework for interactive visualizations
- Plotly - Graphing library for heatmaps
- Pandas - Data loading and manipulation
- PyYAML - Configuration file parsing
- Pydantic - Configuration validation

## Port allocation

Web apps in this project use distinct ports to avoid conflicts:

| App | Port | Config / usage |
|-----|------|----------------|
| Main viewer (`task visualize`) | 8050 | `visualize/config.yaml` (profile: main) |
| Blend viewer (`task visualize-blend`) | 8050 | Same config, `--profile blend` |
| Persistence+delta viewer (`task visualize-persistence-delta`) | 8050 | Same config, `--profile persistence_delta` |
| Time series notebook Dash apps | 8052–8063 | Hardcoded in `time_series/time_series.ipynb` |

All viewers use the **single** `visualize/config.yaml`; profiles (main, blend, persistence_delta) select which prediction data file and viewer mode to use.

When adding a new web app, use a port not in this range (e.g. 8070 or higher).

## Integration

The visualize module uses:
- **`analyze/analyze.py`** - For generating prediction data files

The prediction data file is created automatically when you run `task analyze`.
