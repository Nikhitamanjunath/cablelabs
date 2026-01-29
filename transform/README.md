# Transform Module

Extracts numerical data from preprocessed images by mapping colors to values using the color scale on each image.

## Overview

The `ImageTransformer` class processes images containing frequency spectrum graphs with color-coded values. It automatically detects the color scale on each image, extracts the maximum value using OCR, and converts pixel colors to numerical values.

## Components

- **`transform.py`** - Main transformation script with `ImageTransformer` class
- **`config.yaml`** - Configuration file for date ranges, paths, and debug options

## Usage

### Basic Usage

```bash
task transform
```

Or directly:

```bash
python transform/transform.py
```

### Programmatic Usage

```python
from transform import ImageTransformer, TransformConfig, load_config

# Load config
config = load_config("transform/config.yaml")
transformer = ImageTransformer(config)

# Process an image
image_path = Path("data/preprocessed/2025_01_01.png")
success = transformer.process_image(image_path)

if success:
    # Extract dataframe from the graph
    df = transformer.extract_dataframe(image_path)
    print(f"Extracted {df.shape[0]} rows × {df.shape[1]} columns")
```

## Processing Logic

### Scale Detection

1. **Automatic Region Detection**: Finds the color scale on the right side of the image
   - Analyzes vertical gradients in the rightmost 15% of the image
   - Identifies column with strongest vertical gradient (characteristic of color scales)
   - Uses fixed width (40 pixels) centered on the detected column

2. **Max Value Extraction**: Uses OCR to read the maximum value from the scale
   - Tries multiple image regions (bottom portion, expanded regions, full scale)
   - Applies multiple preprocessing strategies (contrast enhancement, thresholding, adaptive threshold)
   - Tests different OCR page segmentation modes
   - Filters results to reasonable ranges (0-1000)
   - Selects the maximum value found across all attempts
   - Falls back to default value (100.0) if OCR fails

3. **Color Extraction**: Samples colors from the scale bar
   - Finds the actual color bar column using gradient and variance analysis
   - Samples 500 colors evenly distributed from top to bottom
   - Filters out white/border colors
   - Validates color variation to ensure correct scale detection
   - Ensures darkest color is at the bottom (value 0)

### Data Extraction

1. **Graph Region Detection**: Uses fixed coordinates matching the scraper output
   - Graph region: (175, 100) to (908, 664) pixels
   - 24 rows (hours 0-23, with hour 23 at top, hour 0 at bottom)
   - 70 columns (frequency bands from 3.1 to 3.45 GHz)

2. **Pixel Sampling**: Samples center pixel of each cell
   - Divides graph into 24×70 grid
   - Samples center pixel of each cell
   - Converts pixel color to numeric value using input scale

3. **DataFrame Creation**: Creates structured DataFrame
   - Columns: date, hour, and 70 frequency band columns (e.g., "3.100-3.105")
   - Rows: one per hour per date
   - Values: numeric values from color-to-number conversion

### Color-to-Number Conversion

Uses the `ColorScale` class with input scale colors:
- Matches pixel colors to the extracted input scale using LAB color space
- Maps position on scale to numeric value using linear interpolation
- Formula: `value = (1.0 - position_ratio) * max_value`
  - Position 0 (top, yellow) → max_value
  - Position 1 (bottom, dark blue) → 0

## Configuration

Create `transform/config.yaml` from `transform/config.yaml.example`:

```yaml
# Date range for processing
from_date: "2025-01-01"
to_date: "2025-01-31"

# Frequency range for console display (doesn't affect extraction)
frequency_start: 3.1
frequency_end: 3.5

# Paths
preprocessed_folder: "data/preprocessed"
output_folder: "data/transformed"
output_data_file: "transformed_data.parquet"

# Debug modes
scale_debug: false      # Visualize detected scale regions
debug_dataframe: false  # Visualize pixel sampling points
```

### Configuration Values

- **`from_date`, `to_date`**: Date range for processing images (YYYY-MM-DD format)
- **`frequency_start`, `frequency_end`**: Only affects which columns are displayed in console output. All 70 frequency bands are always extracted.
- **`preprocessed_folder`**: Path to folder containing preprocessed images (format: `YYYY_MM_DD.png`)
- **`output_folder`**: Path where transformed data and debug images are saved
- **`output_data_file`**: Filename for combined DataFrame (supports `.parquet` or `.csv`)
- **`scale_debug`**: When enabled, saves debug images showing detected scale regions
- **`debug_dataframe`**: When enabled, saves visualization showing sampled pixels on the graph

## Output

### DataFrame Structure

The output DataFrame contains:
- **`date`**: Date string (YYYY-MM-DD format)
- **`hour`**: Hour of day (0-23)
- **70 frequency band columns**: Named like "3.100-3.105", "3.105-3.110", etc.

Each cell contains the numeric value extracted from the corresponding pixel color.

### Output Files

- **`{output_data_file}`**: Combined DataFrame in Parquet format (also saves CSV version)
- **`debug/debug_{date}.png`**: Scale detection visualization (if `scale_debug: true`)
- **`debug/dataframe/dataframe_{date}.png`**: Pixel sampling visualization (if `debug_dataframe: true`)

## Integration

The transform module is used by:
- **`analyze/analyze.py`** - Uses transformed data for analysis (prediction, confidence, error, anomaly)

## Dependencies

- Pillow (PIL) - Image processing
- NumPy - Numerical operations
- OpenCV - Image analysis and OCR preprocessing
- PyTesseract - OCR for reading scale values
- Pandas - DataFrame operations
- PyYAML - Configuration file parsing
- Pydantic - Configuration validation
- Matplotlib - Debug visualizations
