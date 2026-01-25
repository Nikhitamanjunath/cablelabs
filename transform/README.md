# Transform Script

This script processes images from a preprocessed folder and extracts numerical values from colors using a color scale on the right-hand side of the image.

## Features

- **Automatic Scale Detection**: Automatically identifies the color scale on the right-hand side of images
- **Color-to-Number Mapping**: Maps colors to numerical values based on the scale
- **Gradient Support**: Handles the color gradient from purple → pink → red → orange → yellow
- **Date Range Processing**: Processes images by date range (from_date to to_date)
- **Debug Mode**: Visualize detected scale regions with debug images

## Color Scale

The scale transitions from:
- **Purple** (lowest value, 0)
- **Pink**
- **Red**
- **Orange**
- **Yellow** (highest value, max from scale)

## Setup

1. Copy the example config file:
   ```bash
   cp config.yaml.example config.yaml
   ```

2. Edit `config.yaml` to match your setup:
   - Set `from_date` and `to_date` in YYYY-MM-DD format
   - Set `scale_debug` to `true` to enable debug visualization
   - The script automatically detects the scale region (no manual configuration needed)

## Usage

```bash
python transform.py
```

The script will:
1. Process images for each date in the specified range
2. Automatically detect the scale region on the right-hand side
3. Extract the maximum value from the scale using OCR
4. Set up the color-to-number mapping for each image
5. Save debug images (if `scale_debug: true`) showing the detected scale region

## Functions

### `ColorScale.set_scale(max_value)`
Sets the scale maximum value (minimum is always 0).

### `ColorScale.color_to_number(color, tolerance)`
Converts an RGB color tuple to a number based on the scale.

### `ImageTransformer.process_image(image_path)`
Processes an image to identify the scale and set up the color mapping.

### `ImageTransformer.get_color_value(color)`
Gets the numeric value for a given color using the current scale.

## Configuration

The config file requires three main values:

- **from_date**: Start date for processing (YYYY-MM-DD format)
- **to_date**: End date for processing (YYYY-MM-DD format)  
- **scale_debug**: Boolean flag to enable/disable debug visualization

When `scale_debug` is enabled, the script saves annotated images in the `data/transformed/debug/` folder showing:
- The detected scale region (highlighted in red)
- The identified scale range (0.0 to max_value)

## Example

```python
from transform import ImageTransformer, TransformConfig, ColorScale, load_config

# Load config
config = load_config("config.yaml")
transformer = ImageTransformer(config)

# Process an image to set up the scale
# Images are processed by date (YYYY_MM_DD.png format)
transformer.process_image(Path("data/preprocessed/2024_01_15.png"))

# Convert a color to a number
value = transformer.get_color_value((255, 0, 0))  # Red color
print(f"Value: {value}")
```

## Dependencies

- Pillow (PIL)
- NumPy
- OpenCV
- PyTesseract (for OCR to read scale numbers)
- PyYAML
- Pydantic
