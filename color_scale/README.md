# Color Scale Module

Maps colors to numeric values and vice versa using a reference color scale.

## Overview

The `ColorScale` class converts between RGB colors and numeric values based on a vertical color scale image (`data/scale/scale.png`). The scale ranges from dark blue (0) at the bottom to yellow (max) at the top.

## Components

- **`color_scale.py`** - Main `ColorScale` class for color-to-number and number-to-color conversion
- **`visualize.py`** - Visualizes the color scale by showing colors at different value increments
- **`test.py`** - Tests color scale extraction from preprocessed images

## Usage

### Basic Usage

```python
from color_scale import ColorScale

# Initialize with default reference scale (data/scale/scale.png)
scale = ColorScale()

# Convert color to number
value = scale.color_to_number((255, 255, 0), max_value=60.0)  # Yellow -> high value

# Convert number to color
color = scale.number_to_color(30.0, max_value=60.0)  # Returns RGB tuple
```

### With Input Scale

When processing images, you can provide the scale colors extracted from the image:

```python
# Extract scale from image (done by transform.py)
input_scale_colors = [(r, g, b), ...]  # List of RGB tuples from top to bottom
input_max_value = 60.0  # Max value from OCR

# Convert color using input scale
value = scale.color_to_number(
    color=(200, 150, 100),
    input_scale_colors=input_scale_colors,
    input_max_value=input_max_value
)
```

## Tasks

- **`task visualize-color-scale`** - Visualize the reference color scale with value increments
- **`task test-color-scale`** - Test color scale extraction from preprocessed images

## Configuration

Create `color_scale/config.yaml` from `color_scale/config.yaml.example`:

```yaml
num_test_images: 5  # Number of images to test in test-color-scale
```

## Calculation Logic

The module ensures accurate conversions through several key mechanisms:

### Reference Scale Loading

1. **Vertical Pixel Reading**: Reads every pixel vertically from top to bottom
   - Top pixel = maximum value position
   - Bottom pixel = zero value position
   - For multi-column scales, averages across the middle 50% to avoid edge artifacts

2. **Dark Border Detection**: Automatically detects and handles dark borders/labels at the top
   - If top pixel is very dark (brightness < 50), searches for the first bright yellow color
   - Uses the bright yellow as the actual maximum value position
   - Ensures accurate mapping even when scales have labels or borders

### Color-to-Number Conversion

1. **LAB Color Space Matching**: Converts RGB colors to LAB color space for perceptual accuracy
   - LAB color space better matches human color perception than RGB
   - Uses Euclidean distance in LAB space to find closest matching color

2. **Input Scale Priority**: When `input_scale_colors` and `input_max_value` are provided:
   - Matches the input color directly to the input scale colors (extracted from the image)
   - Maps position on input scale to numeric value using linear interpolation
   - Formula: `value = (1.0 - position_ratio) * max_value`
     - Position 0 (top) → max_value
     - Position 1 (bottom) → 0

3. **Reference Scale Fallback**: When no input scale is provided:
   - Matches color to reference scale using LAB distance
   - Rejects matches with distance > 50.0 (color not on scale)
   - Uses same linear mapping formula

### Number-to-Color Conversion

1. **Value to Position Mapping**: Converts numeric value to position on scale
   - Formula: `position_ratio = 1.0 - (value / max_value)`
   - Value 0 → position 1.0 (bottom, dark blue)
   - Value max_value → position 0.0 (top, yellow)

2. **Color Interpolation**: Maps position to color using smooth interpolation
   - Calculates index in reference color array: `position_idx = position_ratio * (num_colors - 1)`
   - Interpolates between two adjacent colors for smooth transitions
   - Formula: `color = color1 + (color2 - color1) * fraction`
   - Clamps RGB values to valid range [0, 255]

3. **Edge Case Handling**: 
   - For values near maximum (≥ 99% of max_value) with dark top colors:
     - Finds brightest yellow color in the first half of the scale
     - Uses that position for accurate maximum value representation

### Accuracy Guarantees

- **Perceptual Matching**: LAB color space ensures colors that look similar are matched correctly
- **Linear Mapping**: Consistent linear relationship between position and value ensures predictable conversions
- **Interpolation**: Smooth color transitions between discrete scale positions
- **Input Scale Support**: Uses actual scale from each image when available, accounting for image-specific variations
- **Validation**: Rejects colors that are too far from the scale (distance threshold)

## Integration

The `ColorScale` class is used by:
- **`transform/transform.py`** - Extracts numeric data from preprocessed images
- **`analyze/image_visualizer.py`** - Generates visualization images from predicted values
