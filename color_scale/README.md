# Color Scale Module

This module provides the `ColorScale` class for mapping between colors and numeric values, with support for interactive training and testing.

## Overview

The `ColorScale` class manages color-to-number and number-to-color conversions based on a color scale. It uses position-based mapping where colors are sampled from a scale bar and their positions determine their numeric values.

## Usage

### Basic Usage

```python
from color_scale import ColorScale

# Create a color scale with default colors
scale = ColorScale(max_value=60.0)

# Convert color to number
value = scale.color_to_number((255, 255, 0))  # Yellow -> high value

# Convert number to color
color = scale.number_to_color(30.0)  # Returns RGB tuple
```

### Loading from Image

```python
# Scale colors extracted from an image
scale_colors = [(r, g, b), ...]  # List of RGB tuples
scale = ColorScale(max_value=60.0, scale_colors_by_position=scale_colors)
```

### Saving and Loading Trained Models

Models are saved based on the `model_name` in `color_scale/config.yaml`:

```python
# Save a trained model (uses model_name from config)
scale.save_model("data/models/default.pkl")

# Load a saved model
scale = ColorScale.load_model("data/models/default.pkl")
```

## Training

Use the interactive training script to improve color-to-number mapping:

```bash
python color_scale/train.py
```

Or use the task:
```bash
task train-color-scale
```

The trainer will:
1. Load configuration from `color_scale/config.yaml` (or use defaults)
2. Show you colors from the scale
3. Ask you to assign numeric values to each color
4. Learn from your feedback
5. Save the trained model to `data/models/{model_name}.pkl` based on config

## Testing

Test the color scale in both directions:

```bash
python color_scale/test.py
```

The test script will:
- Test `color_to_number()`: Shows colors and predicted values
- Test `number_to_color()`: Shows values and predicted colors
- Allow you to verify accuracy

## Integration

The trained model (`data/models/{model_name}.pkl`) is automatically used by:
- `transform/transform.py` - Uses model specified in `transform/config.yaml` (`model_name` field)
- `predict/predict.py` - Uses model specified in `predict/config.yaml` (`model_name` field)

If no trained model exists, the system falls back to extracting the scale from images automatically.

## Configuration

Create `color_scale/config.yaml` from `color_scale/config.yaml.example`:

```yaml
model_name: "default"  # Will create data/models/default.pkl
max_value: 60.0
num_colors: 25
```

You can create multiple models (e.g., "apple", "banana") by changing `model_name` and training again.
