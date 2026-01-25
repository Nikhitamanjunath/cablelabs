#!/usr/bin/env python3
"""
Color Scale Testing Script

Tests the ColorScale class by extracting scales from preprocessed images
and showing the color scale output for each one.
"""

import sys
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
from PIL import Image
import yaml
import cv2
import pytesseract
import re
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving files
import matplotlib.pyplot as plt

from color_scale import ColorScale

# Import ImageTransformer to extract scales from images
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

transform_path = project_root / "transform"
if str(transform_path) not in sys.path:
    sys.path.insert(0, str(transform_path))

import importlib.util
spec = importlib.util.spec_from_file_location("transform_module", transform_path / "transform.py")
transform_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transform_module)
ImageTransformer = transform_module.ImageTransformer
TransformConfig = transform_module.TransformConfig


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """Convert RGB tuple to hex color string."""
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def find_brightest_yellow(scale_colors: list) -> tuple:
    """
    Find the brightest yellow color in the scale and its position.
    
    Returns:
        Tuple of (position_index, color, brightness_score)
    """
    brightest_idx = 0
    brightest_score = 0
    
    for i, color in enumerate(scale_colors):
        # Score for yellow: high R+G, relatively low B, high brightness
        yellow_score = color[0] + color[1] - color[2]
        brightness = (color[0] + color[1] + color[2]) / 3.0
        combined_score = yellow_score + brightness
        
        if combined_score > brightest_score:
            brightest_score = combined_score
            brightest_idx = i
    
    return (brightest_idx, scale_colors[brightest_idx], brightest_score)


def find_all_label_positions(img_array: np.ndarray, scale_region_dict: dict) -> list:
    """
    Find all number labels on the scale and their vertical positions.
    
    Args:
        img_array: Full image array
        scale_region_dict: Scale region coordinates
    
    Returns:
        List of tuples (value, position_ratio) where position_ratio is 0.0 (top) to 1.0 (bottom)
    """
    scale_height = scale_region_dict["bottom_y"] - scale_region_dict["top_y"]
    scale_width = scale_region_dict["right_x"] - scale_region_dict["left_x"]
    
    # Extract region to the right of the scale where labels are
    right_expansion = 80
    expanded_right_x = min(img_array.shape[1], scale_region_dict["right_x"] + right_expansion)
    label_region = img_array[
        scale_region_dict["top_y"]:scale_region_dict["bottom_y"],
        scale_region_dict["right_x"]:expanded_right_x
    ]
    
    # Try OCR on vertical slices to find all numbers
    num_slices = 40  # More slices for better resolution
    slice_height = scale_height // num_slices
    
    label_positions = []  # List of (value, position_ratio)
    
    for i in range(num_slices):
        y_start = i * slice_height
        y_end = min((i + 1) * slice_height, scale_height)
        slice_region = label_region[y_start:y_end, :]
        
        try:
            # Convert to grayscale and preprocess
            gray = cv2.cvtColor(slice_region, cv2.COLOR_RGB2GRAY)
            processed = cv2.convertScaleAbs(gray, alpha=2.0, beta=50)
            
            # Try OCR
            ocr_text = pytesseract.image_to_string(processed, config='--psm 8')
            numbers = re.findall(r'\d+\.?\d*', ocr_text)
            
            for num_str in numbers:
                try:
                    num = float(num_str)
                    # Filter to reasonable range (0-1000)
                    if 0 <= num <= 1000:
                        # Calculate position ratio (0.0 = top, 1.0 = bottom)
                        position_ratio = (y_start + y_end) / 2 / scale_height
                        label_positions.append((num, position_ratio))
                except ValueError:
                    continue
        except Exception:
            continue
    
    # Remove duplicates and sort by position
    # Group by value and take the average position for each value
    value_to_positions = {}
    for value, pos in label_positions:
        if value not in value_to_positions:
            value_to_positions[value] = []
        value_to_positions[value].append(pos)
    
    # Average positions for each value
    unique_labels = []
    for value, positions in value_to_positions.items():
        avg_position = np.mean(positions)
        unique_labels.append((value, avg_position))
    
    # Sort by position (top to bottom)
    unique_labels.sort(key=lambda x: x[1])
    
    return unique_labels


def calculate_actual_max_value(labeled_max: float, max_label_position: Optional[float], brightest_yellow_position: float) -> float:
    """
    Calculate the actual max value at the top (brightest yellow) based on the labeled max.
    
    Args:
        labeled_max: The max value found by OCR (e.g., 60)
        max_label_position: Position of the max label on the scale (0.0 to 1.0, where 0 = top), or None
        brightest_yellow_position: Position of brightest yellow (0.0 to 1.0, where 0 = top)
    
    Returns:
        Actual max value at the top position
    """
    # If we found the label position, use it
    if max_label_position is not None and max_label_position > 0:
        # Brightest yellow is at position 0 (top), label is at max_label_position
        # Extrapolate: actual_max = labeled_max / max_label_position
        actual_max = labeled_max / max_label_position
    elif brightest_yellow_position < 0.1:
        # Brightest yellow is near the top, assume label is around 0.9 (typical scale layout)
        # This is a fallback if we can't detect the label position
        estimated_label_position = 0.9
        actual_max = labeled_max / estimated_label_position
    else:
        # Brightest yellow is not at top, use labeled max
        actual_max = labeled_max
    
    return actual_max


def show_scale_for_image(image_path: Path, scale: ColorScale, input_scale_colors: list, labeled_max_value: float, scale_region_dict: Optional[dict] = None):
    """
    Show the color scale extracted from an image.
    
    Args:
        image_path: Path to the preprocessed image
        scale: ColorScale instance with reference scale
        input_scale_colors: Colors extracted from the input image scale
        labeled_max_value: Maximum value from the input image scale (from OCR, e.g., 60)
    """
    print("\n" + "="*80)
    print(f"Image: {image_path.name}")
    print("="*80)
    print(f"Labeled max value (from OCR): {labeled_max_value}")
    print(f"Number of colors in input scale: {len(input_scale_colors)}")
    
    # Initialize variables
    max_label_position = 0.9  # Default estimate
    actual_max_value = labeled_max_value  # Default to labeled max
    max_value = actual_max_value
    
    if len(input_scale_colors) > 0:
        print(f"Top color: {input_scale_colors[0]}")
        print(f"Bottom color: {input_scale_colors[-1]}")
        
        # Find the brightest yellow color
        brightest_idx, brightest_color, brightest_score = find_brightest_yellow(input_scale_colors)
        brightest_position_ratio = brightest_idx / (len(input_scale_colors) - 1) if len(input_scale_colors) > 1 else 0.0
        
        print(f"\nBrightest yellow found at:")
        print(f"  Position index: {brightest_idx} (ratio: {brightest_position_ratio:.4f})")
        print(f"  Color: {brightest_color}")
        print(f"  Brightness score: {brightest_score:.1f}")
        
        # Try to find all number labels on the scale
        all_labels = []
        detected_max_label_position = None
        
        if scale_region_dict:
            try:
                # Load the image to find all label positions
                img = Image.open(image_path)
                img_array = np.array(img)
                
                # Find all labels
                all_labels = find_all_label_positions(img_array, scale_region_dict)
                
                if all_labels:
                    print(f"\nDetected {len(all_labels)} number labels on scale:")
                    for val, pos in all_labels:
                        print(f"  Value {val:.1f} at position {pos:.4f}")
                    
                    # Find the max label and its position
                    # Use the label closest to labeled_max_value (from OCR) for better accuracy
                    max_label_candidates = [label for label in all_labels if abs(label[0] - labeled_max_value) < 5.0]
                    if max_label_candidates:
                        # Use the detected label closest to the OCR max value
                        max_label = max(max_label_candidates, key=lambda x: x[0])
                        detected_max_label_position = max_label[1]
                    else:
                        # Fallback to highest detected label
                        max_label = max(all_labels, key=lambda x: x[0])
                        detected_max_label_position = max_label[1]
                    
                    max_label_value = max_label[0]
                    
                    # Use labeled_max_value from OCR if it's close, otherwise use detected value
                    if abs(max_label_value - labeled_max_value) < 5.0:
                        max_label_value = labeled_max_value
            except Exception as e:
                print(f"  Error detecting labels: {e}")
                import traceback
                traceback.print_exc()
        
        # Determine max_label_position: use detected if available, otherwise estimate
        if detected_max_label_position is not None:
            max_label_position = detected_max_label_position
            print(f"\nUsing detected max label position: {max_label_position:.4f}")
        else:
            # If no labels detected, estimate based on typical scale layout
            # On most scales, the max label is near the bottom (85-95% down from top)
            # But the actual max value (brightest yellow) is at the very top (0%)
            # So we assume max label is at 0.9 (90% down), which gives:
            # actual_max = labeled_max / 0.9 ≈ labeled_max * 1.11
            # This is a reasonable estimate
            max_label_position = 0.9
            print(f"\nNo labels detected, using estimated position: {max_label_position:.4f}")
            print(f"  (Brightest yellow at {brightest_position_ratio:.4f})")
            print(f"  WARNING: This is an estimate - label detection may have failed")
        
        # Calculate actual max at top (position 0)
        # The max label (labeled_max_value) is at max_label_position
        # So actual_max = labeled_max_value / max_label_position
        if max_label_position > 0:
            actual_max_value = labeled_max_value / max_label_position
        else:
            actual_max_value = labeled_max_value
        
        print(f"\nCalibration:")
        print(f"  Max label ({labeled_max_value:.1f}) at position: {max_label_position:.4f}")
        print(f"  Calculated actual max (at top): {actual_max_value:.2f}")
        
        # Use the actual max value for display
        max_value = actual_max_value
    
    # Show colors in increments of 5, plus the labeled max and actual max values
    increment = 5
    values = []
    current = 0.0
    while current <= max_value:
        values.append(current)
        current += increment
    
    # Add the labeled max value (from OCR) if it's not already in the list
    if labeled_max_value not in values and labeled_max_value <= max_value:
        values.append(labeled_max_value)
    
    # Add the actual max value if it's not already in the list
    if max_value not in values:
        values.append(max_value)
    
    values.sort()  # Keep sorted
    
    print(f"\n{'Value':<10} {'RGB':<25} {'Hex':<10}")
    print("-" * 80)
    
    colors_list = []
    colors_rgb_normalized = []
    
    # Store the labeled max and actual max for proper mapping
    labeled_max = labeled_max_value
    actual_max = max_value
    
    for value in values:
        # Use single-point calibration: map value to position on the input scale
        # The input scale has:
        # - Top (position 0) = actual_max (brightest yellow)
        # - Position max_label_position = labeled_max (e.g., 60)
        # - Bottom (position 1) = 0 (dark blue)
        
        # Calculate position on input scale
        # Simple linear mapping: map value from 0 to actual_max across position 1.0 to 0.0
        # Top (position 0.0) = actual_max, Bottom (position 1.0) = 0
        if value >= actual_max:
            position_ratio = 0.0  # Top = actual_max
        elif value <= 0:
            position_ratio = 1.0  # Bottom = 0
        else:
            # Linear mapping: value 0 -> position 1.0, value actual_max -> position 0.0
            # position_ratio = 1.0 - (value / actual_max)
            position_ratio = 1.0 - (value / actual_max)
        
        # Clamp position_ratio
        position_ratio = max(0.0, min(1.0, position_ratio))
        
        # Get color from input scale at this position
        # Input scale colors are ordered from top (index 0) to bottom (index -1)
        num_input_colors = len(input_scale_colors)
        position_idx = position_ratio * (num_input_colors - 1)
        idx1 = int(np.clip(position_idx, 0, num_input_colors - 1))
        idx2 = min(idx1 + 1, num_input_colors - 1)
        frac = position_idx - idx1
        frac = max(0.0, min(1.0, frac))
        
        # Interpolate between adjacent colors in input scale
        color1 = np.array(input_scale_colors[idx1], dtype=np.float64)
        color2 = np.array(input_scale_colors[idx2], dtype=np.float64)
        interpolated_color = color1 + (color2 - color1) * frac
        interpolated_color = np.clip(interpolated_color, 0, 255)
        color = tuple(interpolated_color.astype(np.uint8))
        
        hex_color = rgb_to_hex(color)
        
        colors_list.append((value, color, hex_color))
        colors_rgb_normalized.append(tuple(c / 255.0 for c in color))
        
        print(f"{value:<10.1f} {str(color):<25} {hex_color:<10}")
    
    # Create and save matplotlib visualization
    if plt is None:
        print("\nWarning: Matplotlib not available, skipping visualization")
        return
    
    print(f"\nCreating visualization for {len(colors_list)} colors...")
    
    try:
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create a grid of color swatches
        num_colors = len(colors_list)
        cols = 4  # Number of columns
        rows = (num_colors + cols - 1) // cols  # Calculate rows needed
        
        # Create an array to hold the color grid
        grid_height = rows
        grid_width = cols
        color_grid = np.ones((grid_height, grid_width, 3))
        text_grid = []
        
        for idx, (value, color, hex_color) in enumerate(colors_list):
            row = idx // cols
            col = idx % cols
            
            if row < grid_height and col < grid_width:
                # Set color in grid (normalized to 0-1)
                color_grid[row, col] = colors_rgb_normalized[idx]
                text_grid.append((row, col, f"{value:.1f}\n{hex_color}", sum(color) < 400))
        
        # Display the grid using imshow
        ax.imshow(color_grid, aspect='auto', interpolation='nearest', origin='upper')
        
        # Add text labels
        for row, col, text, use_white in text_grid:
            ax.text(
                col, row, text,
                ha='center', va='center',
                fontsize=8,
                fontweight='bold',
                color='white' if use_white else 'black',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.3) if use_white else None
            )
        
        # Remove ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'Color Scale for {image_path.name}\nMax Value: {max_value}', 
                     fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save to data folder
        output_dir = project_root / "data" / "color_scale_test"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"scale_{image_path.stem}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved visualization to: {output_file}")
        
        plt.close(fig)
    except Exception as e:
        print(f"  ✗ Error: Could not create visualization: {e}")
        import traceback
        traceback.print_exc()


def load_config(config_path: str = None) -> dict:
    """Load YAML configuration file."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    
    config_file = Path(config_path)
    
    if not config_file.exists():
        example_path = Path(__file__).parent / "config.yaml.example"
        raise FileNotFoundError(
            f"Configuration file '{config_path}' not found.\n"
            f"Please create '{config_path}' from '{example_path}'."
        )
    
    try:
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML: {e}") from e
    
    if not config_data:
        raise ValueError(f"Configuration file '{config_path}' is empty.")
    
    return config_data


def main():
    """Main entry point."""
    try:
        # Load config
        config = load_config()
        num_test_images = config.get('num_test_images', 5)
        
        print("="*80)
        print("Color Scale Testing")
        print("="*80)
        print(f"\nTesting scales from first {num_test_images} preprocessed images")
        
        # Initialize reference color scale
        try:
            reference_scale = ColorScale()
            print("✓ Loaded reference color scale from data/scale/scale.png")
        except Exception as e:
            print(f"✗ Error loading reference scale: {e}")
            sys.exit(1)
        
        # Find preprocessed images
        preprocessed_path = project_root / "data" / "preprocessed"
        if not preprocessed_path.exists():
            print(f"✗ Preprocessed folder not found: {preprocessed_path}")
            sys.exit(1)
        
        # Get all PNG images, sorted
        image_files = sorted(preprocessed_path.glob("*.png"))
        
        if len(image_files) == 0:
            print(f"✗ No PNG images found in {preprocessed_path}")
            sys.exit(1)
        
        num_to_test = min(num_test_images, len(image_files))
        print(f"Found {len(image_files)} images, testing first {num_to_test}\n")
        
        # Process each image
        for i, image_path in enumerate(image_files[:num_to_test]):
            print(f"\n{'='*80}")
            print(f"Processing image {i+1} of {num_to_test}")
            print(f"{'='*80}")
            
            # Use ImageTransformer to extract scale
            transform_config = TransformConfig(
                from_date="2025-01-01",  # Dummy date, not used
                to_date="2025-01-01",
                scale_debug=False,
                debug_dataframe=False,
                frequency_start=3.1,
                frequency_end=3.5,
                preprocessed_folder=str(preprocessed_path),
                output_folder=str(project_root / "data" / "transformed" / "test")
            )
            
            transformer = ImageTransformer(transform_config)
            success = transformer.process_image(image_path)
            
            if success and transformer.input_scale_colors and transformer.input_max_value:
                # Get scale region dict if available
                scale_region_dict = None
                if hasattr(transformer, '_last_scale_region'):
                    scale_region_dict = transformer._last_scale_region
                
                # Get the labeled max value (before correction)
                # We need to get it from the identify_scale call
                labeled_max = transformer.input_max_value  # This is the OCR value
                
                show_scale_for_image(
                    image_path,
                    reference_scale,
                    transformer.input_scale_colors,
                    labeled_max,  # Pass the labeled max (from OCR)
                    scale_region_dict
                )
            else:
                print(f"✗ Failed to extract scale from {image_path.name}")
        
        print(f"\n{'='*80}")
        print("Testing Complete!")
        print(f"{'='*80}")
        
    except FileNotFoundError as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"✗ Unexpected Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
