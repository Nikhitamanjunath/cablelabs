#!/usr/bin/env python3
"""
ColorScale Class

Manages color-to-number mapping based on a reference color scale.
Loads colors from data/scale/scale.png by reading every pixel vertically.
Top of image = higher numbers, bottom = 0.

The scale goes from very dark blue (0) → light blue → pink → red → orange → yellow (max).
"""

from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import cv2
from PIL import Image


class ColorScale:
    """
    Manages color-to-number mapping based on a reference color scale.
    
    Loads colors from data/scale/scale.png by reading every pixel vertically.
    Top pixel = max value, bottom pixel = 0.
    """
    
    def __init__(self, reference_scale_path: Optional[str] = None):
        """
        Initialize the color scale with reference scale from data/scale/scale.png.
        
        Args:
            reference_scale_path: Path to reference scale image (default: data/scale/scale.png)
        """
        if reference_scale_path is None:
            # Default to data/scale/scale.png relative to project root
            project_root = Path(__file__).parent.parent
            reference_scale_path = str(project_root / "data" / "scale" / "scale.png")
        
        self.reference_scale_path = Path(reference_scale_path)
        self.reference_colors = self._load_reference_scale()
        self._lab_colors = None  # LAB color space for better perceptual matching
        self._build_lab_colors()
    
    def _load_reference_scale(self) -> list:
        """
        Load colors from reference scale image by reading every pixel vertically.
        Top pixel = max value position, bottom pixel = 0.
        
        Returns:
            List of RGB tuples ordered from top (max) to bottom (0)
        """
        if not self.reference_scale_path.exists():
            raise FileNotFoundError(
                f"Reference scale image not found: {self.reference_scale_path}\n"
                f"Please ensure data/scale/scale.png exists."
            )
        
        # Load image
        img = Image.open(self.reference_scale_path)
        img_array = np.array(img)
        
        # Get image dimensions
        height, width = img_array.shape[:2]
        
        # Read every pixel vertically from top to bottom
        # Sample from the middle column (or average across width if narrow)
        colors = []
        
        if width == 1:
            # Single column - read directly
            for y in range(height):
                pixel = img_array[y, 0]
                colors.append(tuple(pixel[:3]))  # RGB only
        else:
            # Multiple columns - average across middle portion
            # Use middle 50% of width to avoid edges
            start_x = width // 4
            end_x = 3 * width // 4
            
            for y in range(height):
                # Average colors across the middle portion
                pixel_row = img_array[y, start_x:end_x]
                avg_color = np.mean(pixel_row, axis=0).astype(int)
                colors.append(tuple(avg_color[:3]))  # RGB only
        
        print(f"  Loaded {len(colors)} colors from reference scale: {self.reference_scale_path}")
        print(f"  Scale dimensions: {width}x{height} pixels")
        print(f"  Top color (max): {colors[0]}, Bottom color (0): {colors[-1]}")
        
        # Check if top color is black/dark (might be border/label, not part of scale)
        top_brightness = sum(colors[0]) / 3.0
        if top_brightness < 50:
            # Find the first bright yellow color (high R, high G, low B)
            for i, color in enumerate(colors):
                brightness = sum(color) / 3.0
                if color[0] > 200 and color[1] > 200 and color[2] < 150 and brightness > 200:
                    if i > 0:
                        bright_color = colors[i]
                        colors = [bright_color] * (i + 1) + colors[i + 1:]
                        print(f"  Adjusted: Found bright color at pixel {i}, using it for top (max value)")
                    break
        
        print(f"  Final: Top color (max): {colors[0]}, Bottom color (0): {colors[-1]}")
        
        return colors
    
    def _rgb_to_lab(self, rgb_color: Tuple[int, int, int]) -> np.ndarray:
        """Convert RGB tuple to LAB color space."""
        rgb = np.array([[rgb_color]], dtype=np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[0, 0]
    
    def _build_lab_colors(self):
        """Convert RGB colors to LAB color space for perceptual matching."""
        if not self.reference_colors:
            self._lab_colors = None
            return
        
        self._lab_colors = [self._rgb_to_lab(color) for color in self.reference_colors]
    
    def color_to_number(
        self, 
        color: Tuple[int, int, int], 
        input_scale_colors: Optional[list] = None,
        input_max_value: Optional[float] = None
    ) -> Optional[float]:
        """
        Convert a color (RGB) to a number based on the scale.
        
        If input_scale_colors and input_max_value are provided, matches the color
        directly to the input scale colors and maps to values.
        
        Args:
            color: RGB tuple (R, G, B)
            input_scale_colors: List of RGB colors from input image scale (top to bottom, optional)
            input_max_value: Max value from input image scale (optional)
            
        Returns:
            Numeric value corresponding to the color, or None if no match found
        """
        if self.reference_colors is None or len(self.reference_colors) == 0:
            return None
        
        # Determine max value to use
        max_value = input_max_value if input_max_value is not None else 60.0
        
        # If input scale is provided, match directly to input scale colors
        if input_scale_colors and input_max_value is not None:
            color_lab = self._rgb_to_lab(color)
            input_scale_lab = [self._rgb_to_lab(scale_color) for scale_color in input_scale_colors]
            
            distances = np.array([np.linalg.norm(color_lab - lab_color) for lab_color in input_scale_lab])
            best_idx = np.argmin(distances)
            
            num_colors = len(input_scale_colors)
            position_ratio = best_idx / (num_colors - 1) if num_colors > 1 else 0.0
            return max(0.0, min(max_value, (1.0 - position_ratio) * max_value))
        
        # No input scale provided, use reference scale directly
        if not self._lab_colors:
            return None
        
        color_lab = self._rgb_to_lab(color)
        distances = np.array([np.linalg.norm(color_lab - lab_color) for lab_color in self._lab_colors])
        best_idx = np.argmin(distances)
        
        if distances[best_idx] > 50.0:
            return None
        
        position_ratio = best_idx / (len(self.reference_colors) - 1) if len(self.reference_colors) > 1 else 0.0
        return (1.0 - position_ratio) * max_value
    
    def number_to_color(self, value: float, max_value: float = 60.0) -> Tuple[int, int, int]:
        """
        Convert a number to a color based on the reference scale.
        
        Args:
            value: Numeric value (should be between 0 and max_value)
            max_value: Maximum value on the scale (default: 60.0)
            
        Returns:
            RGB tuple corresponding to the value
        """
        if self.reference_colors is None or len(self.reference_colors) == 0:
            raise RuntimeError("Reference scale colors not available.")
        
        # Clamp value to valid range
        value = max(0.0, min(max_value, float(value)))
        
        # Convert value to position ratio (0.0 to 1.0) on the scale
        # Top (position 0) = max value, bottom (position 1) = 0
        if max_value <= 0:
            # Invalid max_value, return minimum color (bottom)
            return tuple(self.reference_colors[-1])
        
        value_ratio = value / max_value if max_value > 0 else 0.0
        
        # Map value to position: value max -> position 0 (top), value 0 -> position 1 (bottom)
        position_ratio = 1.0 - value_ratio
        
        # Clamp position_ratio to [0, 1]
        position_ratio = max(0.0, min(1.0, position_ratio))
        
        # Special handling: if max value and top color is dark, find brightest yellow
        if value >= max_value * 0.99 and sum(self.reference_colors[0]) / 3.0 < 100:
            brightest_idx = max(
                range(len(self.reference_colors) // 2),
                key=lambda i: sum(self.reference_colors[i][:2]) - self.reference_colors[i][2] + sum(self.reference_colors[i]) / 3.0
            )
            position_ratio = brightest_idx / (len(self.reference_colors) - 1) if len(self.reference_colors) > 1 else 0.0
        
        # Map position to index in reference colors array
        num_colors = len(self.reference_colors)
        position_idx = position_ratio * (num_colors - 1)
        
        # Clamp indices to valid range
        idx1 = int(np.clip(position_idx, 0, num_colors - 1))
        idx2 = min(idx1 + 1, num_colors - 1)
        
        # Interpolate between two adjacent colors
        frac = position_idx - idx1
        frac = max(0.0, min(1.0, frac))
        
        color1 = np.array(self.reference_colors[idx1], dtype=np.float64)
        color2 = np.array(self.reference_colors[idx2], dtype=np.float64)
        interpolated_color = color1 + (color2 - color1) * frac
        
        # Clamp RGB values to valid range [0, 255]
        interpolated_color = np.clip(interpolated_color, 0, 255)
        
        return tuple(interpolated_color.astype(np.uint8))
    
    def get_scale_colors(self) -> list:
        """Get all colors in the reference scale for visualization."""
        return self.reference_colors if self.reference_colors else []
    
    def calibrate_from_input_scale(
        self,
        input_scale_colors: list,
        input_max_value: float,
        input_max_position: Optional[int] = None
    ) -> dict:
        """
        Calibrate the reference scale to match an input scale.
        
        Finds the color in the input scale that represents the max value,
        then finds the closest match in the reference scale to establish calibration.
        
        Args:
            input_scale_colors: List of RGB colors from input image scale (top to bottom)
            input_max_value: Max value from input image scale
            input_max_position: Position index of max value color in input scale (0 = top)
            
        Returns:
            Dictionary with calibration info
        """
        if not input_scale_colors or len(input_scale_colors) == 0:
            return {"calibrated": False}
        
        # If max position not provided, assume it's at the top (position 0)
        if input_max_position is None:
            input_max_position = 0
        
        # Get the color at the max position in input scale
        if input_max_position >= len(input_scale_colors):
            input_max_position = 0
        
        max_color = input_scale_colors[input_max_position]
        
        max_color_lab = self._rgb_to_lab(max_color)
        distances = np.array([np.linalg.norm(max_color_lab - lab_color) for lab_color in self._lab_colors])
        ref_max_position = np.argmin(distances)
        
        return {
            "calibrated": True,
            "input_max_value": input_max_value,
            "input_max_position": input_max_position,
            "reference_max_position": ref_max_position,
            "max_color": max_color
        }
