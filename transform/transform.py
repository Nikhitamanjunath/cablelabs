#!/usr/bin/env python3
"""
Image Transform Script

This script processes images from a preprocessed folder and extracts
numerical values from colors using a color scale on the right-hand side.

The scale goes from very dark blue → light blue → pink → red → orange → yellow (gradually),
with min=0 and max determined from the scale on the image.
"""

import sys
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from datetime import datetime, timedelta
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import pytesseract
import yaml
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving files
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pydantic import BaseModel, Field, ValidationError

# Add project root to Python path to find color_scale module
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import ColorScale from color_scale module
from color_scale import ColorScale

# Constants
SCALE_RIGHT_START_RATIO = 0.85
SCALE_WIDTH = 40
SCALE_MARGIN_RATIO = 0.05
GRAPH_LEFT = 175
GRAPH_TOP = 100
GRAPH_WIDTH = 733
GRAPH_HEIGHT = 564
FREQ_START = 3.1
FREQ_STEP = 0.005
NUM_ROWS = 24
NUM_COLS = 70
SCALE_SAMPLE_COUNT = 500
DEFAULT_MAX_VALUE = 100.0
MAX_REASONABLE_VALUE = 1000.0


class TransformConfig(BaseModel):
    """Transform configuration."""
    from_date: str = Field(
        ...,
        description="Start date for processing images (YYYY-MM-DD format)"
    )
    to_date: str = Field(
        ...,
        description="End date for processing images (YYYY-MM-DD format)"
    )
    scale_debug: bool = Field(
        default=False,
        description="Enable debug mode to visualize scale detection"
    )
    debug_dataframe: bool = Field(
        default=False,
        description="Enable debug mode to visualize dataframe extraction with matplotlib"
    )
    frequency_start: float = Field(
        default=3.1,
        description="Start frequency in GHz for console display (only affects which columns are printed)"
    )
    frequency_end: float = Field(
        default=3.5,
        description="End frequency in GHz for console display (only affects which columns are printed)"
    )
    preprocessed_folder: str = Field(
        default="data/preprocessed",
        description="Path to folder containing preprocessed images"
    )
    output_folder: str = Field(
        default="data/transformed",
        description="Path to folder for output images"
    )
    output_data_file: str = Field(
        default="transformed_data.parquet",
        description="Path to output file for combined DataFrame (Parquet format). Will be saved in output_folder."
    )


class ImageTransformer:
    """
    Main class for transforming images using color scale mapping.
    """
    
    def __init__(self, config: TransformConfig):
        """
        Initialize the transformer.
        
        Args:
            config: TransformConfig object
        """
        self.config = config
        self.scale: Optional[ColorScale] = None
        self.input_scale_colors: Optional[list] = None
        self.input_max_value: Optional[float] = None
        self.preprocessed_path = Path(config.preprocessed_folder)
        self.output_path = Path(config.output_folder)
        self.preprocessed_path.mkdir(exist_ok=True)
        self.output_path.mkdir(exist_ok=True)
    
    def _filter_columns_by_frequency(self, columns: List[str], freq_start: float, freq_end: float) -> List[str]:
        """Filter DataFrame columns by frequency range."""
        display_cols = []
        for col_name in columns:
            try:
                freq_range = col_name.split('-')
                if len(freq_range) == 2:
                    freq_band_start, freq_band_end = float(freq_range[0]), float(freq_range[1])
                    if freq_band_end >= freq_start and freq_band_start <= freq_end:
                        display_cols.append(col_name)
            except (ValueError, IndexError):
                continue
        return display_cols
    
    def _display_dataframe_preview(self, df: pd.DataFrame, display_cols: List[str], freq_start: float, freq_end: float):
        """Display a preview of the DataFrame with optional frequency filtering."""
        if display_cols and len(df) > 0:
            existing_cols = [col for col in display_cols if col in df.columns]
            if existing_cols:
                df_display = df[existing_cols]
                print(f"\n  Displaying frequency range: {freq_start} - {freq_end} GHz")
                print(f"  ({len(existing_cols)} columns out of {NUM_COLS} total)")
                print(f"\n  DataFrame preview (first 5 rows, filtered columns):")
                print(df_display.iloc[:5].to_string())
            else:
                print(f"\n  No matching columns found in frequency range {freq_start} - {freq_end} GHz")
                print(f"  Showing first 10 columns instead:")
                if len(df.columns) > 0:
                    print(df.iloc[:5, :min(10, len(df.columns))].to_string())
                else:
                    print("  (DataFrame is empty)")
        elif len(df) > 0:
            print(f"\n  No columns found in frequency range {freq_start} - {freq_end} GHz")
            print(f"  Showing first 10 columns instead:")
            print(df.iloc[:5, :min(10, len(df.columns))].to_string())
        else:
            print(f"\n  DataFrame is empty - no data extracted")
    
    def _get_ocr_preprocessing_steps(self):
        """Return list of preprocessing functions for OCR."""
        return [
            lambda x: cv2.convertScaleAbs(x, alpha=2.0, beta=50),
            lambda x: cv2.threshold(x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            lambda x: cv2.adaptiveThreshold(x, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
            lambda x: cv2.convertScaleAbs(x, alpha=1.3, beta=20),
            lambda x: cv2.threshold(x, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
        ]
    
    def _get_ocr_configs(self):
        """Return list of OCR page segmentation mode configs."""
        return ['--psm 6', '--psm 7', '--psm 8', '--psm 11', '--psm 4']
    
    def _extract_numbers_from_region(self, region: np.ndarray, region_name: str) -> List[Dict]:
        """Extract numbers from an image region using OCR with multiple strategies."""
        import re
        results = []
        region_gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        
        for preprocess in self._get_ocr_preprocessing_steps():
            try:
                processed = preprocess(region_gray)
                for ocr_config in self._get_ocr_configs():
                    try:
                        ocr_text = pytesseract.image_to_string(processed, config=ocr_config)
                        numbers = re.findall(r'\d+\.?\d*', ocr_text)
                        if numbers:
                            float_numbers = [float(n) for n in numbers]
                            valid_numbers = [n for n in float_numbers if 0 <= n <= MAX_REASONABLE_VALUE]
                            if valid_numbers:
                                results.append({
                                    'region': region_name,
                                    'config': ocr_config,
                                    'numbers': valid_numbers,
                                    'text': ocr_text.strip(),
                                    'processed_image': processed.copy() if self.config.scale_debug else None
                                })
                    except Exception:
                        continue
            except Exception:
                continue
        return results
    
    def _prepare_ocr_regions(self, img_array: np.ndarray, scale_region_dict: Dict[str, int], scale_region: np.ndarray) -> List[Tuple]:
        """Prepare image regions for OCR analysis."""
        height, width = img_array.shape[:2]
        scale_height = scale_region_dict["bottom_y"] - scale_region_dict["top_y"]
        
        right_expansion = 50
        expanded_right_x = min(width, scale_region_dict["right_x"] + right_expansion)
        wider_right_x = min(width, scale_region_dict["right_x"] + 80)
        
        regions = [
            (img_array[scale_region_dict["top_y"]:scale_region_dict["bottom_y"], 
             scale_region_dict["left_x"]:wider_right_x], "full scale with labels"),
            (img_array[int(scale_region_dict["top_y"] + scale_height * 0.9):scale_region_dict["bottom_y"],
             scale_region_dict["left_x"]:expanded_right_x], "expanded bottom region"),
            (scale_region[int(scale_height * 0.9):, :], "very bottom of scale"),
            (scale_region[int(scale_height * 0.8):, :], "bottom of scale"),
            (img_array[scale_region_dict["top_y"]:scale_region_dict["bottom_y"],
             scale_region_dict["left_x"]:expanded_right_x], "expanded region"),
            (scale_region, "full scale region")
        ]
        return regions
    
    def _detect_scale_region(self, img_array: np.ndarray) -> Dict[str, int]:
        """
        Automatically detect the scale region on the right side of the image.
        
        Args:
            img_array: Image as numpy array
            
        Returns:
            Dictionary with scale region coordinates
        """
        height, width = img_array.shape[:2]
        
        # Start by looking at the rightmost portion of the image
        # Typically the scale is in the rightmost 10-15% of the image
        right_start = int(width * SCALE_RIGHT_START_RATIO)
        right_region = img_array[:, right_start:]
        
        gray_region = cv2.cvtColor(right_region, cv2.COLOR_RGB2GRAY)
        sobel_y = cv2.Sobel(gray_region, cv2.CV_64F, 0, 1, ksize=3)
        column_gradients = np.mean(np.abs(sobel_y), axis=0)
        scale_col_idx = np.argmax(column_gradients)
        
        actual_scale_x = right_start + scale_col_idx
        top_margin = int(height * SCALE_MARGIN_RATIO)
        bottom_margin = int(height * SCALE_MARGIN_RATIO)
        
        return {
            "left_x": max(0, actual_scale_x - SCALE_WIDTH // 2),
            "right_x": min(width, actual_scale_x + SCALE_WIDTH // 2),
            "top_y": top_margin,
            "bottom_y": height - bottom_margin
        }
    
    def identify_scale(self, image_path: Path) -> Tuple[float, Dict[str, int]]:
        """
        Identify the scale from an image and extract the maximum value.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (max_value, scale_region_dict)
        """
        # Load image
        img = Image.open(image_path)
        img_array = np.array(img)
        height, width = img_array.shape[:2]
        
        # Automatically detect scale region
        scale_region_dict = self._detect_scale_region(img_array)
        
        # Extract scale region
        scale_region = img_array[
            scale_region_dict["top_y"]:scale_region_dict["bottom_y"],
            scale_region_dict["left_x"]:scale_region_dict["right_x"]
        ]
        
        self._last_scale_region = scale_region_dict
        self._last_scale_image = img_array.copy()
        
        all_found_numbers = []
        ocr_results = []
        
        ocr_regions = self._prepare_ocr_regions(img_array, scale_region_dict, scale_region)
        for region, region_name in ocr_regions:
            results = self._extract_numbers_from_region(region, region_name)
            ocr_results.extend(results)
            for result in results:
                all_found_numbers.extend(result['numbers'])
        
        # Now find the maximum from all collected numbers
        max_value = None
        if all_found_numbers:
            unique_numbers = sorted(set(all_found_numbers))
            reasonable_numbers = [n for n in unique_numbers if 0 <= n <= MAX_REASONABLE_VALUE]
            
            if reasonable_numbers:
                max_value = max(reasonable_numbers)
                print(f"  OCR found numbers: {unique_numbers}")
                print(f"  Selected max value: {max_value}")
                
                # Print details from the best result (one that found the max)
                for result in ocr_results:
                    if max_value in result['numbers']:
                        print(f"  Best OCR result from ({result['region']}, {result['config']}):")
                        print(f"    Numbers found: {result['numbers']}")
                        print(f"    OCR text: {result['text'][:100]}")
                        
                        # Save OCR debug image if debug mode is enabled
                        if self.config.scale_debug and result['processed_image'] is not None:
                            self._save_ocr_debug_image(
                                result['processed_image'], 
                                result['region'], 
                                result['text'], 
                                result['numbers'], 
                                max_value
                            )
                        break
        
        if max_value is None:
            print(f"  OCR failed to extract max value, trying alternative method...")
        
        if max_value is None:
            print(f"  Warning: Could not extract max value from scale, using default {DEFAULT_MAX_VALUE}")
            max_value = DEFAULT_MAX_VALUE
        
        return max_value, scale_region_dict
    
    def _extract_scale_colors_by_position(self, img_array: np.ndarray, scale_region_dict: Dict[str, int]) -> list:
        """
        Extract colors from the scale bar, ordered by position (top to bottom).
        Finds the actual color bar column by looking for strong vertical gradients.
        
        Args:
            img_array: Full image array
            scale_region_dict: Scale region coordinates
            
        Returns:
            List of RGB tuples sampled from the scale bar, ordered from top to bottom
        """
        scale_region = img_array[
            scale_region_dict["top_y"]:scale_region_dict["bottom_y"],
            scale_region_dict["left_x"]:scale_region_dict["right_x"]
        ]
        
        scale_height = scale_region_dict["bottom_y"] - scale_region_dict["top_y"]
        scale_width = scale_region_dict["right_x"] - scale_region_dict["left_x"]
        
        # Find the actual color bar column by looking for strong vertical gradients
        # The color bar should have the strongest vertical color variation
        gray_region = cv2.cvtColor(scale_region, cv2.COLOR_RGB2GRAY)
        sobel_y = cv2.Sobel(gray_region, cv2.CV_64F, 0, 1, ksize=3)
        gradient_strength = np.abs(sobel_y)
        
        # Find column with strongest average vertical gradient
        column_gradients = np.mean(gradient_strength, axis=0)
        color_bar_x = int(np.argmax(column_gradients))
        
        # Also check color variance - the color bar should have high color variance
        # Convert to LAB for better color distance calculation
        lab_region = cv2.cvtColor(scale_region, cv2.COLOR_RGB2LAB)
        color_variances = []
        for x in range(scale_width):
            column_colors = lab_region[:, x, :]
            # Calculate variance in LAB space
            variance = np.var(column_colors, axis=0).sum()
            color_variances.append(variance)
        
        color_bar_x_by_variance = int(np.argmax(color_variances))
        
        # Use the column that has both strong gradient and high color variance
        # Prefer the one closer to the left (color bar is usually on the left side of scale region)
        if abs(color_bar_x - color_bar_x_by_variance) < 10:
            # Both methods agree, use the average
            color_bar_x = (color_bar_x + color_bar_x_by_variance) // 2
        else:
            # Use the one with higher combined score
            gradient_score = column_gradients[color_bar_x]
            variance_score = color_variances[color_bar_x_by_variance]
            if variance_score > gradient_score * 0.5:
                color_bar_x = color_bar_x_by_variance
        
        print(f"  Found color bar at column {color_bar_x} of {scale_width} (gradient method: {int(np.argmax(column_gradients))}, variance method: {color_bar_x_by_variance})")
        
        scale_colors = []
        
        for i in range(SCALE_SAMPLE_COUNT):
            y_ratio = i / (SCALE_SAMPLE_COUNT - 1) if SCALE_SAMPLE_COUNT > 1 else 0.0
            scale_y = int(y_ratio * scale_height)
            
            if 0 <= scale_y < scale_height:
                # Average a few pixels around the color bar column for better color accuracy
                color_samples = []
                for offset in range(-1, 2):  # Smaller range to stay on color bar
                    x_pos = color_bar_x + offset
                    if 0 <= x_pos < scale_width:
                        color_samples.append(scale_region[scale_y, x_pos])
                
                if color_samples:
                    avg_color = np.mean(color_samples, axis=0).astype(int)
                    color_tuple = tuple(avg_color)
                    
                    # Filter out white or very light colors from the scale
                    # These shouldn't be in the actual color scale (likely from labels/borders)
                    brightness = (color_tuple[0] + color_tuple[1] + color_tuple[2]) / 3.0
                    is_white = (color_tuple[0] > 250 and color_tuple[1] > 250 and color_tuple[2] > 250)
                    
                    # Skip white colors - they're likely from labels or borders, not the scale
                    if not is_white and brightness < 250:
                        scale_colors.append(color_tuple)
                    elif len(scale_colors) > 0:
                        # If we hit white, use the last valid color to maintain continuity
                        scale_colors.append(scale_colors[-1])
                    # If first color is white, we'll skip it (will be handled in validation)
        
        # Validate that we got actual color scale colors (not all white/background)
        if len(scale_colors) > 0:
            # Check color range - should have variation
            first_color = np.array(scale_colors[0])
            last_color = np.array(scale_colors[-1])
            color_diff = np.abs(first_color - last_color).sum()
            
            if color_diff < 50:  # Very little color variation
                print(f"  Warning: Extracted colors have very little variation (diff={color_diff:.1f}), scale region might be wrong")
            else:
                print(f"  Color variation check: first={scale_colors[0]}, last={scale_colors[-1]}, diff={color_diff:.1f}")
            
            # Ensure the last color (minimum for standard scale) is dark enough
            # Find the darkest color in the scale and ensure it's at the end
            last_color = np.array(scale_colors[-1])
            last_brightness = (last_color[0] + last_color[1] + last_color[2]) / 3.0
            
            darkest_color = last_color
            darkest_brightness = last_brightness
            darkest_idx = len(scale_colors) - 1
            
            for idx, scale_color in enumerate(scale_colors):
                brightness = (scale_color[0] + scale_color[1] + scale_color[2]) / 3.0
                if brightness < darkest_brightness:
                    darkest_brightness = brightness
                    darkest_color = np.array(scale_color)
                    darkest_idx = idx
            
            # If the last color is too light (brightness > 150), replace last portion with darkest color
            if last_brightness > 150 and darkest_brightness < 150:
                # Replace last 10% of colors with darkest color to ensure value 0 maps correctly
                replace_count = max(1, len(scale_colors) // 10)
                darkest_tuple = tuple(darkest_color.astype(int))
                for i in range(replace_count):
                    scale_colors[-(i+1)] = darkest_tuple
                print(f"  Replaced last {replace_count} colors with darkest color (brightness {darkest_brightness:.1f}) to ensure value 0 maps correctly")
        
        print(f"  Extracted {len(scale_colors)} colors from scale bar (ordered top to bottom)")
        
        return scale_colors
    
    def _save_debug_image(self, image_path: Path, scale_region_dict: Dict[str, int], max_value: float):
        """
        Save a debug image showing the detected scale region.
        
        Args:
            image_path: Original image path
            scale_region_dict: Scale region coordinates
            max_value: Detected max value
        """
        # Create a copy of the image for annotation
        debug_img = Image.fromarray(self._last_scale_image.copy())
        draw = ImageDraw.Draw(debug_img)
        
        # Draw rectangle around detected scale region
        left = scale_region_dict["left_x"]
        right = scale_region_dict["right_x"]
        top = scale_region_dict["top_y"]
        bottom = scale_region_dict["bottom_y"]
        
        # Draw rectangle in red
        draw.rectangle([left, top, right, bottom], outline="red", width=3)
        
        # Add text annotation
        try:
            # Try to use a default font, fallback to basic if not available
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        except:
            font = ImageFont.load_default()
        
        text = f"Scale: 0.0 - {max_value}"
        # Draw text with background
        bbox = draw.textbbox((left, top - 25), text, font=font)
        draw.rectangle(bbox, fill="red", outline="red")
        draw.text((left, top - 25), text, fill="white", font=font)
        
        # Save debug image
        debug_dir = self.output_path / "debug"
        debug_dir.mkdir(exist_ok=True)
        debug_filename = f"debug_{image_path.stem}.png"
        debug_path = debug_dir / debug_filename
        debug_img.save(debug_path)
        print(f"  Debug image saved: {debug_path}")
    
    def _save_ocr_debug_image(self, processed_image: np.ndarray, region_name: str, ocr_text: str, numbers: list, max_value: float):
        """
        Save a debug image showing the OCR processed region.
        
        Args:
            processed_image: Processed image array used for OCR
            region_name: Name of the region being analyzed
            ocr_text: Text extracted by OCR
            numbers: List of numbers found
            max_value: Detected max value
        """
        debug_dir = self.output_path / "debug" / "ocr"
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy array to PIL Image
        if len(processed_image.shape) == 2:
            # Grayscale
            ocr_img = Image.fromarray(processed_image, mode='L')
        else:
            # Color
            ocr_img = Image.fromarray(processed_image)
        
        # Add text annotation
        draw = ImageDraw.Draw(ocr_img)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        except:
            font = ImageFont.load_default()
        
        # Add annotation text
        annotation = f"{region_name}\nOCR: {ocr_text.strip()[:50]}\nNumbers: {numbers}\nMax: {max_value}"
        draw.text((5, 5), annotation, fill="yellow", font=font, stroke_width=1, stroke_fill="black")
        
        # Save
        safe_region_name = region_name.replace(" ", "_").replace("-", "_")
        debug_filename = f"ocr_debug_{safe_region_name}.png"
        debug_path = debug_dir / debug_filename
        ocr_img.save(debug_path)
        print(f"    OCR debug image saved: {debug_path}")
    
    def process_image(self, image_path: Path) -> bool:
        """
        Process a single image: identify scale and set up color mapping.
        
        Extracts scale from image and uses reference scale for calibration.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            True if successful, False otherwise
        """
        if not image_path.exists():
            print(f"Error: Image not found: {image_path}")
            return False
        
        print(f"Processing image: {image_path.name}")
        
        # Initialize color scale with reference scale
        try:
            self.scale = ColorScale()
            print("  ✓ Loaded reference color scale from data/scale/scale.png")
        except Exception as e:
            print(f"  ✗ Error loading reference scale: {e}")
            return False
        
        # Load image for color extraction
        img = Image.open(image_path)
        img_array = np.array(img)
        
        # Identify the scale and get max value
        max_value, scale_region_dict = self.identify_scale(image_path)
        
        # Extract actual color-to-number mapping from the scale
        print("  Extracting color-to-number mapping from scale...")
        # Extract colors from scale bar by position (top to bottom)
        scale_colors = self._extract_scale_colors_by_position(img_array, scale_region_dict)
        
        # Store input scale info for calibration
        self.input_scale_colors = scale_colors
        self.input_max_value = max_value
        
        print(f"  Scale set: 0.0 to {max_value}")
        print(f"  Scale region: ({scale_region_dict['left_x']}, {scale_region_dict['top_y']}) to ({scale_region_dict['right_x']}, {scale_region_dict['bottom_y']})")
        print(f"  Extracted {len(scale_colors)} colors from input image scale")
        
        # Save debug image if enabled
        if self.config.scale_debug:
            self._save_debug_image(image_path, scale_region_dict, max_value)
        
        return True
    
    def get_color_value(self, color: Tuple[int, int, int]) -> Optional[float]:
        """
        Get the numeric value for a given color using the current scale.
        
        Args:
            color: RGB tuple (R, G, B)
            
        Returns:
            Numeric value or None if scale not set or color not found
        """
        if self.scale is None:
            return None
        
        if self.input_scale_colors is None or len(self.input_scale_colors) == 0:
            return None
        
        if self.input_max_value is None:
            return None
        
        return self.scale.color_to_number(
            color,
            input_scale_colors=self.input_scale_colors,
            input_max_value=self.input_max_value
        )
    
    def _detect_graph_region(self, img_array: np.ndarray, scale_region_dict: Dict[str, int]) -> Dict[str, int]:
        """
        Detect the graph region in the center of the image.
        
        Uses fixed coordinates based on the scraper's crop_image method:
        - x="175" y="100" width="733" height="564"
        - Crop box: (175, 100) to (908, 664)
        
        Args:
            img_array: Full image array
            scale_region_dict: Scale region coordinates (to exclude from graph detection)
            
        Returns:
            Dictionary with graph region coordinates
        """
        height, width = img_array.shape[:2]
        
        left = GRAPH_LEFT
        top = GRAPH_TOP
        right = GRAPH_LEFT + GRAPH_WIDTH
        bottom = GRAPH_TOP + GRAPH_HEIGHT
        
        # Validate coordinates are within image bounds
        left = max(0, min(left, width))
        top = max(0, min(top, height))
        right = max(left, min(right, width))
        bottom = max(top, min(bottom, height))
        
        return {
            "left_x": left,
            "right_x": right,
            "top_y": top,
            "bottom_y": bottom
        }
    
    def extract_dataframe(self, image_path: Path) -> pd.DataFrame:
        """
        Extract pixel values from the graph and create a dataframe.
        
        The graph has:
        - 24 rows (hours 0-23, with hour 0 at bottom, hour 23 at top)
        - 70 columns (frequency bands from 3.1-3.105 to 3.4405-3.450)
        
        Args:
            image_path: Path to the image file
            
        Returns:
            DataFrame with columns: hour, frequency_band, frequency_start, frequency_end, value
        """
        if self.scale is None:
            raise RuntimeError("Scale not set. Call process_image() first.")
        
        # Load image
        img = Image.open(image_path)
        img_array = np.array(img)
        
        # Get scale region from the last processed image
        if not hasattr(self, '_last_scale_region') or self._last_scale_region is None:
            # Re-identify scale if not available
            _, scale_region_dict = self.identify_scale(image_path)
        else:
            scale_region_dict = self._last_scale_region
        
        # Detect graph region
        graph_region_dict = self._detect_graph_region(img_array, scale_region_dict)
        
        graph_region = img_array[
            graph_region_dict["top_y"]:graph_region_dict["bottom_y"],
            graph_region_dict["left_x"]:graph_region_dict["right_x"]
        ]
        
        graph_height = graph_region_dict["bottom_y"] - graph_region_dict["top_y"]
        graph_width = graph_region_dict["right_x"] - graph_region_dict["left_x"]
        
        freq_end = FREQ_START + NUM_COLS * FREQ_STEP
        
        cell_height = graph_height / NUM_ROWS
        cell_width = graph_width / NUM_COLS
        
        data = []
        selected_pixels = []  # For debug visualization
        
        print(f"  Extracting data from graph region: ({graph_region_dict['left_x']}, {graph_region_dict['top_y']}) to ({graph_region_dict['right_x']}, {graph_region_dict['bottom_y']})")
        print(f"  Graph size: {graph_width} x {graph_height} pixels")
        print(f"  Cell size: {cell_width:.1f} x {cell_height:.1f} pixels per cell")
        print(f"  Extracting {NUM_ROWS} rows x {NUM_COLS} columns...")
        
        # Verify scale is set up
        if self.scale is None:
            raise RuntimeError("Scale not set. Call process_image() first.")
        scale_colors = self.scale.get_scale_colors()
        if scale_colors is None or len(scale_colors) == 0:
            raise RuntimeError(f"Scale colors are empty. Expected colors but got {len(scale_colors) if scale_colors else 0} colors.")
        print(f"  Scale has {len(scale_colors)} colors available")
        
        for row_idx in range(NUM_ROWS):
            hour = 23 - row_idx
            y_in_graph = int(row_idx * cell_height + cell_height / 2)
            y_in_image = graph_region_dict["top_y"] + y_in_graph
            
            for col_idx in range(NUM_COLS):
                freq_band_start = FREQ_START + col_idx * FREQ_STEP
                freq_band_end = FREQ_START + (col_idx + 1) * FREQ_STEP
                x_in_graph = int(col_idx * cell_width + cell_width / 2)
                x_in_image = graph_region_dict["left_x"] + x_in_graph
                
                # Get pixel color at the center of this cell
                if 0 <= y_in_graph < graph_height and 0 <= x_in_graph < graph_width:
                    pixel_color = tuple(graph_region[y_in_graph, x_in_graph])
                    
                    # Convert color to number
                    value = self.get_color_value(pixel_color)
                    
                    # Debug: print first few conversions to see what's happening
                    if row_idx == 0 and col_idx < 5:
                        print(f"    Debug: hour={hour}, freq_band={col_idx+1}, cell=({x_in_graph},{y_in_graph}), color={pixel_color}, value={value}")
                    
                    data.append({
                        'hour': hour,
                        'frequency_band': col_idx + 1,
                        'frequency_start': freq_band_start,
                        'frequency_end': freq_band_end,
                        'value': value if value is not None else np.nan
                    })
                    
                    # Store for debug visualization
                    if self.config.debug_dataframe:
                        selected_pixels.append({
                            'x': x_in_image,
                            'y': y_in_image,
                            'value': value,
                            'hour': hour,
                            'freq_band': col_idx + 1
                        })
        
        # Create dataframe
        df = pd.DataFrame(data)
        
        print(f"  Created dataframe with {len(df)} rows before pivot")
        if len(df) > 0:
            non_nan_count = df['value'].notna().sum()
            print(f"  Non-NaN values: {non_nan_count} out of {len(df)} total")
        
        all_hours = list(range(23, -1, -1))
        all_freq_bands = list(range(1, NUM_COLS + 1))
        
        freq_band_labels = [
            f"{FREQ_START + col_idx * FREQ_STEP:.3f}-{FREQ_START + (col_idx + 1) * FREQ_STEP:.3f}"
            for col_idx in range(NUM_COLS)
        ]
        
        # Create empty dataframe with full structure (24 rows × 70 columns)
        df_pivot = pd.DataFrame(index=all_hours, columns=all_freq_bands, dtype=float)
        
        # Reshape to 24 rows (hours) × 70 columns (frequency bands)
        # Fill in values from the data
        if len(df) > 0:
            for _, row in df.iterrows():
                hour = int(row['hour'])
                freq_band = int(row['frequency_band'])
                value = row['value']
                if hour in df_pivot.index and freq_band in df_pivot.columns:
                    df_pivot.loc[hour, freq_band] = value
        
        # Don't sort - keep order: hour 23 first, hour 0 last (matching graph layout)
        # df_pivot is already in the correct order (23, 22, ..., 1, 0)
        
        # Rename columns to actual frequency ranges (e.g., "3.1-3.105")
        df_pivot.columns = freq_band_labels
        
        print(f"  Pivoted dataframe shape: {df_pivot.shape}")
        
        # Debug visualization
        if self.config.debug_dataframe:
            print(f"  Debug mode enabled: Creating visualization with {len(selected_pixels)} selected pixels...")
            try:
                self._visualize_dataframe_extraction(
                    img_array, 
                    graph_region_dict, 
                    selected_pixels, 
                    image_path
                )
                print(f"  ✓ Debug visualization created successfully")
            except Exception as e:
                print(f"  ✗ Warning: Debug visualization failed: {e}")
                import traceback
                traceback.print_exc()
        
        return df_pivot
    
    def _visualize_dataframe_extraction(
        self, 
        img_array: np.ndarray, 
        graph_region_dict: Dict[str, int],
        selected_pixels: List[Dict],
        image_path: Path
    ):
        """
        Visualize the dataframe extraction with matplotlib.
        
        Args:
            img_array: Full image array
            graph_region_dict: Graph region coordinates
            selected_pixels: List of selected pixels with their values
            image_path: Original image path
        """
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        
        # Display the image
        ax.imshow(img_array)
        ax.set_title(f"Dataframe Extraction: {image_path.name}", fontsize=14, fontweight='bold')
        
        # Draw rectangle around graph region
        rect_width = graph_region_dict["right_x"] - graph_region_dict["left_x"]
        rect_height = graph_region_dict["bottom_y"] - graph_region_dict["top_y"]
        
        rect = patches.Rectangle(
            (graph_region_dict["left_x"], graph_region_dict["top_y"]),
            rect_width,
            rect_height,
            linewidth=3, 
            edgecolor='cyan', 
            facecolor='none',
            linestyle='--'
        )
        ax.add_patch(rect)
        
        # Add label for graph region
        ax.text(
            graph_region_dict["left_x"], 
            graph_region_dict["top_y"] - 15,
            f"Graph Region: ({graph_region_dict['left_x']}, {graph_region_dict['top_y']}) to ({graph_region_dict['right_x']}, {graph_region_dict['bottom_y']})",
            color='cyan',
            fontsize=12,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7)
        )
        
        # Draw ALL selected pixels (all 1680 points: 24 rows × 70 columns)
        print(f"  Drawing {len(selected_pixels)} sampled points on visualization...")
        valid_count = 0
        nan_count = 0
        
        for i, pixel in enumerate(selected_pixels):
            x, y = pixel['x'], pixel['y']
            value = pixel['value']
            
            # Draw a dot at every sampled point
            if value is not None and not np.isnan(value):
                # Valid values: yellow dot with black border
                circle = plt.Circle((x, y), 2, facecolor='yellow', alpha=0.8, edgecolor='black', linewidth=0.5)
                valid_count += 1
            else:
                # NaN values: gray dot
                circle = plt.Circle((x, y), 2, facecolor='gray', alpha=0.4, edgecolor='black', linewidth=0.3)
                nan_count += 1
            
            ax.add_patch(circle)
            
            # Add text annotation for every third point to show values
            if i % 3 == 0 and value is not None and not np.isnan(value):
                ax.text(
                    x + 4, y,
                    f"{value:.1f}",
                    color='yellow',
                    fontsize=5,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7)
                )
        
        print(f"    Valid values: {valid_count}, NaN values: {nan_count}")
        
        ax.axis('off')
        
        # Save the visualization
        debug_dir = self.output_path / "debug" / "dataframe"
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_filename = f"dataframe_{image_path.stem}.png"
        debug_path = debug_dir / debug_filename
        plt.savefig(debug_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  DataFrame extraction visualization saved: {debug_path}")


def load_config(config_path: str = None) -> TransformConfig:
    """
    Load and validate YAML configuration file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Validated TransformConfig object
    """
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
    
    return TransformConfig(**config_data)


def generate_date_range(from_date: str, to_date: str) -> list[str]:
    """
    Generate a list of dates from from_date to to_date (inclusive).
    
    Args:
        from_date: Start date in YYYY-MM-DD format
        to_date: End date in YYYY-MM-DD format
    
    Returns:
        List of date strings in YYYY-MM-DD format
    """
    start = datetime.strptime(from_date, "%Y-%m-%d")
    end = datetime.strptime(to_date, "%Y-%m-%d")
    
    if start > end:
        raise ValueError(f"from_date ({from_date}) must be <= to_date ({to_date})")
    
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    
    return dates


def main():
    """Main entry point."""
    try:
        print("Loading config...")
        config = load_config()
        print("✓ Configuration loaded")
        
        print("Initializing transformer...")
        transformer = ImageTransformer(config)
        
        # Generate date range
        dates = generate_date_range(config.from_date, config.to_date)
        print(f"Processing dates: {config.from_date} to {config.to_date} ({len(dates)} date(s))")
        
        # Find images matching the date range
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        preprocessed_path = Path(config.preprocessed_folder)
        
        if not preprocessed_path.exists():
            print(f"Error: Preprocessed folder not found: {preprocessed_path}")
            return
        
        # Process each date and collect DataFrames
        processed_count = 0
        all_dataframes = []  # List to store (date, dataframe) tuples
        
        for date in dates:
            # Convert date to filename format (YYYY-MM-DD -> YYYY_MM_DD.png)
            filename = date.replace("-", "_") + ".png"
            image_path = preprocessed_path / filename
            
            if not image_path.exists():
                print(f"\n{'='*60}")
                print(f"Image not found for date {date}: {filename}")
                continue
            
            print(f"\n{'='*60}")
            print(f"Processing date: {date} ({filename})")
            print(f"{'='*60}")
            
            success = transformer.process_image(image_path)
            
            if success:
                processed_count += 1
                print(f"✓ Successfully processed {image_path.name}")
                print(f"  Scale: 0.0 to {transformer.input_max_value if transformer.input_max_value else 'N/A'}")
                
                # Extract dataframe from the graph
                print("\n  Extracting dataframe from graph...")
                try:
                    df = transformer.extract_dataframe(image_path)
                    print(f"✓ DataFrame extracted: {df.shape[0]} rows × {df.shape[1]} columns")
                    print(f"\n  DataFrame shape: {df.shape} (should be 24 rows × 70 columns)")
                    
                    display_cols = transformer._filter_columns_by_frequency(df.columns, config.frequency_start, config.frequency_end)
                    
                    transformer._display_dataframe_preview(df, display_cols, config.frequency_start, config.frequency_end)
                    
                    print(f"\n  Value statistics (all data):")
                    # Flatten values for statistics (use full dataframe)
                    values_flat = df.values.flatten()
                    values_flat = values_flat[~np.isnan(values_flat)]
                    if len(values_flat) > 0:
                        print(f"    Count (non-NaN): {len(values_flat)}")
                        print(f"    Min: {np.min(values_flat):.2f}")
                        print(f"    Max: {np.max(values_flat):.2f}")
                        print(f"    Mean: {np.mean(values_flat):.2f}")
                        print(f"    Median: {np.median(values_flat):.2f}")
                    else:
                        print("    No valid values found")
                    
                    if display_cols:
                        existing_cols = [col for col in display_cols if col in df.columns]
                        if existing_cols:
                            print(f"\n  Full DataFrame (filtered to {config.frequency_start} - {config.frequency_end} GHz):")
                            print(df[existing_cols].to_string())
                        else:
                            print(f"\n  Full DataFrame (first 20 columns, all {NUM_COLS} columns available):")
                            print(df.iloc[:, :20].to_string())
                    else:
                        print(f"\n  Full DataFrame (first 20 columns, all {NUM_COLS} columns available):")
                        print(df.iloc[:, :20].to_string())
                    
                    # Add date column and reset index to make hour a column
                    df_with_date = df.reset_index()  # Reset index to make hour a column
                    df_with_date.rename(columns={'index': 'hour'}, inplace=True)
                    df_with_date.insert(0, 'date', date)  # Add date as first column
                    
                    # Store for later combination
                    all_dataframes.append(df_with_date)
                    print(f"\n  ✓ DataFrame stored for combination (shape: {df_with_date.shape})")
                    
                except Exception as e:
                    print(f"✗ Failed to extract dataframe: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"✗ Failed to process {image_path.name}")
        
        # Combine all DataFrames into one
        if all_dataframes:
            print(f"\n{'='*60}")
            print(f"Combining {len(all_dataframes)} DataFrame(s)...")
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            print(f"✓ Combined DataFrame shape: {combined_df.shape}")
            print(f"  Columns: {list(combined_df.columns[:5])}... (and {len(combined_df.columns) - 5} more)")
            
            # Save to file
            output_file_path = transformer.output_path / config.output_data_file
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as Parquet (efficient for ML) or CSV
            if config.output_data_file.endswith('.parquet'):
                try:
                    combined_df.to_parquet(output_file_path, index=False, engine='pyarrow')
                    print(f"✓ Saved combined DataFrame to: {output_file_path} (Parquet format)")
                    
                    # Also save as CSV for human readability
                    csv_path = output_file_path.with_suffix('.csv')
                    combined_df.to_csv(csv_path, index=False)
                    print(f"✓ Also saved as CSV: {csv_path}")
                except ImportError:
                    print(f"⚠ PyArrow not available, saving as CSV instead")
                    csv_path = output_file_path.with_suffix('.csv')
                    combined_df.to_csv(csv_path, index=False)
                    print(f"✓ Saved combined DataFrame to: {csv_path}")
            else:
                # Save as CSV if extension is .csv or other format
                combined_df.to_csv(output_file_path, index=False)
                print(f"✓ Saved combined DataFrame to: {output_file_path}")
            
            # Print summary statistics
            print(f"\n  Combined DataFrame Summary:")
            print(f"    Total rows: {len(combined_df)}")
            print(f"    Total columns: {len(combined_df.columns)}")
            print(f"    Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
            print(f"    Hours: {sorted(combined_df['hour'].unique())}")
            
            # Value statistics
            value_cols = [col for col in combined_df.columns if col not in ['date', 'hour']]
            values_flat = combined_df[value_cols].values.flatten()
            values_flat = values_flat[~np.isnan(values_flat)]
            if len(values_flat) > 0:
                print(f"    Value statistics (all frequency bands):")
                print(f"      Count (non-NaN): {len(values_flat)}")
                print(f"      Min: {np.min(values_flat):.2f}")
                print(f"      Max: {np.max(values_flat):.2f}")
                print(f"      Mean: {np.mean(values_flat):.2f}")
                print(f"      Median: {np.median(values_flat):.2f}")
        else:
            print(f"\n{'='*60}")
            print(f"⚠ No DataFrames to combine - no data was extracted")
        
        print(f"\n{'='*60}")
        print(f"✓ Processing complete: {processed_count}/{len(dates)} image(s) processed")
        
    except FileNotFoundError as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValidationError as e:
        print(f"✗ Configuration Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"✗ Unexpected Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
