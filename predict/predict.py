#!/usr/bin/env python3
"""
Prediction Script

This script loads transformed data from the transform step and performs
online learning to predict the next day's data. It uses the color scale
from preprocessed images to generate visualization images showing actual
vs predicted side by side.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFont
import cv2
import pytesseract
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field, ValidationError

# Add project root to Python path to find color_scale and transform modules
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import ColorScale from color_scale module
from color_scale import ColorScale

# Import ImageTransformer and TransformConfig from transform module
transform_path = project_root / "transform"
if str(transform_path) not in sys.path:
    sys.path.insert(0, str(transform_path))
# Import the module directly
import importlib.util
spec = importlib.util.spec_from_file_location("transform_module", transform_path / "transform.py")
transform_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transform_module)
ImageTransformer = transform_module.ImageTransformer
TransformConfig = transform_module.TransformConfig

# Constants
GRAPH_LEFT = 175
GRAPH_TOP = 100
GRAPH_WIDTH = 733
GRAPH_HEIGHT = 564
NUM_ROWS = 24
NUM_COLS = 70
CELL_SIZE = 8
ERROR_CELL_SIZE = 20
SCALE_WIDTH = 50
PADDING = 20
GAP = 20
SCALE_GAP = 20
ERROR_MAP_GAP = 20
DEFAULT_MAX_VALUE = 60.0
ZERO_ERROR_THRESHOLD = 0.1
PEAK_THRESHOLD_MULTIPLIER = 1.5
VOLATILITY_THRESHOLD = 0.35
PEAK_DETECTION_THRESHOLD = 0.80
MIN_LOOKBACK_FOR_PEAK = 4
MAX_LOOKBACK_FOR_PEAK = 6
# Scale detection constants (matching transform module)
SCALE_RIGHT_START_RATIO = 0.85
SCALE_MARGIN_RATIO = 0.05


class PredictConfig(BaseModel):
    """Prediction configuration."""
    transformed_data_file: str = Field(
        ...,
        description="Path to transformed data file (CSV or Parquet) from transform step"
    )
    preprocessed_folder: str = Field(
        default="data/preprocessed",
        description="Path to folder containing preprocessed images"
    )
    output_folder: str = Field(
        default="data/predictions",
        description="Path to folder for prediction results"
    )
    lookback_days: int = Field(
        default=6,
        ge=1,
        description="Number of previous days to use for prediction"
    )
    prediction_method: str = Field(
        default="exponential_smoothing",
        description="Prediction method: 'linear', 'exponential_smoothing', or 'moving_average'"
    )
    visualize_predictions: bool = Field(
        default=True,
        description="Enable visualization of predictions"
    )


class OnlinePredictor:
    """
    Online learning predictor for time series data.
    """
    
    def __init__(self, config: PredictConfig):
        self.config = config
        self.lookback_days = config.lookback_days
        self.prediction_method = config.prediction_method
        
    def predict_next_day(self, historical_data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict the next day's data using historical data.
        
        Args:
            historical_data: DataFrame with columns [date, hour, freq_bands...]
                           Should contain data for the last N days
            
        Returns:
            DataFrame with predicted values for the next day (24 rows, 70 frequency columns)
        """
        # Get frequency band columns (all columns except date and hour)
        freq_columns = [col for col in historical_data.columns if col not in ['date', 'hour']]
        
        # Get unique dates and hours
        dates = sorted(historical_data['date'].unique())
        hours = sorted(historical_data['hour'].unique())
        
        # Predict for each hour and frequency band
        predictions = []
        
        for hour in hours:
            # Get data for this hour across all historical days
            hour_data = historical_data[historical_data['hour'] == hour]
            
            # Predict each frequency band
            predicted_row = {'hour': hour}
            
            for freq_col in freq_columns:
                # Get values for this frequency band across historical days
                values = hour_data[freq_col].values
                
                # Remove NaN values
                valid_values = values[~np.isnan(values)]
                
                if len(valid_values) == 0:
                    # No valid data, predict NaN
                    predicted_row[freq_col] = np.nan
                elif len(valid_values) == 1:
                    # Only one value, use it
                    predicted_row[freq_col] = valid_values[0]
                else:
                    predicted = self._predict_value(valid_values)
                    
                    # Ensure prediction is non-negative
                    predicted = max(0.0, predicted)
                    
                    predicted_row[freq_col] = predicted
            
            predictions.append(predicted_row)
        
        # Create DataFrame
        pred_df = pd.DataFrame(predictions)
        
        # Sort by hour in reverse order (23 to 0) to match original image layout
        # Original image: hour 23 at top (row 0), hour 0 at bottom (row 23)
        pred_df = pred_df.sort_values('hour', ascending=False).reset_index(drop=True)
        
        return pred_df
    
    def _predict_value(self, values: np.ndarray) -> float:
        """Predict next value using peak detection or standard method."""
        if len(values) >= MIN_LOOKBACK_FOR_PEAK:
            lookback = min(MAX_LOOKBACK_FOR_PEAK, len(values))
            recent = values[-lookback:]
            recent_max, recent_mean, recent_std = np.max(recent), np.mean(recent), np.std(recent)
            
            if recent_max > recent_mean * PEAK_THRESHOLD_MULTIPLIER and recent_std > recent_mean * VOLATILITY_THRESHOLD:
                peak_threshold = recent_max * PEAK_DETECTION_THRESHOLD
                peak_positions = [i for i, v in enumerate(values) if v >= peak_threshold]
                if peak_positions:
                    peak_idx = peak_positions[-1]
                    peak_value = values[peak_idx]
                    positions_from_end = len(values) - 1 - peak_idx
                    last_value = values[-1]
                    
                    if positions_from_end <= 1:
                        predicted = (peak_value * 0.97 + last_value * 0.03) if last_value > recent_mean * PEAK_THRESHOLD_MULTIPLIER else peak_value * 0.97
                    elif positions_from_end <= 3:
                        predicted = peak_value * 0.94
                    elif positions_from_end <= 5:
                        predicted = peak_value * 0.90
                    else:
                        predicted = peak_value * 0.88
                    return max(0.0, predicted)
        
        # Use standard prediction method
        if self.prediction_method == "exponential_smoothing":
            return self._exponential_smoothing(values)
        elif self.prediction_method == "moving_average":
            return self._moving_average(values)
        elif self.prediction_method == "linear":
            return self._linear_regression(values)
        else:
            return self._exponential_smoothing(values)
    
    def _exponential_smoothing(self, values: np.ndarray) -> float:
        """Exponential smoothing prediction with strong emphasis on recent values."""
        if len(values) == 0:
            return np.nan
        
        if len(values) == 1:
            return values[0]
        
        lookback = min(MAX_LOOKBACK_FOR_PEAK, len(values))
        recent = values[-lookback:]
        max_val, min_val = np.max(recent), np.min(recent)
        range_val, mean_val, std_val = max_val - min_val, np.mean(recent), np.std(recent)
        
        if range_val > 40 and std_val > mean_val * 0.8:
            max_idx_in_recent = np.argmax(recent)
            if max_idx_in_recent >= lookback // 2:
                return max_val * (0.95 if max_idx_in_recent >= lookback - 2 else 0.90)
            return max_val * 0.82
        
        # For less volatile data, use exponential smoothing with heavy recent weight
        if len(values) >= 3:
            # Very heavy weight on most recent: [0.05, 0.15, 0.80]
            weights = np.array([0.05, 0.15, 0.80])
            return np.sum(values[-3:] * weights)
        elif len(values) == 2:
            return 0.92 * values[-1] + 0.08 * values[-2]
        
        return values[-1]
    
    def _moving_average(self, values: np.ndarray) -> float:
        """Moving average prediction with emphasis on recent values."""
        if len(values) == 0:
            return np.nan
        
        # Use weighted moving average - more weight on recent values
        window = min(self.lookback_days, len(values))
        recent_values = values[-window:]
        
        # Create weights that favor recent values
        weights = np.linspace(0.5, 1.0, len(recent_values))
        weights = weights / weights.sum()
        
        return np.sum(recent_values * weights)
    
    def _linear_regression(self, values: np.ndarray) -> float:
        """Simple linear regression prediction."""
        if len(values) < 2:
            return values[0] if len(values) == 1 else np.nan
        
        # Simple linear extrapolation
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        
        # Predict next value
        next_x = len(values)
        predicted = coeffs[0] * next_x + coeffs[1]
        
        return predicted


class PredictionVisualizer:
    """
    Visualizes predictions by generating images from predicted values.
    """
    
    def __init__(self, config: PredictConfig):
        self.config = config
        self.output_path = Path(config.output_folder)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.preprocessed_path = Path(config.preprocessed_folder)
        self.input_scale_colors: Optional[list] = None
        self.input_max_value: Optional[float] = None
        # Fallback color scale for when input scale is not available
        self._color_scale_fallback = ColorScale()
    
    def get_scale_for_date(self, date: str) -> Optional[ColorScale]:
        """
        Get the color scale for a given date.
        
        Extracts scale from preprocessed image and uses reference scale for calibration.
        
        Args:
            date: Date in YYYY-MM-DD format
            
        Returns:
            ColorScale object or None if image not found
        """
        # Convert date to filename format
        filename = date.replace("-", "_") + ".png"
        image_path = self.preprocessed_path / filename
        
        if not image_path.exists():
            print(f"  Warning: Preprocessed image not found for {date}: {filename}")
            return None
        
        # Initialize color scale with reference scale
        try:
            scale = ColorScale()
            print("  ✓ Loaded reference color scale from data/scale/scale.png")
        except Exception as e:
            print(f"  ✗ Error loading reference scale: {e}")
            return None
        
        # Use ImageTransformer to extract scale from input image
        transform_config = TransformConfig(
            from_date=date,
            to_date=date,
            scale_debug=False,
            debug_dataframe=False,
            frequency_start=3.1,
            frequency_end=3.5,
            preprocessed_folder=self.config.preprocessed_folder,
            output_folder=str(self.output_path / "temp")
        )
        
        transformer = ImageTransformer(transform_config)
        success = transformer.process_image(image_path)
        
        if success and transformer.scale:
            # Store input scale info for calibration
            self.input_scale_colors = transformer.input_scale_colors
            self.input_max_value = transformer.input_max_value
            
            if self.input_scale_colors is None or len(self.input_scale_colors) == 0:
                print(f"  Warning: Scale colors are empty for {date}, scale extraction may have failed")
                return None
            
            print(f"  ✓ Extracted scale from image (max_value: {self.input_max_value})")
            return scale
        else:
            print(f"  Warning: Failed to process image for {date}")
            return None
    
    def load_original_image(self, date: str) -> Optional[np.ndarray]:
        """
        Load the original preprocessed image for a given date.
        
        Args:
            date: Date in YYYY-MM-DD format
            
        Returns:
            Image array or None if not found
        """
        filename = date.replace("-", "_") + ".png"
        image_path = self.preprocessed_path / filename
        
        if not image_path.exists():
            return None
        
        img = Image.open(image_path)
        return np.array(img)
    
    def extract_graph_region(self, img_array: np.ndarray) -> np.ndarray:
        """
        Extract the graph region from the original image.
        Uses the same fixed coordinates as the transform script.
        
        Args:
            img_array: Full image array
            
        Returns:
            Graph region array (24x70 grid)
        """
        height, width = img_array.shape[:2]
        
        left = GRAPH_LEFT
        top = GRAPH_TOP
        right = GRAPH_LEFT + GRAPH_WIDTH
        bottom = GRAPH_TOP + GRAPH_HEIGHT
        
        # Validate coordinates
        left = max(0, min(left, width))
        top = max(0, min(top, height))
        right = max(left, min(right, width))
        bottom = max(top, min(bottom, height))
        
        # Extract graph region
        graph_region = img_array[top:bottom, left:right]
        
        graph_height, graph_width = graph_region.shape[:2]
        cell_height = graph_height / NUM_ROWS
        cell_width = graph_width / NUM_COLS
        
        resized_graph = np.zeros((NUM_ROWS, NUM_COLS, 3), dtype=np.uint8)
        
        for row_idx in range(NUM_ROWS):
            y_in_graph = min(int(row_idx * cell_height + cell_height / 2), graph_height - 1)
            for col_idx in range(NUM_COLS):
                x_in_graph = min(int(col_idx * cell_width + cell_width / 2), graph_width - 1)
                
                # Sample pixel at center of cell
                resized_graph[row_idx, col_idx] = graph_region[y_in_graph, x_in_graph]
        
        return resized_graph
    
    def _detect_scale_region(self, img_array: np.ndarray) -> Dict[str, int]:
        """
        Automatically detect the scale region on the right side of the image.
        Uses the same logic as the transform module.
        
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
    
    def extract_scale_region(self, img_array: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract the scale region from the original image (including labels).
        
        Args:
            img_array: Full image array
            
        Returns:
            Scale region image array or None if detection fails
        """
        try:
            scale_region_dict = self._detect_scale_region(img_array)
            
            # Extract scale region (includes the color bar and labels)
            scale_region = img_array[
                scale_region_dict["top_y"]:scale_region_dict["bottom_y"],
                scale_region_dict["left_x"]:scale_region_dict["right_x"]
            ]
            
            return scale_region
        except Exception as e:
            print(f"  Warning: Failed to extract scale region: {e}")
            return None
    
    def _get_min_color(self, color_scale: ColorScale) -> Tuple[int, int, int]:
        """Get the minimum (darkest) color from the scale."""
        scale_colors = color_scale.get_scale_colors()
        if not scale_colors:
            return (0, 0, 0)
        
        min_color = tuple(scale_colors[-1])
        min_brightness = sum(float(c) for c in min_color) / 3.0
        
        if min_brightness > 200:
            darkest = min_color
            darkest_brightness = min_brightness
            for scale_color in scale_colors:
                brightness = sum(float(c) for c in scale_color) / 3.0
                if brightness < darkest_brightness:
                    darkest_brightness = brightness
                    darkest = tuple(scale_color)
            return darkest
        return min_color
    
    def _get_color_for_value(self, value: float, color_scale: ColorScale, max_value: float) -> Tuple[int, int, int]:
        """Convert value to color, ensuring never white/too light."""
        min_color = self._get_min_color(color_scale)
        clamped_value = max(0.0, min(max_value, float(value)))
        
        try:
            if self.input_scale_colors and len(self.input_scale_colors) > 0:
                color = self._number_to_color_using_input_scale(clamped_value, self.input_scale_colors, max_value)
            else:
                color = color_scale.number_to_color(clamped_value, max_value=max_value)
        except Exception:
            return min_color
        
        is_white = all(c > 250 for c in color)
        brightness = sum(float(c) for c in color) / 3.0
        is_too_light = brightness > (150 if value < 5.0 else 230)
        
        return min_color if (is_white or is_too_light) else color
    
    def _number_to_color_using_input_scale(self, value: float, input_scale_colors: list, max_value: float) -> Tuple[int, int, int]:
        """
        Convert a number to color using the input scale colors directly.
        This ensures colors match what the original image's scale represents.
        
        Args:
            value: Numeric value
            input_scale_colors: List of RGB colors from input image scale (top to bottom)
            max_value: Maximum value on the scale
            
        Returns:
            RGB tuple
        """
        if not input_scale_colors or len(input_scale_colors) == 0:
            # Fallback to reference scale
            return self._color_scale_fallback.number_to_color(value, max_value=max_value)
        
        # Clamp value
        value = max(0.0, min(max_value, float(value)))
        
        # Map value to position on input scale
        # Top (position 0) = max value, bottom (position 1) = 0
        value_ratio = value / max_value if max_value > 0 else 0.0
        position_ratio = 1.0 - value_ratio  # Invert: max value -> position 0 (top)
        position_ratio = max(0.0, min(1.0, position_ratio))
        
        # Get color from input scale at this position
        num_colors = len(input_scale_colors)
        position_idx = position_ratio * (num_colors - 1)
        idx1 = int(np.clip(position_idx, 0, num_colors - 1))
        idx2 = min(idx1 + 1, num_colors - 1)
        frac = position_idx - idx1
        frac = max(0.0, min(1.0, frac))
        
        # Interpolate between adjacent colors
        color1 = np.array(input_scale_colors[idx1], dtype=np.float64)
        color2 = np.array(input_scale_colors[idx2], dtype=np.float64)
        interpolated_color = color1 + (color2 - color1) * frac
        interpolated_color = np.clip(interpolated_color, 0, 255)
        
        return tuple(interpolated_color.astype(np.uint8))
    
    def generate_prediction_image(
        self,
        predicted_df: pd.DataFrame,
        actual_df: Optional[pd.DataFrame],
        date: str,
        color_scale: ColorScale
    ) -> Path:
        """
        Generate an image from predicted values using the color scale.
        Shows original image, predicted image, and error map side by side.
        
        Args:
            predicted_df: DataFrame with predicted values (24 rows, 70 frequency columns)
            actual_df: Optional DataFrame with actual values for comparison
            date: Date string (YYYY-MM-DD)
            color_scale: ColorScale object for converting values to colors
            
        Returns:
            Tuple of (Path to saved image, mean_error, max_error, min_error, zero_error_percentage)
            All error values are None if actual_df is not provided
        """
        # Get frequency columns
        freq_columns = [col for col in predicted_df.columns if col not in ['date', 'hour']]
        num_freq_bands = len(freq_columns)
        num_hours = len(predicted_df)
        
        # Ensure predicted_df is in the correct order (hour 23 to 0)
        # Sort by hour descending to match original image (23 at top, 0 at bottom)
        predicted_df_sorted = predicted_df.sort_values('hour', ascending=False).reset_index(drop=True)
        
        # Calculate error map if actual data is available
        error_map = None
        mean_error = None
        max_error = None
        min_error = None
        zero_error_percentage = None
        if actual_df is not None:
            # Sort actual_df to match predicted_df order (hour 23 to 0)
            actual_df_sorted = actual_df.sort_values('hour', ascending=False).reset_index(drop=True)
            
            # Calculate absolute error for each cell: |actual - predicted|
            error_map = np.zeros((num_hours, num_freq_bands))
            error_values = []
            zero_error_count = 0
            total_valid_cells = 0
            
            for row_idx in range(min(len(predicted_df_sorted), len(actual_df_sorted))):
                pred_row = predicted_df_sorted.iloc[row_idx]
                actual_row = actual_df_sorted.iloc[row_idx]
                
                for col_idx, freq_col in enumerate(freq_columns):
                    pred_val = pred_row[freq_col]
                    actual_val = actual_row[freq_col]
                    
                    # Calculate signed error: actual - predicted
                    # Positive = predicted too low (actual > predicted)
                    # Negative = predicted too high (actual < predicted)
                    if not np.isnan(actual_val) and not np.isnan(pred_val):
                        signed_error = actual_val - pred_val
                        error_map[row_idx, col_idx] = signed_error
                        
                        # For statistics, use absolute error
                        abs_error = abs(signed_error)
                        error_values.append(abs_error)
                        total_valid_cells += 1
                        
                        if abs_error < ZERO_ERROR_THRESHOLD:
                            zero_error_count += 1
                    else:
                        # NaN values - mark as NaN
                        error_map[row_idx, col_idx] = np.nan
            
            # Calculate error statistics
            mean_error = None
            max_error = None
            min_error = None
            zero_error_percentage = None
            
            if error_values:
                mean_error = np.mean(error_values)
                
                non_zero_errors = [e for e in error_values if e >= ZERO_ERROR_THRESHOLD]
                if non_zero_errors:
                    max_error = np.max(non_zero_errors)
                    min_error = np.min(non_zero_errors)
                
                if total_valid_cells > 0:
                    zero_error_percentage = (zero_error_count / total_valid_cells) * 100.0
        
        
        graph_width = num_freq_bands * CELL_SIZE
        graph_height = num_hours * CELL_SIZE
        error_map_gap = ERROR_MAP_GAP if error_map is not None else 0
        error_map_height_estimate = (num_hours * ERROR_CELL_SIZE) if error_map is not None else 0
        stats_height = 30 if error_map is not None else 0
        
        effective_max = self.input_max_value if self.input_max_value is not None else DEFAULT_MAX_VALUE
        original_img_array = self.load_original_image(date)
        
        # Determine actual scale width (may be wider if it includes labels)
        # Extract scale region early to determine width
        actual_scale_width = SCALE_WIDTH
        scale_region = None
        if original_img_array is not None:
            scale_region = self.extract_scale_region(original_img_array)
            if scale_region is not None:
                actual_scale_width = scale_region.shape[1]
        
        total_width = graph_width * 2 + actual_scale_width + GAP + SCALE_GAP + PADDING * 2
        total_height = graph_height + error_map_height_estimate + stats_height + PADDING * 3 + 30 + error_map_gap
        
        # Create image
        img = Image.new('RGB', (total_width, total_height), color='white')
        draw = ImageDraw.Draw(img)
        orig_x_start = PADDING
        orig_y_start = PADDING + 25
        
        if original_img_array is not None:
            original_graph = self.extract_graph_region(original_img_array)
            
            # Create a transformer to convert original image colors to values
            # This ensures the original image uses the same color mapping as predictions
            # ImageTransformer and TransformConfig are already imported at the top
            transform_config = TransformConfig(
                from_date=date,
                to_date=date,
                scale_debug=False,
                debug_dataframe=False,
                frequency_start=3.1,
                frequency_end=3.5,
                preprocessed_folder=self.config.preprocessed_folder,
                output_folder=str(self.output_path / "temp")
            )
            orig_transformer = ImageTransformer(transform_config)
            orig_image_path = self.preprocessed_path / f"{date.replace('-', '_')}.png"
            if orig_image_path.exists():
                orig_transformer.process_image(orig_image_path)
            
            for row_idx in range(num_hours):
                y = orig_y_start + row_idx * CELL_SIZE
                for col_idx in range(num_freq_bands):
                    x = orig_x_start + col_idx * CELL_SIZE
                    # Get pixel color from original graph
                    raw_color = tuple(original_graph[row_idx, col_idx])
                    
                    # Convert color to value using the original image's scale
                    if orig_transformer.scale and orig_transformer.input_scale_colors and orig_transformer.input_max_value:
                        value = orig_transformer.get_color_value(raw_color)
                        if value is None:
                            value = 0.0
                        # Convert value back to color using the same process as predictions
                        clamped_value = max(0.0, min(effective_max, float(value)))
                        if self.input_scale_colors and len(self.input_scale_colors) > 0:
                            color = self._number_to_color_using_input_scale(
                                clamped_value,
                                self.input_scale_colors,
                                effective_max
                            )
                        else:
                            color = color_scale.number_to_color(clamped_value, max_value=effective_max)
                    else:
                        # Fallback to raw color if transformer failed
                        color = raw_color
                    
                    draw.rectangle([x, y, x + CELL_SIZE, y + CELL_SIZE], fill=color)
        else:
            print(f"  Warning: Could not load original image for {date}")
            draw.rectangle(
                [orig_x_start, orig_y_start, orig_x_start + graph_width, orig_y_start + graph_height],
                fill='lightgray', outline='black', width=2
            )
        
        pred_x_start = orig_x_start + graph_width + GAP
        pred_y_start = PADDING + 25
        
        for row_idx, row in predicted_df_sorted.iterrows():
            y = pred_y_start + row_idx * CELL_SIZE
            for col_idx, freq_col in enumerate(freq_columns):
                value = row[freq_col]
                x = pred_x_start + col_idx * CELL_SIZE
                
                value = 0.0 if (np.isnan(value) or value is None) else float(value)
                color = self._get_color_for_value(value, color_scale, effective_max)
                draw.rectangle([x, y, x + CELL_SIZE, y + CELL_SIZE], fill=color)
        
        scale_x = pred_x_start + graph_width + SCALE_GAP
        scale_y_start = PADDING + 25
        
        # Use the scale region already extracted (or try again if not available)
        if scale_region is None and original_img_array is not None:
            scale_region = self.extract_scale_region(original_img_array)
        
        if scale_region is not None:
            # Resize the scale region to match the graph height
            scale_img = Image.fromarray(scale_region)
            scale_height = graph_height
            # Maintain aspect ratio, but use the detected width or a reasonable default
            scale_region_width = scale_region.shape[1]
            scale_img_resized = scale_img.resize((scale_region_width, scale_height), Image.Resampling.LANCZOS)
            
            # Paste the actual scale image (includes labels with max value)
            img.paste(scale_img_resized, (scale_x, scale_y_start))
        else:
            # Fallback: generate scale if extraction failed
            num_scale_samples = max(200, graph_height)
            
            for i in range(num_scale_samples):
                y_ratio = i / (num_scale_samples - 1) if num_scale_samples > 1 else 0.0
                y = int(scale_y_start + y_ratio * graph_height)
                value = (1.0 - y_ratio) * effective_max
                color = self._get_color_for_value(value, color_scale, effective_max)
                y_next = int(scale_y_start + ((i + 1) / num_scale_samples) * graph_height) if i < num_scale_samples - 1 else scale_y_start + graph_height
                draw.rectangle([scale_x, y, scale_x + SCALE_WIDTH, y_next], fill=color)
        
        # Add labels
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
            small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        draw.text((PADDING, 5), f"Date: {date}", fill='black', font=font)
        draw.text((orig_x_start, PADDING + 10), "Original", fill='black', font=small_font)
        draw.text((pred_x_start, PADDING + 10), "Predicted", fill='black', font=small_font)
        
        # Draw error map if available (simple grid with numbers)
        # Error map spans the full width of both original and predicted images
        if error_map is not None:
            error_y_start = orig_y_start + graph_height + error_map_gap
            
            
            # Load font for error map (larger size for readability)
            try:
                error_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
            except:
                try:
                    error_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 10)
                except:
                    error_font = ImageFont.load_default()
            
            error_map_width = graph_width * 2 + GAP
            error_total_cells = num_freq_bands * 2
            
            for row_idx in range(num_hours):
                y = error_y_start + row_idx * ERROR_CELL_SIZE
                for col_idx in range(error_total_cells):
                    x = orig_x_start + col_idx * ERROR_CELL_SIZE
                    
                    if x + ERROR_CELL_SIZE <= orig_x_start + error_map_width:
                        freq_col_idx = col_idx % num_freq_bands
                        error_pct = error_map[row_idx, freq_col_idx]
                        
                        draw.rectangle(
                            [x, y, x + ERROR_CELL_SIZE, y + ERROR_CELL_SIZE],
                            fill='white', outline='lightgray', width=1
                        )
                        
                        if not np.isnan(error_pct):
                            error_text = f"+{error_pct:.1f}" if error_pct >= 0 else f"{error_pct:.1f}"
                            try:
                                bbox = draw.textbbox((0, 0), error_text, font=error_font)
                                text_x = x + max(0, (ERROR_CELL_SIZE - (bbox[2] - bbox[0])) // 2)
                                text_y = y + max(0, (ERROR_CELL_SIZE - (bbox[3] - bbox[1])) // 2)
                                draw.text((text_x, text_y), error_text, fill='black', font=error_font)
                            except Exception:
                                pass
            
            error_map_height = num_hours * ERROR_CELL_SIZE
            
            # Label for error map
            draw.text((orig_x_start, error_y_start - 15), "Error Map", fill='black', font=small_font)
            
            stats_y = error_y_start + error_map_height + 5
            stats_x = orig_x_start
            stats_spacing = 30
            
            if mean_error is not None:
                error_text = f"Mean: {mean_error:.2f}"
                draw.text((stats_x, stats_y), error_text, fill='black', font=font)
                # Get text width to position next stat
                try:
                    bbox = draw.textbbox((0, 0), error_text, font=font)
                    stats_x += (bbox[2] - bbox[0]) + stats_spacing
                except:
                    stats_x += 80 + stats_spacing
            
            if max_error is not None:
                max_text = f"Max: {max_error:.2f}"
                draw.text((stats_x, stats_y), max_text, fill='black', font=font)
                try:
                    bbox = draw.textbbox((0, 0), max_text, font=font)
                    stats_x += (bbox[2] - bbox[0]) + stats_spacing
                except:
                    stats_x += 80 + stats_spacing
            
            if min_error is not None:
                min_text = f"Min: {min_error:.2f}"
                draw.text((stats_x, stats_y), min_text, fill='black', font=font)
                try:
                    bbox = draw.textbbox((0, 0), min_text, font=font)
                    stats_x += (bbox[2] - bbox[0]) + stats_spacing
                except:
                    stats_x += 80 + stats_spacing
            
            if zero_error_percentage is not None:
                zero_text = f"Zero Error: {zero_error_percentage:.1f}%"
                draw.text((stats_x, stats_y), zero_text, fill='black', font=font)
        
        # Save image
        output_file = self.output_path / f"prediction_{date.replace('-', '_')}.png"
        img.save(output_file)
        
        return output_file, mean_error, max_error, min_error, zero_error_percentage


def load_config(config_path: str = None) -> PredictConfig:
    """Load and validate YAML configuration file."""
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
    
    return PredictConfig(**config_data)


def main():
    """Main entry point."""
    try:
        print("Loading config...")
        config = load_config()
        print("✓ Configuration loaded")
        
        # Load transformed data
        print(f"\nLoading transformed data from: {config.transformed_data_file}")
        data_path = Path(config.transformed_data_file)
        
        if not data_path.exists():
            print(f"✗ Error: Transformed data file not found: {data_path}")
            return
        
        # Load data
        if data_path.suffix == '.parquet':
            try:
                df = pd.read_parquet(data_path)
            except ImportError:
                print("⚠ PyArrow not available, trying CSV...")
                csv_path = data_path.with_suffix('.csv')
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                else:
                    print(f"✗ Error: Neither Parquet nor CSV file found")
                    return
        else:
            df = pd.read_csv(data_path)
        
        print(f"✓ Loaded data: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  Hours: {sorted(df['hour'].unique())}")
        
        # Sort by date and hour
        df = df.sort_values(['date', 'hour']).reset_index(drop=True)
        
        # Get unique dates
        dates = sorted(df['date'].unique())
        print(f"  Total dates: {len(dates)}")
        
        if len(dates) < config.lookback_days + 1:
            print(f"⚠ Warning: Need at least {config.lookback_days + 1} days for prediction, but only have {len(dates)}")
            print("  Will use available data")
        
        # Initialize predictor and visualizer
        predictor = OnlinePredictor(config)
        visualizer = PredictionVisualizer(config)
        
        # Process each day (starting from lookback_days + 1)
        predictions_made = 0
        
        for i in range(config.lookback_days, len(dates)):
            target_date = dates[i]
            print(f"\n{'='*60}")
            print(f"Predicting for date: {target_date}")
            print(f"{'='*60}")
            
            # Get historical data (previous N days)
            historical_dates = dates[max(0, i - config.lookback_days):i]
            historical_data = df[df['date'].isin(historical_dates)].copy()
            
            print(f"  Using historical data from {len(historical_dates)} day(s): {historical_dates[0]} to {historical_dates[-1]}")
            
            # Predict next day
            predicted_df = predictor.predict_next_day(historical_data)
            
            # Add date column
            predicted_df.insert(0, 'date', target_date)
            
            # Get actual data for comparison (if available)
            actual_data = df[df['date'] == target_date].copy()
            actual_df = actual_data if len(actual_data) > 0 else None
            
            if actual_df is not None:
                print(f"  ✓ Actual data available for comparison")
            else:
                print(f"  ⚠ No actual data available for {target_date}")
            
            # Get color scale for this date
            print(f"  Loading color scale from preprocessed image...")
            color_scale = visualizer.get_scale_for_date(target_date)
            
            if color_scale is None:
                print(f"  ⚠ Warning: Could not load color scale, skipping visualization")
            else:
                max_val = visualizer.input_max_value if visualizer.input_max_value is not None else DEFAULT_MAX_VALUE
                print(f"  ✓ Color scale loaded (max: {max_val})")
                
                # Generate visualization
                if config.visualize_predictions:
                    print(f"  Generating visualization (original vs predicted + error map)...")
                    output_path, mean_error, max_error, min_error, zero_error_percentage = visualizer.generate_prediction_image(
                        predicted_df,
                        actual_df,
                        target_date,
                        color_scale
                    )
                    print(f"  ✓ Visualization saved: {output_path}")
                    if mean_error is not None:
                        print(f"    Mean Error: {mean_error:.2f}")
                    if max_error is not None:
                        print(f"    Max Error: {max_error:.2f} (excluding 0)")
                    if min_error is not None:
                        print(f"    Min Error: {min_error:.2f} (excluding 0)")
                    if zero_error_percentage is not None:
                        print(f"    Zero Error Cells: {zero_error_percentage:.1f}%")
            
            predictions_made += 1
        
        print(f"\n{'='*60}")
        print(f"✓ Prediction complete: {predictions_made} prediction(s) made")
        
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
