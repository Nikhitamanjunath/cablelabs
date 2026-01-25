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
    learning_rate: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Learning rate for online learning (0.0 to 1.0)"
    )
    prediction_method: str = Field(
        default="exponential_smoothing",
        description="Prediction method: 'linear', 'exponential_smoothing', or 'moving_average'"
    )
    exponential_smoothing_alpha: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Alpha parameter for exponential smoothing"
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
        self.learning_rate = config.learning_rate
        self.prediction_method = config.prediction_method
        self.alpha = config.exponential_smoothing_alpha
        
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
                    # Improved prediction strategy
                    # For volatile data, use the maximum of recent values (peaks are important)
                    # But apply decay based on how recent the peak was
                    
                    # Smart prediction: For volatile data with peaks, peaks are the signal
                    # Check if there are significant peaks in the data
                    if len(valid_values) >= 4:
                        # Look at last 6 values if available for better pattern detection
                        lookback = min(6, len(valid_values))
                        recent = valid_values[-lookback:]
                        recent_max = np.max(recent)
                        recent_mean = np.mean(recent)
                        recent_std = np.std(recent)
                        
                        # If there's a significant peak (max >> mean) and high volatility
                        # Lower the threshold to catch more peak cases
                        if recent_max > recent_mean * 1.5 and recent_std > recent_mean * 0.35:
                            # Peak-based prediction: use the peak with minimal decay
                            # Find most recent occurrence of values close to the peak
                            peak_threshold = recent_max * 0.80  # 80% of max (even more lenient to catch peaks)
                            peak_positions = [i for i, v in enumerate(valid_values) if v >= peak_threshold]
                            if peak_positions:
                                most_recent_peak_idx = peak_positions[-1]
                                peak_value = valid_values[most_recent_peak_idx]
                                # Calculate how recent the peak is
                                positions_from_end = len(valid_values) - 1 - most_recent_peak_idx
                                # If peak was very recent (last 2 values), use it almost directly
                                # Also consider the most recent value - if it's also high, trust it more
                                last_value = valid_values[-1]
                                if positions_from_end <= 1:
                                    # Peak is very recent - use peak with minimal decay
                                    if last_value > recent_mean * 1.5:
                                        # Last value is also high - average peak and last
                                        predicted = (peak_value * 0.97 + last_value * 0.03)
                                    else:
                                        predicted = peak_value * 0.97
                                elif positions_from_end <= 3:
                                    predicted = peak_value * 0.94
                                elif positions_from_end <= 5:
                                    predicted = peak_value * 0.90
                                else:
                                    predicted = peak_value * 0.88
                            else:
                                # Use standard method
                                if self.prediction_method == "exponential_smoothing":
                                    predicted = self._exponential_smoothing(valid_values)
                                else:
                                    predicted = self._exponential_smoothing(valid_values)
                        else:
                            # Use the selected prediction method
                            if self.prediction_method == "exponential_smoothing":
                                predicted = self._exponential_smoothing(valid_values)
                            elif self.prediction_method == "moving_average":
                                predicted = self._moving_average(valid_values)
                            elif self.prediction_method == "linear":
                                predicted = self._linear_regression(valid_values)
                            else:
                                predicted = self._exponential_smoothing(valid_values)
                    else:
                        # Use the selected prediction method
                        if self.prediction_method == "exponential_smoothing":
                            predicted = self._exponential_smoothing(valid_values)
                        elif self.prediction_method == "moving_average":
                            predicted = self._moving_average(valid_values)
                        elif self.prediction_method == "linear":
                            predicted = self._linear_regression(valid_values)
                        else:
                            predicted = self._exponential_smoothing(valid_values)
                    
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
    
    def _exponential_smoothing(self, values: np.ndarray) -> float:
        """Exponential smoothing prediction with strong emphasis on recent values."""
        if len(values) == 0:
            return np.nan
        
        if len(values) == 1:
            return values[0]
        
        # Strategy: For volatile data, peaks are the signal
        # Look at last 6 values if available to better detect patterns
        lookback = min(6, len(values))
        recent = values[-lookback:]
        max_val = np.max(recent)
        min_val = np.min(recent)
        range_val = max_val - min_val
        mean_val = np.mean(recent)
        std_val = np.std(recent)
        
        # If very volatile (high range and std), peaks are likely the signal
        if range_val > 40 and std_val > mean_val * 0.8:
            # Find the maximum and its position
            max_idx_in_recent = np.argmax(recent)
            max_absolute_idx = len(values) - lookback + max_idx_in_recent
            
            # If max is in the most recent half of the lookback window
            if max_idx_in_recent >= lookback // 2:
                # Recent peak - use it with minimal decay (0.90-0.95)
                decay = 0.95 if max_idx_in_recent >= lookback - 2 else 0.90
                return max_val * decay
            else:
                # Older peak - still use it but with more decay
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
        
        # Use fixed coordinates from transform script (same as scraper)
        left = 175
        top = 100
        right = 908  # 175 + 733
        bottom = 664  # 100 + 564
        
        # Validate coordinates
        left = max(0, min(left, width))
        top = max(0, min(top, height))
        right = max(left, min(right, width))
        bottom = max(top, min(bottom, height))
        
        # Extract graph region
        graph_region = img_array[top:bottom, left:right]
        
        # Resize to 24x70 grid (one pixel per cell)
        graph_height, graph_width = graph_region.shape[:2]
        num_rows = 24
        num_cols = 70
        
        # Calculate cell dimensions
        cell_height = graph_height / num_rows
        cell_width = graph_width / num_cols
        
        # Create resized graph (24x70 pixels)
        resized_graph = np.zeros((num_rows, num_cols, 3), dtype=np.uint8)
        
        for row_idx in range(num_rows):
            # Row 0 is hour 23 (top), row 23 is hour 0 (bottom)
            # Calculate y position: center of the row cell
            y_in_graph = int(row_idx * cell_height + cell_height / 2)
            y_in_graph = min(y_in_graph, graph_height - 1)
            
            for col_idx in range(num_cols):
                # Calculate x position: center of the column cell
                x_in_graph = int(col_idx * cell_width + cell_width / 2)
                x_in_graph = min(x_in_graph, graph_width - 1)
                
                # Sample pixel at center of cell
                resized_graph[row_idx, col_idx] = graph_region[y_in_graph, x_in_graph]
        
        return resized_graph
    
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
                        
                        # Count zero errors (use threshold for floating point comparison)
                        # Use a more lenient threshold - consider errors < 0.1 as effectively zero
                        # This accounts for small rounding differences in color-to-number conversion
                        if abs_error < 0.1:  # Consider errors < 0.1 as zero (more lenient)
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
                
                # Max and min error (ignoring cells with effectively 0 error)
                non_zero_errors = [e for e in error_values if e >= 0.1]  # Use same threshold as zero detection
                if non_zero_errors:
                    max_error = np.max(non_zero_errors)
                    min_error = np.min(non_zero_errors)
                
                # Percentage of cells with effectively 0 error (< 0.1)
                if total_valid_cells > 0:
                    zero_error_percentage = (zero_error_count / total_valid_cells) * 100.0
                    print(f"    Debug: zero_error_count={zero_error_count}, total_valid_cells={total_valid_cells}, percentage={zero_error_percentage:.1f}%")
                    print(f"    Debug: Sample errors - min={np.min(error_values):.3f}, max={np.max(error_values):.3f}, mean={np.mean(error_values):.3f}")
                    # Count how many are exactly 0 vs very small
                    exact_zeros = sum(1 for e in error_values if e == 0.0)
                    very_small = sum(1 for e in error_values if 0.0 < e < 0.1)
                    print(f"    Debug: exact_zeros={exact_zeros}, very_small (<0.1)={very_small}")
        
        # Image dimensions - one pixel per cell (24x70)
        cell_size = 8  # Size of each cell in pixels
        padding = 20
        scale_width = 50
        gap = 20  # Gap between original and predicted
        scale_gap = 20  # Gap between predicted and scale
        
        # Calculate image dimensions
        graph_width = num_freq_bands * cell_size
        graph_height = num_hours * cell_size
        
        # Side by side: original | predicted | scale
        # Add error map below if available (spans full width of both graphs)
        # Error map will use larger cells (20px instead of 8px) for better readability
        error_map_gap = 20 if error_map is not None else 0
        # Error map height will be calculated when drawing (uses larger cells)
        # Estimate: if error map exists, it will be taller than graph_height
        error_map_height_estimate = (num_hours * 20) if error_map is not None else 0  # 20px per cell
        # Extra space for statistics (displayed horizontally)
        stats_height = 30 if error_map is not None else 0
        
        total_width = graph_width * 2 + scale_width + gap + scale_gap + padding * 2
        total_height = graph_height + error_map_height_estimate + stats_height + padding * 3 + 30 + error_map_gap  # Extra space for labels, error map, and stats
        
        # Create image
        img = Image.new('RGB', (total_width, total_height), color='white')
        draw = ImageDraw.Draw(img)
        
        # Calculate actual data range for proper scaling FIRST
        # This needs to be done before drawing either original or predicted
        # Get all values from both predicted and actual dataframes to determine the true range
        all_values = []
        for _, row in predicted_df_sorted.iterrows():
            for freq_col in freq_columns:
                val = row[freq_col]
                if not np.isnan(val) and val is not None:
                    all_values.append(float(val))
        
        if actual_df is not None:
            actual_df_sorted_for_range = actual_df.sort_values('hour', ascending=False).reset_index(drop=True)
            for _, row in actual_df_sorted_for_range.iterrows():
                for freq_col in freq_columns:
                    val = row[freq_col]
                    if not np.isnan(val) and val is not None:
                        all_values.append(float(val))
        
        # Determine the effective max value for visualization
        # CRITICAL: Use the original image's scale max_value EXACTLY as-is
        # The original image's scale defines what colors mean, so predictions must use the same scale
        # This ensures predicted colors are directly comparable to the original image
        effective_max = self.input_max_value if self.input_max_value is not None else 60.0
        
        # Don't modify based on data range - use the scale as-is
        # The scale is the ground truth for what colors represent
        effective_min = 0.0
        
        # Load and extract original graph region
        original_img_array = self.load_original_image(date)
        orig_x_start = padding  # Define even if original image not found
        orig_y_start = padding + 25  # Space for label
        
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
            
            # Draw original graph - convert colors to values, then back to colors
            # This ensures both original and predicted use the same color mapping
            for row_idx in range(num_hours):
                y = orig_y_start + row_idx * cell_size
                for col_idx in range(num_freq_bands):
                    x = orig_x_start + col_idx * cell_size
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
                    
                    draw.rectangle(
                        [x, y, x + cell_size, y + cell_size],
                        fill=color
                    )
        else:
            print(f"  Warning: Could not load original image for {date}")
            # Draw placeholder for original
            draw.rectangle(
                [orig_x_start, orig_y_start, orig_x_start + graph_width, orig_y_start + graph_height],
                fill='lightgray',
                outline='black',
                width=2
            )
        
        # Draw predicted graph
        # Note: predicted_df_sorted is already sorted with hour 23 first (row 0), hour 0 last (row 23)
        # This matches the original image layout
        pred_x_start = orig_x_start + graph_width + gap
        pred_y_start = padding + 25  # Space for label
        
        for row_idx, row in predicted_df_sorted.iterrows():
            hour = int(row['hour'])
            y = pred_y_start + row_idx * cell_size
            
            for col_idx, freq_col in enumerate(freq_columns):
                value = row[freq_col]
                x = pred_x_start + col_idx * cell_size
                
                # Convert value to color - ALWAYS use a color, never white
                # If NaN or invalid, use minimum value (0.0 = dark blue)
                if np.isnan(value) or value is None:
                    value = 0.0  # Use minimum value (dark blue) instead of white
                
                # Convert to float - number_to_color will handle values above max_value
                # by mapping them to the maximum color (top of scale)
                value = float(value)
                
                # Get minimum color from scale first (for fallback)
                scale_colors = color_scale.get_scale_colors()
                if scale_colors and len(scale_colors) > 0:
                    min_color = tuple(scale_colors[-1])  # Bottom = minimum (dark blue/purple)
                    
                    # Verify minimum color is not white - if it is, find the darkest color
                    min_brightness = (min_color[0] + min_color[1] + min_color[2]) / 3.0
                    if min_brightness > 200:  # Minimum color is too light, find darkest
                        darkest_color = min_color
                        darkest_brightness = min_brightness
                        for scale_color in scale_colors:
                            brightness = (scale_color[0] + scale_color[1] + scale_color[2]) / 3.0
                            if brightness < darkest_brightness:
                                darkest_brightness = brightness
                                darkest_color = tuple(scale_color)
                        min_color = darkest_color
                else:
                    # No scale colors - skip this cell
                    continue
                
                # Convert value to color using the INPUT scale colors directly
                # This ensures colors match what the original image's scale represents
                # Clamp value to valid range
                clamped_value = max(0.0, min(effective_max, float(value)))
                
                try:
                    # Use input scale colors if available, otherwise fallback to reference scale
                    if self.input_scale_colors and len(self.input_scale_colors) > 0:
                        color = self._number_to_color_using_input_scale(
                            clamped_value, 
                            self.input_scale_colors, 
                            effective_max
                        )
                    else:
                        color = color_scale.number_to_color(clamped_value, max_value=effective_max)
                except Exception as e:
                    # If number_to_color fails, use minimum color from scale
                    print(f"    Warning: number_to_color failed for value {value}: {e}")
                    color = min_color
                
                # Aggressive check: Never use white or very light colors
                # Check multiple conditions for white/light colors
                is_white = (color[0] > 250 and color[1] > 250 and color[2] > 250)
                # Use float64 to avoid overflow warnings
                brightness = (float(color[0]) + float(color[1]) + float(color[2])) / 3.0
                is_too_light = brightness > 230  # Lower threshold
                
                # For low values, be even more strict
                if value < 5.0:
                    is_too_light = brightness > 150  # Very strict for low values
                
                # If color is white/too light, ALWAYS use minimum color from scale
                if is_white or is_too_light:
                    color = min_color
                
                # Final check: if still white after all checks, force minimum color
                if color[0] > 250 and color[1] > 250 and color[2] > 250:
                    color = min_color
                
                # Draw cell - always draw with a color, never white
                draw.rectangle(
                    [x, y, x + cell_size, y + cell_size],
                    fill=color
                )
        
        # Draw color scale on the right (with gap from predicted image)
        # This visualization uses the trained color scale directly - no transformations
        scale_x = pred_x_start + graph_width + scale_gap
        scale_y_start = padding + 25
        scale_height = graph_height
        num_scale_samples = max(200, scale_height)
        
        for i in range(num_scale_samples):
            y_ratio = i / (num_scale_samples - 1) if num_scale_samples > 1 else 0.0
            y = int(scale_y_start + y_ratio * scale_height)
            
            # Calculate value based on position
            # Scale is always standard: top (y_ratio=0) = max_value, bottom (y_ratio=1) = 0
            # Top of scale bar = yellow (max value), bottom = dark blue (0)
            # Use effective_max for consistency with the graph visualization
            max_val = effective_max
            value = (1.0 - y_ratio) * max_val
            
            # Get color for this value using the input scale colors if available
            if self.input_scale_colors and len(self.input_scale_colors) > 0:
                color = self._number_to_color_using_input_scale(value, self.input_scale_colors, max_val)
            else:
                color = color_scale.number_to_color(value, max_value=max_val)
            if color:
                # Draw a line segment for this part of the scale
                y_next = int(scale_y_start + ((i + 1) / num_scale_samples) * scale_height) if i < num_scale_samples - 1 else scale_y_start + scale_height
                draw.rectangle(
                    [scale_x, y, scale_x + scale_width, y_next],
                    fill=color
                )
        
        # Add labels
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
            small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Title
        title = f"Date: {date}"
        draw.text((padding, 5), title, fill='black', font=font)
        
        # Labels for each graph
        draw.text((orig_x_start, padding + 10), "Original", fill='black', font=small_font)
        draw.text((pred_x_start, padding + 10), "Predicted", fill='black', font=small_font)
        
        # Draw error map if available (simple grid with numbers)
        # Error map spans the full width of both original and predicted images
        if error_map is not None:
            error_y_start = orig_y_start + graph_height + error_map_gap
            
            # Use larger cell size for error map to make numbers clearly visible
            error_cell_size = 20  # Larger cells for better readability
            
            # Load font for error map (larger size for readability)
            try:
                error_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
            except:
                try:
                    error_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 10)
                except:
                    error_font = ImageFont.load_default()
            
            # Error map width spans both graphs (original + predicted)
            error_map_width = graph_width * 2 + gap
            # Calculate how many cells fit in the error map width
            error_cells_per_graph = num_freq_bands
            error_total_cells = error_cells_per_graph * 2  # Span both graphs
            
            # Draw grid background (white cells with borders)
            # Each cell in error map corresponds to a cell in the graphs
            for row_idx in range(num_hours):
                y = error_y_start + row_idx * error_cell_size
                for col_idx in range(error_total_cells):
                    x = orig_x_start + col_idx * error_cell_size
                    
                    # Only draw if within error map width
                    if x + error_cell_size <= orig_x_start + error_map_width:
                        # Get error from the corresponding position
                        # Map column index back to original frequency band
                        freq_col_idx = col_idx % num_freq_bands
                        error_pct = error_map[row_idx, freq_col_idx]
                        
                        # Draw white cell with light gray border
                        draw.rectangle(
                            [x, y, x + error_cell_size, y + error_cell_size],
                            fill='white',
                            outline='lightgray',
                            width=1
                        )
                        
                        # Draw error as text for every cell (with sign)
                        if not np.isnan(error_pct):
                            # Show signed error with + or - sign
                            if error_pct >= 0:
                                error_text = f"+{error_pct:.1f}"
                            else:
                                error_text = f"{error_pct:.1f}"  # Negative already has - sign
                            
                            # Center text in cell
                            # Get text size to center properly
                            try:
                                bbox = draw.textbbox((0, 0), error_text, font=error_font)
                                text_width = bbox[2] - bbox[0]
                                text_height = bbox[3] - bbox[1]
                            except:
                                # Fallback if textbbox fails
                                text_width = 15
                                text_height = 10
                            
                            text_x = x + max(0, (error_cell_size - text_width) // 2)
                            text_y = y + max(0, (error_cell_size - text_height) // 2)
                            
                            try:
                                draw.text((text_x, text_y), error_text, fill='black', font=error_font)
                            except Exception as e:
                                # If font rendering fails, skip this text
                                pass
            
            # Update error map height based on new cell size
            error_map_height = num_hours * error_cell_size
            
            # Label for error map
            draw.text((orig_x_start, error_y_start - 15), "Error Map", fill='black', font=small_font)
            
            # Display error statistics horizontally (next to each other)
            stats_y = error_y_start + error_map_height + 5
            stats_x = orig_x_start
            stats_spacing = 30  # Space between each stat
            
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
                max_val = visualizer.input_max_value if visualizer.input_max_value is not None else 60.0
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
