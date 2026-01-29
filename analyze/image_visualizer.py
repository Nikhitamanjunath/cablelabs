"""
Image visualization for prediction analysis.

Generates side-by-side images (original vs predicted, error map, confidence map)
using ColorScale and transform module. Uses ErrorCalculator for error map and stats.
"""

from pathlib import Path
from .error import ErrorCalculator
from typing import Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import cv2

# Constants (matching original predict module)
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
SCALE_RIGHT_START_RATIO = 0.85
SCALE_MARGIN_RATIO = 0.05


def _get_transform_module(project_root: Path):
    """Load transform module from project."""
    import importlib.util
    transform_path = project_root / "transform"
    spec = importlib.util.spec_from_file_location(
        "transform_module", transform_path / "transform.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ImageVisualizerConfig:
    """Config needed for image visualizer (paths + confidence thresholds for colors)."""
    def __init__(
        self,
        preprocessed_folder: str = "data/preprocessed",
        output_folder: str = "data/predictions",
        confidence_high_threshold: float = 0.7,
        confidence_low_threshold: float = 0.4,
    ):
        self.preprocessed_folder = preprocessed_folder
        self.output_folder = output_folder
        self.confidence_high_threshold = confidence_high_threshold
        self.confidence_low_threshold = confidence_low_threshold


class ImageVisualizer:
    """
    Visualizes predictions by generating images (original vs predicted, error map, confidence map).
    Uses ErrorCalculator for error map and statistics.
    """

    def __init__(
        self,
        config: ImageVisualizerConfig,
        error_calculator: ErrorCalculator,
        project_root: Optional[Path] = None,
    ):
        self.config = config
        self.error_calculator = error_calculator
        self.output_path = Path(config.output_folder)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.preprocessed_path = Path(config.preprocessed_folder)
        self.input_scale_colors: Optional[list] = None
        self.input_max_value: Optional[float] = None

        root = project_root or Path(__file__).resolve().parent.parent
        if str(root) not in __import__("sys").path:
            __import__("sys").path.insert(0, str(root))
        from color_scale import ColorScale
        self._ColorScale = ColorScale
        self._color_scale_fallback = ColorScale()

        transform_module = _get_transform_module(root)
        self.ImageTransformer = transform_module.ImageTransformer
        self.TransformConfig = transform_module.TransformConfig

    def get_scale_for_date(self, date: str) -> Optional[Any]:
        """Get ColorScale for a given date by extracting scale from preprocessed image."""
        filename = date.replace("-", "_") + ".png"
        image_path = self.preprocessed_path / filename
        if not image_path.exists():
            print(f"  Warning: Preprocessed image not found for {date}: {filename}")
            return None
        try:
            scale = self._ColorScale()
            print("  ✓ Loaded reference color scale from data/scale/scale.png")
        except Exception as e:
            print(f"  ✗ Error loading reference scale: {e}")
            return None

        transform_config = self.TransformConfig(
            from_date=date,
            to_date=date,
            scale_debug=False,
            debug_dataframe=False,
            frequency_start=3.1,
            frequency_end=3.5,
            preprocessed_folder=self.config.preprocessed_folder,
            output_folder=str(self.output_path / "temp"),
        )
        transformer = self.ImageTransformer(transform_config)
        success = transformer.process_image(image_path)
        if success and transformer.scale:
            self.input_scale_colors = transformer.input_scale_colors
            self.input_max_value = transformer.input_max_value
            if self.input_scale_colors is None or len(self.input_scale_colors) == 0:
                print(f"  Warning: Scale colors are empty for {date}")
                return None
            print(f"  ✓ Extracted scale from image (max_value: {self.input_max_value})")
            return scale
        print(f"  Warning: Failed to process image for {date}")
        return None

    def load_original_image(self, date: str) -> Optional[np.ndarray]:
        """Load original preprocessed image for a date."""
        filename = date.replace("-", "_") + ".png"
        image_path = self.preprocessed_path / filename
        if not image_path.exists():
            return None
        img = Image.open(image_path)
        return np.array(img)

    def extract_graph_region(self, img_array: np.ndarray) -> np.ndarray:
        """Extract graph region from image (24x70 grid)."""
        height, width = img_array.shape[:2]
        left = max(0, min(GRAPH_LEFT, width))
        top = max(0, min(GRAPH_TOP, height))
        right = max(left, min(GRAPH_LEFT + GRAPH_WIDTH, width))
        bottom = max(top, min(GRAPH_TOP + GRAPH_HEIGHT, height))
        graph_region = img_array[top:bottom, left:right]
        graph_height, graph_width = graph_region.shape[:2]
        cell_height = graph_height / NUM_ROWS
        cell_width = graph_width / NUM_COLS
        resized_graph = np.zeros((NUM_ROWS, NUM_COLS, 3), dtype=np.uint8)
        for row_idx in range(NUM_ROWS):
            y_in_graph = min(int(row_idx * cell_height + cell_height / 2), graph_height - 1)
            for col_idx in range(NUM_COLS):
                x_in_graph = min(int(col_idx * cell_width + cell_width / 2), graph_width - 1)
                resized_graph[row_idx, col_idx] = graph_region[y_in_graph, x_in_graph]
        return resized_graph

    def _detect_scale_region(self, img_array: np.ndarray) -> Dict[str, int]:
        """Detect scale region on the right side of the image."""
        height, width = img_array.shape[:2]
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
            "bottom_y": height - bottom_margin,
        }

    def extract_scale_region(self, img_array: np.ndarray) -> Optional[np.ndarray]:
        """Extract scale region from image."""
        try:
            d = self._detect_scale_region(img_array)
            return img_array[d["top_y"]:d["bottom_y"], d["left_x"]:d["right_x"]]
        except Exception as e:
            print(f"  Warning: Failed to extract scale region: {e}")
            return None

    def _get_min_color(self, color_scale: Any) -> Tuple[int, int, int]:
        """Get minimum (darkest) color from scale."""
        scale_colors = color_scale.get_scale_colors()
        if not scale_colors:
            return (0, 0, 0)
        min_color = tuple(scale_colors[-1])
        min_brightness = sum(float(c) for c in min_color) / 3.0
        if min_brightness > 200:
            darkest = min_color
            darkest_brightness = min_brightness
            for sc in scale_colors:
                b = sum(float(c) for c in sc) / 3.0
                if b < darkest_brightness:
                    darkest_brightness = b
                    darkest = tuple(sc)
            return darkest
        return min_color

    def _get_color_for_value(self, value: float, color_scale: Any, max_value: float) -> Tuple[int, int, int]:
        """Convert value to color."""
        min_color = self._get_min_color(color_scale)
        clamped_value = max(0.0, min(max_value, float(value)))
        try:
            if self.input_scale_colors and len(self.input_scale_colors) > 0:
                color = self._number_to_color_using_input_scale(
                    clamped_value, self.input_scale_colors, max_value
                )
            else:
                color = color_scale.number_to_color(clamped_value, max_value=max_value)
        except Exception:
            return min_color
        is_white = all(c > 250 for c in color)
        brightness = sum(float(c) for c in color) / 3.0
        is_too_light = brightness > (150 if value < 5.0 else 230)
        return min_color if (is_white or is_too_light) else color

    def _number_to_color_using_input_scale(
        self, value: float, input_scale_colors: list, max_value: float
    ) -> Tuple[int, int, int]:
        """Convert number to color using input scale colors."""
        if not input_scale_colors or len(input_scale_colors) == 0:
            return self._color_scale_fallback.number_to_color(value, max_value=max_value)
        value = max(0.0, min(max_value, float(value)))
        value_ratio = value / max_value if max_value > 0 else 0.0
        position_ratio = max(0.0, min(1.0, 1.0 - value_ratio))
        num_colors = len(input_scale_colors)
        position_idx = position_ratio * (num_colors - 1)
        idx1 = int(np.clip(position_idx, 0, num_colors - 1))
        idx2 = min(idx1 + 1, num_colors - 1)
        frac = max(0.0, min(1.0, position_idx - idx1))
        color1 = np.array(input_scale_colors[idx1], dtype=np.float64)
        color2 = np.array(input_scale_colors[idx2], dtype=np.float64)
        interpolated = color1 + (color2 - color1) * frac
        interpolated = np.clip(interpolated, 0, 255)
        return tuple(interpolated.astype(np.uint8))

    def generate_prediction_image(
        self,
        predicted_df: pd.DataFrame,
        actual_df: Optional[pd.DataFrame],
        date: str,
        color_scale: Any,
        confidence_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[Path, Optional[float], Optional[float], Optional[float], Optional[float]]:
        """
        Generate image: original, predicted, error map, confidence matrix.
        Uses ErrorCalculator for error map and stats.
        Returns (output_path, mean_error, max_error, min_error, zero_error_percentage).
        """
        freq_columns = [col for col in predicted_df.columns if col not in ['date', 'hour']]
        num_freq_bands = len(freq_columns)
        num_hours = len(predicted_df)
        predicted_df_sorted = predicted_df.sort_values('hour', ascending=False).reset_index(drop=True)

        error_map = None
        error_map_height = 0
        mean_error = None
        max_error = None
        min_error = None
        zero_error_percentage = None

        if actual_df is not None:
            mean_error, max_error, min_error, zero_error_percentage, error_map = (
                self.error_calculator.compute_map_stats(
                    predicted_df, actual_df, freq_columns
                )
            )
            if error_map is not None:
                error_map_height = num_hours * ERROR_CELL_SIZE

        graph_width = num_freq_bands * CELL_SIZE
        graph_height = num_hours * CELL_SIZE
        error_map_gap = ERROR_MAP_GAP if error_map is not None else 0
        confidence_map_gap = ERROR_MAP_GAP if confidence_df is not None else 0
        confidence_map_height_est = (num_hours * ERROR_CELL_SIZE) if confidence_df is not None else 0
        stats_height = 30 if (error_map is not None or confidence_df is not None) else 0
        stats_gap_extra = 20 if (error_map is not None and confidence_df is not None) else 0
        total_height = (
            graph_height + (num_hours * ERROR_CELL_SIZE if error_map is not None else 0)
            + confidence_map_height_est + stats_height + stats_gap_extra
            + PADDING * 3 + 30 + error_map_gap + confidence_map_gap
        )

        effective_max = self.input_max_value if self.input_max_value is not None else DEFAULT_MAX_VALUE
        original_img_array = self.load_original_image(date)
        actual_scale_width = SCALE_WIDTH
        scale_region = None
        if original_img_array is not None:
            scale_region = self.extract_scale_region(original_img_array)
            if scale_region is not None:
                actual_scale_width = scale_region.shape[1]

        total_width = graph_width * 2 + actual_scale_width + GAP + SCALE_GAP + PADDING * 2
        img = Image.new('RGB', (total_width, int(total_height)), color='white')
        draw = ImageDraw.Draw(img)
        orig_x_start = PADDING
        orig_y_start = PADDING + 25

        if original_img_array is not None:
            original_graph = self.extract_graph_region(original_img_array)
            transform_config = self.TransformConfig(
                from_date=date, to_date=date, scale_debug=False, debug_dataframe=False,
                frequency_start=3.1, frequency_end=3.5,
                preprocessed_folder=self.config.preprocessed_folder,
                output_folder=str(self.output_path / "temp"),
            )
            orig_transformer = self.ImageTransformer(transform_config)
            orig_image_path = self.preprocessed_path / f"{date.replace('-', '_')}.png"
            if orig_image_path.exists():
                orig_transformer.process_image(orig_image_path)
            for row_idx in range(num_hours):
                y = orig_y_start + row_idx * CELL_SIZE
                for col_idx in range(num_freq_bands):
                    x = orig_x_start + col_idx * CELL_SIZE
                    raw_color = tuple(original_graph[row_idx, col_idx])
                    if orig_transformer.scale and getattr(orig_transformer, 'input_scale_colors', None) and getattr(orig_transformer, 'input_max_value', None):
                        value = orig_transformer.get_color_value(raw_color) or 0.0
                        clamped_value = max(0.0, min(effective_max, float(value)))
                        if self.input_scale_colors and len(self.input_scale_colors) > 0:
                            color = self._number_to_color_using_input_scale(clamped_value, self.input_scale_colors, effective_max)
                        else:
                            color = color_scale.number_to_color(clamped_value, max_value=effective_max)
                    else:
                        color = raw_color
                    draw.rectangle([x, y, x + CELL_SIZE, y + CELL_SIZE], fill=color)
        else:
            draw.rectangle(
                [orig_x_start, orig_y_start, orig_x_start + graph_width, orig_y_start + graph_height],
                fill='lightgray', outline='black', width=2,
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
        if scale_region is None and original_img_array is not None:
            scale_region = self.extract_scale_region(original_img_array)
        if scale_region is not None:
            scale_img = Image.fromarray(scale_region)
            scale_region_width = scale_region.shape[1]
            scale_img_resized = scale_img.resize((scale_region_width, graph_height), Image.Resampling.LANCZOS)
            img.paste(scale_img_resized, (scale_x, scale_y_start))
        else:
            num_scale_samples = max(200, graph_height)
            for i in range(num_scale_samples):
                y_ratio = i / (num_scale_samples - 1) if num_scale_samples > 1 else 0.0
                y = int(scale_y_start + y_ratio * graph_height)
                value = (1.0 - y_ratio) * effective_max
                color = self._get_color_for_value(value, color_scale, effective_max)
                y_next = int(scale_y_start + ((i + 1) / num_scale_samples) * graph_height) if i < num_scale_samples - 1 else scale_y_start + graph_height
                draw.rectangle([scale_x, y, scale_x + SCALE_WIDTH, y_next], fill=color)

        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
            small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
        except Exception:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()

        draw.text((PADDING, 5), f"Date: {date}", fill='black', font=font)
        draw.text((orig_x_start, PADDING + 10), "Original", fill='black', font=small_font)
        draw.text((pred_x_start, PADDING + 10), "Predicted", fill='black', font=small_font)

        if error_map is not None:
            error_y_start = orig_y_start + graph_height + error_map_gap
            try:
                error_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
            except Exception:
                try:
                    error_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 10)
                except Exception:
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
                        draw.rectangle([x, y, x + ERROR_CELL_SIZE, y + ERROR_CELL_SIZE], fill='white', outline='lightgray', width=1)
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
            draw.text((orig_x_start, error_y_start - 15), "Error Map", fill='black', font=small_font)

            if confidence_df is None:
                stats_y = error_y_start + error_map_height + 5
                stats_x = orig_x_start
                stats_spacing = 30
                if mean_error is not None:
                    draw.text((stats_x, stats_y), f"Mean: {mean_error:.2f}", fill='black', font=font)
                    stats_x += 80 + stats_spacing
                if max_error is not None:
                    draw.text((stats_x, stats_y), f"Max: {max_error:.2f}", fill='black', font=font)
                    stats_x += 80 + stats_spacing
                if min_error is not None:
                    draw.text((stats_x, stats_y), f"Min: {min_error:.2f}", fill='black', font=font)
                    stats_x += 80 + stats_spacing
                if zero_error_percentage is not None:
                    draw.text((stats_x, stats_y), f"Zero Error: {zero_error_percentage:.1f}%", fill='black', font=font)

        if confidence_df is not None:
            confidence_df_sorted = confidence_df.sort_values('hour', ascending=False).reset_index(drop=True)
            if error_map is not None:
                confidence_y_start = orig_y_start + graph_height + error_map_height + error_map_gap + confidence_map_gap
            else:
                confidence_y_start = orig_y_start + graph_height + confidence_map_gap
            try:
                confidence_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
            except Exception:
                confidence_font = ImageFont.load_default()
            high_t = self.config.confidence_high_threshold
            low_t = self.config.confidence_low_threshold
            for row_idx in range(num_hours):
                y = confidence_y_start + row_idx * ERROR_CELL_SIZE
                for col_idx in range(num_freq_bands * 2):
                    x = orig_x_start + col_idx * ERROR_CELL_SIZE
                    if x + ERROR_CELL_SIZE <= orig_x_start + graph_width * 2 + GAP:
                        freq_col_idx = col_idx % num_freq_bands
                        if row_idx < len(confidence_df_sorted):
                            conf_row = confidence_df_sorted.iloc[row_idx]
                            freq_col = freq_columns[freq_col_idx]
                            confidence = conf_row.get(freq_col, 0.0)
                            if not np.isnan(confidence):
                                if confidence >= high_t:
                                    bg_color = (200, 255, 200)
                                elif confidence >= low_t:
                                    bg_color = (255, 255, 200)
                                else:
                                    bg_color = (255, 200, 200)
                                draw.rectangle([x, y, x + ERROR_CELL_SIZE, y + ERROR_CELL_SIZE], fill=bg_color, outline='lightgray', width=1)
                                conf_text = f"{confidence * 100:.0f}%"
                                try:
                                    bbox = draw.textbbox((0, 0), conf_text, font=confidence_font)
                                    text_x = x + max(0, (ERROR_CELL_SIZE - (bbox[2] - bbox[0])) // 2)
                                    text_y = y + max(0, (ERROR_CELL_SIZE - (bbox[3] - bbox[1])) // 2)
                                    draw.text((text_x, text_y), conf_text, fill='black', font=confidence_font)
                                except Exception:
                                    pass
                            else:
                                draw.rectangle([x, y, x + ERROR_CELL_SIZE, y + ERROR_CELL_SIZE], fill='lightgray', outline='lightgray', width=1)
                        else:
                            draw.rectangle([x, y, x + ERROR_CELL_SIZE, y + ERROR_CELL_SIZE], fill='lightgray', outline='lightgray', width=1)
            draw.text((orig_x_start, confidence_y_start - 15), "Confidence Map", fill='black', font=small_font)

            stats_gap = 25
            if error_map is not None:
                stats_y = confidence_y_start + num_hours * ERROR_CELL_SIZE + stats_gap
                stats_x = orig_x_start
                if mean_error is not None:
                    draw.text((stats_x, stats_y), f"Mean Error: {mean_error:.2f}", fill='black', font=font)
                    stats_x += 100 + 40
                if max_error is not None:
                    draw.text((stats_x, stats_y), f"Max Error: {max_error:.2f}", fill='black', font=font)
                    stats_x += 100 + 40
                if min_error is not None:
                    draw.text((stats_x, stats_y), f"Min Error: {min_error:.2f}", fill='black', font=font)
                    stats_x += 100 + 40
                if zero_error_percentage is not None:
                    draw.text((stats_x, stats_y), f"Zero Error: {zero_error_percentage:.1f}%", fill='black', font=font)

        output_file = self.output_path / f"prediction_{date.replace('-', '_')}.png"
        img.save(output_file)
        return output_file, mean_error, max_error, min_error, zero_error_percentage
