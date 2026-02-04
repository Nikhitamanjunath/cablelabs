#!/usr/bin/env python3
"""
Analyze Script

Loads transformed data from the transform step and runs prediction, confidence,
error, and anomaly analysis. Produces an output file consumable by the visualize
module (same schema as before). Optional image visualization (original vs predicted,
error map, confidence map) can be enabled via config.
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field, ValidationError

# Ensure project root is on path when run as script
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from analyze.error import ErrorCalculator, ErrorConfig
from analyze.anomaly import AnomalyDetector, AnomalyConfig
from analyze.confidence import ConfidenceCalculator, ConfidenceConfig
from analyze.prediction import Predictor, PredictionConfig
from analyze.image_visualizer import ImageVisualizer, ImageVisualizerConfig
from analyze.online_learning import OnlineLearner, OnlineLearningConfig


class AnalyzeConfig(BaseModel):
    """Analysis configuration (paths + flags + sub-configs via flat keys)."""
    transformed_data_file: str = Field(..., description="Path to transformed data file from transform step")
    preprocessed_folder: str = Field(default="data/preprocessed", description="Path to preprocessed images folder")
    output_folder: str = Field(default="data/predictions", description="Path to output folder")
    prediction_data_file: str = Field(
        default="data/predictions/predictions_data.parquet",
        description="Path to save prediction data for visualize module",
    )
    lookback_days: int = Field(default=6, ge=1, description="Number of previous days for prediction")
    prediction_method: str = Field(
        default="exponential_smoothing",
        description="Prediction method: linear, exponential_smoothing, moving_average",
    )
    visualize_predictions: bool = Field(default=True, description="Enable PNG visualization of predictions")

    # Prediction hyperparameters
    peak_threshold_multiplier: float = Field(default=1.5, ge=0.0)
    volatility_threshold: float = Field(default=0.35, ge=0.0)
    peak_detection_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    min_lookback_for_peak: int = Field(default=4, ge=1)
    max_lookback_for_peak: int = Field(default=6, ge=1)

    # Confidence hyperparameters
    confidence_data_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    confidence_variation_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    confidence_consistency_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    confidence_method_weight: float = Field(default=0.1, ge=0.0, le=1.0)
    confidence_accuracy_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    confidence_initial: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence_min_samples: int = Field(default=3, ge=1)
    confidence_accuracy_decay: float = Field(default=0.2, ge=0.0, le=1.0)
    confidence_high_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    confidence_low_threshold: float = Field(default=0.4, ge=0.0, le=1.0)

    # Error hyperparameters
    zero_error_threshold: float = Field(default=0.1, ge=0.0)

    # Anomaly hyperparameters
    anomaly_method: str = Field(default="z_score", description="z_score or iqr")
    anomaly_z_score_threshold: float = Field(default=2.5, ge=0.0)
    anomaly_min_samples: int = Field(default=5, ge=1)

    # Online learning (adaptive method and lookback)
    online_learning_state_file: str = Field(
        default="data/online_learning_state.json",
        description="Path to state file for online learning persistence",
    )
    online_learning_exploration_epsilon: float = Field(default=0.1, ge=0.0, le=1.0)
    online_learning_lookback_candidates: list = Field(
        default_factory=lambda: [4, 5, 6, 7],
        description="Lookback values to learn; empty to disable lookback learning",
    )

    # Blend and persistence-delta (used by analyze_blend.py and analyze_persistence_delta.py)
    blend_input_file: str = Field(
        default="data/time_series/tuned_blend_predictions.parquet",
        description="Tuned blend parquet from time_series notebook (input for analyze_blend)",
    )
    blend_output_file: str = Field(
        default="data/predictions_blend/predictions.parquet",
        description="Output parquet for blend predictions (for visualize)",
    )
    persistence_delta_output_file: str = Field(
        default="data/predictions_persistence_delta/predictions.parquet",
        description="Output parquet for persistence+hour-delta predictions (for visualize)",
    )


def load_config(config_path: Optional[str] = None) -> AnalyzeConfig:
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
    with open(config_file, "r") as f:
        config_data = yaml.safe_load(f)
    if not config_data:
        raise ValueError(f"Configuration file '{config_path}' is empty.")
    return AnalyzeConfig(**config_data)


def run(config: AnalyzeConfig) -> None:
    """Run analysis: load data, run predictor/confidence/error/anomaly, save output."""
    print("Loading config...")
    print("✓ Configuration loaded")

    print(f"\nLoading transformed data from: {config.transformed_data_file}")
    data_path = Path(config.transformed_data_file)
    if not data_path.exists():
        print(f"✗ Error: Transformed data file not found: {data_path}")
        return

    if data_path.suffix == ".parquet":
        try:
            df = pd.read_parquet(data_path)
        except ImportError:
            print("⚠ PyArrow not available, trying CSV...")
            csv_path = data_path.with_suffix(".csv")
            if csv_path.exists():
                df = pd.read_csv(csv_path)
            else:
                print("✗ Error: Neither Parquet nor CSV file found")
                return
    else:
        df = pd.read_csv(data_path)

    print(f"✓ Loaded data: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    df = df.sort_values(["date", "hour"]).reset_index(drop=True)
    dates = sorted(df["date"].unique())
    print(f"  Total dates: {len(dates)}")

    if len(dates) < config.lookback_days + 1:
        print(f"⚠ Warning: Need at least {config.lookback_days + 1} days, but only have {len(dates)}")

    # Build sub-configs and components
    error_config = ErrorConfig(zero_error_threshold=config.zero_error_threshold)
    anomaly_config = AnomalyConfig(
        anomaly_method=config.anomaly_method,
        anomaly_z_score_threshold=config.anomaly_z_score_threshold,
        anomaly_min_samples=config.anomaly_min_samples,
    )
    confidence_config = ConfidenceConfig(
        confidence_data_weight=config.confidence_data_weight,
        confidence_variation_weight=config.confidence_variation_weight,
        confidence_consistency_weight=config.confidence_consistency_weight,
        confidence_method_weight=config.confidence_method_weight,
        confidence_accuracy_weight=config.confidence_accuracy_weight,
        confidence_initial=config.confidence_initial,
        confidence_min_samples=config.confidence_min_samples,
        confidence_accuracy_decay=config.confidence_accuracy_decay,
        confidence_high_threshold=config.confidence_high_threshold,
        confidence_low_threshold=config.confidence_low_threshold,
    )
    prediction_config = PredictionConfig(
        lookback_days=config.lookback_days,
        prediction_method=config.prediction_method,
        peak_threshold_multiplier=config.peak_threshold_multiplier,
        volatility_threshold=config.volatility_threshold,
        peak_detection_threshold=config.peak_detection_threshold,
        min_lookback_for_peak=config.min_lookback_for_peak,
        max_lookback_for_peak=config.max_lookback_for_peak,
    )

    error_calculator = ErrorCalculator(error_config)
    anomaly_detector = AnomalyDetector(anomaly_config)
    confidence_calculator = ConfidenceCalculator(confidence_config, config.lookback_days)
    predictor = Predictor(prediction_config, confidence_calculator)

    online_learning_config = OnlineLearningConfig(
        state_file=config.online_learning_state_file,
        exploration_epsilon=config.online_learning_exploration_epsilon,
        lookback_candidates=config.online_learning_lookback_candidates,
        default_lookback_days=config.lookback_days,
        default_prediction_method=config.prediction_method,
    )
    online_learner = OnlineLearner(online_learning_config)

    image_visualizer: Optional[ImageVisualizer] = None
    if config.visualize_predictions:
        img_config = ImageVisualizerConfig(
            preprocessed_folder=config.preprocessed_folder,
            output_folder=config.output_folder,
            confidence_high_threshold=config.confidence_high_threshold,
            confidence_low_threshold=config.confidence_low_threshold,
        )
        image_visualizer = ImageVisualizer(img_config, error_calculator, project_root)

    all_prediction_data = []
    predictions_made = 0
    max_lookback = max(
        config.lookback_days,
        max(config.online_learning_lookback_candidates or [config.lookback_days]),
    )

    for i in range(max_lookback, len(dates)):
        target_date = dates[i]
        print(f"\n{'='*60}")
        print(f"Predicting for date: {target_date}")
        print(f"{'='*60}")

        lookback_used = online_learner.get_lookback_days()
        historical_dates = dates[max(0, i - lookback_used) : i]
        historical_data = df[df["date"].isin(historical_dates)].copy()
        print(f"  Using historical data from {len(historical_dates)} day(s): {historical_dates[0]} to {historical_dates[-1]} (lookback={lookback_used})")

        result = predictor.predict_next_day(historical_data, online_learner=online_learner)
        predicted_df = result[0]
        confidence_df = result[1]
        method_df = result[2] if len(result) > 2 else None
        all_predictions = result[3] if len(result) > 3 else None

        predicted_df.insert(0, "date", target_date)
        confidence_df.insert(0, "date", target_date)
        if method_df is not None:
            method_df.insert(0, "date", target_date)

        actual_data = df[df["date"] == target_date].copy()
        actual_df = actual_data if len(actual_data) > 0 else None

        if actual_df is not None:
            print("  ✓ Actual data available for comparison")
            confidence_calculator.update_accuracy_history(predicted_df, actual_df)
            if all_predictions is not None:
                day_accuracies = []
                actual_sorted = actual_df.sort_values("hour").reset_index(drop=True)
                pred_sorted = predicted_df.sort_values("hour").reset_index(drop=True)
                for row_idx in range(min(len(pred_sorted), len(actual_sorted))):
                    pred_row = pred_sorted.iloc[row_idx]
                    actual_row = actual_sorted.iloc[row_idx]
                    hour = int(pred_row["hour"])
                    for freq_col in [c for c in predicted_df.columns if c not in ["date", "hour"]]:
                        actual_val = actual_row.get(freq_col)
                        if actual_val is None or np.isnan(actual_val):
                            continue
                        cell_preds = all_predictions.get((hour, freq_col), {})
                        for method_name, pred_val in cell_preds.items():
                            if pred_val is not None and not np.isnan(pred_val):
                                online_learner.record_result(hour, freq_col, method_name, float(actual_val), float(pred_val))
                        pred_val = pred_row.get(freq_col)
                        if pred_val is not None and not np.isnan(pred_val):
                            acc = 1.0 - error_calculator.normalized_error(float(actual_val), float(pred_val))
                            day_accuracies.append(max(0.0, min(1.0, acc)))
                if day_accuracies:
                    online_learner.record_lookback_result(lookback_used, sum(day_accuracies) / len(day_accuracies))
            online_learner.save()
            print("  ✓ Updated accuracy history for online learning")
        else:
            print(f"  ⚠ No actual data available for {target_date}")

        if image_visualizer is not None:
            print("  Loading color scale from preprocessed image...")
            color_scale = image_visualizer.get_scale_for_date(target_date)
            if color_scale is None:
                print("  ⚠ Warning: Could not load color scale, skipping visualization")
            else:
                print("  ✓ Color scale loaded")
                print("  Generating visualization (original vs predicted + error map + confidence map)...")
                output_path, mean_error, max_error, min_error, zero_error_percentage = (
                    image_visualizer.generate_prediction_image(
                        predicted_df, actual_df, target_date, color_scale, confidence_df
                    )
                )
                print(f"  ✓ Visualization saved: {output_path}")
                if mean_error is not None:
                    print(f"    Mean Error: {mean_error:.2f}")
                if max_error is not None:
                    print(f"    Max Error: {max_error:.2f}")
                if min_error is not None:
                    print(f"    Min Error: {min_error:.2f}")
                if zero_error_percentage is not None:
                    print(f"    Zero Error Cells: {zero_error_percentage:.1f}%")

        freq_columns = [col for col in predicted_df.columns if col not in ["date", "hour"]]
        pred_sorted = predicted_df.sort_values("hour").reset_index(drop=True)
        conf_sorted = confidence_df.sort_values("hour").reset_index(drop=True)
        method_sorted = method_df.sort_values("hour").reset_index(drop=True) if method_df is not None else None
        historical_data_for_anomaly = df[df["date"].isin(historical_dates)].copy()

        for row_idx in range(len(pred_sorted)):
            pred_row = pred_sorted.iloc[row_idx]
            conf_row = conf_sorted.iloc[row_idx]
            method_row = method_sorted.iloc[row_idx] if method_sorted is not None else None
            date_val = pred_row["date"]
            hour = int(pred_row["hour"])

            for freq_col in freq_columns:
                predicted_val = pred_row.get(freq_col)
                confidence_val = conf_row.get(freq_col)
                prediction_method_used = (
                    method_row.get(freq_col, config.prediction_method)
                    if method_row is not None
                    else config.prediction_method
                )

                is_anomaly = False
                if predicted_val is not None and not np.isnan(predicted_val):
                    hour_historical = historical_data_for_anomaly[
                        historical_data_for_anomaly["hour"] == hour
                    ]
                    if len(hour_historical) > 0:
                        historical_values = hour_historical[freq_col].values
                        is_anomaly = anomaly_detector.detect(historical_values, float(predicted_val))

                actual_val = None
                error_val = None
                if actual_df is not None:
                    actual_sorted = actual_df.sort_values("hour").reset_index(drop=True)
                    if row_idx < len(actual_sorted):
                        actual_row = actual_sorted.iloc[row_idx]
                        actual_val = actual_row.get(freq_col)
                        if (
                            predicted_val is not None
                            and not np.isnan(predicted_val)
                            and actual_val is not None
                            and not np.isnan(actual_val)
                        ):
                            error_val = error_calculator.signed_error(float(actual_val), float(predicted_val))

                all_prediction_data.append({
                    "date": date_val,
                    "hour": hour,
                    "frequency_band": freq_col,
                    "predicted": float(predicted_val) if predicted_val is not None and not np.isnan(predicted_val) else None,
                    "confidence": float(confidence_val) if confidence_val is not None and not np.isnan(confidence_val) else None,
                    "actual": float(actual_val) if actual_val is not None and not np.isnan(actual_val) else None,
                    "error": float(error_val) if error_val is not None and not np.isnan(error_val) else None,
                    "is_anomaly": is_anomaly,
                    "prediction_method": prediction_method_used if isinstance(prediction_method_used, str) else config.prediction_method,
                    "lookback_days": lookback_used,
                })

        predictions_made += 1

    if all_prediction_data:
        print(f"\n{'='*60}")
        print("Saving prediction data for interactive visualization...")
        prediction_data_df = pd.DataFrame(all_prediction_data)
        data_file_path = Path(config.prediction_data_file)
        data_file_path.parent.mkdir(parents=True, exist_ok=True)
        if data_file_path.suffix == ".parquet":
            try:
                prediction_data_df.to_parquet(data_file_path, index=False)
                print(f"✓ Prediction data saved: {data_file_path}")
            except ImportError:
                print("⚠ PyArrow not available, saving as CSV instead...")
                csv_path = data_file_path.with_suffix(".csv")
                prediction_data_df.to_csv(csv_path, index=False)
                print(f"✓ Prediction data saved: {csv_path}")
        elif data_file_path.suffix == ".json":
            prediction_data_df.to_json(data_file_path, orient="records", indent=2)
            print(f"✓ Prediction data saved: {data_file_path}")
        else:
            csv_path = data_file_path.with_suffix(".csv")
            prediction_data_df.to_csv(csv_path, index=False)
            print(f"✓ Prediction data saved: {csv_path}")

    print(f"\n{'='*60}")
    print(f"✓ Analysis complete: {predictions_made} prediction(s) made")


def main() -> None:
    """Main entry point."""
    try:
        config = load_config()
        run(config)
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
