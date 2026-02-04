#!/usr/bin/env python3
"""
Analyze Persistence + Hour Delta

Replicates the logic from time_series_14_persistence_plus_hour_delta.ipynb:
- Load transformed data, 60/10/30 train/val/test split, optional low-pass filter.
- Fit delta[j,h] = mean(y_{i,j} - y_{i-1,j}) over training steps i where hour(i) = h.
- Predict on test: hat(y)_{t,j} = y_{prev,j} + delta[j, hour(t)].
Outputs predictions in the visualize schema to data/predictions_persistence_delta/predictions.parquet.
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Ensure project root is on path when run as script
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from analyze.analyze import load_config

# Same values as time_series_14_persistence_plus_hour_delta.ipynb
TRAIN_RATIO = 0.6
VAL_RATIO = 0.1
TEST_RATIO = 0.3
USE_LOWPASS = True
CUTOFF = 0.5
FILTER_ORDER = 3
N_HOURS = 24


def _lowpass_per_channel(Z_mat: np.ndarray, cutoff: float = 0.5, order: int = 3) -> np.ndarray:
    from scipy.signal import butter, filtfilt
    normal_cutoff = min(float(cutoff), 0.99)
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    out = np.zeros_like(Z_mat)
    for j in range(Z_mat.shape[1]):
        out[:, j] = filtfilt(b, a, Z_mat[:, j])
    return out


def run(config=None) -> None:
    """Load data from config, fit persistence+delta, predict on test, save for visualize."""
    if config is None:
        config = load_config()
    data_path = Path(config.transformed_data_file)
    if not data_path.is_absolute():
        data_path = project_root / data_path
    if not data_path.exists():
        data_path = data_path.with_suffix(".csv")
    if not data_path.exists():
        print(f"✗ Error: Transform data not found at {data_path}. Run task transform first.")
        return

    output_file = Path(config.persistence_delta_output_file)
    if not output_file.is_absolute():
        output_file = project_root / output_file

    print(f"Loading transformed data from: {data_path}")
    df = pd.read_parquet(data_path) if data_path.suffix == ".parquet" else pd.read_csv(data_path)
    df = df.sort_values(["date", "hour"]).reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["date"]) + pd.to_timedelta(df["hour"], unit="h")

    freq_bands = sorted([c for c in df.columns if c not in ("date", "hour", "datetime")])
    N = len(freq_bands)
    Z = df[freq_bands].values.astype(np.float64)
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    hours = df["hour"].values.astype(int)
    T_total = Z.shape[0]

    n_train = int(T_total * TRAIN_RATIO)
    n_val = int(T_total * VAL_RATIO)
    n_test = T_total - n_train - n_val

    if USE_LOWPASS:
        Z_full = _lowpass_per_channel(Z, cutoff=CUTOFF, order=FILTER_ORDER)
        print(f"Applied low-pass filter (cutoff={CUTOFF}, order={FILTER_ORDER}).")
    else:
        Z_full = Z.copy()

    Z_train = Z_full[:n_train]
    Z_val = Z_full[n_train : n_train + n_val]
    Z_test = Z_full[n_train + n_val :]
    hours_train = hours[:n_train]
    test_start = n_train + n_val

    # Fit delta per (channel, hour): delta[j,h] = mean(y_{i,j} - y_{i-1,j}) over i where hour(i)=h
    delta = np.zeros((N, N_HOURS))
    for j in range(N):
        for h in range(N_HOURS):
            mask = hours_train[1:] == h
            if np.any(mask):
                diff = Z_train[1:, j] - Z_train[:-1, j]
                delta[j, h] = np.mean(diff[mask])

    # Predict on test: pred[t,j] = Z_full[prev_idx, j] + delta[j, h]
    pred_test = np.zeros((n_test, N), dtype=np.float64)
    for t in range(n_test):
        prev_idx = test_start + t - 1 if t > 0 else test_start - 1
        h = hours[test_start + t]
        if h < 0 or h >= N_HOURS:
            h = 0
        for j in range(N):
            pred_test[t, j] = Z_full[prev_idx, j] + delta[j, h]

    # Build output in visualize schema: date, hour, frequency_band, predicted, actual, error, confidence, is_anomaly
    test_df = df.iloc[test_start : test_start + n_test].reset_index(drop=True)
    rows = []
    for t in range(n_test):
        date_str = str(test_df.loc[t, "date"])
        if hasattr(test_df.loc[t, "date"], "strftime"):
            date_str = test_df.loc[t, "date"].strftime("%Y-%m-%d")
        hour_val = int(test_df.loc[t, "hour"])
        for j in range(N):
            actual_val = float(Z_test[t, j])
            pred_val = float(pred_test[t, j])
            rows.append({
                "date": date_str,
                "hour": hour_val,
                "frequency_band": freq_bands[j],
                "predicted": pred_val,
                "actual": actual_val,
                "error": actual_val - pred_val,
                "confidence": np.nan,
                "is_anomaly": False,
            })

    out = pd.DataFrame(rows)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_file, index=False)
    print(f"✓ Persistence+delta prediction data saved: {output_file}")
    print(f"  Rows: {len(out)}, dates: {out['date'].min()} to {out['date'].max()}")


def main() -> None:
    """Entry point."""
    try:
        config = load_config()
        run(config)
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
