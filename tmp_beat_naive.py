"""
Experiment: try to beat naive 1-step-ahead MAE on cablelabs time series.
Uses same data and test window as the notebook. Runs several candidate models.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "transformed" / "transformed_data.parquet"
if not DATA_PATH.exists():
    DATA_PATH = DATA_PATH.with_suffix(".csv")
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Data not found at {DATA_PATH}")

df = pd.read_parquet(DATA_PATH) if DATA_PATH.suffix == ".parquet" else pd.read_csv(DATA_PATH)
df = df.sort_values(["date", "hour"]).reset_index(drop=True)
df["datetime"] = pd.to_datetime(df["date"]) + pd.to_timedelta(df["hour"], unit="h")
freq_bands = sorted([c for c in df.columns if c not in ("date", "hour", "datetime")])

TEST_STEPS = 24 * 365
# With 9192 rows we have 432 hours before test; use 14 days validation so test+val+24 <= 9192
VAL_STEPS = 24 * 14
TRAIN_MIN = 24 * 14

def _series_for_freq(freq):
    """One series per band: datetime index, values. Drops NA."""
    sub = pd.DataFrame({
        "datetime": pd.to_datetime(df["date"]) + pd.to_timedelta(df["hour"], unit="h"),
        "value": df[freq],
    }).dropna(subset=["value"]).sort_values("datetime").reset_index(drop=True)
    return sub

def run_naive():
    """Baseline: ŷ(t) = y(t-1). Returns MAE and predictions list for alignment."""
    rows = []
    for freq in freq_bands:
        sub = _series_for_freq(freq)
        if len(sub) < TEST_STEPS + 1:
            continue
        vals = sub["value"].values
        dts = sub["datetime"].values
        for i in range(-TEST_STEPS, 0):
            rows.append({"datetime": dts[i], "frequency_band": freq, "actual": float(vals[i]), "predicted": float(vals[i - 1])})
    pred_df = pd.DataFrame(rows)
    mae = (pred_df["actual"] - pred_df["predicted"]).abs().mean()
    return mae, pred_df

def run_tuned_blend():
    """ŷ = α*y(t-1) + (1-α)*y(t-24). Tune α per band on validation (last VAL_STEPS of train)."""
    rows = []
    for freq in freq_bands:
        sub = _series_for_freq(freq)
        n = len(sub)
        if n < TEST_STEPS + TRAIN_MIN + 24:
            continue
        dts = sub["datetime"].values
        vals = sub["value"].values.astype(float)
        # Validation: indices [n - TEST_STEPS - VAL_STEPS : n - TEST_STEPS]
        val_start = n - TEST_STEPS - VAL_STEPS
        val_end = n - TEST_STEPS
        if val_end - 24 <= val_start:
            continue
        best_mae = np.inf
        best_alpha = 1.0
        for alpha in np.linspace(0.5, 1.0, 51):  # 0.5 to 1.0 (naive-heavy)
            preds = alpha * vals[val_start + 23 : val_end - 1] + (1 - alpha) * vals[val_start : val_end - 24]
            actuals = vals[val_start + 24 : val_end]
            mae = np.mean(np.abs(actuals - preds))
            if mae < best_mae:
                best_mae = mae
                best_alpha = alpha
        for i in range(-TEST_STEPS, 0):
            pred = best_alpha * vals[i - 1] + (1 - best_alpha) * vals[i - 24]
            rows.append({"datetime": dts[i], "frequency_band": freq, "actual": vals[i], "predicted": pred})
    pred_df = pd.DataFrame(rows)
    mae = (pred_df["actual"] - pred_df["predicted"]).abs().mean()
    return mae, pred_df

def run_per_hour_blend():
    """Per hour-of-day α_h: ŷ = α_h*y(t-1) + (1-α_h)*y(t-24). Tune 24 alphas per band on validation."""
    rows = []
    for freq in freq_bands:
        sub = _series_for_freq(freq)
        n = len(sub)
        if n < TEST_STEPS + TRAIN_MIN + 24:
            continue
        dts = sub["datetime"].values
        vals = sub["value"].values.astype(float)
        val_start = n - TEST_STEPS - VAL_STEPS
        val_end = n - TEST_STEPS
        if val_end - 24 <= val_start:
            continue
        # hour 0..23 from datetime; we need hour for each index
        hours = np.arange(len(sub)) % 24
        alphas = np.ones(24)
        for h in range(24):
            mask = (hours[val_start + 24 : val_end] == h)
            if mask.sum() < 10:
                alphas[h] = 1.0
                continue
            best_mae = np.inf
            best_a = 1.0
            for alpha in np.linspace(0.5, 1.0, 26):
                preds = alpha * vals[val_start + 23 : val_end - 1] + (1 - alpha) * vals[val_start : val_end - 24]
                actuals = vals[val_start + 24 : val_end]
                mae = np.mean(np.abs(actuals[mask] - preds[mask]))
                if mae < best_mae:
                    best_mae = mae
                    best_a = alpha
            alphas[h] = best_a
        for i in range(-TEST_STEPS, 0):
            h = (len(sub) + i) % 24
            pred = alphas[h] * vals[i - 1] + (1 - alphas[h]) * vals[i - 24]
            rows.append({"datetime": dts[i], "frequency_band": freq, "actual": vals[i], "predicted": pred})
    pred_df = pd.DataFrame(rows)
    mae = (pred_df["actual"] - pred_df["predicted"]).abs().mean()
    return mae, pred_df

def run_ridge_blend():
    """Ridge regression: ŷ = b0 + b1*y(t-1) + b2*y(t-24). Fit on train, predict test."""
    rows = []
    for freq in freq_bands:
        sub = _series_for_freq(freq)
        n = len(sub)
        if n < TEST_STEPS + TRAIN_MIN + 24:
            continue
        dts = sub["datetime"].values
        vals = sub["value"].values.astype(float)
        train_end = n - TEST_STEPS
        # Train: y = b0 + b1*x1 + b2*x24
        y_tr = vals[24:train_end]
        x1 = vals[23:train_end - 1]
        x24 = vals[0:train_end - 24]
        X_tr = np.column_stack([np.ones_like(y_tr), x1, x24])
        # Ridge: (X'X + λI)^{-1} X' y
        lam = 1e-4
        try:
            beta = np.linalg.solve(X_tr.T @ X_tr + lam * np.eye(3), X_tr.T @ y_tr)
        except np.linalg.LinAlgError:
            beta = np.array([0.0, 1.0, 0.0])
        for i in range(-TEST_STEPS, 0):
            pred = beta[0] + beta[1] * vals[i - 1] + beta[2] * vals[i - 24]
            rows.append({"datetime": dts[i], "frequency_band": freq, "actual": vals[i], "predicted": pred})
    pred_df = pd.DataFrame(rows)
    mae = (pred_df["actual"] - pred_df["predicted"]).abs().mean()
    return mae, pred_df

def run_three_lag_ridge():
    """Ridge: ŷ = b0 + b1*y(t-1) + b2*y(t-2) + b3*y(t-24). More lags."""
    rows = []
    for freq in freq_bands:
        sub = _series_for_freq(freq)
        n = len(sub)
        if n < TEST_STEPS + TRAIN_MIN + 24:
            continue
        dts = sub["datetime"].values
        vals = sub["value"].values.astype(float)
        train_end = n - TEST_STEPS
        y_tr = vals[24:train_end]
        x1 = vals[23:train_end - 1]
        x2 = vals[22:train_end - 2]
        x24 = vals[0:train_end - 24]
        X_tr = np.column_stack([np.ones_like(y_tr), x1, x2, x24])
        lam = 1e-3
        try:
            beta = np.linalg.solve(X_tr.T @ X_tr + lam * np.eye(4), X_tr.T @ y_tr)
        except np.linalg.LinAlgError:
            beta = np.array([0.0, 1.0, 0.0, 0.0])
        for i in range(-TEST_STEPS, 0):
            pred = beta[0] + beta[1] * vals[i - 1] + beta[2] * vals[i - 2] + beta[3] * vals[i - 24]
            rows.append({"datetime": dts[i], "frequency_band": freq, "actual": vals[i], "predicted": pred})
    pred_df = pd.DataFrame(rows)
    mae = (pred_df["actual"] - pred_df["predicted"]).abs().mean()
    return mae, pred_df

def run_tuned_blend_refined():
    """Same as tuned_blend but alpha only in [0.97, 0.98, 0.99, 1.0] (very naive-heavy)."""
    rows = []
    for freq in freq_bands:
        sub = _series_for_freq(freq)
        n = len(sub)
        if n < TEST_STEPS + TRAIN_MIN + 24:
            continue
        dts = sub["datetime"].values
        vals = sub["value"].values.astype(float)
        val_start = n - TEST_STEPS - VAL_STEPS
        val_end = n - TEST_STEPS
        if val_end - 24 <= val_start:
            continue
        best_mae = np.inf
        best_alpha = 1.0
        for alpha in np.linspace(0.97, 1.0, 31):
            preds = alpha * vals[val_start + 23 : val_end - 1] + (1 - alpha) * vals[val_start : val_end - 24]
            actuals = vals[val_start + 24 : val_end]
            mae = np.mean(np.abs(actuals - preds))
            if mae < best_mae:
                best_mae = mae
                best_alpha = alpha
        for i in range(-TEST_STEPS, 0):
            pred = best_alpha * vals[i - 1] + (1 - best_alpha) * vals[i - 24]
            rows.append({"datetime": dts[i], "frequency_band": freq, "actual": float(vals[i]), "predicted": float(pred)})
    pred_df = pd.DataFrame(rows)
    return (pred_df["actual"] - pred_df["predicted"]).abs().mean(), pred_df

def run_median_two_lag():
    """ŷ = median(y(t-1), y(t-2)). Robust to one lag being an outlier."""
    rows = []
    for freq in freq_bands:
        sub = _series_for_freq(freq)
        if len(sub) < TEST_STEPS + 2:
            continue
        dts = sub["datetime"].values
        vals = sub["value"].values.astype(float)
        for i in range(-TEST_STEPS, 0):
            pred = float(np.median([vals[i - 1], vals[i - 2]]))
            rows.append({"datetime": dts[i], "frequency_band": freq, "actual": float(vals[i]), "predicted": pred})
    pred_df = pd.DataFrame(rows)
    return (pred_df["actual"] - pred_df["predicted"]).abs().mean(), pred_df

def run_ols_val():
    """Find (a,b) in grid such that ŷ = a*y(t-1) + b*y(t-24) minimizes MAE on validation; then predict test."""
    rows = []
    for freq in freq_bands:
        sub = _series_for_freq(freq)
        n = len(sub)
        if n < TEST_STEPS + TRAIN_MIN + 24:
            continue
        dts = sub["datetime"].values
        vals = sub["value"].values.astype(float)
        val_start = n - TEST_STEPS - VAL_STEPS
        val_end = n - TEST_STEPS
        if val_end - 24 <= val_start:
            continue
        y_val = vals[val_start + 24 : val_end]
        x1_val = vals[val_start + 23 : val_end - 1]
        x24_val = vals[val_start : val_end - 24]
        best_mae = np.inf
        best_a, best_b = 1.0, 0.0
        for a in np.linspace(0.8, 1.0, 11):
            for b in np.linspace(-0.1, 0.2, 16):
                preds = a * x1_val + b * x24_val
                mae = np.mean(np.abs(y_val - preds))
                if mae < best_mae:
                    best_mae = mae
                    best_a, best_b = a, b
        for i in range(-TEST_STEPS, 0):
            pred = best_a * vals[i - 1] + best_b * vals[i - 24]
            rows.append({"datetime": dts[i], "frequency_band": freq, "actual": float(vals[i]), "predicted": float(pred)})
    pred_df = pd.DataFrame(rows)
    return (pred_df["actual"] - pred_df["predicted"]).abs().mean(), pred_df

def main():
    print("Loading data and running models (same test window as notebook)...")
    mae_naive, pred_naive = run_naive()
    print(f"Naive MAE: {mae_naive:.6f}  (n={len(pred_naive)})")

    mae_tuned, _ = run_tuned_blend()
    print(f"Tuned blend (α*y(t-1)+(1-α)*y(t-24)) MAE: {mae_tuned:.6f}  diff={mae_tuned - mae_naive:.6f}")

    mae_perhour, _ = run_per_hour_blend()
    print(f"Per-hour blend MAE: {mae_perhour:.6f}  diff={mae_perhour - mae_naive:.6f}")

    mae_ridge, _ = run_ridge_blend()
    print(f"Ridge (1,24) MAE: {mae_ridge:.6f}  diff={mae_ridge - mae_naive:.6f}")

    mae_3lag, _ = run_three_lag_ridge()
    print(f"Ridge (1,2,24) MAE: {mae_3lag:.6f}  diff={mae_3lag - mae_naive:.6f}")

    # Refined blend: alpha in [0.97, 1.0] only (very naive-heavy)
    mae_refined, pred_refined = run_tuned_blend_refined()
    print(f"Tuned blend refined (α in [0.97,1]) MAE: {mae_refined:.6f}  diff={mae_refined - mae_naive:.6f}")

    # Median of y(t-1) and y(t-2) - robust to one bad lag
    mae_med2, _ = run_median_two_lag()
    print(f"Median(y(t-1), y(t-2)) MAE: {mae_med2:.6f}  diff={mae_med2 - mae_naive:.6f}")

    # OLS on validation: min MAE by grid search over simple combos
    mae_ols_val, _ = run_ols_val()
    print(f"OLS-style weights (val MAE grid) MAE: {mae_ols_val:.6f}  diff={mae_ols_val - mae_naive:.6f}")

    best = min([
        ("naive", mae_naive),
        ("tuned_blend", mae_tuned),
        ("per_hour_blend", mae_perhour),
        ("ridge_blend", mae_ridge),
        ("ridge_3lag", mae_3lag),
        ("tuned_blend_refined", mae_refined),
        ("median_two_lag", mae_med2),
        ("ols_val", mae_ols_val),
    ], key=lambda x: x[1])
    print(f"\nBest: {best[0]} with MAE {best[1]:.6f}")
    if best[0] != "naive":
        print("We beat naive!")
    else:
        print("Naive still best.")

if __name__ == "__main__":
    main()
