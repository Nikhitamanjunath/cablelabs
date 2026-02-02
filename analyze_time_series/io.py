"""
I/O for analyze_time_series: load transform parquet, save results for visualize.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def load_transform_parquet(path: Path) -> pd.DataFrame:
    """Load transform output parquet; sort by date, hour."""
    if not path.exists():
        raise FileNotFoundError(f"Transformed data file not found: {path}")
    if path.suffix == ".parquet":
        try:
            df = pd.read_parquet(path)
        except ImportError:
            csv_path = path.with_suffix(".csv")
            if csv_path.exists():
                df = pd.read_csv(csv_path)
            else:
                raise FileNotFoundError(f"Neither Parquet nor CSV found: {path}")
    else:
        df = pd.read_csv(path)
    df = df.sort_values(["date", "hour"]).reset_index(drop=True)
    return df


def get_value_columns(df: pd.DataFrame) -> List[str]:
    """Return column names that are value columns (exclude date, hour)."""
    return [c for c in df.columns if c not in ("date", "hour")]


def extract_series(
    df: pd.DataFrame,
    hour: int,
    freq_col: str,
) -> pd.Series:
    """
    Extract a single daily time series for (hour, freq_col).
    Returns a Series with date index and values.
    """
    subset = df[df["hour"] == hour][["date", freq_col]].copy()
    subset = subset.sort_values("date").drop_duplicates("date", keep="last")
    subset["date"] = pd.to_datetime(subset["date"])
    subset = subset.set_index("date")[freq_col]
    return subset


def get_all_series_keys(df: pd.DataFrame) -> List[Tuple[int, str]]:
    """Return list of (hour, freq_col) for which we have data."""
    value_cols = get_value_columns(df)
    hours = sorted(df["hour"].unique())
    return [(h, c) for h in hours for c in value_cols]


def save_forecasts_parquet(
    forecasts: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save forecasts DataFrame to parquet (or CSV fallback)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        try:
            forecasts.to_parquet(output_path, index=False)
        except Exception:
            forecasts.to_csv(output_path.with_suffix(".csv"), index=False)
    else:
        forecasts.to_csv(output_path, index=False)


def save_metrics_json(metrics: Dict[str, Any], output_path: Path) -> None:
    """Save per-model metrics to JSON."""
    import json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)


def save_decomposition_parquet(
    decomposition: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save trend/seasonality decomposition to parquet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        try:
            decomposition.to_parquet(output_path, index=False)
        except Exception:
            decomposition.to_csv(output_path.with_suffix(".csv"), index=False)
    else:
        decomposition.to_csv(output_path, index=False)


def load_analysis_output(
    forecasts_path: Path,
    metrics_path: Optional[Path] = None,
    decomposition_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Optional[Dict], Optional[pd.DataFrame]]:
    """
    Load analyze_time_series output for visualize_time_series.
    Returns (forecasts_df, metrics_dict, decomposition_df).
    """
    if not forecasts_path.exists():
        raise FileNotFoundError(f"Forecasts file not found: {forecasts_path}")
    if forecasts_path.suffix == ".parquet":
        try:
            forecasts = pd.read_parquet(forecasts_path)
        except Exception:
            forecasts = pd.read_csv(forecasts_path.with_suffix(".csv"))
    else:
        forecasts = pd.read_csv(forecasts_path)

    metrics = None
    if metrics_path and metrics_path.exists():
        import json
        with open(metrics_path) as f:
            metrics = json.load(f)

    decomposition = None
    if decomposition_path and decomposition_path.exists():
        try:
            decomposition = pd.read_parquet(decomposition_path)
        except Exception:
            if decomposition_path.with_suffix(".csv").exists():
                decomposition = pd.read_csv(decomposition_path.with_suffix(".csv"))
    return forecasts, metrics, decomposition
