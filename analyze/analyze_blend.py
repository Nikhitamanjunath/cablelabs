#!/usr/bin/env python3
"""
Analyze Blend

Reads the tuned blend predictions from the time_series notebook (path from analyze config)
and converts them into the same schema as the analyze module output, so the visualize
module can display predicted, actual, and error maps the same way.
Uses the single analyze config (config.yaml).
"""

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

# Ensure project root is on path when run as script
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from analyze.analyze import load_config


def run(config=None) -> None:
    """Load tuned blend parquet from config paths, add date/hour/error, save for visualize."""
    if config is None:
        config = load_config()
    tuned_blend_file = Path(config.blend_input_file)
    if not tuned_blend_file.is_absolute():
        tuned_blend_file = project_root / tuned_blend_file
    output_file = Path(config.blend_output_file)
    if not output_file.is_absolute():
        output_file = project_root / output_file

    if not tuned_blend_file.exists():
        print(f"✗ Error: Tuned blend predictions not found: {tuned_blend_file}")
        print("  Run the time_series notebook and execute the Tuned blend cell to generate it.")
        return

    print(f"Loading tuned blend predictions from: {tuned_blend_file}")
    df = pd.read_parquet(tuned_blend_file)
    df["datetime"] = pd.to_datetime(df["datetime"])

    if len(df) == 0:
        print("✗ Error: Tuned blend file is empty")
        return

    # Convert to same schema as analyze output: date, hour, frequency_band, predicted, actual, error
    out = pd.DataFrame({
        "date": df["datetime"].dt.strftime("%Y-%m-%d"),
        "hour": df["datetime"].dt.hour,
        "frequency_band": df["frequency_band"],
        "predicted": df["predicted"].astype(float),
        "actual": df["actual"].astype(float),
    })
    out["error"] = out["actual"] - out["predicted"]
    out["confidence"] = np.nan
    out["is_anomaly"] = False

    output_file.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_file, index=False)
    print(f"✓ Blend prediction data saved: {output_file}")
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
