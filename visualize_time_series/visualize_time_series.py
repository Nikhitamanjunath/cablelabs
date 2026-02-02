#!/usr/bin/env python3
"""
Interactive time series visualization: select frequency and hour, view
time series with prediction intervals, trend/seasonality, and model error comparison.
"""

import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field, ValidationError

try:
    import dash
    from dash import dcc, html, Input, Output, State
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("Error: Dash and Plotly are required. pip install dash plotly")
    sys.exit(1)

# Project root
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from analyze_time_series.io import load_analysis_output, load_transform_parquet, extract_series


class VisualizeTimeSeriesConfig(BaseModel):
    """Configuration for time series visualization."""
    analysis_output_folder: str = Field(
        ...,
        description="Path to analyze_time_series output folder",
    )
    transformed_data_file: Optional[str] = Field(
        default=None,
        description="Optional path to transformed data for full actuals",
    )
    forecasts_file: str = Field(default="forecasts.parquet")
    metrics_file: str = Field(default="metrics.json")
    decomposition_file: str = Field(default="decomposition.parquet")
    port: int = Field(default=8051, ge=1024, le=65535)
    host: str = Field(default="127.0.0.1")


def load_config(config_path: Optional[str] = None) -> VisualizeTimeSeriesConfig:
    """Load and validate YAML config."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    path = Path(config_path)
    if not path.exists():
        example = Path(__file__).parent / "config.yaml.example"
        raise FileNotFoundError(
            f"Config not found: {config_path}. Create from {example}."
        )
    with open(path) as f:
        data = yaml.safe_load(f)
    if not data:
        raise ValueError("Config is empty.")
    return VisualizeTimeSeriesConfig(**data)


def load_data(config: VisualizeTimeSeriesConfig) -> Tuple[
    pd.DataFrame,
    Optional[Dict[str, Any]],
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
]:
    """
    Load forecasts, metrics, decomposition, and optionally full transform data.
    Returns (forecasts_df, metrics_dict, decomposition_df, transform_df).
    """
    out_dir = Path(config.analysis_output_folder)
    forecasts_path = out_dir / config.forecasts_file
    metrics_path = out_dir / config.metrics_file
    decomposition_path = out_dir / config.decomposition_file
    forecasts, metrics, decomposition = load_analysis_output(
        forecasts_path, metrics_path, decomposition_path
    )
    transform_df = None
    if config.transformed_data_file:
        try:
            transform_df = load_transform_parquet(Path(config.transformed_data_file))
        except Exception:
            pass
    return forecasts, metrics, decomposition, transform_df


# Colors for models (baselines first, then statistical)
MODEL_COLORS = {
    "naive": "#1f77b4",
    "historic_avg": "#ff7f0e",
    "window_avg": "#2ca02c",
    "seasonal_naive": "#d62728",
    "exponential_smoothing": "#bcbd22",
    "arima": "#9467bd",
    "sarima": "#8c564b",
    "ets": "#e377c2",
    "prophet": "#7f7f7f",
}


def _extract_hourly_actuals(transform_df: pd.DataFrame, freq_band: str) -> Optional[pd.Series]:
    """Extract hourly actuals for a frequency band: datetime index (date + hour) -> value."""
    if freq_band not in transform_df.columns:
        return None
    df = transform_df[["date", "hour", freq_band]].copy()
    df = df.dropna(subset=[freq_band]).sort_values(["date", "hour"])
    df["datetime"] = pd.to_datetime(df["date"]) + pd.to_timedelta(df["hour"], unit="h")
    return df.set_index("datetime")[freq_band]


def build_timeseries_plot(
    forecasts: pd.DataFrame,
    hour: Any,
    freq_band: str,
    confidence_pct: float,
    actuals_series: Optional[pd.Series] = None,
    actuals_hourly: Optional[pd.Series] = None,
    selected_model: Optional[str] = None,
) -> go.Figure:
    """
    Build main time series plot: actuals + model point forecasts + PIs.
    - hour: int = one value per day for that hour (daily view). hour == "all" = hourly view.
    - actuals_series: daily actuals (date index) when hour is int.
    - actuals_hourly: hourly actuals (datetime index) when hour == "all".
    - selected_model: if not None and != "all", show only this model (plus actuals).
    """
    hourly_view = hour == "all"
    if hourly_view:
        sub = forecasts[forecasts["frequency_band"] == freq_band].copy()
        if len(sub) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No data for this frequency. Run analyze-time-series first.",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            )
            return fig
        sub["datetime"] = pd.to_datetime(sub["date"]) + pd.to_timedelta(sub["hour"], unit="h")
        sub = sub.sort_values("datetime")
        x_label = "Date & time (hourly)"
    else:
        sub = forecasts[(forecasts["hour"] == int(hour)) & (forecasts["frequency_band"] == freq_band)]
        if len(sub) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No data for this (hour, frequency). Run analyze-time-series first.",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            )
            return fig
        x_label = "Date (one point per day for selected hour)"

    fig = go.Figure()

    # Use datetime x for zoom/range slider; collect full x range for default view
    all_x: List[pd.Timestamp] = []

    def _add_trace(x_vals, y_vals, name, **kwargs):
        x_dt = pd.to_datetime(x_vals)
        all_x.extend(x_dt.dropna().tolist())
        fig.add_trace(
            go.Scatter(x=x_dt, y=y_vals, name=name, **kwargs)
        )

    # Actuals
    if hourly_view and actuals_hourly is not None and len(actuals_hourly) > 0:
        _add_trace(
            actuals_hourly.index,
            actuals_hourly.values,
            "Actual",
            line=dict(color="black", width=2, dash="solid"),
            mode="lines+markers",
        )
    elif not hourly_view and actuals_series is not None and len(actuals_series) > 0:
        _add_trace(
            actuals_series.index,
            actuals_series.values,
            "Actual",
            line=dict(color="black", width=2, dash="solid"),
            mode="lines+markers",
        )
    elif not hourly_view:
        act_by_date = sub.dropna(subset=["actual"]).drop_duplicates("date")[["date", "actual"]]
        if len(act_by_date) > 0:
            _add_trace(
                act_by_date["date"],
                act_by_date["actual"],
                "Actual",
                line=dict(color="black", width=2, dash="solid"),
                mode="lines+markers",
            )
    elif hourly_view and len(sub.dropna(subset=["actual"])) > 0:
        act_sub = sub.dropna(subset=["actual"]).drop_duplicates(["datetime"])[["datetime", "actual"]].sort_values("datetime")
        _add_trace(
            act_sub["datetime"],
            act_sub["actual"],
            "Actual",
            line=dict(color="black", width=2, dash="solid"),
            mode="lines+markers",
        )

    # Per model: point forecast and PI band (filter to selected_model if set)
    models = sub["model"].unique()
    if selected_model and selected_model != "all":
        models = [m for m in models if m == selected_model]
    for model in models:
        msub = sub[sub["model"] == model]
        msub = msub.sort_values("datetime" if hourly_view else "date")
        x_vals = msub["datetime"] if hourly_view else msub["date"]
        x_dt = pd.to_datetime(x_vals)
        all_x.extend(x_dt.dropna().tolist())
        point = msub["point"].tolist()
        lower = msub["lower_95"].tolist()
        upper = msub["upper_95"].tolist()
        color = MODEL_COLORS.get(model, "#17becf")
        fig.add_trace(
            go.Scatter(
                x=x_dt, y=point,
                name=model,
                line=dict(color=color, width=1.5),
                mode="lines+markers",
            )
        )
        if any(l is not None and not np.isnan(l) for l in lower) and any(
            u is not None and not np.isnan(u) for u in upper
        ):
            fig.add_trace(
                go.Scatter(
                    x=x_dt.tolist() + x_dt.tolist()[::-1],
                    y=upper + lower[::-1],
                    fill="toself",
                    fillcolor=f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.2)",
                    line=dict(width=0),
                    name=f"{model} ({confidence_pct:.0f}% PI)",
                    showlegend=True,
                )
            )

    # Default view: last 90 days when we have more than 90 days (avoid "block" view)
    xaxis_range = None
    if all_x:
        x_min, x_max = min(all_x), max(all_x)
        if (x_max - x_min).days > 90:
            range_start = x_max - pd.Timedelta(days=90)
            xaxis_range = [range_start, x_max]

    fig.update_layout(
        title=f"Time series: {'all hours (hourly)' if hourly_view else f'hour={hour}'}, frequency={freq_band}",
        xaxis_title=x_label,
        yaxis_title="Value",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(b=80, t=80),
        xaxis=dict(
            type="date",
            rangeslider=dict(visible=True, thickness=0.05),
            range=xaxis_range,
        ),
        yaxis=dict(
            autorange=True,
            rangemode="normal",
            fixedrange=False,
        ),
    )
    fig.update_xaxes(tickangle=-45)
    return fig


def build_trend_seasonality_plot(
    decomposition: pd.DataFrame,
    hour: int,
    freq_band: str,
) -> Tuple[Optional[go.Figure], Optional[go.Figure]]:
    """Build trend and seasonality subplots from decomposition (Prophet output)."""
    sub = decomposition[
        (decomposition["hour"] == hour) & (decomposition["frequency_band"] == freq_band)
    ]
    if len(sub) == 0:
        return None, None
    sub = sub.sort_values("date")
    x = sub["date"].astype(str).tolist()
    trend_fig = go.Figure()
    trend_fig.add_trace(
        go.Scatter(x=x, y=sub["trend"], name="Trend", line=dict(color="blue", width=2))
    )
    trend_fig.update_layout(
        title=f"Trend (hour={hour}, freq={freq_band})",
        xaxis_title="Date",
        yaxis_title="Trend",
        margin=dict(b=60, t=50),
    )
    trend_fig.update_xaxes(tickangle=-45)
    season_fig = go.Figure()
    if "seasonal_weekly" in sub.columns:
        season_fig.add_trace(
            go.Scatter(
                x=x, y=sub["seasonal_weekly"],
                name="Seasonal (weekly)",
                line=dict(color="green", width=2),
            )
        )
    season_fig.update_layout(
        title=f"Seasonality (hour={hour}, freq={freq_band})",
        xaxis_title="Date",
        yaxis_title="Seasonal",
        margin=dict(b=60, t=50),
    )
    season_fig.update_xaxes(tickangle=-45)
    return trend_fig, season_fig


def build_metrics_table(metrics: Optional[Dict[str, Any]]) -> List[dict]:
    """Build table rows for model comparison: MAE, RMSE, PI coverage, PI width."""
    if not metrics:
        return []
    # Order: baselines first, then statistical
    order = [
        "naive", "historic_avg", "window_avg", "seasonal_naive",
        "exponential_smoothing",
        "arima", "sarima", "ets", "prophet",
    ]
    rows = []
    for model in order:
        if model not in metrics:
            continue
        m = metrics[model]
        mae_val = m.get("mae")
        rmse_val = m.get("rmse")
        cov = m.get("pi_coverage")
        width = m.get("pi_width")
        rows.append({
            "model": model,
            "mae": f"{mae_val:.4f}" if mae_val is not None else "—",
            "rmse": f"{rmse_val:.4f}" if rmse_val is not None else "—",
            "pi_coverage": f"{cov:.2%}" if cov is not None else "—",
            "pi_width": f"{width:.4f}" if width is not None else "—",
        })
    for model, m in metrics.items():
        if model not in order:
            mae_val = m.get("mae")
            rmse_val = m.get("rmse")
            cov = m.get("pi_coverage")
            width = m.get("pi_width")
            rows.append({
                "model": model,
                "mae": f"{mae_val:.4f}" if mae_val is not None else "—",
                "rmse": f"{rmse_val:.4f}" if rmse_val is not None else "—",
                "pi_coverage": f"{cov:.2%}" if cov is not None else "—",
                "pi_width": f"{width:.4f}" if width is not None else "—",
            })
    return rows


def build_raw_timeseries_plot(transform_df: pd.DataFrame, freq_band: str) -> go.Figure:
    """Build raw time series plot for one frequency band (no forecasts). Same as notebook plot_band."""
    if freq_band not in transform_df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Unknown band: {freq_band}",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )
        return fig
    sub = (
        transform_df[["date", "hour", freq_band]]
        .dropna(subset=[freq_band])
        .copy()
    )
    sub["datetime"] = pd.to_datetime(sub["date"]) + pd.to_timedelta(sub["hour"], unit="h")
    sub = sub.sort_values("datetime")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sub["datetime"],
            y=sub[freq_band],
            mode="lines+markers",
            name=freq_band,
            line=dict(width=1.5),
        )
    )
    fig.update_layout(
        title=f"Raw time series: {freq_band}",
        xaxis_title="Date & time",
        yaxis_title="Value",
        xaxis=dict(type="date", rangeslider=dict(visible=True, thickness=0.05)),
        yaxis=dict(autorange=True, rangemode="normal"),
        height=450,
        margin=dict(b=80, t=60),
    )
    return fig


def build_metrics_bar(metrics: Optional[Dict[str, Any]]) -> go.Figure:
    """Bar chart: MAE and RMSE per model (baselines first)."""
    if not metrics:
        return go.Figure()
    order = [
        "naive", "historic_avg", "window_avg", "seasonal_naive",
        "exponential_smoothing",
        "arima", "sarima", "ets", "prophet",
    ]
    models = [m for m in order if m in metrics]
    mae_vals = [metrics[m].get("mae") for m in models]
    rmse_vals = [metrics[m].get("rmse") for m in models]
    mae_vals = [float(x) if x is not None else np.nan for x in mae_vals]
    rmse_vals = [float(x) if x is not None else np.nan for x in rmse_vals]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(name="MAE", x=models, y=mae_vals, marker_color="#1f77b4")
    )
    fig.add_trace(
        go.Bar(name="RMSE", x=models, y=rmse_vals, marker_color="#ff7f0e")
    )
    fig.update_layout(
        title="Model comparison: MAE and RMSE",
        barmode="group",
        xaxis_title="Model",
        yaxis_title="Error",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(b=80, t=60),
    )
    fig.update_xaxes(tickangle=-45)
    return fig


def create_app(
    forecasts: pd.DataFrame,
    metrics: Optional[Dict[str, Any]],
    decomposition: Optional[pd.DataFrame],
    transform_df: Optional[pd.DataFrame],
) -> dash.Dash:
    """Create Dash app with frequency/hour selectors and plots."""
    freq_bands = sorted(forecasts["frequency_band"].unique())
    hours = sorted(forecasts["hour"].unique())
    if not freq_bands or not hours:
        freq_bands = ["(no data)"]
        hours = [0]

    hour_options = [{"label": "All hours (hourly)", "value": "all"}]
    hour_options += [{"label": str(h), "value": h} for h in hours]

    model_names = sorted(forecasts["model"].unique())
    model_options = [{"label": "All models", "value": "all"}]
    model_options += [{"label": m.replace("_", " ").title(), "value": m} for m in model_names]
    default_model = "prophet" if "prophet" in model_names else (model_names[0] if model_names else "all")

    # Raw data tab: frequency bands from transformed data (same as notebook)
    raw_freq_bands = (
        sorted([c for c in transform_df.columns if c not in ("date", "hour")])
        if transform_df is not None and len(transform_df) > 0
        else []
    )
    raw_freq_options = [{"label": f, "value": f} for f in raw_freq_bands]
    raw_default = raw_freq_bands[0] if raw_freq_bands else None

    app = dash.Dash(__name__, title="Time Series Viewer")
    analysis_tab = html.Div([
        html.Div([
            html.Label("Frequency band:"),
            dcc.Dropdown(
                id="freq-band",
                options=[{"label": f, "value": f} for f in freq_bands],
                value=freq_bands[0] if freq_bands[0] != "(no data)" else None,
                clearable=False,
                style={"width": "300px"},
            ),
        ], style={"margin": "10px"}),
        html.Div([
            html.Label("Hour:"),
            dcc.Dropdown(
                id="hour",
                options=hour_options,
                value="all",
                clearable=False,
                style={"width": "200px"},
            ),
        ], style={"margin": "10px"}),
        html.Div([
            html.Label("Model:"),
            dcc.Dropdown(
                id="model-select",
                options=model_options,
                value=default_model,
                clearable=False,
                style={"width": "220px"},
            ),
        ], style={"margin": "10px"}),
        html.Div([
            html.Label("PI confidence:"),
            dcc.Dropdown(
                id="confidence-pct",
                options=[
                    {"label": "90%", "value": 90},
                    {"label": "95%", "value": 95},
                    {"label": "99%", "value": 99},
                ],
                value=95,
                clearable=False,
                style={"width": "100px"},
            ),
        ], style={"margin": "10px"}),
        dcc.Graph(
            id="timeseries-plot",
            style={"height": "450px"},
            config=dict(scrollZoom=True, displayModeBar=True),
        ),
        html.H3("Trend and seasonality"),
        html.Div([
            dcc.Graph(id="trend-plot", style={"height": "280px", "display": "inline-block", "width": "48%"}),
            dcc.Graph(id="seasonality-plot", style={"height": "280px", "display": "inline-block", "width": "48%"}),
        ]),
        html.H3("Model comparison (errors)"),
        dcc.Graph(id="metrics-bar", style={"height": "350px"}),
        html.Div(id="metrics-table-container", style={"margin": "20px", "overflowX": "auto"}),
    ])
    raw_tab = html.Div([
        html.P(
            "Select frequency band to view raw transformed time series (no forecasts). "
            "Uses the same data as the notebook.",
            style={"margin": "10px"},
        ),
        html.Div([
            html.Label("Frequency band:"),
            dcc.Dropdown(
                id="raw-freq-band",
                options=raw_freq_options,
                value=raw_default,
                clearable=False,
                style={"width": "300px"},
                placeholder="(no transformed data)" if not raw_freq_options else None,
            ),
        ], style={"margin": "10px"}),
        dcc.Graph(
            id="raw-timeseries-plot",
            style={"height": "450px"},
            config=dict(scrollZoom=True, displayModeBar=True),
        ),
    ])

    app.layout = html.Div([
        html.H1("Time Series Viewer", style={"textAlign": "center"}),
        dcc.Tabs(id="main-tabs", value="analysis", children=[
            dcc.Tab(label="Analysis", value="analysis", children=analysis_tab),
            dcc.Tab(label="Raw data", value="raw", children=raw_tab),
        ]),
    ])

    def get_actuals_series(h: Any, freq: str) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
        """Return (daily_actuals, hourly_actuals). Daily when h is int, hourly when h == 'all'."""
        if transform_df is None:
            return None, None
        try:
            if h == "all":
                return None, _extract_hourly_actuals(transform_df, freq)
            return extract_series(transform_df, int(h), freq), None
        except Exception:
            return None, None

    @app.callback(
        Output("timeseries-plot", "figure"),
        Input("freq-band", "value"),
        Input("hour", "value"),
        Input("model-select", "value"),
        Input("confidence-pct", "value"),
    )
    def update_timeseries(freq_band: str, hour: Any, selected_model: Any, confidence_pct: float):
        if freq_band is None or hour is None:
            return go.Figure()
        actuals_daily, actuals_hourly = get_actuals_series(hour, freq_band)
        return build_timeseries_plot(
            forecasts, hour, freq_band, confidence_pct or 95,
            actuals_series=actuals_daily,
            actuals_hourly=actuals_hourly,
            selected_model=selected_model,
        )

    @app.callback(
        [Output("trend-plot", "figure"), Output("seasonality-plot", "figure")],
        Input("freq-band", "value"),
        Input("hour", "value"),
    )
    def update_decomposition(freq_band: str, hour: Any):
        if decomposition is None or len(decomposition) == 0 or freq_band is None:
            empty = go.Figure()
            empty.add_annotation(
                text="No decomposition data (run Prophet in analyze-time-series).",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            )
            return empty, empty
        decomp_hour = hour
        if hour == "all":
            sub = decomposition[decomposition["frequency_band"] == freq_band]
            if len(sub) == 0:
                decomp_hour = decomposition["hour"].iloc[0]
            else:
                decomp_hour = sub["hour"].iloc[0]
        trend_fig, season_fig = build_trend_seasonality_plot(decomposition, decomp_hour, freq_band)
        if trend_fig is None:
            trend_fig = go.Figure()
            trend_fig.add_annotation(text="No decomposition for this series.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        if season_fig is None:
            season_fig = go.Figure()
            season_fig.add_annotation(text="No decomposition for this series.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return trend_fig, season_fig

    @app.callback(
        Output("raw-timeseries-plot", "figure"),
        Input("raw-freq-band", "value"),
    )
    def update_raw_timeseries(freq_band: Optional[str]):
        if transform_df is None or freq_band is None:
            fig = go.Figure()
            fig.add_annotation(
                text="No transformed data. Set transformed_data_file in config and run transform.",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            )
            return fig
        return build_raw_timeseries_plot(transform_df, freq_band)

    @app.callback(
        Output("metrics-bar", "figure"),
        Input("freq-band", "value"),
    )
    def update_metrics_bar(_):
        return build_metrics_bar(metrics)

    @app.callback(
        Output("metrics-table-container", "children"),
        Input("freq-band", "value"),
    )
    def update_metrics_table(_):
        rows = build_metrics_table(metrics)
        if not rows:
            return html.P("No metrics (run analyze-time-series first).")
        return html.Table(
            [html.Tr([html.Th(k) for k in ["model", "mae", "rmse", "pi_coverage", "pi_width"]])]
            + [html.Tr([html.Td(r[k]) for k in ["model", "mae", "rmse", "pi_coverage", "pi_width"]]) for r in rows],
            style={"borderCollapse": "collapse", "border": "1px solid #ccc"},
        )

    return app


def main() -> None:
    try:
        config = load_config()
        print("Loading analysis output...")
        forecasts, metrics, decomposition, transform_df = load_data(config)
        print(f"  Forecasts: {len(forecasts)} rows")
        print(f"  Metrics: {len(metrics) if metrics else 0} models")
        print(f"  Decomposition: {len(decomposition) if decomposition is not None else 0} rows")
        if transform_df is not None:
            print("  Transformed data loaded for full actuals")
        app = create_app(forecasts, metrics, decomposition, transform_df)
        url = f"http://{config.host}:{config.port}"
        print(f"Starting server at {url}")
        print("  Opening browser automatically...")
        print("  Press Ctrl+C to stop the server")

        def open_browser():
            time.sleep(1.5)
            webbrowser.open(url)

        browser_thread = threading.Thread(target=open_browser, daemon=True)
        browser_thread.start()

        app.run(host=config.host, port=config.port, debug=False)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValidationError as e:
        print(f"Config error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
