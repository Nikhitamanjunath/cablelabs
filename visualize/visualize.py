#!/usr/bin/env python3
"""
Interactive Prediction Data Visualizer

Provides an interactive web-based interface for exploring prediction data.
Allows toggling between predicted values, confidence scores, actual values, and errors.
"""

import sys
import webbrowser
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import yaml
from pydantic import BaseModel, Field, ValidationError

try:
    import dash
    from dash import dcc, html, Input, Output, callback_context
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("✗ Error: Dash and Plotly are required for interactive visualization")
    print("  Install with: pip install dash plotly")
    sys.exit(1)

# Constants
NUM_ROWS = 24  # Hours 0-23
NUM_COLS = 70  # Frequency bands


class VisualizeConfig(BaseModel):
    """Visualization configuration."""
    prediction_data_file: str = Field(
        ...,
        description="Path to prediction data file (Parquet, CSV, or JSON) created by predict task"
    )
    port: int = Field(
        default=8050,
        ge=1024,
        le=65535,
        description="Port for the Dash web server"
    )
    host: str = Field(
        default="127.0.0.1",
        description="Host for the Dash web server"
    )


def load_config(config_path: Optional[str] = None) -> VisualizeConfig:
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
    
    return VisualizeConfig(**config_data)


def load_prediction_data(data_file: Path) -> pd.DataFrame:
    """Load prediction data from file."""
    if not data_file.exists():
        raise FileNotFoundError(f"Prediction data file not found: {data_file}")
    
    if data_file.suffix == '.parquet':
        try:
            df = pd.read_parquet(data_file)
        except ImportError:
            print("⚠ PyArrow not available, trying CSV...")
            csv_path = data_file.with_suffix('.csv')
            if csv_path.exists():
                df = pd.read_csv(csv_path)
            else:
                raise FileNotFoundError(f"Neither Parquet nor CSV file found")
    elif data_file.suffix == '.json':
        df = pd.read_json(data_file, orient='records')
    else:
        df = pd.read_csv(data_file)
    
    return df


def create_heatmap_data(df: pd.DataFrame, selected_date: str, view_mode: str) -> tuple:
    """
    Create heatmap data for the selected date and view mode.
    
    Returns:
        Tuple of (heatmap_matrix, color_scale, title, value_range)
    """
    # Filter by date
    date_data = df[df['date'] == selected_date].copy()
    
    if len(date_data) == 0:
        return None, None, f"No data for {selected_date}", None
    
    # Get unique hours and frequency bands
    hours = sorted(date_data['hour'].unique(), reverse=True)  # 23 to 0 (top to bottom)
    freq_bands = sorted(date_data['frequency_band'].unique())
    
    # Handle anomaly view mode separately (returns early)
    if view_mode == 'anomaly':
        anomaly_matrix = np.full((NUM_ROWS, NUM_COLS), np.nan)
        for _, row in date_data.iterrows():
            hour = int(row['hour'])
            freq_band = row['frequency_band']
            is_anomaly = row.get('is_anomaly', False)
            
            hour_idx = NUM_ROWS - 1 - hours.index(hour) if hour in hours else None
            freq_idx = freq_bands.index(freq_band) if freq_band in freq_bands else None
            
            if hour_idx is not None and freq_idx is not None:
                anomaly_matrix[hour_idx, freq_idx] = 1.0 if is_anomaly else 0.0
        
        # Use red for anomalies, gray for normal
        color_scale = [
            [0, 'rgb(200, 200, 200)'],  # Gray (normal)
            [1.0, 'rgb(255, 0, 0)']  # Red (anomaly)
        ]
        value_range = [0.0, 1.0]
        title = f"{selected_date} - Anomalies (Red = Anomaly, Gray = Normal)"
        return anomaly_matrix, color_scale, title, value_range
    
    # Create matrix (24 rows × 70 columns)
    matrix = np.full((NUM_ROWS, NUM_COLS), np.nan)
    
    # Map values based on view mode
    value_column = {
        'predicted': 'predicted',
        'confidence': 'confidence',
        'actual': 'actual',
        'error': 'error',
        'anomaly': 'is_anomaly'  # Special handling for anomaly
    }.get(view_mode, 'predicted')
    
    # Handle anomaly view mode separately (returns early)
    if view_mode == 'anomaly':
        anomaly_matrix = np.full((NUM_ROWS, NUM_COLS), np.nan)
        for _, row in date_data.iterrows():
            hour = int(row['hour'])
            freq_band = row['frequency_band']
            is_anomaly = row.get('is_anomaly', False)
            
            hour_idx = NUM_ROWS - 1 - hours.index(hour) if hour in hours else None
            freq_idx = freq_bands.index(freq_band) if freq_band in freq_bands else None
            
            if hour_idx is not None and freq_idx is not None:
                anomaly_matrix[hour_idx, freq_idx] = 1.0 if is_anomaly else 0.0
        
        # Use red for anomalies, gray for normal
        color_scale = [
            [0, 'rgb(200, 200, 200)'],  # Gray (normal)
            [1.0, 'rgb(255, 0, 0)']  # Red (anomaly)
        ]
        value_range = [0.0, 1.0]
        title = f"{selected_date} - Anomalies (Red = Anomaly, Gray = Normal)"
        return anomaly_matrix, color_scale, title, value_range
    
    for _, row in date_data.iterrows():
        hour = int(row['hour'])
        freq_band = row['frequency_band']
        
        # Find row index (hour 23 = row 0, hour 0 = row 23)
        hour_idx = NUM_ROWS - 1 - hours.index(hour) if hour in hours else None
        
        # Find column index
        freq_idx = freq_bands.index(freq_band) if freq_band in freq_bands else None
        
        if hour_idx is not None and freq_idx is not None:
            value = row.get(value_column)
            if value is not None and not np.isnan(value):
                matrix[hour_idx, freq_idx] = float(value)
    
    # Determine color scale and range based on view mode
    if view_mode == 'confidence':
        # Confidence: 0.0 to 1.0, use green-yellow-red scale with darker shades
        # Red: below 0.7 (70%), Yellow: 0.7-0.9 (70%-90%), Green: 0.9+ (90%+)
        color_scale = [
            [0, 'rgb(180, 0, 0)'],  # Dark red (low, below 70%)
            [0.7, 'rgb(200, 180, 0)'],  # Dark yellow (medium, 70%-90%)
            [0.9, 'rgb(0, 150, 0)'],  # Dark green (high, 90%+)
            [1.0, 'rgb(0, 200, 0)']  # Darker bright green (very high)
        ]
        value_range = [0.0, 1.0]
        title = f"{selected_date} - Confidence Scores"
    elif view_mode == 'error':
        # Error: signed values, use diverging scale (blue-white-red)
        valid_values = matrix[~np.isnan(matrix)]
        if len(valid_values) > 0:
            abs_max = max(abs(np.min(valid_values)), abs(np.max(valid_values)))
            value_range = [-abs_max, abs_max]
        else:
            value_range = [-10, 10]
        color_scale = [
            [0, 'rgb(0, 0, 255)'],  # Blue (negative error)
            [0.5, 'rgb(255, 255, 255)'],  # White (zero error)
            [1.0, 'rgb(255, 0, 0)']  # Red (positive error)
        ]
        title = f"{selected_date} - Error (Actual - Predicted)"
    else:
        # Predicted/Actual: use value range from data
        valid_values = matrix[~np.isnan(matrix)]
        if len(valid_values) > 0:
            value_range = [float(np.min(valid_values)), float(np.max(valid_values))]
        else:
            value_range = [0.0, 100.0]
        
        # Use a color scale similar to the original (blue to yellow)
        color_scale = [
            [0, 'rgb(0, 0, 100)'],  # Dark blue (low)
            [0.25, 'rgb(0, 100, 200)'],  # Blue
            [0.5, 'rgb(100, 200, 255)'],  # Light blue
            [0.75, 'rgb(255, 200, 100)'],  # Orange
            [1.0, 'rgb(255, 255, 0)']  # Yellow (high)
        ]
        title = f"{selected_date} - {view_mode.capitalize()} Values"
    
    return matrix, color_scale, title, value_range


def calculate_stats(df: pd.DataFrame, selected_date: str) -> dict:
    """Calculate statistics for the selected date."""
    date_data = df[df['date'] == selected_date].copy()
    
    if len(date_data) == 0:
        return {}
    
    stats = {}
    
    # Mean confidence
    confidences = date_data['confidence'].dropna()
    if len(confidences) > 0:
        stats['mean_confidence'] = float(confidences.mean())
        stats['min_confidence'] = float(confidences.min())
        stats['max_confidence'] = float(confidences.max())
    
    # Error statistics (if actual data available)
    errors = date_data['error'].dropna()
    if len(errors) > 0:
        abs_errors = errors.abs()
        stats['mean_error'] = float(abs_errors.mean())
        stats['max_error'] = float(abs_errors.max())
        stats['min_error'] = float(abs_errors.min())
        
        # Zero error percentage (errors < 0.1)
        zero_error_count = (abs_errors < 0.1).sum()
        stats['zero_error_percentage'] = float((zero_error_count / len(errors)) * 100.0)

    # ML prediction error metrics (from actual vs predicted pairs)
    mask = date_data['actual'].notna() & date_data['predicted'].notna()
    if mask.sum() > 0:
        actual_vals = date_data.loc[mask, 'actual'].astype(float).values
        pred_vals = date_data.loc[mask, 'predicted'].astype(float).values
        residuals = actual_vals - pred_vals
        abs_res = np.abs(residuals)
        stats['mae'] = float(np.mean(abs_res))
        stats['rmse'] = float(np.sqrt(np.mean(residuals ** 2)))
        stats['bias'] = float(np.mean(residuals))
        stats['medae'] = float(np.median(abs_res))
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((actual_vals - np.mean(actual_vals)) ** 2)
        if ss_tot > 0:
            stats['r2'] = float(1.0 - ss_res / ss_tot)
        if len(actual_vals) > 1 and np.std(actual_vals) > 0 and np.std(pred_vals) > 0:
            stats['pearson_r'] = float(np.corrcoef(actual_vals, pred_vals)[0, 1])

    # Anomaly statistics
    if 'is_anomaly' in date_data.columns:
        anomalies = date_data['is_anomaly']
        if len(anomalies) > 0:
            anomaly_count = anomalies.sum() if anomalies.dtype == bool else (anomalies == True).sum()
            total_cells = len(anomalies)
            stats['anomaly_count'] = int(anomaly_count)
            stats['anomaly_percentage'] = float((anomaly_count / total_cells) * 100.0) if total_cells > 0 else 0.0
    
    return stats


def main():
    """Main entry point for interactive visualization."""
    try:
        print("Loading configuration...")
        config = load_config()
        print("✓ Configuration loaded")
        
        # Load prediction data
        print(f"\nLoading prediction data from: {config.prediction_data_file}")
        data_file = Path(config.prediction_data_file)
        df = load_prediction_data(data_file)
        print(f"✓ Loaded {len(df)} data points")
        
        # Get available dates
        available_dates = sorted(df['date'].unique())
        print(f"  Available dates: {len(available_dates)}")
        print(f"  Date range: {available_dates[0]} to {available_dates[-1]}")
        
        if len(available_dates) == 0:
            print("✗ Error: No prediction data found")
            return
        
        # Initialize Dash app
        app = dash.Dash(__name__)
        
        # App layout
        app.layout = html.Div([
            html.H1("Prediction Data Interactive Viewer", style={'textAlign': 'center', 'marginBottom': '20px'}),
            
            html.Div([
                html.Label("Select Date:", style={'marginRight': '10px', 'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='date-selector',
                    options=[{'label': date, 'value': date} for date in available_dates],
                    value=available_dates[-1] if available_dates else None,
                    style={'width': '200px', 'display': 'inline-block'}
                ),
                html.Div(style={'width': '30px', 'display': 'inline-block'}),  # Spacing
                html.Label("View Mode:", style={'marginRight': '10px', 'fontWeight': 'bold'}),
                dcc.RadioItems(
                    id='view-toggle',
                    options=[
                        {'label': 'Predicted', 'value': 'predicted'},
                        {'label': 'Confidence', 'value': 'confidence'},
                        {'label': 'Actual', 'value': 'actual'},
                        {'label': 'Error', 'value': 'error'},
                        {'label': 'Anomaly', 'value': 'anomaly'}
                    ],
                    value='predicted',
                    inline=True,
                    style={'display': 'inline-block'}
                )
            ], style={'marginBottom': '20px', 'padding': '10px', 'backgroundColor': '#f0f0f0'}),
            
            dcc.Graph(id='heatmap-grid', style={'height': '600px'}),
            
            html.Div(id='stats-display', style={
                'marginTop': '20px',
                'padding': '15px',
                'backgroundColor': '#f9f9f9',
                'border': '1px solid #ddd',
                'borderRadius': '5px'
            })
        ], style={'padding': '20px', 'maxWidth': '1400px', 'margin': '0 auto'})
        
        @app.callback(
            Output('heatmap-grid', 'figure'),
            Output('stats-display', 'children'),
            Input('date-selector', 'value'),
            Input('view-toggle', 'value')
        )
        def update_visualization(selected_date: str, view_mode: str):
            """Update heatmap and stats based on selected date and view mode."""
            if selected_date is None:
                return go.Figure(), html.Div("No date selected")
            
            # Filter data for selected date
            date_data = df[df['date'] == selected_date].copy()
            
            if len(date_data) == 0:
                fig = go.Figure()
                fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
                return fig, html.Div(f"No data for {selected_date}")
            
            # Create heatmap data
            matrix, color_scale, title, value_range = create_heatmap_data(df, selected_date, view_mode)
            
            if matrix is None:
                fig = go.Figure()
                fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
                return fig, html.Div(f"No data for {selected_date}")
            
            # Get frequency band labels for x-axis
            freq_bands = sorted(date_data['frequency_band'].unique())
            hours = sorted(date_data['hour'].unique(), reverse=True)  # 23 to 0
            
            # Create x-axis labels (use frequency band names, but show only every 10th to avoid crowding)
            x_tickvals = list(range(0, len(freq_bands), max(1, len(freq_bands) // 10)))
            x_ticktext = [freq_bands[i] for i in x_tickvals]
            
            # Build lookup for prediction_method if present (from online learning)
            method_lookup = {}
            if 'prediction_method' in date_data.columns:
                for _, r in date_data.iterrows():
                    method_lookup[(int(r['hour']), r['frequency_band'])] = r.get('prediction_method', '')
            # Create custom hover text matrix with proper frequency band labels
            hover_text = []
            for hour_idx in range(len(hours)):
                row = []
                for freq_idx in range(len(freq_bands)):
                    value = matrix[hour_idx, freq_idx]
                    hour = hours[hour_idx]
                    freq_band = freq_bands[freq_idx]
                    method_str = ""
                    if method_lookup:
                        method_str = method_lookup.get((hour, freq_band), '')
                        method_str = f"<br>Method: {method_str}" if method_str else ""
                    if view_mode == 'anomaly':
                        # Special handling for anomaly view
                        if not np.isnan(value):
                            status = "Anomaly" if value >= 0.5 else "Normal"
                            row.append(f"Hour: {hour}<br>Frequency: {freq_band}<br>Status: {status}{method_str}")
                        else:
                            row.append(f"Hour: {hour}<br>Frequency: {freq_band}<br>Status: N/A{method_str}")
                    elif not np.isnan(value):
                        row.append(f"Hour: {hour}<br>Frequency: {freq_band}<br>Value: {value:.2f}{method_str}")
                    else:
                        row.append(f"Hour: {hour}<br>Frequency: {freq_band}<br>Value: N/A{method_str}")
                hover_text.append(row)
            
            # Create heatmap with numeric x-axis (we'll label it with frequency bands)
            fig = go.Figure(data=go.Heatmap(
                z=matrix,
                x=list(range(len(freq_bands))),
                y=[str(hour) for hour in hours],
                colorscale=color_scale,
                zmin=value_range[0],
                zmax=value_range[1],
                colorbar=dict(title=view_mode.capitalize()),
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                showscale=True
            ))
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title="Frequency Band",
                yaxis_title="Hour (23 at top, 0 at bottom)",
                height=600,
                xaxis=dict(
                    tickmode='array',
                    tickvals=x_tickvals,
                    ticktext=x_ticktext,
                    tickangle=-45,
                    showgrid=True
                ),
                yaxis=dict(
                    showgrid=True,
                    autorange='reversed'  # Hour 23 at top
                )
            )
            
            # Calculate and display stats
            stats = calculate_stats(df, selected_date)
            stats_elements = []
            
            if 'mean_confidence' in stats:
                stats_elements.append(html.Div([
                    html.Strong("Mean Confidence: "),
                    html.Span(f"{stats['mean_confidence']*100:.1f}%")
                ], style={'marginRight': '30px', 'display': 'inline-block'}))
            
            if 'anomaly_count' in stats:
                stats_elements.append(html.Div([
                    html.Strong("Anomalies: "),
                    html.Span(f"{stats['anomaly_count']} ({stats['anomaly_percentage']:.1f}%)")
                ], style={'marginRight': '30px', 'display': 'inline-block'}))
            
            if 'mean_error' in stats:
                stats_elements.append(html.Div([
                    html.Strong("Mean Error: "),
                    html.Span(f"{stats['mean_error']:.2f}")
                ], style={'marginRight': '30px', 'display': 'inline-block'}))
                
                stats_elements.append(html.Div([
                    html.Strong("Max Error: "),
                    html.Span(f"{stats['max_error']:.2f}")
                ], style={'marginRight': '30px', 'display': 'inline-block'}))
                
                stats_elements.append(html.Div([
                    html.Strong("Min Error: "),
                    html.Span(f"{stats['min_error']:.2f}")
                ], style={'marginRight': '30px', 'display': 'inline-block'}))
                
                if 'zero_error_percentage' in stats:
                    stats_elements.append(html.Div([
                        html.Strong("Zero Error: "),
                        html.Span(f"{stats['zero_error_percentage']:.1f}%")
                    ], style={'display': 'inline-block'}))

            # Prediction error metrics (ML) subsection
            ml_elements = []
            if 'mae' in stats:
                ml_elements.append(html.Div([
                    html.Strong("MAE: "),
                    html.Span(f"{stats['mae']:.2f}")
                ], style={'marginRight': '30px', 'display': 'inline-block'}))
            if 'rmse' in stats:
                ml_elements.append(html.Div([
                    html.Strong("RMSE: "),
                    html.Span(f"{stats['rmse']:.2f}")
                ], style={'marginRight': '30px', 'display': 'inline-block'}))
            if 'r2' in stats:
                ml_elements.append(html.Div([
                    html.Strong("R²: "),
                    html.Span(f"{stats['r2']:.3f}")
                ], style={'marginRight': '30px', 'display': 'inline-block'}))
            if 'bias' in stats:
                ml_elements.append(html.Div([
                    html.Strong("Bias: "),
                    html.Span(f"{stats['bias']:.2f}")
                ], style={'marginRight': '30px', 'display': 'inline-block'}))
            if 'medae' in stats:
                ml_elements.append(html.Div([
                    html.Strong("MedAE: "),
                    html.Span(f"{stats['medae']:.2f}")
                ], style={'marginRight': '30px', 'display': 'inline-block'}))
            if 'pearson_r' in stats:
                ml_elements.append(html.Div([
                    html.Strong("Pearson r: "),
                    html.Span(f"{stats['pearson_r']:.3f}")
                ], style={'display': 'inline-block'}))

            if ml_elements:
                stats_elements.append(html.Div([
                    html.H4("Prediction error metrics (ML)", style={'marginTop': '15px', 'marginBottom': '8px'}),
                    html.Div(ml_elements)
                ]))

            stats_display = html.Div([
                html.H3("Statistics", style={'marginBottom': '10px'}),
                html.Div(stats_elements)
            ]) if stats_elements else html.Div("No statistics available")
            
            return fig, stats_display
        
        # Run server
        print(f"\n{'='*60}")
        print(f"Starting interactive visualization server...")
        url = f"http://{config.host}:{config.port}"
        print(f"  Opening browser to: {url}")
        print(f"  Press Ctrl+C to stop the server")
        print(f"{'='*60}\n")
        
        # Open browser automatically
        # Use a small delay to ensure server is ready
        import threading
        import time
        
        def open_browser():
            time.sleep(1.5)  # Wait for server to start
            webbrowser.open(url)
        
        browser_thread = threading.Thread(target=open_browser, daemon=True)
        browser_thread.start()
        
        app.run(host=config.host, port=config.port, debug=False)
        
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
