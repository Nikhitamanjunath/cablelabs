#!/usr/bin/env python3
"""
Color Scale Visualizer

Visualizes the reference color scale by showing colors at different value increments.
Displays colors in increments of 5 (0, 5, 10, 15, ..., up to max_value).
Shows pixel-by-pixel information from the reference scale.
"""

import sys
from pathlib import Path
from typing import Tuple
from PIL import Image, ImageDraw, ImageFont
import argparse
import matplotlib
plt = None
for backend in ['TkAgg', 'Qt5Agg', 'MacOSX', 'QtAgg']:
    try:
        matplotlib.use(backend)
        import matplotlib.pyplot as plt
        plt.figure().close()  # Test backend
        print(f"  Using matplotlib backend: {backend}")
        break
    except Exception:
        continue

if plt is None:
    try:
        import matplotlib.pyplot as plt
        print("  Using default matplotlib backend")
    except Exception as e:
        print(f"  Warning: Could not set matplotlib backend: {e}")

import numpy as np

from color_scale import ColorScale


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """Convert RGB tuple to hex color string."""
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def show_colors_in_increments(max_value: float, increment: int = 5):
    """
    Show colors in increments based on the reference scale.
    
    Args:
        max_value: Maximum value for the scale
        increment: Increment step (default: 5)
    """
    print("="*60)
    print("Color Scale - Sample Colors")
    print("="*60)
    print(f"\nShowing colors in increments of {increment} from 0 to {max_value}")
    print(f"Based on reference scale: data/scale/scale.png\n")
    
    # Initialize color scale
    try:
        scale = ColorScale()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Generate values in increments
    values = []
    current = 0.0
    while current <= max_value:
        values.append(current)
        current += increment
    
    colors_list = []
    print(f"{'Value':<10} {'RGB':<20} {'Hex':<10} {'Color Preview'}")
    print("-" * 80)
    
    for value in values:
        color = scale.number_to_color(value, max_value=max_value)
        hex_color = rgb_to_hex(color)
        colors_list.append((value, color, hex_color))
        print(f"{value:<10.1f} {str(color):<20} {hex_color:<10} {'█' * 10}")
    
    print("\n" + "="*60)
    print("Color Scale Visualization Complete")
    print("="*60)
    
    # Create matplotlib visualization
    if plt is None:
        print("\nWarning: Matplotlib not available, skipping visual display")
        return
    
    print("\nDisplaying colors in matplotlib window...")
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
    except Exception as e:
        print(f"  Warning: Could not create matplotlib figure: {e}")
        return
    
    cols, swatch_size, spacing = 4, 1.0, 0.1
    rows = (len(colors_list) + cols - 1) // cols
    
    for idx, (value, color, hex_color) in enumerate(colors_list):
        row, col = idx // cols, idx % cols
        x = col * (swatch_size + spacing)
        y = (rows - row - 1) * (swatch_size + spacing)
        
        ax.add_patch(plt.Rectangle(
            (x, y), swatch_size, swatch_size,
            facecolor=tuple(c / 255.0 for c in color),
            edgecolor='black',
            linewidth=2
        ))
        
        ax.text(
            x + swatch_size / 2, y + swatch_size / 2,
            f"{value:.1f}\n{hex_color}",
            ha='center', va='center',
            fontsize=9,
            fontweight='bold',
            color='white' if sum(color) < 400 else 'black'
        )
    
    # Set axis limits and remove ticks
    ax.set_xlim(-spacing, cols * (swatch_size + spacing))
    ax.set_ylim(-spacing, rows * (swatch_size + spacing))
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'Color Scale - Increments of {increment} (0 to {max_value})', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.show(block=True)  # Block until window is closed
    
    # Also show pixel-by-pixel information
    print(f"\nReference scale has {len(scale.get_scale_colors())} pixels")
    print("Going through scale pixel by pixel:\n")
    
    scale_colors = scale.get_scale_colors()
    num_samples = min(20, len(scale_colors))  # Show first 20 pixels as samples
    
    print(f"{'Pixel Index':<15} {'Position Ratio':<20} {'RGB':<20} {'Hex'}")
    print("-" * 80)
    
    for i in range(0, len(scale_colors), max(1, len(scale_colors) // num_samples)):
        pixel_color = scale_colors[i]
        position_ratio = i / (len(scale_colors) - 1) if len(scale_colors) > 1 else 0.0
        value_at_position = (1.0 - position_ratio) * max_value
        hex_color = rgb_to_hex(pixel_color)
        
        print(f"{i:<15} {position_ratio:<20.4f} {str(pixel_color):<20} {hex_color}")
    
    if len(scale_colors) > num_samples:
        print(f"\n... and {len(scale_colors) - num_samples} more pixels")
        # Show last pixel
        last_idx = len(scale_colors) - 1
        last_color = scale_colors[last_idx]
        last_position_ratio = 1.0
        last_value = 0.0
        last_hex = rgb_to_hex(last_color)
        print(f"{last_idx:<15} {last_position_ratio:<20.4f} {str(last_color):<20} {last_hex}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Visualize color scale colors in increments")
    parser.add_argument(
        '--max-value',
        type=float,
        default=None,
        help='Maximum value for the scale (will prompt if not provided)'
    )
    parser.add_argument(
        '--increment',
        type=int,
        default=5,
        help='Increment step for showing colors (default: 5)'
    )
    
    args = parser.parse_args()
    
    # Prompt for max_value if not provided
    max_value = args.max_value
    if max_value is None:
        while True:
            try:
                user_input = input("\nEnter the maximum value for the scale (or press Enter for default 60.0): ").strip()
                if not user_input:
                    max_value = 60.0
                    break
                max_value = float(user_input)
                if max_value > 0:
                    break
                else:
                    print("  Please enter a positive number.")
            except ValueError:
                print("  Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                print("\n\nInterrupted by user.")
                sys.exit(0)
    
    print(f"\nUsing max value: {max_value}")
    
    try:
        show_colors_in_increments(max_value, args.increment)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
