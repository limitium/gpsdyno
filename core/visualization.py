#!/usr/bin/env python3
# GPSDyno - GPS-based vehicle power calculator
# Copyright (C) 2024 GPSDyno Contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
GPSDyno visualization module.

Contains functions for power and track chart generation using matplotlib.
"""
import logging
import os
from bisect import bisect_left

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from PIL import Image
from scipy.signal import find_peaks

try:
    from .. import config
except ImportError:
    import config

try:
    from locales.strings import LABELS
except ImportError:
    from ..locales.strings import LABELS

from .density import plot_power_methods
from .structures import (
    SPEED_ABS_MS,
    SPEED_REL_MS,
    SPEED_SPEED_KMH,
    ALT_ABS_MS,
    ALT_ALTITUDE_M,
    KMH_TO_MS,
    MS_TO_S,
)

logger = logging.getLogger(__name__)

# Cache for watermark logo to avoid repeated file I/O
_WATERMARK_CACHE = {'logo': None, 'available': False, 'initialized': False}

# Constants
WATERMARK_ALPHA = 0.15
POWER_TIME_THRESHOLD_SECONDS = 0.5
SECONDS_PER_MINUTE = 60
Y_AXIS_MARGIN_MULTIPLIER = 1.1


def format_weather_info(weather_data):
    """Format weather information for legends.

    Args:
        weather_data: dict with weather data

    Returns:
        dict: {
            'items': list of str - formatted weather strings,
            'location': str or None - location
        }
    """
    if not weather_data:
        return {'items': [], 'location': None}

    items = []
    if weather_data.get('temperature_c') is not None:
        items.append(f"T: {weather_data.get('temperature_c')}°C")
    if weather_data.get('pressure_hpa') is not None:
        items.append(f"P: {weather_data.get('pressure_hpa')} hPa")
    if weather_data.get('humidity_percent') is not None:
        items.append(f"RH: {weather_data.get('humidity_percent')}%")
    if weather_data.get('wind_speed_kph') is not None:
        wind_speed_kph = weather_data.get('wind_speed_kph')
        wind_speed_ms = wind_speed_kph / KMH_TO_MS
        items.append(f"Wind: {wind_speed_kph:.1f} km/h ({wind_speed_ms:.1f} m/s)")

    return {
        'items': items,
        'location': weather_data.get('location')
    }


def load_watermark_logo():
    """Load and prepare logo for watermark.
    
    Uses caching to avoid repeated file I/O operations.

    Returns:
        tuple: (logo_img_pil, logo_available)
    """
    # Return cached value if already loaded
    if _WATERMARK_CACHE['initialized']:
        return _WATERMARK_CACHE['logo'], _WATERMARK_CACHE['available']
    
    try:
        logo_path = config.LOGO_PATH
        logo_img_pil = Image.open(logo_path)
        logo_img_pil = logo_img_pil.convert("RGBA")

        alpha_value = int(255 * WATERMARK_ALPHA)

        # Apply alpha transparency to non-transparent pixels
        new_data = [
            (item[0], item[1], item[2], alpha_value) if item[3] > 0 else item
            for item in logo_img_pil.getdata()
        ]
        logo_img_pil.putdata(new_data)

        # Cache the result
        _WATERMARK_CACHE['logo'] = logo_img_pil
        _WATERMARK_CACHE['available'] = True
        _WATERMARK_CACHE['initialized'] = True
        return logo_img_pil, True
    except FileNotFoundError:
        logger.warning(f"Watermark file '{config.LOGO_PATH}' not found.")
        _WATERMARK_CACHE['initialized'] = True
        return None, False
    except Exception as e:
        logger.warning(f"Error processing watermark file: {e}")
        _WATERMARK_CACHE['initialized'] = True
        return None, False


def _add_logo_watermark(fig, logo_img_pil):
    """Add watermark logo to figure (centered).
    
    Args:
        fig: Matplotlib figure object
        logo_img_pil: PIL Image object or None
    """
    if logo_img_pil is None:
        return
    fig_width_px, fig_height_px = fig.get_size_inches() * fig.dpi
    logo_width_px, logo_height_px = logo_img_pil.size
    xo = (fig_width_px - logo_width_px) / 2
    yo = (fig_height_px - logo_height_px) / 2
    fig.figimage(logo_img_pil, xo=xo, yo=yo, alpha=None, zorder=10)


def _format_duration_ms(duration_ms):
    """Format duration from milliseconds to MM:SS.mmm string.
    
    Args:
        duration_ms: Duration in milliseconds
        
    Returns:
        Formatted string like "05:23.456" or empty string if None
    """
    if duration_ms is None:
        return ""
    seconds_total = duration_ms / MS_TO_S
    minutes = int(seconds_total // SECONDS_PER_MINUTE)
    seconds = int(seconds_total % SECONDS_PER_MINUTE)
    milliseconds = int((seconds_total - int(seconds_total)) * MS_TO_S)
    return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def _separate_valid_invalid_points(power_speed_data):
    """Separate valid and invalid power-speed points using NumPy.
    
    Args:
        power_speed_data: List of tuples (speed, power, is_valid)
        
    Returns:
        tuple: (speeds_valid, power_valid, speeds_invalid, power_invalid, power_at_speeds)
    """
    if not power_speed_data:
        empty_array = np.array([])
        return empty_array, empty_array, empty_array, empty_array, empty_array
    
    power_speed_array = np.array(power_speed_data)
    speeds_arr = power_speed_array[:, 0]
    powers_arr = power_speed_array[:, 1]
    
    # Create boolean mask for valid points
    if power_speed_array.shape[1] >= 3:
        valid_mask = power_speed_array[:, 2].astype(bool)
    else:
        valid_mask = np.ones(len(power_speed_array), dtype=bool)
    
    # Separate using boolean indexing
    speeds_valid = speeds_arr[valid_mask]
    power_valid = powers_arr[valid_mask]
    speeds_invalid = speeds_arr[~valid_mask]
    power_invalid = powers_arr[~valid_mask]
    
    return speeds_valid, power_valid, speeds_invalid, power_invalid, powers_arr


def _create_combined_legend(ax, power_estimation, uncertainty, filter_info, dt_info, method_legend_items):
    """Create combined legend with filter info, estimation methods, and uncertainty.
    
    Args:
        ax: Matplotlib axes object
        power_estimation: dict with power estimation data
        uncertainty: dict with uncertainty data
        filter_info: str - filter type information
        dt_info: float - time step information
        method_legend_items: list of (line, label) tuples from plot_power_methods
    """
    combined_items = []
    
    # Filter information
    combined_items.append(plt.Line2D([0], [0], marker='', linestyle='none',
                                      label=f"Filter: {filter_info}, dt={dt_info:.2f}s", color='none'))
    
    # Estimation methods
    if method_legend_items:
        combined_items.append(plt.Line2D([0], [0], marker='', linestyle='none',
                                          label='─' * 25, color='none'))
        for line, label in method_legend_items:
            line.set_label(label)
            combined_items.append(line)
    
    # Uncertainty breakdown
    if uncertainty and uncertainty.get('components'):
        combined_items.append(plt.Line2D([0], [0], marker='', linestyle='none',
                                          label='─' * 25, color='none'))
        components = uncertainty['components']
        parts = []
        if components.get('gps', {}).get('hp'):
            parts.append(f"GPS ±{components['gps']['hp']:.0f}")
        if components.get('mass', {}).get('hp'):
            parts.append(f"M ±{components['mass']['hp']:.0f}")
        
        if parts:
            breakdown_text = f"Uncertainty: {' | '.join(parts)} WHP"
            combined_items.append(plt.Line2D([0], [0], marker='', linestyle='none',
                                              label=breakdown_text, color='none'))
        
        # Consistency score
        if power_estimation and power_estimation.get('consistency_score') is not None:
            score = power_estimation['consistency_score']
            combined_items.append(plt.Line2D([0], [0], marker='', linestyle='none',
                                              label=f"Stability: {score:.0%}",
                                              color='none'))
    
    # Create legend
    if combined_items:
        combined_handles = []
        combined_labels = []
        for item in combined_items:
            if isinstance(item, tuple):
                combined_handles.append(item[0])
                combined_labels.append(item[1])
            else:
                combined_handles.append(item)
                combined_labels.append(item.get_label())
        
        combined_legend = ax.legend(
            combined_handles, combined_labels,
            loc='lower right',
            bbox_to_anchor=(1.0, 0.0),
            frameon=True,
            fontsize=config.ANNOTATION_FONTSIZE,
            fancybox=True,
            shadow=True
        )
        frame = combined_legend.get_frame()
        frame.set_facecolor('#f5f5dc')  # Beige
        frame.set_edgecolor('#8B4513')  # Brown
        frame.set_linewidth(1.2)
        ax.add_artist(combined_legend)


def _create_weather_legend(ax, weather_data, session_duration_ms):
    """Create weather and location legend.
    
    Args:
        ax: Matplotlib axes object
        weather_data: dict with weather data
        session_duration_ms: session duration in milliseconds
    """
    if not weather_data:
        return
    
    weather_info = format_weather_info(weather_data)
    legend_items = []
    
    if weather_info['location']:
        legend_items.append(plt.Line2D([0], [0], marker='', linestyle='none',
                                     label=f"Location: {weather_info['location']}", color='none'))
    
    if session_duration_ms is not None:
        formatted_duration = _format_duration_ms(session_duration_ms)
        legend_items.append(plt.Line2D([0], [0], marker='', linestyle='none',
                                       label=f"Session time: {formatted_duration}", color='none'))
    
    if weather_data.get('weather_datetime'):
        legend_items.append(plt.Line2D([0], [0], marker='', linestyle='none',
                                       label=f"Measurement time: {weather_data.get('weather_datetime')}", color='none'))
    
    if weather_info['items']:
        weather_text = "Weather: " + ", ".join(weather_info['items'])
        legend_items.append(plt.Line2D([0], [0], marker='', linestyle='none',
                                     label=weather_text, color='none'))
    
    if legend_items:
        location_legend = plt.legend(
            handles=legend_items,
            loc='lower left',
            frameon=True,
            fontsize=config.LEGEND_FONTSIZE,
            bbox_to_anchor=(-0.01, 0.0),
            fancybox=True,
            shadow=True,
        )
        frame = location_legend.get_frame()
        frame.set_facecolor('#e6f2ff')
        frame.set_edgecolor('#3399ff')
        frame.set_linewidth(1.5)
        ax.add_artist(location_legend)


def _create_warnings_legend(ax, warnings):
    """Create warnings legend.
    
    Args:
        ax: Matplotlib axes object
        warnings: list of warning messages
    """
    if not warnings:
        return
    
    warning_handles = [plt.Line2D([0], [0], marker='', linestyle='none', label=w, color='none')
                       for w in warnings]
    warn_leg = plt.legend(
        handles=warning_handles,
        loc='upper left',
        frameon=True,
        fontsize=config.ANNOTATION_FONTSIZE,
        bbox_to_anchor=(-0.01, 1.0),
        fancybox=True,
        shadow=True,
    )
    warn_frame = warn_leg.get_frame()
    warn_frame.set_facecolor('#ffe6e6')
    warn_frame.set_edgecolor('#ff9999')
    warn_frame.set_linewidth(1.2)
    ax.add_artist(warn_leg)


def _calculate_y_bounds(powers, *series_arrays):
    """Calculate Y-axis bounds from power and component series.
    
    Args:
        powers: NumPy array of power values
        *series_arrays: Variable number of series arrays (air, rolling, acceleration, slope)
        
    Returns:
        tuple: (y_lower_bound, y_upper_bound) or None if no data
    """
    if len(powers) == 0:
        return None
    
    all_y_values = powers.astype(float) if isinstance(powers, np.ndarray) else np.array(powers, dtype=float)
    
    # Collect y values from all series
    series_y_values = []
    for series_array in series_arrays:
        if series_array is not None:
            if series_array is series_arrays[2]:  # acceleration_force_array (index 2)
                series_y_values.append(np.maximum(0, series_array[:, 1]))
            else:
                series_y_values.append(series_array[:, 1])
    
    if series_y_values:
        all_y_values = np.concatenate([all_y_values] + series_y_values)
    
    if len(all_y_values) == 0:
        return None
    
    min_y = float(np.min(all_y_values))
    max_y = float(np.max(all_y_values))
    y_margin_factor = config.Y_MARGIN_FACTOR
    
    y_lower_bound = 0 if min_y >= 0 else min_y - abs(min_y * y_margin_factor)
    y_upper_bound = max_y + abs(max_y * y_margin_factor) if max_y > 0 else 1
    
    if min_y == max_y:
        y_lower_bound = min_y - 1 if min_y != 0 else -1
        y_upper_bound = max_y + 1 if max_y != 0 else 1
    
    return y_lower_bound, y_upper_bound


def plot_power_chart(power_data, car_info, output_file=None, session_duration_ms=None,
                     warnings_dict=None, cautions_dict=None, weather_data=None,
                     show_legends_on_second=False):
    """Build power charts (by speed and by time).

    Args:
        power_data: dict with power data from calculate_power()
        car_info: dict with vehicle parameters
        output_file: base path for saving (without extension creates _speed.png and _time.png)
        session_duration_ms: session duration in ms
        warnings_dict: dict with warnings
        cautions_dict: dict with cautions
        weather_data: dict with weather data
        show_legends_on_second: show legends on second chart (Power vs Time)

    Returns:
        dict: {
            'chart_paths': dict with file paths,
            'power_estimation': dict
        }
    """
    if not power_data or not power_data.get('power_time'):
        logger.warning("No power data for chart building")
        return None

    logo_img_pil, logo_available = load_watermark_logo()
    power_time_data = power_data['power_time']
    result = {'chart_paths': {}}

    # Prepare output filenames
    if output_file:
        base_filename = os.path.splitext(output_file)[0]
        extension = os.path.splitext(output_file)[1] or '.png'
        power_by_time_filename = f"{base_filename}_time{extension}"
        power_by_speed_filename = f"{base_filename}_speed{extension}"
        result['chart_paths']['power_by_time'] = power_by_time_filename
        result['chart_paths']['power_by_speed'] = power_by_speed_filename
    else:
        power_by_time_filename = None
        power_by_speed_filename = None

    # Extract car information
    car_name = car_info.get("name", "Unknown vehicle")
    mass = car_info.get("mass", "?")
    drag_coef = car_info.get("drag_coefficient", "?")
    frontal_area = car_info.get("frontal_area", "?")
    rolling_resistance = car_info.get("rolling_resistance", 0.015)

    # Format title with car parameters
    rolling_resistance_str = f"{rolling_resistance:.3f}" if rolling_resistance is not None else "?"
    if isinstance(drag_coef, (int, float)) and isinstance(frontal_area, (int, float)):
        aero_final_coef = round(drag_coef * frontal_area, 3)
    else:
        aero_final_coef = "?"
    title = (f"{car_name} (Mass: {mass} kg, Cd: {drag_coef}, A: {frontal_area} m², "
             f"Aero:{aero_final_coef}, Crr: {rolling_resistance_str})")

    # Collect warnings and cautions
    warnings = []
    if warnings_dict:
        warnings.extend(msg for msg in warnings_dict.values() if msg)
    if cautions_dict:
        warnings.extend(msg for msg in cautions_dict.values() if msg)

    # ===== CHART 1: Power by Speed (Power vs Speed) =====
    plt.figure(figsize=(12, 8))
    fig1 = plt.gcf()
    if logo_available:
        _add_logo_watermark(fig1, logo_img_pil)

    # Separate valid and invalid points
    power_speed_data = power_data['power_speed']
    speeds_valid, power_valid, speeds_invalid, power_invalid, power_at_speeds = \
        _separate_valid_invalid_points(power_speed_data)

    # Plot scatter points
    if len(speeds_valid) > 0:
        plt.scatter(speeds_valid, power_valid, color='#2980b9', s=15, alpha=config.ALPHA_VALID)
    if len(speeds_invalid) > 0:
        plt.scatter(speeds_invalid, power_invalid, color='red', s=15, alpha=config.ALPHA_INVALID)

    plt.grid(True, linestyle='--', alpha=0.7)

    # Draw power estimation method lines
    power_estimation = power_data.get('power_estimation')
    uncertainty = power_data.get('uncertainty')
    ax = plt.gca()
    filter_info = power_data.get('filter', 'unknown')
    dt_info = power_data.get('dt', 0.0)
    method_legend_items = []

    if power_estimation and len(speeds_valid) > 0:
        try:
            x_min = float(np.min(speeds_valid)) if len(speeds_valid) > 0 else 0
            x_max = float(np.max(speeds_valid)) if len(speeds_valid) > 0 else 200
            method_legend_items = plot_power_methods(ax, power_estimation, (x_min, x_max), uncertainty)
            result['power_estimation'] = power_estimation
        except Exception as e:
            logger.error(f"Error building estimation method lines: {e}")

    # Create legends
    _create_combined_legend(ax, power_estimation, uncertainty, filter_info, dt_info, method_legend_items)
    _create_weather_legend(ax, weather_data, session_duration_ms)
    _create_warnings_legend(ax, warnings)

    # Set chart labels and limits
    plt.title(f"{title}\n{LABELS['power_by_speed_title']}")
    plt.xlabel(LABELS['speed_axis'])
    plt.ylabel(LABELS['power_axis'])

    if len(power_at_speeds) > 0:
        plt.ylim(0, float(np.max(power_at_speeds)) * Y_AXIS_MARGIN_MULTIPLIER)

    # Add estimation method values as Y-axis ticks
    if power_estimation:
        method_values = [
            round(m['value']) for m in power_estimation.get('methods', {}).values()
            if m.get('value')
        ]
        if method_values:
            current_ticks = list(plt.gca().get_yticks())
            new_ticks = sorted(set(current_ticks + method_values))
            plt.gca().set_yticks(new_ticks)

    plt.tight_layout()

    if power_by_speed_filename:
        plt.savefig(power_by_speed_filename, dpi=150, bbox_inches='tight', facecolor='white')
    elif not output_file:
        plt.show()

    plt.close()

    # ===== CHART 2: Power by Time (Power vs Time) =====
    plt.figure(figsize=(12, 8))
    fig2 = plt.gcf()
    if logo_available:
        _add_logo_watermark(fig2, logo_img_pil)

    # Extract time and power data
    if power_time_data:
        power_time_array = np.array(power_time_data)
        times = power_time_array[:, 0]
        powers = power_time_array[:, 1]
    else:
        times = np.array([])
        powers = np.array([])

    # Plot main power line
    plt.plot(times, powers, '-', linewidth=1, color='#2980b9', alpha=0.7, label=LABELS['total_power'])
    plt.grid(True, linestyle='--', alpha=0.7)

    # Get component series and convert to arrays
    air_resistance_series = power_data.get('air_resistance_time', [])
    rolling_force_series = power_data.get('rolling_force_time', [])
    acceleration_force_series = power_data.get('acceleration_force_time', [])
    slope_force_series = power_data.get('slope_force_time', [])
    
    air_resistance_array = np.array(air_resistance_series) if air_resistance_series else None
    rolling_force_array = np.array(rolling_force_series) if rolling_force_series else None
    acceleration_force_array = np.array(acceleration_force_series) if acceleration_force_series else None
    slope_force_array = np.array(slope_force_series) if slope_force_series else None

    # Plot component series
    if air_resistance_array is not None:
        plt.plot(air_resistance_array[:, 0], air_resistance_array[:, 1], 
                label=LABELS['air_resistance'], color='blue', linestyle='--', alpha=0.7)

    if rolling_force_array is not None:
        plt.plot(rolling_force_array[:, 0], rolling_force_array[:, 1], 
                label=LABELS['rolling_resistance'], color='green', linestyle=':', alpha=0.7)

    if acceleration_force_array is not None:
        plt.plot(acceleration_force_array[:, 0], np.maximum(0, acceleration_force_array[:, 1]), 
                label=LABELS['acceleration_power'], color='red', linestyle='-.', alpha=0.7)

    if slope_force_array is not None:
        plt.plot(slope_force_array[:, 0], slope_force_array[:, 1], 
                label=LABELS['slope_resistance'], color='orange', linestyle='-.', alpha=0.7)

    # Create main legend
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        main_leg = plt.legend(handles=handles, labels=labels, loc=config.MAIN_LEGEND_LOCATION,
                             frameon=True, fontsize=config.ANNOTATION_FONTSIZE, fancybox=True, shadow=True)
        frame_main_leg = main_leg.get_frame()
        frame_main_leg.set_facecolor('#f8f8f8')
        frame_main_leg.set_edgecolor('#aaaaaa')
        frame_main_leg.set_linewidth(0.8)
        plt.gca().add_artist(main_leg)

    plt.title(f"{title}\n{LABELS['power_by_time_title']}")
    plt.xlabel(LABELS['time_axis'])
    plt.ylabel(LABELS['power_axis'])

    # Set Y-axis bounds
    y_bounds = _calculate_y_bounds(powers, air_resistance_array, rolling_force_array, 
                                   acceleration_force_array, slope_force_array)
    if y_bounds:
        plt.ylim(y_bounds[0], y_bounds[1])

    plt.tight_layout()

    if power_by_time_filename:
        plt.savefig(power_by_time_filename, dpi=150, bbox_inches='tight', facecolor='white')
    elif not output_file:
        plt.show()

    plt.close()

    return result


def correlate_track_data(coords_data, speed_data, altitude_data, power_data):
    """Correlate coordinates with power/speed/altitude by timestamp.

    Args:
        coords_data: list of (timestamp_ms, lat, lon)
        speed_data: list of tuples from NMEA parser
        altitude_data: list of tuples from NMEA parser
        power_data: result of calculate_power()

    Returns:
        dict: {'coords': [...], 'speeds': [...], 'altitudes': [...], 'powers': [...]}
    """
    # Build timestamp -> value maps for O(1) lookup
    speed_by_ts = {
        s[SPEED_ABS_MS]: (s[SPEED_SPEED_KMH], s[SPEED_REL_MS]) for s in speed_data
    }
    alt_by_ts = {a[ALT_ABS_MS]: a[ALT_ALTITUDE_M] for a in altitude_data}

    # Build sorted arrays for power lookup (binary search - O(log n))
    power_time_data = power_data.get('power_time', [])
    first_ts = power_data.get('first_timestamp_ms', coords_data[0][0] if coords_data else 0)
    
    if power_time_data:
        power_times = np.array([p[0] for p in power_time_data])
        power_values = np.array([p[1] for p in power_time_data])
        sort_idx = np.argsort(power_times)
        power_times_sorted = power_times[sort_idx]
        power_values_sorted = power_values[sort_idx]
    else:
        power_times_sorted = np.array([])
        power_values_sorted = np.array([])

    result = {'coords': [], 'speeds': [], 'altitudes': [], 'powers': []}

    for ts_ms, lat, lon in coords_data:
        result['coords'].append((lat, lon))

        # Speed lookup
        if ts_ms in speed_by_ts:
            speed, _ = speed_by_ts[ts_ms]
            result['speeds'].append(speed)
        else:
            result['speeds'].append(0)

        # Altitude lookup
        result['altitudes'].append(alt_by_ts.get(ts_ms, 0))

        # Power lookup using binary search
        rel_s = (ts_ms - first_ts) / MS_TO_S
        if len(power_times_sorted) > 0:
            idx = bisect_left(power_times_sorted, rel_s)
            
            # Check neighbors for closest match
            candidates = []
            if idx > 0:
                candidates.append((abs(power_times_sorted[idx - 1] - rel_s), power_values_sorted[idx - 1]))
            if idx < len(power_times_sorted):
                candidates.append((abs(power_times_sorted[idx] - rel_s), power_values_sorted[idx]))
            
            if candidates:
                closest_dist, closest_power = min(candidates, key=lambda x: x[0])
                if closest_dist < POWER_TIME_THRESHOLD_SECONDS:
                    result['powers'].append(max(0, closest_power))
                else:
                    result['powers'].append(0)
            else:
                result['powers'].append(0)
        else:
            result['powers'].append(0)

    return result


def plot_track_pseudo3d(coords, altitudes, powers, output_file=None, location=None):
    """Build pseudo-3D track map with isometry and shadow.

    Track is displayed with upward offset proportional to altitude,
    with gray "shadow" below and vertical pillars.
    Line color encodes power.

    Args:
        coords: list of tuples (lat, lon)
        altitudes: list of altitudes in meters
        powers: list of powers in WHP
        output_file: base path for saving
        location: location name for title

    Returns:
        str: path to saved file or None
    """
    if not coords or len(coords) < 2:
        return None

    # Extract and normalize coordinates
    coords_array = np.array(coords)
    lats = coords_array[:, 0]
    lons = coords_array[:, 1]
    
    # Normalize altitude for offset
    altitudes_array = np.array(altitudes, dtype=float)
    alt_min = float(np.min(altitudes_array))
    alt_max = float(np.max(altitudes_array))
    alt_range = max(1.0, alt_max - alt_min)
    alt_norm = ((altitudes_array - alt_min) / alt_range).tolist()

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Shadow (gray, no offset)
    ax.plot(lons, lats, color='#CCCCCC',
            linewidth=config.TRACK_SHADOW_WIDTH,
            alpha=config.TRACK_SHADOW_ALPHA, zorder=1)

    # Vertical "pillars" every N points
    step = max(1, len(lats) // config.TRACK_PILLAR_STEP)
    for i in range(0, len(lats), step):
        elevated_lat = lats[i] + alt_norm[i] * config.TRACK_OFFSET_SCALE
        ax.plot([lons[i], lons[i]], [lats[i], elevated_lat],
               color='gray', alpha=config.TRACK_PILLAR_ALPHA,
               linewidth=0.8, zorder=2)

    # Elevated track coordinates
    lats_elevated = (lats + np.array(alt_norm) * config.TRACK_OFFSET_SCALE).tolist()

    # Create line segments for LineCollection
    points = np.array([lons, lats_elevated]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Custom colormap: blue -> red (burgundy)
    track_colors = ['#2166ac', '#4393c3', '#92c5de', '#f4a582', '#d6604d', '#b2182b', '#67001f']
    track_cmap = LinearSegmentedColormap.from_list('power_track', track_colors)

    # Color normalization by power
    powers_array = np.array(powers, dtype=float)
    power_min = float(np.min(powers_array))
    power_max = float(np.max(powers_array))
    if power_max <= power_min:
        power_max = power_min + 1.0  # Avoid division by zero

    norm = plt.Normalize(power_min, power_max)
    lc = LineCollection(segments, cmap=track_cmap, norm=norm,
                        linewidth=config.TRACK_LINE_WIDTH, zorder=3)
    lc.set_array(np.array(powers[:-1]))

    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect('equal')
    ax.axis('off')

    # Colorbar
    cbar = fig.colorbar(lc, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label(LABELS.get('track_power_label', 'Power (HP)'), fontsize=12)

    # Find and label power peaks
    peaks, _ = find_peaks(
        powers_array,
        prominence=config.TRACK_PEAK_PROMINENCE,
        distance=config.TRACK_PEAK_DISTANCE
    )

    # Filter peaks by minimum power threshold
    min_power_threshold = power_max * config.TRACK_PEAK_MIN_POWER_RATIO
    peak_powers = powers_array[peaks]
    valid_peaks = peaks[peak_powers >= min_power_threshold]
    
    for peak_idx in valid_peaks:
        power_val = float(powers_array[peak_idx])
        x = lons[peak_idx]
        y = lats_elevated[peak_idx]
        ax.annotate(
            f'{power_val:.0f}',
            xy=(x, y),
            fontsize=config.TRACK_PEAK_FONTSIZE,
            fontweight='bold',
            color=config.TRACK_PEAK_COLOR,
            ha='center',
            va='bottom',
            zorder=6
        )

    # Title
    if location:
        ax.set_title(location, fontsize=14, fontweight='bold')

    # Add small watermark in bottom left corner
    logo_img, logo_available = load_watermark_logo()
    if logo_available and logo_img:
        new_width = logo_img.width // 4
        new_height = logo_img.height // 4
        logo_small = logo_img.resize((new_width, new_height))
        fig.figimage(logo_small, xo=20, yo=20, alpha=0.6, zorder=10)

    # Save or show
    track_filename = None
    if output_file:
        base_filename = os.path.splitext(output_file)[0]
        track_filename = f"{base_filename}_track.png"
        plt.savefig(track_filename, dpi=150, bbox_inches='tight', facecolor='white')
    else:
        plt.show()

    plt.close()
    return track_filename


def plot_track(coords, output_file=None, location=None):
    """Build simple track map (legacy function for backward compatibility).

    Args:
        coords: list of tuples (lat, lon)
        output_file: base path for saving
        location: location name for title

    Returns:
        str: path to saved file or None
    """
    if not coords:
        return None

    lats, lons = zip(*coords)

    plt.figure(figsize=(8, 8))
    plt.plot(lons, lats, '-', linewidth=1, color='blue')

    if location:
        plt.title(location)
    else:
        plt.title('Track')

    plt.axis('equal')
    plt.axis('off')
    plt.grid(False)

    track_filename = None
    if output_file:
        base_filename = os.path.splitext(output_file)[0]
        track_filename = f"{base_filename}_track.png"
        plt.savefig(track_filename, dpi=150, transparent=True, bbox_inches='tight')
    else:
        plt.show()

    plt.close()
    return track_filename
