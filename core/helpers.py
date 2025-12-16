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
Helper functions for power calculation.
Extracted from calculator.py to improve readability and maintainability.
"""
import logging

import numpy as np

try:
    from .. import config
except ImportError:
    import config

from .structures import PERCENT_TO_DECIMAL, MS_TO_S

logger = logging.getLogger(__name__)


def calculate_air_density_from_weather(weather_data, r_dry_air, r_vapor):
    """
    Calculate air density from weather data using psychrometric equations.

    Args:
        weather_data: dict with temperature_c, pressure_hpa, humidity_percent
        r_dry_air: Gas constant for dry air (J/(kg·K))
        r_vapor: Gas constant for water vapor (J/(kg·K))

    Returns:
        float: Air density in kg/m³
    """
    temp_c = weather_data.get("temperature_c", 20.0)
    pressure_hpa = weather_data.get("pressure_hpa", 1013.25)
    humidity = weather_data.get("humidity_percent", 50.0)

    temp_k = temp_c + 273.15
    pressure_pa = pressure_hpa * PERCENT_TO_DECIMAL

    # Saturation vapor pressure (Magnus formula)
    P_sat_hpa = 6.1094 * np.exp((17.625 * temp_c) / (temp_c + 243.04))
    P_sat_pa = P_sat_hpa * PERCENT_TO_DECIMAL

    # Partial pressures
    P_v_pa = (humidity / PERCENT_TO_DECIMAL) * P_sat_pa
    P_d_pa = pressure_pa - P_v_pa

    # Air density from ideal gas law
    rho = (P_d_pa / (r_dry_air * temp_k)) + (P_v_pa / (r_vapor * temp_k))
    return rho


def process_altitude_data_for_slopes(
    altitude_data,
    time_seconds,
    speeds_smooth,
    first_timestamp_ms_arg,
    kalman_altitude_func,
    interp1d_func,
    calculate_savgol_window_size_func,
    apply_savgol_filter_func,
    savgol_window_length,
    savgol_window_divisor,
    min_distance_for_slope,
    max_safe_slope
):
    """
    Process altitude data to calculate road slopes.

    Args:
        altitude_data: list of altitude tuples
        time_seconds: array of time values in seconds
        speeds_smooth: array of smoothed speeds in m/s
        first_timestamp_ms_arg: first timestamp in ms
        kalman_altitude_func: function to smooth altitude
        interp1d_func: interpolation function
        calculate_savgol_window_size_func: function to calculate window size
        apply_savgol_filter_func: function to apply Savitzky-Golay filter
        savgol_window_length: base window length
        savgol_window_divisor: divisor for dynamic sizing
        min_distance_for_slope: minimum distance for slope calculation
        max_safe_slope: maximum safe slope value

    Returns:
        np.ndarray: slopes array (same length as time_seconds)
    """
    from .structures import ALT_ABS_MS, ALT_ALTITUDE_M

    slopes = np.zeros_like(speeds_smooth)

    if not altitude_data or len(altitude_data) < 3:
        return slopes

    try:
        # Pre-allocate arrays for altitude data
        n_alt = len(altitude_data)
        alt_abs_times_ms = np.empty(n_alt, dtype=np.int64)
        alt_values = np.empty(n_alt, dtype=float)

        for i, point in enumerate(altitude_data):
            alt_abs_times_ms[i] = point[ALT_ABS_MS]
            alt_values[i] = point[ALT_ALTITUDE_M]

        if len(alt_values) < 3:
            return slopes

        # Calculate first timestamp
        first_timestamp_sec_val = 0.0
        if first_timestamp_ms_arg is not None:
            first_timestamp_sec_val = first_timestamp_ms_arg / MS_TO_S
        elif len(alt_abs_times_ms) > 0:
            first_timestamp_sec_val = alt_abs_times_ms[0] / MS_TO_S

        # Convert to relative time in seconds
        alt_times_sec = (alt_abs_times_ms / MS_TO_S) - first_timestamp_sec_val

        # Smooth altitude with Kalman filter
        altitude_smooth = kalman_altitude_func(alt_values, alt_times_sec)

        # Handle duplicate timestamps
        if len(alt_times_sec) != len(np.unique(alt_times_sec)):
            unique_times, unique_indices = np.unique(alt_times_sec, return_inverse=True)
            unique_values = np.zeros_like(unique_times)
            for i, t in enumerate(unique_times):
                mask = unique_indices == i
                unique_values[i] = np.mean(altitude_smooth[mask])
            alt_times_sec = unique_times
            altitude_smooth = unique_values

        if len(alt_times_sec) < 2:
            return slopes

        # Interpolate altitude to speed time grid
        alt_interp = interp1d_func(
            alt_times_sec, altitude_smooth,
            kind='linear', bounds_error=False,
            fill_value=(altitude_smooth[0], altitude_smooth[-1])
        )
        altitude_at_speed_times = alt_interp(time_seconds)

        # Vectorized distance calculation
        dt_array = np.diff(time_seconds, prepend=time_seconds[0])
        distance_traveled = speeds_smooth * dt_array

        # Smooth distance with Savitzky-Golay
        window_size = calculate_savgol_window_size_func(
            len(distance_traveled),
            savgol_window_length,
            savgol_window_divisor
        )
        distance_traveled_smooth = apply_savgol_filter_func(distance_traveled, window_size)
        distance_traveled_smooth = np.maximum(distance_traveled_smooth, 0)

        # Vectorized slope calculation
        alt_diff = np.diff(altitude_at_speed_times, prepend=altitude_at_speed_times[0])
        valid_distance_mask = distance_traveled_smooth > min_distance_for_slope
        tan_slope = np.zeros_like(slopes)
        tan_slope[valid_distance_mask] = alt_diff[valid_distance_mask] / distance_traveled_smooth[valid_distance_mask]
        slopes = tan_slope / np.sqrt(1 + tan_slope**2)

        # Clip and smooth slopes
        slopes = np.clip(slopes, -max_safe_slope, max_safe_slope)
        window_size = calculate_savgol_window_size_func(
            len(slopes),
            savgol_window_length,
            savgol_window_divisor
        )
        slopes = apply_savgol_filter_func(slopes, window_size)

    except Exception as e:
        logger.error(f"Error processing altitude data: {e}", exc_info=True)

    return slopes


def calculate_uncertainty_data(
    power_estimation,
    valid_flags,
    power_hp_array,
    speed_kmh_array,
    acceleration_smooth,
    slopes,
    time_seconds,
    hdop_statistics,
    weather_data,
    drag_coefficient,
    frontal_area,
    rolling_resistance,
    rho,
    uncertainty_mass_kg
):
    """
    Extract data needed for uncertainty calculation from power calculation results.

    Args:
        power_estimation: dict with power estimation results
        valid_flags: boolean array of valid points
        power_hp_array: array of power values in HP
        speed_kmh_array: array of speeds in km/h
        acceleration_smooth: array of smoothed accelerations
        slopes: array of road slopes
        time_seconds: array of time values in seconds
        hdop_statistics: dict with HDOP statistics
        weather_data: dict with weather data
        drag_coefficient: drag coefficient
        frontal_area: frontal area
        rolling_resistance: rolling resistance coefficient
        rho: air density
        uncertainty_mass_kg: mass uncertainty in kg

    Returns:
        dict with uncertainty calculation inputs, or None if cannot calculate
    """
    from .structures import (
        KMH_TO_MS,
        HIGH_POWER_THRESHOLD_RATIO,
        DEFAULT_ACCELERATION_MS2,
        DEFAULT_GPS_FREQUENCY_HZ,
        DEFAULT_HDOP_MEAN,
        DEFAULT_CONSISTENCY_SCORE,
        FALLBACK_MAX_POWER_SPEED_KMH,
    )

    if not power_estimation or not power_estimation.get('recommended_value'):
        return None

    recommended_power = power_estimation['recommended_value']

    # Find max power point among valid ones
    if np.any(valid_flags):
        valid_powers = power_hp_array[valid_flags]
        valid_speeds_kmh = speed_kmh_array[valid_flags]
        max_power_val = np.max(valid_powers)
        max_power_idx_in_valid = np.argmax(valid_powers)
        max_power_speed_kmh = valid_speeds_kmh[max_power_idx_in_valid]
    else:
        max_power_val = 0.0
        max_power_speed_kmh = FALLBACK_MAX_POWER_SPEED_KMH

    v_car_ms = max_power_speed_kmh / KMH_TO_MS

    # Wind speed from weather
    wind_kph = 0.0
    if weather_data:
        wind_kph = weather_data.get('wind_speed_kph', 0.0)

    # Vectorized: average acceleration in high power zone
    high_power_threshold = max_power_val * HIGH_POWER_THRESHOLD_RATIO if max_power_val > 0 else 0
    high_power_mask = valid_flags & (power_hp_array >= high_power_threshold)
    valid_indices = np.where(high_power_mask)[0]
    
    if len(valid_indices) > 0 and len(acceleration_smooth) > 0:
        bounded_indices = valid_indices[valid_indices < len(acceleration_smooth)]
        if len(bounded_indices) > 0:
            a_ms2 = float(np.mean(np.abs(acceleration_smooth[bounded_indices])))
        else:
            a_ms2 = DEFAULT_ACCELERATION_MS2
    else:
        a_ms2 = DEFAULT_ACCELERATION_MS2

    # Vectorized: average slope in high power zone
    if len(valid_indices) > 0 and len(slopes) > 0:
        bounded_indices = valid_indices[valid_indices < len(slopes)]
        if len(bounded_indices) > 0:
            slope_avg = float(np.mean(slopes[bounded_indices]))
        else:
            slope_avg = 0.0
    else:
        slope_avg = 0.0

    # GPS frequency
    gps_frequency_hz = DEFAULT_GPS_FREQUENCY_HZ
    if len(time_seconds) > 1:
        time_diffs = np.diff(time_seconds)
        valid_diffs = time_diffs[time_diffs > 0]
        if len(valid_diffs) > 0:
            avg_interval = np.mean(valid_diffs)
            if avg_interval > 0:
                gps_frequency_hz = 1.0 / avg_interval

    # Average HDOP
    hdop_mean = hdop_statistics['mean'] if hdop_statistics else DEFAULT_HDOP_MEAN

    # Consistency score from estimation methods
    consistency_score = power_estimation.get('consistency_score', DEFAULT_CONSISTENCY_SCORE)

    return {
        'v_car_ms': v_car_ms,
        'wind_kph': wind_kph,
        'rho': rho,
        'cd': drag_coefficient,
        'frontal_area': frontal_area,
        'a_ms2': a_ms2,
        'crr': rolling_resistance,
        'slope_avg': slope_avg,
        'power_hp': recommended_power,
        'hdop_mean': hdop_mean,
        'gps_frequency_hz': gps_frequency_hz,
        'consistency_score': consistency_score,
        'uncertainty_mass_kg': uncertainty_mass_kg,
    }

