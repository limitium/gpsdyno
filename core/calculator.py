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
Main GPSDyno calculation engine.
Contains calculate_power function for wheel horsepower calculation from GPS data.
"""
import numpy as np
import logging
from scipy.interpolate import interp1d

try:
    from .. import config
except ImportError:
    import config

from .filters import (
    kalman_cv,
    kalman_cv_adaptive,
    kalman_altitude,
    validate_param,
    apply_savgol_filter,
    calculate_savgol_window_size,
    pre_kalman_filter,
)
from .density import calculate_power_estimation
from .uncertainty import calculate_total_uncertainty
from .helpers import (
    calculate_air_density_from_weather,
    process_altitude_data_for_slopes,
    calculate_uncertainty_data,
    calculate_headings_from_coords,
    calculate_relative_wind_speed,
)
from .structures import (
    SPEED_ABS_MS,
    SPEED_REL_MS,
    SPEED_TIME_STR,
    SPEED_SPEED_KMH,
    SPEED_NUM_SATS,
    SPEED_HDOP,
    ALT_ABS_MS,
    ALT_ALTITUDE_M,
    KMH_TO_MS,
    MS_TO_S,
    PERCENT_TO_DECIMAL,
    HIGH_POWER_THRESHOLD_RATIO,
    DEFAULT_ACCELERATION_MS2,
    DEFAULT_GPS_FREQUENCY_HZ,
    DEFAULT_HDOP_MEAN,
    DEFAULT_CONSISTENCY_SCORE,
    FALLBACK_MAX_POWER_SPEED_KMH,
    AERODYNAMIC_FORCE_COEFFICIENT,
)

logger = logging.getLogger(__name__)


def calculate_power(speed_data, car_info, weather_data=None, altitude_data=None,
                    first_timestamp_ms_arg=None, methods=None, coords_data=None):
    """
    Calculate wheel horsepower from GPS data.

    Uses two-pass adaptive Kalman filter with HDOP consideration.

    Args:
        speed_data: list of tuples (abs_ms, rel_ms, time_str, speed_kmh, nmea, sats, hdop)
        car_info: dict with vehicle params (mass, drag_coefficient, frontal_area, rolling_resistance)
        weather_data: dict with weather data (temperature_c, pressure_hpa, humidity_percent, wind_speed_kph, wind_direction_deg)
        altitude_data: list of tuples with altitude data
        first_timestamp_ms_arg: first timestamp in ms (for synchronization)
        methods: list of power estimation methods (default: ['peak_detection'])
        coords_data: list of coordinate tuples (timestamp_ms, lat, lon) for heading calculation

    Returns:
        dict: {
            'power_time': [(time_s, power_hp), ...],
            'power_speed': [(speed_kmh, power_hp, is_valid), ...],
            'air_resistance_time': [(time_s, power_hp), ...],
            'rolling_force_time': [(time_s, power_hp), ...],
            'acceleration_force_time': [(time_s, power_hp), ...],
            'slope_force_time': [(time_s, power_hp), ...],
            'power_estimation': dict or None,
            'filter': str,
            'dt': float,
            'time_map': dict,
            'first_timestamp_ms': int,
            'hdop_statistics': dict,
            'pre_kalman_stats': dict,
            'filtered_ratio': float,
            'valid_points': int,
            'total_points': int
        }
        or None if insufficient data
    """
    if not speed_data or len(speed_data) < 10:
        logger.warning("Insufficient speed data for power calculation")
        return None

    # Cache frequently-used config values for performance
    hdop_fallback = getattr(config, 'HDOP_FALLBACK_VALUE', 0.7)
    hdop_min_bound = getattr(config, 'HDOP_MIN_BOUND', 0.5)
    hdop_max_bound = getattr(config, 'HDOP_MAX_BOUND', 2.0)
    min_satellites = config.MIN_SATELLITES
    hdop_percentile = config.HDOP_PERCENTILE
    pre_kalman_min_valid_points = getattr(config, 'PRE_KALMAN_MIN_VALID_POINTS', 10)
    watts_to_hp = getattr(config, 'WATTS_TO_HP', 735.5)
    speed_percentile_low = config.SPEED_PERCENTILE_LOW
    speed_percentile_high = config.SPEED_PERCENTILE_HIGH
    gravity = config.GRAVITY
    air_density_default = config.AIR_DENSITY
    r_dry_air = getattr(config, 'R_DRY_AIR', 287.058)
    r_vapor = getattr(config, 'R_VAPOR', 461.495)
    savgol_window_length = getattr(config, 'SAVGOL_WINDOW_LENGTH', 11)
    savgol_window_divisor = getattr(config, 'SAVGOL_WINDOW_DIVISOR', 5)
    min_distance_for_slope = getattr(config, 'MIN_DISTANCE_FOR_SLOPE', 0.5)
    max_safe_slope = config.MAX_SAFE_SLOPE
    kalman_low_power_filter_percentile = getattr(config, 'KALMAN_LOW_POWER_FILTER_PERCENTILE', 95)
    kalman_hdop_factor = getattr(config, 'KALMAN_HDOP_FACTOR', 0.5)
    kalman_speed_r = getattr(config, 'KALMAN_SPEED_R', 0.2)
    kalman_r_min_multiplier = getattr(config, 'KALMAN_R_MIN_MULTIPLIER', 0.5)
    kalman_r_max_multiplier = getattr(config, 'KALMAN_R_MAX_MULTIPLIER', 3.0)
    uncertainty_mass_kg = getattr(config, 'UNCERTAINTY_MASS_KG', 20)

    n_points = len(speed_data)

    # Pre-allocate arrays/lists for speed_data fields
    abs_times_ms = np.empty(n_points, dtype=np.int64)
    rel_times_ms = np.empty(n_points, dtype=np.int64)
    speeds_kmh = np.empty(n_points, dtype=float)
    times_str = [None] * n_points
    num_sats_arr = np.empty(n_points, dtype=np.int32)
    
    # Collect HDOP values for percentile calculation while unpacking tuples
    # Use fallback value for missing HDOP (matching old behavior)
    hdop_per_point = np.full(n_points, hdop_fallback, dtype=float)
    hdop_values = []
    for i, point in enumerate(speed_data):
        abs_times_ms[i] = point[SPEED_ABS_MS]
        rel_times_ms[i] = point[SPEED_REL_MS]
        times_str[i] = point[SPEED_TIME_STR]
        speeds_kmh[i] = point[SPEED_SPEED_KMH]

        num_sats = point[SPEED_NUM_SATS] if len(point) > SPEED_NUM_SATS else None
        hdop_val = point[SPEED_HDOP] if len(point) > SPEED_HDOP else None
        num_sats_arr[i] = -1 if num_sats is None else int(num_sats)

        # Use fallback for missing HDOP (matching old behavior for Kalman filter)
        if hdop_val is not None and hdop_val > 0:
            hdop_per_point[i] = float(hdop_val)
            hdop_values.append(float(hdop_val))
        else:
            hdop_per_point[i] = hdop_fallback

    # Calculate adaptive HDOP threshold
    hdop_threshold = hdop_fallback  # Default fallback
    hdop_statistics = None
    if hdop_values:
        hdop_threshold = np.percentile(hdop_values, hdop_percentile)
        # Ensure reasonable bounds
        hdop_threshold = max(hdop_min_bound, min(hdop_threshold, hdop_max_bound))
        hdop_statistics = {
            'threshold': hdop_threshold,
            'percentile': config.HDOP_PERCENTILE,
            'min': float(np.min(hdop_values)),
            'max': float(np.max(hdop_values)),
            'mean': float(np.mean(hdop_values)),
            'median': float(np.median(hdop_values)),
            'count': len(hdop_values)
        }

    # Build validity mask using vectorized operations
    valid_flags = np.ones(n_points, dtype=bool)

    # Satellite-based validity: only when value is present (>=0)
    sats_known = num_sats_arr >= 0
    sats_low = sats_known & (num_sats_arr < min_satellites)
    valid_flags[sats_low] = False

    # HDOP-based validity: check against threshold
    hdop_bad = hdop_per_point > hdop_threshold
    valid_flags[hdop_bad] = False

    # Convert times/speeds to NumPy arrays
    time_seconds = rel_times_ms.astype(float) / MS_TO_S
    speeds_ms = speeds_kmh / KMH_TO_MS

    # Extract vehicle parameters (needed for two-pass calculation)
    mass = validate_param("mass", car_info.get("mass"), 1500, min_value=100)
    drag_coefficient = validate_param("drag_coefficient", car_info.get("drag_coefficient"), 0.30, min_value=0.1)
    frontal_area = validate_param("frontal_area", car_info.get("frontal_area"), 2.2, min_value=0.5)

    # === PRE-KALMAN FILTERING ===
    pre_kalman_stats = pre_kalman_filter(speed_data, hdop_threshold)

    pre_kalman_valid_indices = pre_kalman_stats['valid_indices']

    # Check: enough valid points?
    if len(pre_kalman_valid_indices) >= pre_kalman_min_valid_points:
        # Extract valid data for Kalman
        time_seconds_valid = time_seconds[pre_kalman_valid_indices]
        speeds_ms_valid = speeds_ms[pre_kalman_valid_indices]
        hdop_valid = hdop_per_point[pre_kalman_valid_indices]

        # === TWO-PASS CALCULATION WITH ADAPTIVE KALMAN ===

        # FIRST PASS: basic Kalman to identify high power zones
        speeds_smooth_pass1, acceleration_smooth_pass1 = kalman_cv(speeds_ms_valid, time_seconds_valid)

        # Rough power estimate for first pass (inertia + aerodynamics only)
        # Vectorized calculation
        v_pass1 = speeds_smooth_pass1
        a_pass1 = acceleration_smooth_pass1
        p_inertia = mass * a_pass1 * v_pass1
        p_aero = AERODYNAMIC_FORCE_COEFFICIENT * air_density_default * drag_coefficient * frontal_area * v_pass1**3
        rough_power = np.maximum(p_inertia + p_aero, 0.0) / watts_to_hp

        # === FILTER AND DISCARD LOW POWER ZONE POINTS ===
        if len(rough_power) > 0 and np.max(rough_power) > 0:
            cutoff_percentile = PERCENT_TO_DECIMAL - kalman_low_power_filter_percentile
            low_power_threshold = np.percentile(rough_power[rough_power > 0], cutoff_percentile)
            valid_power_mask = rough_power >= low_power_threshold

            hdop_in_valid_power = hdop_valid[valid_power_mask]
            if len(hdop_in_valid_power) > 0:
                median_hdop_high_power = np.median(hdop_in_valid_power)
            else:
                median_hdop_high_power = hdop_fallback
        else:
            median_hdop_high_power = hdop_fallback
            valid_power_mask = np.ones(len(rough_power), dtype=bool)

        # Mark low-power points as invalid (vectorized)
        invalid_power_indices = pre_kalman_valid_indices[~valid_power_mask]
        valid_flags[invalid_power_indices] = False

        # SECOND PASS: adaptive Kalman with HDOP-based R
        if median_hdop_high_power > 0:
            r_array_valid = kalman_speed_r * (1 + kalman_hdop_factor * (hdop_valid / median_hdop_high_power))
        else:
            r_array_valid = np.full_like(hdop_valid, kalman_speed_r)

        r_array_valid = np.clip(
            r_array_valid,
            kalman_speed_r * kalman_r_min_multiplier,
            kalman_speed_r * kalman_r_max_multiplier
        )

        speeds_smooth_valid, acceleration_smooth_valid = kalman_cv_adaptive(
            speeds_ms_valid, time_seconds_valid, r_array=r_array_valid
        )

        # Interpolate back to full time grid
        if len(time_seconds_valid) > 1:
            speed_interp = interp1d(
                time_seconds_valid, speeds_smooth_valid,
                kind='linear', bounds_error=False,
                fill_value=(speeds_smooth_valid[0], speeds_smooth_valid[-1])
            )
            acc_interp = interp1d(
                time_seconds_valid, acceleration_smooth_valid,
                kind='linear', bounds_error=False,
                fill_value=(acceleration_smooth_valid[0], acceleration_smooth_valid[-1])
            )

            speeds_smooth = speed_interp(time_seconds)
            acceleration_smooth = acc_interp(time_seconds)
        else:
            speeds_smooth, acceleration_smooth = kalman_cv(speeds_ms, time_seconds)
            pre_kalman_stats['filtered_ratio'] = 0.0

        pre_kalman_stats['adaptive_kalman'] = {
            'median_hdop_valid_power': float(median_hdop_high_power),
            'valid_power_points': int(np.sum(valid_power_mask)),
            'low_power_filter_percentile': kalman_low_power_filter_percentile,
            'hdop_factor': kalman_hdop_factor
        }
    else:
        logger.warning(f"Too few valid points after pre-Kalman filtering: {len(pre_kalman_valid_indices)}")
        speeds_smooth, acceleration_smooth = kalman_cv(speeds_ms, time_seconds)
        pre_kalman_stats['filtered_ratio'] = 0.0

    # === PERCENTILE FILTERING BY SMOOTHED SPEED ===
    speeds_smooth_kmh = speeds_smooth * KMH_TO_MS

    # Vectorized percentile filtering
    valid_speeds = speeds_smooth_kmh[valid_flags]
    if len(valid_speeds) > 0:
        speed_cutoff_low = PERCENT_TO_DECIMAL - speed_percentile_low
        speed_percentile_low_val = np.percentile(valid_speeds, speed_cutoff_low)
        speed_percentile_high_val = np.percentile(valid_speeds, speed_percentile_high)

        # Vectorized filtering: mark points outside percentile range as invalid
        speed_out_of_range = (speeds_smooth_kmh < speed_percentile_low_val) | (speeds_smooth_kmh > speed_percentile_high_val)
        valid_flags[speed_out_of_range] = False

    rolling_resistance = car_info.get("rolling_resistance", 0.015)

    # Calculate air density from weather data if available
    if weather_data:
        rho = calculate_air_density_from_weather(weather_data, r_dry_air, r_vapor)
    else:
        rho = air_density_default

    # Process altitude data to calculate slopes
    slopes = process_altitude_data_for_slopes(
        altitude_data,
        time_seconds,
        speeds_smooth,
        first_timestamp_ms_arg,
        kalman_altitude,
        interp1d,
        calculate_savgol_window_size,
        apply_savgol_filter,
        savgol_window_length,
        savgol_window_divisor,
        min_distance_for_slope,
        max_safe_slope
    )

    # === CALCULATE HEADINGS AND RELATIVE WIND SPEED ===
    # Calculate headings from GPS coordinates if available
    # Pass filter functions for aggressive smoothing (power is calculated on straights)
    headings = None
    if coords_data and len(coords_data) >= 2:
        headings = calculate_headings_from_coords(
            coords_data, time_seconds, first_timestamp_ms_arg,
            calculate_savgol_window_size_func=calculate_savgol_window_size,
            apply_savgol_filter_func=apply_savgol_filter
        )
    
    # Calculate relative wind speed (accounting for wind direction)
    # If wind direction is not available, fall back to using just car speed
    wind_speed_kph = 0.0
    wind_direction_deg = None
    if weather_data:
        wind_speed_kph = weather_data.get('wind_speed_kph', 0.0)
        wind_direction_deg = weather_data.get('wind_direction_deg', None)
    
    # Calculate relative airspeed for each point
    if headings is not None and wind_direction_deg is not None and wind_speed_kph > 0:
        # Vectorized calculation of relative wind speed
        # Both use same angular convention (0° = North):
        # - Car heading: direction car is traveling TO (navigation convention)
        # - Wind direction: direction wind is coming FROM (meteorological convention)
        wind_speed_ms = wind_speed_kph / KMH_TO_MS
        # To get the component along car's direction:
        # - If wind comes FROM direction θ_wind and car goes TO direction θ_car
        # - The component = wind_speed * cos(θ_car - θ_wind)
        # - Positive = headwind (wind opposes car), negative = tailwind (wind assists car)
        angle_diff_deg = headings - wind_direction_deg
        # Normalize to -180 to 180
        angle_diff_deg = ((angle_diff_deg + 180) % 360) - 180
        # Convert to radians
        angle_diff_rad = np.radians(angle_diff_deg)
        # Component of wind along car's direction (positive = headwind, negative = tailwind)
        wind_component_ms = wind_speed_ms * np.cos(angle_diff_rad)
        # Relative airspeed = car speed + wind component
        # (headwind increases effective speed, tailwind decreases it)
        relative_airspeeds = speeds_smooth + wind_component_ms
        # Ensure non-negative (minimum 0.1 m/s to avoid numerical issues)
        relative_airspeeds = np.maximum(relative_airspeeds, 0.1)
    else:
        # No wind direction data or no headings - use car speed only
        relative_airspeeds = speeds_smooth

    # === VECTORIZED POWER CALCULATION ===
    v = speeds_smooth

    # Use relative airspeed (accounting for wind) in air resistance calculation
    air_resistance_force = AERODYNAMIC_FORCE_COEFFICIENT * rho * drag_coefficient * frontal_area * relative_airspeeds**2
    rolling_force = mass * gravity * rolling_resistance
    acceleration_force = mass * acceleration_smooth
    slope_force = mass * gravity * slopes

    total_force = air_resistance_force + rolling_force + acceleration_force + slope_force
    power_watts = np.maximum(total_force * v, 0.0)
    power_hp_array = power_watts / watts_to_hp

    # Component powers (for time plots)
    air_resistance_power_hp = (air_resistance_force * v) / watts_to_hp
    rolling_force_power_hp = (rolling_force * v) / watts_to_hp
    acceleration_force_power_hp = (acceleration_force * v) / watts_to_hp
    slope_force_power_hp = (slope_force * v) / watts_to_hp

    # Convert to Python lists of tuples for public API
    # Use np.column_stack for efficient tuple creation, then convert to list
    time_list = time_seconds.tolist()
    power_list = power_hp_array.tolist()
    
    # Create time series tuples more efficiently
    power_time_series = list(zip(time_list, power_list))
    air_resistance_series = list(zip(time_list, air_resistance_power_hp.tolist()))
    rolling_force_series = list(zip(time_list, rolling_force_power_hp.tolist()))
    acceleration_force_series = list(zip(time_list, acceleration_force_power_hp.tolist()))
    slope_force_series = list(zip(time_list, slope_force_power_hp.tolist()))

    # Create power_speed_data efficiently
    speed_kmh_array = v * KMH_TO_MS
    power_speed_data = list(zip(
        speed_kmh_array.tolist(),
        power_list,
        valid_flags.tolist()
    ))

    valid_power_speed = [
        p for p in power_speed_data
        if len(p) <= 2 or p[2]
    ]
    powers = [p[1] for p in valid_power_speed]

    total_points = len(power_speed_data)
    valid_points = len(valid_power_speed)
    filtered_ratio = 1.0 - (valid_points / total_points) if total_points > 0 else 1.0

    power_estimation = calculate_power_estimation(powers, methods=methods) if powers else None

    # === UNCERTAINTY CALCULATION ===
    uncertainty = None
    uncertainty_data = calculate_uncertainty_data(
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
    )

    if uncertainty_data:
        # Call uncertainty calculation
        uncertainty_result = calculate_total_uncertainty(
            v_car_ms=uncertainty_data['v_car_ms'],
            a_ms2=uncertainty_data['a_ms2'],
            crr=uncertainty_data['crr'],
            slope_avg=uncertainty_data['slope_avg'],
            power_hp=uncertainty_data['power_hp'],
            hdop_mean=uncertainty_data['hdop_mean'],
            gps_frequency_hz=uncertainty_data['gps_frequency_hz'],
            consistency_score=uncertainty_data['consistency_score'],
            delta_mass_kg=uncertainty_data['uncertainty_mass_kg']
        )

        # Format output structure
        hdop_mean = uncertainty_data['hdop_mean']
        gps_frequency_hz = uncertainty_data['gps_frequency_hz']
        recommended_power = uncertainty_data['power_hp']
        uncertainty_mass_kg = uncertainty_data.get('uncertainty_mass_kg', 20)
        
        uncertainty = {
            'total_hp': uncertainty_result.total_hp,
            'type': 'worst-case',
            'display': f"±{int(round(uncertainty_result.total_hp))}",
            'components': {
                'mass': {
                    'hp': uncertainty_result.mass_hp,
                    'note': f"±{uncertainty_mass_kg} kg"
                },
                'gps': {
                    'hp': uncertainty_result.gps_hp,
                    'note': f"HDOP={hdop_mean:.1f}, {gps_frequency_hz:.0f} Hz"
                }
            },
            'consistency_multiplier': uncertainty_result.consistency_multiplier,
            'range': [
                round(recommended_power - uncertainty_result.total_hp, 1),
                round(recommended_power + uncertainty_result.total_hp, 1)
            ]
        }

    # Use dict comprehension for better performance
    time_map = {
        time_seconds[i]: ts
        for i, ts in enumerate(times_str)
        if i < len(time_seconds)
    }

    return {
        'power_time': power_time_series,
        'power_speed': power_speed_data,
        'air_resistance_time': air_resistance_series,
        'rolling_force_time': rolling_force_series,
        'acceleration_force_time': acceleration_force_series,
        'slope_force_time': slope_force_series,
        'power_estimation': power_estimation,
        'uncertainty': uncertainty,
        'filter': 'kalman',
        'dt': float(time_seconds[1] - time_seconds[0]) if len(time_seconds) > 1 else 0.0,
        'time_map': time_map,
        'first_timestamp_ms': int(abs_times_ms[0]) if len(abs_times_ms) > 0 else None,
        'hdop_statistics': hdop_statistics,
        'pre_kalman_stats': pre_kalman_stats,
        'filtered_ratio': filtered_ratio,
        'valid_points': valid_points,
        'total_points': total_points
    }
