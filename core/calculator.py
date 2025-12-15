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
    pre_kalman_filter,
)
from .density import calculate_power_estimation
from .uncertainty import calculate_total_uncertainty
from .structures import (
    SPEED_ABS_MS,
    SPEED_REL_MS,
    SPEED_TIME_STR,
    SPEED_SPEED_KMH,
    SPEED_NUM_SATS,
    SPEED_HDOP,
    ALT_ABS_MS,
    ALT_ALTITUDE_M,
)

logger = logging.getLogger(__name__)


def calculate_power(speed_data, car_info, weather_data=None, altitude_data=None,
                    first_timestamp_ms_arg=None, methods=None):
    """
    Calculate wheel horsepower from GPS data.

    Uses two-pass adaptive Kalman filter with HDOP consideration.

    Args:
        speed_data: list of tuples (abs_ms, rel_ms, time_str, speed_kmh, nmea, sats, hdop)
        car_info: dict with vehicle params (mass, drag_coefficient, frontal_area, rolling_resistance)
        weather_data: dict with weather data (temperature_c, pressure_hpa, humidity_percent, wind_speed_kph)
        altitude_data: list of tuples with altitude data
        first_timestamp_ms_arg: first timestamp in ms (for synchronization)
        methods: list of power estimation methods (default: ['peak_detection'])

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

    n_points = len(speed_data)

    # Pre-allocate arrays/lists for speed_data fields
    abs_times_ms = np.empty(n_points, dtype=np.int64)
    rel_times_ms = np.empty(n_points, dtype=np.int64)
    speeds_kmh = np.empty(n_points, dtype=float)
    times_str = [None] * n_points
    num_sats_arr = np.empty(n_points, dtype=np.int32)
    
    # Collect HDOP values for percentile calculation while unpacking tuples
    # Use fallback value for missing HDOP (matching old behavior)
    hdop_fallback = getattr(config, 'HDOP_FALLBACK_VALUE', 0.7)
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
        hdop_threshold = np.percentile(hdop_values, config.HDOP_PERCENTILE)
        # Ensure reasonable bounds
        hdop_min = getattr(config, 'HDOP_MIN_BOUND', 0.5)
        hdop_max = getattr(config, 'HDOP_MAX_BOUND', 2.0)
        hdop_threshold = max(hdop_min, min(hdop_threshold, hdop_max))
        hdop_statistics = {
            'threshold': hdop_threshold,
            'percentile': config.HDOP_PERCENTILE,
            'min': float(np.min(hdop_values)),
            'max': float(np.max(hdop_values)),
            'mean': float(np.mean(hdop_values)),
            'median': float(np.median(hdop_values)),
            'count': len(hdop_values)
        }

    # Build validity mask using vectorised operations
    valid_flags = np.ones(n_points, dtype=bool)

    # Satellite-based validity: only when value is present (>=0)
    min_sats = config.MIN_SATELLITES
    sats_known = num_sats_arr >= 0
    sats_low = sats_known & (num_sats_arr < min_sats)
    valid_flags[sats_low] = False

    # HDOP-based validity: check against threshold
    hdop_array = hdop_per_point
    hdop_bad = hdop_array > hdop_threshold
    valid_flags[hdop_bad] = False

    # Convert times/speeds to NumPy arrays
    time_seconds = rel_times_ms.astype(float) / 1000.0
    speeds_ms = speeds_kmh / 3.6

    # Extract vehicle parameters (needed for two-pass calculation)
    mass = validate_param("mass", car_info.get("mass"), 1500, min_value=100)
    drag_coefficient = validate_param("drag_coefficient", car_info.get("drag_coefficient"), 0.30, min_value=0.1)
    frontal_area = validate_param("frontal_area", car_info.get("frontal_area"), 2.2, min_value=0.5)

    # === PRE-KALMAN FILTERING ===
    pre_kalman_stats = pre_kalman_filter(speed_data, hdop_threshold)

    pre_kalman_valid_indices = pre_kalman_stats['valid_indices']
    min_valid_points = getattr(config, 'PRE_KALMAN_MIN_VALID_POINTS', 10)

    # Check: enough valid points?
    if len(pre_kalman_valid_indices) >= min_valid_points:
        # Extract valid data for Kalman
        time_seconds_valid = time_seconds[pre_kalman_valid_indices]
        speeds_ms_valid = speeds_ms[pre_kalman_valid_indices]
        hdop_valid = hdop_per_point[pre_kalman_valid_indices]

        # === TWO-PASS CALCULATION WITH ADAPTIVE KALMAN ===

        # FIRST PASS: basic Kalman to identify high power zones
        speeds_smooth_pass1, acceleration_smooth_pass1 = kalman_cv(speeds_ms_valid, time_seconds_valid)

        # Rough power estimate for first pass (inertia + aerodynamics only)
        watts_to_hp = getattr(config, 'WATTS_TO_HP', 735.5)
        rough_power = []
        for i in range(len(speeds_smooth_pass1)):
            v = speeds_smooth_pass1[i]
            a = acceleration_smooth_pass1[i]
            # Simplified formula: P ≈ m*a*v + 0.5*rho*Cd*A*v³
            p_inertia = mass * a * v
            p_aero = 0.5 * config.AIR_DENSITY * drag_coefficient * frontal_area * v**3
            rough_power.append(max(p_inertia + p_aero, 0) / watts_to_hp)
        rough_power = np.array(rough_power)

        # === FILTER AND DISCARD LOW POWER ZONE POINTS ===
        low_power_filter_percentile = getattr(config, 'KALMAN_LOW_POWER_FILTER_PERCENTILE', 95)
        if len(rough_power) > 0 and np.max(rough_power) > 0:
            cutoff_percentile = 100 - low_power_filter_percentile
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

        # Mark low-power points as invalid
        for idx, original_idx in enumerate(pre_kalman_valid_indices):
            if not valid_power_mask[idx]:
                valid_flags[original_idx] = False

        # SECOND PASS: adaptive Kalman with HDOP-based R
        hdop_factor = getattr(config, 'KALMAN_HDOP_FACTOR', 0.5)
        base_r = getattr(config, 'KALMAN_SPEED_R', 0.2)
        r_min_mult = getattr(config, 'KALMAN_R_MIN_MULTIPLIER', 0.5)
        r_max_mult = getattr(config, 'KALMAN_R_MAX_MULTIPLIER', 3.0)

        if median_hdop_high_power > 0:
            r_array_valid = base_r * (1 + hdop_factor * (hdop_valid / median_hdop_high_power))
        else:
            r_array_valid = np.full_like(hdop_valid, base_r)

        r_array_valid = np.clip(r_array_valid, base_r * r_min_mult, base_r * r_max_mult)

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
            'low_power_filter_percentile': low_power_filter_percentile,
            'hdop_factor': hdop_factor
        }
    else:
        logger.warning(f"Too few valid points after pre-Kalman filtering: {len(pre_kalman_valid_indices)}")
        speeds_smooth, acceleration_smooth = kalman_cv(speeds_ms, time_seconds)
        pre_kalman_stats['filtered_ratio'] = 0.0

    # === PERCENTILE FILTERING BY SMOOTHED SPEED ===
    speeds_smooth_kmh = speeds_smooth * 3.6

    valid_speeds = [speeds_smooth_kmh[i] for i in range(len(speeds_smooth_kmh)) if valid_flags[i]]

    if valid_speeds:
        speed_cutoff_low = 100 - config.SPEED_PERCENTILE_LOW
        speed_percentile_low = np.percentile(valid_speeds, speed_cutoff_low)
        speed_percentile_high = np.percentile(valid_speeds, config.SPEED_PERCENTILE_HIGH)

        for i, spd in enumerate(speeds_smooth_kmh):
            if valid_flags[i]:
                if spd < speed_percentile_low or spd > speed_percentile_high:
                    valid_flags[i] = False

    g = config.GRAVITY
    rolling_resistance = car_info.get("rolling_resistance", 0.015)
    rho = config.AIR_DENSITY
    watts_to_hp = getattr(config, 'WATTS_TO_HP', 735.5)

    if weather_data:
        temp_c = weather_data.get("temperature_c", 20)
        pressure_hpa = weather_data.get("pressure_hpa", 1013.25)
        humidity = weather_data.get("humidity_percent", 50)

        temp_k = temp_c + 273.15

        R_d = getattr(config, 'R_DRY_AIR', 287.058)
        R_v = getattr(config, 'R_VAPOR', 461.495)

        pressure_pa = pressure_hpa * 100

        P_sat_hpa = 6.1094 * np.exp((17.625 * temp_c) / (temp_c + 243.04))
        P_sat_pa = P_sat_hpa * 100

        P_v_pa = (humidity / 100.0) * P_sat_pa
        P_d_pa = pressure_pa - P_v_pa

        rho = (P_d_pa / (R_d * temp_k)) + (P_v_pa / (R_v * temp_k))

    slopes = np.zeros_like(speeds_smooth)

    if altitude_data and len(altitude_data) > 2:
        try:
            alt_abs_times_ms = []
            alt_values = []

            for point in altitude_data:
                alt_abs_times_ms.append(point[ALT_ABS_MS])
                alt_values.append(point[ALT_ALTITUDE_M])

            if len(alt_values) > 2:
                first_timestamp_sec_val = 0.0
                if first_timestamp_ms_arg is not None:
                    first_timestamp_sec_val = first_timestamp_ms_arg / 1000.0
                else:
                    if alt_abs_times_ms:
                        first_timestamp_sec_val = alt_abs_times_ms[0] / 1000.0

                alt_times_sec = np.array([(t / 1000.0) - first_timestamp_sec_val for t in alt_abs_times_ms])
                alt_values = np.array(alt_values)

                altitude_smooth = kalman_altitude(alt_values, alt_times_sec)

                if len(alt_times_sec) != len(np.unique(alt_times_sec)):
                    unique_times = np.unique(alt_times_sec)
                    unique_values = np.zeros_like(unique_times)

                    for i, t in enumerate(unique_times):
                        idx = np.where(alt_times_sec == t)[0]
                        unique_values[i] = np.mean(altitude_smooth[idx])

                    alt_times_sec = unique_times
                    altitude_smooth = unique_values

                if len(alt_times_sec) > 1:
                    alt_interp = interp1d(alt_times_sec, altitude_smooth,
                                     kind='linear', bounds_error=False,
                                     fill_value=(altitude_smooth[0], altitude_smooth[-1]))

                    altitude_at_speed_times = alt_interp(time_seconds)

                    distance_traveled = np.zeros_like(speeds_smooth)
                    for i in range(1, len(speeds_smooth)):
                        distance_traveled[i] = speeds_smooth[i] * (time_seconds[i] - time_seconds[i-1])

                    savgol_window = getattr(config, 'SAVGOL_WINDOW_LENGTH', 11)
                    savgol_divisor = getattr(config, 'SAVGOL_WINDOW_DIVISOR', 5)
                    min_distance = getattr(config, 'MIN_DISTANCE_FOR_SLOPE', 0.5)

                    distance_traveled_smooth = np.copy(distance_traveled)
                    window_size = min(savgol_window, len(distance_traveled) // savgol_divisor)
                    if window_size % 2 == 0:
                        window_size = max(3, window_size - 1)
                    distance_traveled_smooth = apply_savgol_filter(distance_traveled, window_size)

                    distance_traveled_smooth = np.maximum(distance_traveled_smooth, 0)

                    for i in range(1, len(speeds_smooth)):
                        if distance_traveled_smooth[i] > min_distance:
                            alt_change = altitude_at_speed_times[i] - altitude_at_speed_times[i-1]
                            tan_slope = alt_change / distance_traveled_smooth[i]
                            slopes[i] = tan_slope / np.sqrt(1 + tan_slope**2)

                    max_safe_slope = config.MAX_SAFE_SLOPE
                    slopes = np.clip(slopes, -max_safe_slope, max_safe_slope)

                    window_size = min(savgol_window, len(slopes) // savgol_divisor)
                    if window_size % 2 == 0:
                        window_size = max(3, window_size - 1)
                    slopes = apply_savgol_filter(slopes, window_size)

        except Exception as e:
            logger.error(f"Error processing altitude data: {e}")

    # === VECTORISED POWER CALCULATION ===
    v = speeds_smooth

    air_resistance_force = 0.5 * rho * drag_coefficient * frontal_area * v**2
    rolling_force = mass * g * rolling_resistance
    acceleration_force = mass * acceleration_smooth
    slope_force = mass * g * slopes

    total_force = air_resistance_force + rolling_force + acceleration_force + slope_force
    power_watts = np.maximum(total_force * v, 0.0)
    power_hp_array = power_watts / watts_to_hp

    # Component powers (for time plots)
    air_resistance_power_hp = (air_resistance_force * v) / watts_to_hp
    rolling_force_power_hp = (rolling_force * v) / watts_to_hp
    acceleration_force_power_hp = (acceleration_force * v) / watts_to_hp
    slope_force_power_hp = (slope_force * v) / watts_to_hp

    # Convert to Python lists of tuples for public API
    time_list = time_seconds.tolist()
    power_list = power_hp_array.tolist()
    air_list = air_resistance_power_hp.tolist()
    roll_list = rolling_force_power_hp.tolist()
    accel_list = acceleration_force_power_hp.tolist()
    slope_list = slope_force_power_hp.tolist()

    power_time_series = list(zip(time_list, power_list))
    air_resistance_series = list(zip(time_list, air_list))
    rolling_force_series = list(zip(time_list, roll_list))
    acceleration_force_series = list(zip(time_list, accel_list))
    slope_force_series = list(zip(time_list, slope_list))

    speed_kmh_array = v * 3.6
    valid_flags_list = valid_flags.tolist()
    power_speed_data = list(zip(speed_kmh_array.tolist(), power_list, valid_flags_list))

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
    if power_estimation and power_estimation.get('recommended_value'):
        # Collect data for uncertainty calculation
        recommended_power = power_estimation['recommended_value']

        # Speed at peak power (find point with max power among valid ones)
        max_power_speed_kmh = 150  # fallback
        max_power_idx = None
        max_power_val = 0
        for idx, (spd, pwr, is_valid) in enumerate(power_speed_data):
            if is_valid and pwr > max_power_val:
                max_power_val = pwr
                max_power_speed_kmh = spd
                max_power_idx = idx

        v_car_ms = max_power_speed_kmh / 3.6

        # Wind speed from weather
        wind_kph = 0.0
        if weather_data:
            wind_kph = weather_data.get('wind_speed_kph', 0.0)

        # Average acceleration in high power zone
        # Use acceleration at points with power > 90% of maximum
        high_power_threshold = max_power_val * 0.9 if max_power_val > 0 else 0
        high_power_accelerations = []
        for idx, (spd, pwr, is_valid) in enumerate(power_speed_data):
            if is_valid and pwr >= high_power_threshold and idx < len(acceleration_smooth):
                high_power_accelerations.append(abs(acceleration_smooth[idx]))
        a_ms2 = np.mean(high_power_accelerations) if high_power_accelerations else 1.0

        # Average slope in high power zone
        high_power_slopes = []
        for idx, (spd, pwr, is_valid) in enumerate(power_speed_data):
            if is_valid and pwr >= high_power_threshold and idx < len(slopes):
                high_power_slopes.append(slopes[idx])
        slope_avg = np.mean(high_power_slopes) if high_power_slopes else 0.0

        # GPS frequency
        gps_frequency_hz = 10.0  # default
        if len(time_seconds) > 1:
            time_diffs = np.diff(time_seconds)
            valid_diffs = time_diffs[time_diffs > 0]
            if len(valid_diffs) > 0:
                avg_interval = np.mean(valid_diffs)
                if avg_interval > 0:
                    gps_frequency_hz = 1.0 / avg_interval

        # Average HDOP
        hdop_mean = hdop_statistics['mean'] if hdop_statistics else 1.0

        # Consistency score from estimation methods
        consistency_score = power_estimation.get('consistency_score', 0.85)

        # Call uncertainty calculation
        uncertainty_result = calculate_total_uncertainty(
            v_car_ms=v_car_ms,
            wind_kph=wind_kph,
            rho=rho,
            cd=drag_coefficient,
            frontal_area=frontal_area,
            a_ms2=a_ms2,
            crr=rolling_resistance,
            slope_avg=slope_avg,
            power_hp=recommended_power,
            hdop_mean=hdop_mean,
            gps_frequency_hz=gps_frequency_hz,
            consistency_score=consistency_score
        )

        # Format output structure
        uncertainty = {
            'total_hp': uncertainty_result.total_hp,
            'type': 'worst-case',
            'display': f"±{int(round(uncertainty_result.total_hp))}",
            'components': {
                'wind': {
                    'hp': uncertainty_result.wind_hp,
                    'note': f"headwind/tailwind {wind_kph:.0f} km/h" if wind_kph > 0 else "wind not accounted"
                },
                'mass': {
                    'hp': uncertainty_result.mass_hp,
                    'note': f"±{getattr(config, 'UNCERTAINTY_MASS_KG', 20)} kg"
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

    time_map = {}
    for i, ts in enumerate(times_str):
        if i < len(time_seconds):
            time_map[time_seconds[i]] = ts

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
