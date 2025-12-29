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

from .structures import PERCENT_TO_DECIMAL, MS_TO_S, KMH_TO_MS

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


def calculate_bearing_from_coords(lat1, lon1, lat2, lon2):
    """
    Calculate bearing (heading) from two GPS coordinates.
    
    Uses the haversine formula to calculate the initial bearing from point 1 to point 2.
    
    Convention: Navigation bearing (direction vehicle is traveling TO)
    - 0° = North
    - 90° = East
    - 180° = South
    - 270° = West
    
    This matches the standard navigation convention used for vehicle headings.
    
    Args:
        lat1: Latitude of first point (degrees)
        lon1: Longitude of first point (degrees)
        lat2: Latitude of second point (degrees)
        lon2: Longitude of second point (degrees)
    
    Returns:
        float: Bearing in degrees (0-360, where 0 is North, 90 is East)
    """
    import math
    
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlon_rad = math.radians(lon2 - lon1)
    
    # Calculate bearing
    y = math.sin(dlon_rad) * math.cos(lat2_rad)
    x = (math.cos(lat1_rad) * math.sin(lat2_rad) - 
         math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad))
    
    bearing_rad = math.atan2(y, x)
    bearing_deg = math.degrees(bearing_rad)
    
    # Normalize to 0-360
    bearing_deg = (bearing_deg + 360) % 360
    
    return bearing_deg


def calculate_headings_from_coords(coords_data, time_seconds, first_timestamp_ms_arg,
                                    calculate_savgol_window_size_func=None,
                                    apply_savgol_filter_func=None):
    """
    Calculate vehicle headings from GPS coordinates with aggressive filtering.
    
    Since power calculations are done on straights, headings are heavily filtered
    to reduce GPS noise. Uses Savitzky-Golay filter with large window size.
    
    Args:
        coords_data: list of coordinate tuples (timestamp_ms, lat, lon)
        time_seconds: array of time values in seconds (for interpolation)
        first_timestamp_ms_arg: first timestamp in ms
        calculate_savgol_window_size_func: function to calculate Savitzky-Golay window size
        apply_savgol_filter_func: function to apply Savitzky-Golay filter
    
    Returns:
        np.ndarray: headings in degrees (0-360), same length as time_seconds, or None if insufficient data
    """
    if not coords_data or len(coords_data) < 2:
        return None
    
    try:
        # Import filter functions if not provided
        if calculate_savgol_window_size_func is None or apply_savgol_filter_func is None:
            from .filters import calculate_savgol_window_size, apply_savgol_filter
            if calculate_savgol_window_size_func is None:
                calculate_savgol_window_size_func = calculate_savgol_window_size
            if apply_savgol_filter_func is None:
                apply_savgol_filter_func = apply_savgol_filter
        
        # Get heading filter parameters (more aggressive than speed/altitude)
        heading_window_length = getattr(config, 'HEADING_FILTER_WINDOW_LENGTH', 21)  # Larger default window
        heading_window_divisor = getattr(config, 'HEADING_FILTER_WINDOW_DIVISOR', 3)  # More aggressive
        heading_polyorder = getattr(config, 'HEADING_FILTER_POLYORDER', 2)  # Higher order for smoothness
        
        # Extract coordinates and timestamps
        n_coords = len(coords_data)
        coord_times_ms = np.empty(n_coords, dtype=np.int64)
        lats = np.empty(n_coords, dtype=float)
        lons = np.empty(n_coords, dtype=float)
        
        for i, point in enumerate(coords_data):
            coord_times_ms[i] = point[0]  # timestamp_ms is first element
            lats[i] = point[1]  # lat is second element
            lons[i] = point[2]  # lon is third element
        
        # Calculate first timestamp
        first_timestamp_sec_val = 0.0
        if first_timestamp_ms_arg is not None:
            first_timestamp_sec_val = first_timestamp_ms_arg / MS_TO_S
        elif len(coord_times_ms) > 0:
            first_timestamp_sec_val = coord_times_ms[0] / MS_TO_S
        
        # Convert to relative time in seconds
        coord_times_sec = (coord_times_ms / MS_TO_S) - first_timestamp_sec_val
        
        # Calculate bearings between consecutive points
        headings = np.zeros(n_coords, dtype=float)
        for i in range(1, n_coords):
            bearing = calculate_bearing_from_coords(
                lats[i-1], lons[i-1],
                lats[i], lons[i]
            )
            headings[i] = bearing
        
        # First point uses same heading as second
        if n_coords > 1:
            headings[0] = headings[1]
        
        # Handle duplicate timestamps
        if len(coord_times_sec) != len(np.unique(coord_times_sec)):
            unique_times, unique_indices = np.unique(coord_times_sec, return_inverse=True)
            unique_headings = np.zeros_like(unique_times)
            for i, t in enumerate(unique_times):
                mask = unique_indices == i
                # Average headings for duplicate timestamps
                unique_headings[i] = np.mean(headings[mask])
            coord_times_sec = unique_times
            headings = unique_headings
        
        if len(coord_times_sec) < 2:
            return None
        
        # === AGGRESSIVE FILTERING FOR HEADINGS ===
        # Since power is calculated on straights, we can use heavy filtering
        # Convert to continuous angle space (unwrap) for filtering
        headings_rad = np.radians(headings)
        headings_unwrapped = np.unwrap(headings_rad)
        
        # Apply Savitzky-Golay filter with aggressive window size
        window_size = calculate_savgol_window_size_func(
            len(headings_unwrapped),
            heading_window_length,
            heading_window_divisor
        )
        
        # Ensure minimum window size for effective filtering, but not larger than data
        min_window = getattr(config, 'HEADING_FILTER_MIN_WINDOW', 5)
        window_size = max(window_size, min(min_window, len(headings_unwrapped)))
        window_size = min(window_size, len(headings_unwrapped))  # Ensure not larger than data
        
        # Apply filter
        headings_filtered = apply_savgol_filter_func(
            headings_unwrapped,
            window_size,
            polyorder=heading_polyorder
        )
        
        # Interpolate filtered headings to speed time grid
        from scipy.interpolate import interp1d
        heading_interp = interp1d(
            coord_times_sec, headings_filtered,
            kind='linear', bounds_error=False,
            fill_value=(headings_filtered[0], headings_filtered[-1])
        )
        headings_at_speed_times_rad = heading_interp(time_seconds)
        
        # Convert back to degrees and normalize to 0-360
        headings_at_speed_times = np.degrees(headings_at_speed_times_rad)
        headings_at_speed_times = (headings_at_speed_times + 360) % 360
        
        return headings_at_speed_times
        
    except Exception as e:
        logger.error(f"Error calculating headings from coordinates: {e}", exc_info=True)
        return None


def calculate_relative_wind_speed(
    car_speed_ms, car_heading_deg, wind_speed_kph, wind_direction_deg
):
    """
    Calculate relative wind speed along the car's direction of travel.
    
    Both use the same angular convention (0° = North):
    - Car heading: Navigation convention - direction car is traveling TO
      (0° = North, 90° = East, 180° = South, 270° = West)
    - Wind direction: Meteorological convention - direction wind is coming FROM
      (0° = wind FROM North, 90° = wind FROM East, 180° = wind FROM South, 270° = wind FROM West)
    
    Examples:
    - Car heading 0° (North) + Wind direction 0° (FROM North) = Headwind
    - Car heading 0° (North) + Wind direction 180° (FROM South) = Tailwind
    - Car heading 90° (East) + Wind direction 90° (FROM East) = Headwind
    
    Args:
        car_speed_ms: Car speed in m/s
        car_heading_deg: Car heading in degrees (0-360, 0 = North, direction TO)
        wind_speed_kph: Wind speed in km/h
        wind_direction_deg: Wind direction in degrees (0-360, 0 = North, direction FROM)
    
    Returns:
        float: Relative airspeed in m/s
    """
    if wind_speed_kph <= 0:
        return car_speed_ms
    
    # Convert wind speed to m/s
    wind_speed_ms = wind_speed_kph / KMH_TO_MS
    
    # Wind direction is where wind comes FROM
    # To get the component along car's direction:
    # - If wind comes FROM direction θ_wind and car goes TO direction θ_car
    # - The component = wind_speed * cos(θ_car - θ_wind)
    # - Positive = headwind (wind opposes car), negative = tailwind (wind assists car)
    angle_diff_deg = car_heading_deg - wind_direction_deg
    
    # Normalize to -180 to 180
    angle_diff_deg = ((angle_diff_deg + 180) % 360) - 180
    
    # Convert to radians
    angle_diff_rad = np.radians(angle_diff_deg)
    
    # Component of wind along car's direction
    # cos(angle) gives the projection: positive = headwind, negative = tailwind
    wind_component_ms = wind_speed_ms * np.cos(angle_diff_rad)
    
    # Relative airspeed = car speed + wind component
    # (headwind increases effective speed, tailwind decreases it)
    relative_airspeed_ms = car_speed_ms + wind_component_ms
    
    # Ensure non-negative (minimum 0.1 m/s to avoid numerical issues)
    relative_airspeed_ms = max(relative_airspeed_ms, 0.1)
    
    return relative_airspeed_ms


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

