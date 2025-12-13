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
GPS and altitude data filtering module.
Contains Kalman filter implementations for speed/acceleration and altitude,
as well as GPS data pre-filtering functions.
"""

import numpy as np
from scipy.signal import savgol_filter

# Import config for default values
try:
    from .. import config
except ImportError:
    import config


def kalman_cv(meas, times, q=None, r=None):
    """
    Kalman filter for constant velocity model with acceleration estimation.

    State vector: [speed, acceleration]
    Model assumes smooth velocity changes with derivative (acceleration).
    Initialization uses average of first N points for robustness against outliers.

    Args:
        meas: array of measurements (speed in m/s)
        times: array of timestamps (seconds)
        q: process noise parameter (default: config.KALMAN_SPEED_Q)
        r: measurement noise parameter (default: config.KALMAN_SPEED_R)

    Returns:
        tuple: (smoothed speeds, accelerations)
    """
    # Use config values if not explicitly set
    if q is None:
        q = getattr(config, 'KALMAN_SPEED_Q', 0.5)
    if r is None:
        r = getattr(config, 'KALMAN_SPEED_R', 0.2)

    N = len(meas)
    # Initialize with average of first N points for robustness against outliers
    init_points = getattr(config, 'KALMAN_INIT_POINTS', 5)
    init_count = min(init_points, N)
    initial_speed = np.mean(meas[:init_count])
    initial_accel = getattr(config, 'KALMAN_INITIAL_ACCELERATION', 0.0)
    x = np.array([initial_speed, initial_accel])  # [speed, acceleration]
    initial_cov = getattr(config, 'KALMAN_INITIAL_COVARIANCE', 1.0)
    P = np.eye(2) * initial_cov
    speeds_kf = np.zeros(N)
    acc_kf = np.zeros(N)
    H = np.array([[1.0, 0.0]])

    # Get minimum dt from config
    min_dt = getattr(config, 'KALMAN_MIN_DT', 1e-6)

    for i in range(N):
        if i == 0:
            speeds_kf[0] = x[0]
            acc_kf[0] = x[1]
            continue

        dti = max(times[i] - times[i-1], min_dt)  # Protection against duplicate timestamps
        F = np.array([[1.0, dti], [0.0, 1.0]])
        Q = np.array([[dti**3/3, dti**2/2], [dti**2/2, dti]]) * q**2
        x = F.dot(x)
        P = F.dot(P).dot(F.T) + Q
        z = np.array([meas[i]])
        R = np.array([[r**2]])
        y = z - H.dot(x)
        S = H.dot(P).dot(H.T) + R
        K = P.dot(H.T).dot(np.linalg.inv(S))
        x = x + K.dot(y)
        P = (np.eye(2) - K.dot(H)).dot(P)
        speeds_kf[i] = x[0]
        acc_kf[i] = x[1]

    return speeds_kf, acc_kf


def kalman_cv_adaptive(meas, times, q=None, r_array=None, r_default=None):
    """
    Kalman filter for constant velocity model with adaptive R.

    Allows using different R values for each point,
    useful for accounting GPS quality (HDOP) on each measurement.
    Initialization uses average of first N points for robustness against outliers.

    Args:
        meas: array of measurements (speed in m/s)
        times: array of timestamps (seconds)
        q: process noise parameter (default: config.KALMAN_SPEED_Q)
        r_array: array of R values for each point (or None for constant)
        r_default: default R value if r_array not provided (default: config.KALMAN_SPEED_R)

    Returns:
        tuple: (smoothed speeds, accelerations)
    """
    # Use config values if not explicitly set
    if q is None:
        q = getattr(config, 'KALMAN_SPEED_Q', 0.5)
    if r_default is None:
        r_default = getattr(config, 'KALMAN_SPEED_R', 0.2)

    N = len(meas)
    # Initialize with average of first N points for robustness against outliers
    init_points = getattr(config, 'KALMAN_INIT_POINTS', 5)
    init_count = min(init_points, N)
    initial_speed = np.mean(meas[:init_count])
    initial_accel = getattr(config, 'KALMAN_INITIAL_ACCELERATION', 0.0)
    x = np.array([initial_speed, initial_accel])  # [speed, acceleration]
    initial_cov = getattr(config, 'KALMAN_INITIAL_COVARIANCE', 1.0)
    P = np.eye(2) * initial_cov
    speeds_kf = np.zeros(N)
    acc_kf = np.zeros(N)
    H = np.array([[1.0, 0.0]])

    # Get minimum dt from config
    min_dt = getattr(config, 'KALMAN_MIN_DT', 1e-6)

    for i in range(N):
        if i == 0:
            speeds_kf[0] = x[0]
            acc_kf[0] = x[1]
            continue

        dti = max(times[i] - times[i-1], min_dt)  # Protection against duplicate timestamps
        F = np.array([[1.0, dti], [0.0, 1.0]])
        Q = np.array([[dti**3/3, dti**2/2], [dti**2/2, dti]]) * q**2
        x = F.dot(x)
        P = F.dot(P).dot(F.T) + Q
        z = np.array([meas[i]])

        # Use adaptive R if provided, otherwise use default
        if r_array is not None and i < len(r_array):
            r = r_array[i]
        else:
            r = r_default

        R = np.array([[r**2]])
        y = z - H.dot(x)
        S = H.dot(P).dot(H.T) + R
        K = P.dot(H.T).dot(np.linalg.inv(S))
        x = x + K.dot(y)
        P = (np.eye(2) - K.dot(H)).dot(P)
        speeds_kf[i] = x[0]
        acc_kf[i] = x[1]

    return speeds_kf, acc_kf


def kalman_altitude(meas, times, q=None, r=None):
    """
    One-dimensional Kalman filter for altitude smoothing.

    Model assumes altitude remains relatively constant,
    with more aggressive smoothing to compensate for GPS noise.

    Args:
        meas: array of altitude measurements (meters)
        times: array of timestamps (seconds)
        q: process noise parameter (default: config.KALMAN_ALTITUDE_Q)
        r: measurement noise parameter (default: config.KALMAN_ALTITUDE_R)

    Returns:
        array of smoothed altitudes
    """
    # Use config values if not explicitly set
    if q is None:
        q = getattr(config, 'KALMAN_ALTITUDE_Q', 0.05)
    if r is None:
        r = getattr(config, 'KALMAN_ALTITUDE_R', 1.0)

    N = len(meas)
    x = meas[0]  # Initial altitude
    P = 1.0       # Initial covariance
    alt_kf = np.zeros(N)
    alt_kf[0] = x

    # Get minimum dt from config
    min_dt = getattr(config, 'KALMAN_MIN_DT', 1e-6)
    
    for i in range(1, N):
        dti = max(times[i] - times[i-1], min_dt)  # Protection against duplicate timestamps
        # Prediction - assume altitude remains constant
        # Covariance increases over time
        P = P + q**2 * dti

        # Correction - update altitude estimate based on measurement
        K = P / (P + r**2)  # Kalman gain
        x = x + K * (meas[i] - x)
        P = (1 - K) * P

        alt_kf[i] = x

    return alt_kf


def validate_param(name, value, default, min_value=0.0, max_value=None):
    """
    Validates parameter and replaces with default if invalid.

    Args:
        name: parameter name
        value: value to validate
        default: default value
        min_value: minimum allowed value
        max_value: maximum allowed value

    Returns:
        valid parameter value
    """
    if value is None or (isinstance(value, (int, float)) and value <= min_value) or (max_value is not None and value > max_value):
        return default
    return value


def apply_savgol_filter(data, window_size, polyorder=None):
    """
    Applies Savitzky-Golay filter for data smoothing.

    Args:
        data: array of data to smooth
        window_size: filter window size
        polyorder: polynomial order (default: config.SAVGOL_POLYORDER)

    Returns:
        smoothed data or original data in case of error
    """
    if polyorder is None:
        polyorder = getattr(config, 'SAVGOL_POLYORDER', 1)
    try:
        if window_size > 2 and window_size % 2 == 1:  # Window must be odd
            return savgol_filter(data, window_size, polyorder)
    except (ValueError, IndexError, TypeError):
        pass
    return data


def pre_kalman_filter(speed_data, hdop_threshold):
    """
    Filters low-quality GPS points BEFORE applying Kalman filter.

    Excludes points with very poor GPS signal to prevent
    false acceleration peaks in the Kalman filter.

    Args:
        speed_data: list of tuples (abs_ms, rel_ms, time_str, speed_kmh, nmea, sats, hdop)
        hdop_threshold: adaptive HDOP threshold (calculated earlier)

    Returns:
        dict: {
            'valid_mask': np.array of bool - mask of valid points
            'valid_indices': np.array of int - indices of valid points
            'filtered_count': int - number of filtered points
            'filtered_ratio': float - ratio of filtered points
            'reasons': dict - filtering reason statistics
        }
    """
    n = len(speed_data)
    valid_mask = np.ones(n, dtype=bool)
    reasons = {'low_sats': 0, 'high_hdop': 0, 'both': 0}

    # Parameters from config
    min_sats = getattr(config, 'PRE_KALMAN_MIN_SATELLITES', 6)
    hdop_multiplier = getattr(config, 'PRE_KALMAN_MAX_HDOP_MULTIPLIER', 1.5)

    # HDOP threshold for pre-Kalman filtering (softer than final)
    pre_kalman_hdop = hdop_threshold * hdop_multiplier

    for i, point in enumerate(speed_data):
        num_sats = point[5] if len(point) > 5 else None
        hdop_val = point[6] if len(point) > 6 else None

        low_sats = num_sats is not None and num_sats < min_sats
        high_hdop = hdop_val is not None and hdop_val > pre_kalman_hdop

        if low_sats and high_hdop:
            valid_mask[i] = False
            reasons['both'] += 1
        elif low_sats:
            valid_mask[i] = False
            reasons['low_sats'] += 1
        elif high_hdop:
            valid_mask[i] = False
            reasons['high_hdop'] += 1

    valid_indices = np.where(valid_mask)[0]
    filtered_count = n - len(valid_indices)
    filtered_ratio = filtered_count / n if n > 0 else 0.0

    return {
        'valid_mask': valid_mask,
        'valid_indices': valid_indices,
        'filtered_count': filtered_count,
        'filtered_ratio': filtered_ratio,
        'reasons': reasons
    }