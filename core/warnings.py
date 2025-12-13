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
Warning and notification generation module.
Single point for all warning logic (eliminates duplication).
"""

import logging

# Import config for threshold values
try:
    from .. import config
    from ..locales.strings import WARNINGS, CAUTIONS
except ImportError:
    import config
    from locales.strings import WARNINGS, CAUTIONS

logger = logging.getLogger(__name__)


def compute_warnings(power_data, weather_data=None, gps_frequency=None,
                     nmea_file=None, pre_kalman_stats=None):
    """
    Unified function for computing all warnings.

    Args:
        power_data: dict with power calculation results
        weather_data: dict with weather data
        gps_frequency: float, GPS frequency in Hz
        nmea_file: str, path to NMEA file (for additional analysis)
        pre_kalman_stats: dict with pre-Kalman filtering statistics

    Returns:
        tuple: (warnings: dict, cautions: dict)
    """
    warnings = {}
    cautions = {}

    # 1. Pre-Kalman filtering
    _check_pre_kalman_filtering(pre_kalman_stats, warnings, cautions)

    # 2. Post-filtering (final)
    _check_post_filtering(power_data, warnings, cautions)

    # 3. Percentile instability
    _check_percentile_stability(power_data, warnings, cautions)

    # 4. Weather
    _check_weather(weather_data, warnings, cautions)

    # 5. GPS frequency
    _check_gps_frequency(gps_frequency, warnings, cautions)

    # 6. NMEA file analysis (if provided)
    if nmea_file:
        _check_nmea_quality(nmea_file, power_data, warnings, cautions)

    # 7. Interpolation check
    if nmea_file:
        _check_interpolation(nmea_file, gps_frequency, cautions)

    return warnings, cautions


def _check_pre_kalman_filtering(pre_kalman_stats, warnings, cautions):
    """Check pre-Kalman filtering statistics."""
    if not pre_kalman_stats:
        return

    ratio = pre_kalman_stats.get('filtered_ratio', 0)
    threshold = getattr(config, 'PRE_KALMAN_WARNING_THRESHOLD', 0.2)

    if ratio > threshold:
        count = pre_kalman_stats.get('filtered_count', 0)
        reasons = pre_kalman_stats.get('reasons', {})

        reason_parts = []
        if reasons.get('low_sats', 0) > 0:
            reason_parts.append(f"low satellites: {reasons['low_sats']}")
        if reasons.get('high_hdop', 0) > 0:
            reason_parts.append(f"high HDOP: {reasons['high_hdop']}")
        if reasons.get('both', 0) > 0:
            reason_parts.append(f"both factors: {reasons['both']}")

        reason_str = ', '.join(reason_parts) if reason_parts else 'low GPS quality'

        cautions["pre_kalman_filtered"] = CAUTIONS['pre_kalman_filtered'].format(
            ratio=ratio, count=count, reasons=reason_str
        )


def _check_post_filtering(power_data, warnings, cautions):
    """Check final data filtering."""
    if not power_data:
        return

    filtered_ratio = power_data.get("filtered_ratio", 0)
    total_points = power_data.get("total_points", 0)
    valid_points = power_data.get("valid_points", 0)
    filtered_count = total_points - valid_points

    if filtered_ratio >= 1.0 and total_points > 0:
        warnings["all_filtered"] = WARNINGS['all_filtered'].format(total=total_points)
    elif filtered_ratio > 0.9:
        warnings["mostly_filtered"] = WARNINGS['high_filtered_ratio'].format(
            ratio=filtered_ratio, filtered=filtered_count, total=total_points
        )
    elif filtered_ratio > 0.8:
        cautions["many_filtered"] = CAUTIONS['many_filtered'].format(
            ratio=filtered_ratio, filtered=filtered_count, total=total_points
        )


def _check_percentile_stability(power_data, warnings, cautions):
    """Check data stability via consistency score."""
    if not power_data:
        return

    power_estimation = power_data.get("power_estimation", {})
    if not power_estimation:
        return

    # consistency_score is always computed (even if method not requested)
    score = power_estimation.get("consistency_score")

    if score is None:
        return

    if score < 0.5:
        warnings["unstable"] = WARNINGS['unstable'].format(score=score)
    elif score < 0.8:
        cautions["unstable"] = CAUTIONS['unstable'].format(score=score)


def _check_weather(weather_data, warnings, cautions):
    """Check weather conditions."""
    if not weather_data:
        return

    # Check for default weather usage
    if weather_data.get('weather_source') == 'default':
        cautions["default_weather"] = CAUTIONS['default_weather']

    wind_speed = weather_data.get("wind_speed_kph", 0)
    high_wind_threshold = getattr(config, 'HIGH_WIND_SPEED_KPH', 30)

    if wind_speed > high_wind_threshold:
        warnings["wind"] = WARNINGS['high_wind'].format(
            threshold=high_wind_threshold, speed=wind_speed
        )
    elif wind_speed >= 14:
        if "wind" not in warnings:
            cautions["wind"] = CAUTIONS['wind'].format(speed=wind_speed)

    conditions = weather_data.get("conditions", "").lower()
    bad_keywords = getattr(config, 'BAD_WEATHER_KEYWORDS', [
        'rain', 'snow', 'дождь', 'снег', 'ливень', 'осадки',
        'гроза', 'гололед', 'thunderstorm', 'sleet', 'ice', 'showers'
    ])

    for keyword in bad_keywords:
        if keyword in conditions:
            cautions["bad_weather"] = CAUTIONS['bad_weather'].format(conditions=conditions)
            break


def _check_gps_frequency(gps_frequency, warnings, cautions):
    """Check GPS frequency."""
    if gps_frequency is None:
        return

    rounded = round(gps_frequency)
    low_freq = getattr(config, 'LOW_GPS_FREQUENCY_HZ', 8.0)
    medium_freq = getattr(config, 'MEDIUM_GPS_FREQUENCY_HZ', 10.0)

    if rounded < low_freq:
        warnings["gps_frequency"] = WARNINGS['low_gps_frequency'].format(
            freq=gps_frequency, threshold=low_freq
        )
    elif rounded < medium_freq:
        if "gps_frequency" not in warnings:
            cautions["gps_frequency"] = CAUTIONS['gps_frequency'].format(freq=gps_frequency)


def _check_nmea_quality(nmea_file, power_data, warnings, cautions):
    """Analyze NMEA file quality."""
    try:
        # Import here to avoid circular dependencies
        from ..parsers.nmea_handler import analyze_nmea_quality, analyze_nmea_timestamps
    except ImportError:
        try:
            from parsers.nmea_handler import analyze_nmea_quality, analyze_nmea_timestamps
        except ImportError:
            logger.warning("Failed to import nmea_handler for quality analysis")
            return

    try:
        quality_metrics = analyze_nmea_quality(nmea_file)
        _, gaps = analyze_nmea_timestamps(nmea_file)
    except Exception as e:
        logger.warning(f"Error analyzing NMEA file: {e}")
        return

    # Check GPS validity
    if quality_metrics:
        invalid_ratio = quality_metrics.get("invalid_ratio", 0)
        if invalid_ratio > 0.1:
            warnings["gps_validity"] = WARNINGS['gps_validity']
        elif invalid_ratio > 0.05:
            if "gps_validity" not in warnings:
                cautions["gps_validity"] = CAUTIONS['gps_validity']

        # Check GPS quality by problematic points percentage
        total_points = quality_metrics.get("total_points", 0)
        if total_points > 0:
            low_fix_count = quality_metrics.get("low_fix_count", 0)
            low_sat_count = quality_metrics.get("low_sat_count", 0)
            high_hdop_count = quality_metrics.get("high_hdop_count", 0)

            # Total problematic points count (without duplication)
            problem_count = max(low_fix_count, low_sat_count, high_hdop_count)
            problem_ratio = problem_count / total_points

            if problem_ratio > 0.10:
                # > 10% problematic points - serious issue
                warnings["gps_quality"] = WARNINGS['gps_quality'].format(ratio=problem_ratio)
            elif problem_ratio > 0.02:
                # > 2% problematic points - warning
                if "gps_quality" not in warnings:
                    cautions["gps_quality"] = CAUTIONS['gps_quality'].format(ratio=problem_ratio)

    # Check for gaps
    if gaps and len(gaps) > 0:
        gap_count = len(gaps)
        if gap_count > 50:  # >50 gaps (~5 sec at 10 Hz) — serious issue
            warnings["log_issue"] = WARNINGS['log_issue'].format(count=gap_count)
        else:
            cautions["log_issue"] = CAUTIONS['log_issue'].format(count=gap_count)


def _check_interpolation(nmea_file, gps_frequency, cautions):
    """Check for interpolated data."""
    try:
        # Import here to avoid circular dependencies
        try:
            from ..parsers.nmea_handler import detect_interpolation
        except ImportError:
            from parsers.nmea_handler import detect_interpolation

        result = detect_interpolation(nmea_file, gps_frequency)

        if result['is_interpolated']:
            reasons = ', '.join(result['reasons'])
            cautions['interpolated_data'] = CAUTIONS['interpolated_data'].format(reasons=reasons)

    except Exception as e:
        logger.warning(f"Error checking interpolation: {e}")


