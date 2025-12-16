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
GPSDyno CLI entry point.

Calculates vehicle wheel horsepower from GPS data in NMEA files.
"""
import json
import os
import sys
import argparse
import logging

# Add script directory to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from core.calculator import calculate_power
from core.visualization import plot_power_chart, plot_track_pseudo3d, correlate_track_data
from core.warnings import compute_warnings as compute_warnings_unified
from core.structures import SPEED_ABS_MS
from parsers.nmea_handler import (
    extract_speed_altitude_data,
    parse_nmea_file,
)
import config
from locales.strings import ERRORS

# Configure logging (basicConfig is sufficient, no need for duplicate handler)
logging.basicConfig(
    level=logging.ERROR,
    format='%(levelname)s: %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger('gpsdyno_calculator')


def format_duration_ms(duration_ms):
    """Format duration from milliseconds to mm:ss.mmm string."""
    if duration_ms is None:
        return ""
    seconds_total = duration_ms / 1000.0
    minutes = int(seconds_total // 60)
    seconds = int(seconds_total % 60)
    milliseconds = int((seconds_total - int(seconds_total)) * 1000)
    return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def analyze_nmea_file(nmea_file_path, weather_json_path=None):
    """
    Analyze NMEA file: parse coordinates, time, speed and altitude.

    Args:
        nmea_file_path: path to NMEA file
        weather_json_path: path to JSON file with weather data (optional)

    Returns:
        dict with analysis results
    """
    result = {
        'file': os.path.basename(nmea_file_path),
        'datetime': None,
        'latitude': None,
        'longitude': None,
        'weather': None,
        'speed_data': [],
        'altitude_data': [],
        'coords': [],
        'gps_frequency': 0,
        'session_duration_ms': None,
        'error': None
    }

    try:
        nmea_data = parse_nmea_file(nmea_file_path, return_dict=True)
        lat = nmea_data['latitude']
        lon = nmea_data['longitude']
        dt_object = nmea_data['datetime']

        result['latitude'] = lat
        result['longitude'] = lon
        result['datetime'] = dt_object.isoformat() if dt_object else None

        # Weather: from external JSON or defaults
        weather_data = None
        if weather_json_path and os.path.exists(weather_json_path):
            try:
                with open(weather_json_path, 'r') as f:
                    external_weather = json.load(f)
                if external_weather.get('success', True):
                    weather_data = external_weather
                    result['weather_source'] = 'external_json'
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error reading weather.json: {e}")

        if weather_data:
            result['weather'] = weather_data
        else:
            # Default values (standard conditions)
            result['weather'] = config.DEFAULT_WEATHER.copy()
            result['weather_source'] = 'default'

        speed_data, altitude_data, gps_frequency, nmea_lines, first_timestamp_ms, coords_data = extract_speed_altitude_data(nmea_file_path)
        if speed_data:
            result['speed_data'] = speed_data
            result['gps_frequency'] = round(gps_frequency, 2)
            result['nmea_lines'] = nmea_lines
            result['first_timestamp_ms'] = first_timestamp_ms

        if altitude_data:
            result['altitude_data'] = altitude_data

        if coords_data:
            result['coords'] = coords_data

        if speed_data and first_timestamp_ms is not None:
            last_timestamp_ms = speed_data[-1][SPEED_ABS_MS]
            result['session_duration_ms'] = last_timestamp_ms - first_timestamp_ms

    except ValueError as ve:
        result['error'] = str(ve)
    except Exception as e:
        result['error'] = f"Unexpected error: {str(e)}"

    return result


def format_json_response(car_info, power_data, result, weather_data=None, gps_frequency=None,
                         nmea_file=None, session_duration_ms=None, warnings_dict=None, cautions_dict=None):
    """
    Format JSON response for CLI output.

    Args:
        car_info: vehicle information
        power_data: power calculation results
        result: chart generation results
        weather_data: weather data
        gps_frequency: GPS frequency
        nmea_file: NMEA file name
        session_duration_ms: session duration in ms
        warnings_dict: warnings
        cautions_dict: cautions

    Returns:
        dict with JSON response
    """
    response = {
        "success": True,
        "car_info": {
            "name": car_info.get("name", "Unknown vehicle"),
            "mass": car_info.get("mass"),
            "drag_coefficient": car_info.get("drag_coefficient"),
            "frontal_area": car_info.get("frontal_area"),
            "rolling_resistance": car_info.get("rolling_resistance")
        },
        "results": {
            "power_estimation": result.get("power_estimation"),
            "uncertainty": power_data.get("uncertainty") if power_data else None
        },
        "graphs": result.get("chart_paths", {}),
        "session_info": {}
    }

    if gps_frequency is not None:
        response["gps_frequency"] = gps_frequency

    if session_duration_ms is not None:
        response["session_info"]["duration_ms"] = session_duration_ms
        response["session_info"]["duration_formatted"] = format_duration_ms(session_duration_ms)

    if power_data and power_data.get("hdop_statistics"):
        response["gps_quality"] = {
            "hdop_statistics": power_data["hdop_statistics"]
        }

    if weather_data:
        response["weather"] = {
            "temperature_c": weather_data.get("temperature_c"),
            "pressure_hpa": weather_data.get("pressure_hpa"),
            "humidity_percent": weather_data.get("humidity_percent"),
            "wind_speed_kph": weather_data.get("wind_speed_kph"),
            "conditions": weather_data.get("conditions"),
            "location": weather_data.get("location"),
            "weather_datetime": weather_data.get("weather_datetime")
        }

    # Pre-Kalman filtering statistics
    pre_kalman_stats = power_data.get("pre_kalman_stats") if power_data else None
    if pre_kalman_stats:
        response["pre_kalman_filtering"] = {
            "filtered_count": pre_kalman_stats.get("filtered_count", 0),
            "filtered_ratio": round(pre_kalman_stats.get("filtered_ratio", 0) * 100, 1),
            "reasons": pre_kalman_stats.get("reasons", {})
        }

    # Use passed warnings and cautions (computed via compute_warnings_unified)
    if warnings_dict:
        response["warning"] = warnings_dict

    if cautions_dict:
        response["caution"] = cautions_dict

    return response


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Vehicle power calculation from NMEA data')
    parser.add_argument('nmea_file', help='Path to NMEA file')
    parser.add_argument('--car', help='Vehicle name', default=None)
    parser.add_argument('--mass', type=float, help='Vehicle mass in kg', default=None)
    parser.add_argument('--drag_coefficient', type=float, help='Drag coefficient (Cd)', default=None)
    parser.add_argument('--frontal_area', type=float, help='Frontal area in m²', default=None)
    parser.add_argument('--rolling_resistance', type=float, help='Rolling resistance coefficient', default=0.015)
    parser.add_argument('--output', help='Output path for charts', default=None)
    parser.add_argument('--file_type', help='File type (nmea)', default="nmea")
    parser.add_argument('--weather-json', dest='weather_json', help='Path to JSON file with weather data', default=None)
    parser.add_argument('--methods', help='Power estimation methods, comma-separated. Available: robust_mean, percentile_99 (p99), mode_kde, peak_detection, consistency, all', default='peak_detection')
    args = parser.parse_args()

    try:
        if not os.path.exists(args.nmea_file):
            error_response = {
                "success": False,
                "error": ERRORS['file_not_found'].format(file_path=args.nmea_file)
            }
            print(json.dumps(error_response, ensure_ascii=False, indent=2))
            sys.exit(1)

        if args.file_type.lower() != "nmea":
            raise ValueError(ERRORS['only_nmea_supported'])

        result = analyze_nmea_file(args.nmea_file, weather_json_path=args.weather_json)

        if 'error' in result and result['error']:
            error_response = {
                "success": False,
                "error": result['error']
            }
            print(json.dumps(error_response, ensure_ascii=False, indent=2))
            sys.exit(1)

        car_name = args.car if args.car else "unknown"

        car_info = {
            "name": car_name,
            "mass": args.mass,
            "drag_coefficient": args.drag_coefficient,
            "frontal_area": args.frontal_area,
            "rolling_resistance": args.rolling_resistance,
        }

        weather_data = result.get('weather')
        # Add weather source to weather_data for warnings
        if weather_data:
            weather_data['weather_source'] = result.get('weather_source', 'unknown')
        if weather_data and not weather_data.get('success', True):
            error_response = {
                "success": False,
                "error": ERRORS['weather_server_error']
            }
            print(json.dumps(error_response, ensure_ascii=False, indent=2))
            sys.exit(1)

        altitude_data = None
        if 'altitude_data' in result and result['altitude_data']:
            altitude_data = result.get('altitude_data')

        # Check minimum speed data points
        speed_data = result.get('speed_data', [])
        min_points = getattr(config, 'MIN_SPEED_POINTS', 100)
        if len(speed_data) < min_points:
            error_response = {
                "success": False,
                "error": ERRORS['insufficient_speed_data'].format(min_points=min_points)
            }
            print(json.dumps(error_response, ensure_ascii=False, indent=2))
            sys.exit(1)

        power_data = calculate_power(
            speed_data,
            car_info,
            weather_data,
            altitude_data,
            result.get('first_timestamp_ms'),
            methods=args.methods
        )

        if not power_data:
            error_response = {
                "success": False,
                "error": ERRORS['calculation_failed']
            }
            print(json.dumps(error_response, ensure_ascii=False, indent=2))
            sys.exit(1)

        warnings_dict, cautions_dict = compute_warnings_unified(
            power_data,
            weather_data,
            result.get('gps_frequency'),
            args.nmea_file,
            pre_kalman_stats=power_data.get('pre_kalman_stats')
        )

        chart_result = plot_power_chart(
            power_data,
            car_info,
            args.output,
            result.get('session_duration_ms'),
            warnings_dict,
            cautions_dict,
            weather_data
        )

        if not chart_result:
            error_response = {
                "success": False,
                "error": ERRORS['chart_failed']
            }
            print(json.dumps(error_response, ensure_ascii=False, indent=2))
            sys.exit(1)

        chart_result['nmea_lines'] = result.get('nmea_lines', {})

        # Build track (if coordinates available)
        if result.get('coords') and result.get('speed_data') and result.get('altitude_data'):
            track_data = correlate_track_data(
                result['coords'],
                result['speed_data'],
                result['altitude_data'],
                power_data
            )
            location = weather_data.get('location') if weather_data else None
            track_path = plot_track_pseudo3d(
                track_data['coords'],
                track_data['altitudes'],
                track_data['powers'],
                args.output,
                location
            )
            if track_path:
                chart_result['chart_paths']['track'] = track_path

        response = format_json_response(
            car_info,
            power_data,
            chart_result,
            weather_data,
            result.get('gps_frequency'),
            args.nmea_file,
            result.get('session_duration_ms'),
            warnings_dict,
            cautions_dict
        )

        print(json.dumps(response, ensure_ascii=False, indent=2))

    except Exception as e:
        error_response = {
            "success": False,
            "error": f"Error: {str(e)}"
        }
        print(json.dumps(error_response, ensure_ascii=False, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
