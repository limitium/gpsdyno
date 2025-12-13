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
NMEA file handler - parsing and data extraction from NMEA files.
"""
import pynmea2
import logging
import os
import sys
from datetime import datetime

# Add parent directory to path for config import
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

import config

logger = logging.getLogger('nmea_handler')


def convert_nmea_to_milliseconds(msg):
    """Returns message time in milliseconds from epoch."""
    try:
        if hasattr(msg, 'timestamp'):
            if hasattr(msg, 'datestamp'):
                dt = datetime.combine(msg.datestamp, msg.timestamp)
            elif hasattr(msg, '_date_from_rmc') and msg._date_from_rmc:
                dt = datetime.combine(msg._date_from_rmc, msg.timestamp)
            else:
                dt = datetime.combine(datetime.now().date(), msg.timestamp)
            return int(dt.timestamp() * 1000)
    except (ValueError, AttributeError) as e:
        logger.error(f"Error converting NMEA time: {e}")
    return None


import re
from typing import Tuple, Optional, Dict, Any, Union


def parse_nmea_file(file_path: str, return_dict: bool = False) -> Union[Tuple[float, float, Optional[str]], Dict[str, Any]]:
    """
    Parses NMEA file and extracts coordinates and time.

    Args:
        file_path: path to NMEA file
        return_dict: if True, returns dict; if False - tuple (backward compatibility)

    Returns:
        If return_dict=False (default):
            tuple: (latitude, longitude, datetime_str)
        If return_dict=True:
            dict: {
                'datetime': datetime object,
                'latitude': float,
                'longitude': float,
                'nmea_messages': {'rmc': msg, 'gga': msg}
            }
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    if not lines:
        raise ValueError(f"File {file_path} is empty")

    gprmc_msg = None
    gpgga_msg = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        try:
            msg = pynmea2.parse(line)
            msg.nmea_str = line
            if msg.sentence_type == 'RMC' and not gprmc_msg:
                gprmc_msg = msg
            elif msg.sentence_type == 'GGA' and not gpgga_msg:
                gpgga_msg = msg
            if gprmc_msg and gpgga_msg:
                break
        except pynmea2.ParseError:
            continue
        except Exception as e:
            logger.error(f"Error parsing line '{line}': {e}")
            continue

    if not gprmc_msg and not gpgga_msg:
        raise ValueError(f"No GPRMC or GPGGA message found in file {file_path}")

    # Extract datetime
    file_datetime = None
    if gprmc_msg and hasattr(gprmc_msg, 'datestamp') and hasattr(gprmc_msg, 'timestamp'):
        if gprmc_msg.datestamp and gprmc_msg.timestamp:
            file_datetime = datetime.combine(gprmc_msg.datestamp, gprmc_msg.timestamp)

    # Fallback: extract date from filename (session_YYYYMMDD_HHMMSS)
    if not file_datetime:
        filename = os.path.basename(file_path)
        datetime_match = re.search(r'session_(\d{8})_(\d{6})', filename)
        if datetime_match:
            date_str = datetime_match.group(1)
            time_str = datetime_match.group(2)
            try:
                file_datetime = datetime.strptime(f"{date_str} {time_str}", '%Y%m%d %H%M%S')
            except ValueError:
                pass

    if not file_datetime and not return_dict:
        # For tuple mode datetime_str can be None
        pass
    elif not file_datetime and return_dict:
        raise ValueError(f"Failed to determine date and time from file {file_path}")

    # Extract coordinates
    lat = lon = None
    if gprmc_msg and hasattr(gprmc_msg, 'latitude') and hasattr(gprmc_msg, 'longitude'):
        lat = gprmc_msg.latitude
        lon = gprmc_msg.longitude
    elif gpgga_msg and hasattr(gpgga_msg, 'latitude') and hasattr(gpgga_msg, 'longitude'):
        lat = gpgga_msg.latitude
        lon = gpgga_msg.longitude

    if lat is None or lon is None:
        raise ValueError(f"Failed to extract coordinates from file {file_path}")

    if return_dict:
        return {
            'datetime': file_datetime,
            'latitude': lat,
            'longitude': lon,
            'nmea_messages': {'rmc': gprmc_msg, 'gga': gpgga_msg}
        }
    else:
        datetime_str = file_datetime.isoformat() if file_datetime else None
        return lat, lon, datetime_str


def analyze_nmea_timestamps(file_path):
    """
    Analyzes timestamps in NMEA file to find gaps.
    """
    all_timestamps = []
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            if line.startswith('$GPRMC') or line.startswith('$GPGGA'):
                try:
                    msg = pynmea2.parse(line.strip())
                    if hasattr(msg, 'timestamp'):
                        hours = msg.timestamp.hour
                        minutes = msg.timestamp.minute
                        seconds = msg.timestamp.second
                        milliseconds = int(msg.timestamp.microsecond / 1000)
                        timestamp_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
                        all_timestamps.append((timestamp_str, line.strip(), msg.timestamp))
                except pynmea2.ParseError:
                    continue
                except Exception as e:
                    logger.debug(f"Error analyzing timestamp: {line.strip()} - {str(e)}")
                    continue

        gaps = []
        if len(all_timestamps) >= 2:
            for i in range(1, len(all_timestamps)):
                prev_time_str = all_timestamps[i-1][0]
                curr_time_str = all_timestamps[i][0]
                prev_time_obj = all_timestamps[i-1][2]
                curr_time_obj = all_timestamps[i][2]

                try:
                    today = datetime.today().date()
                    prev_dt = datetime.combine(today, prev_time_obj)
                    curr_dt = datetime.combine(today, curr_time_obj)
                    time_diff = (curr_dt - prev_dt).total_seconds()
                    if time_diff < 0:
                        time_diff += 24 * 3600

                    gap_threshold = getattr(config, 'GAP_THRESHOLD', 0.5)
                    if time_diff > gap_threshold:
                        gaps.append({
                            'index': i,
                            'time_before': prev_time_str,
                            'time_after': curr_time_str,
                            'duration': time_diff,
                            'nmea_before': all_timestamps[i-1][1],
                            'nmea_after': all_timestamps[i][1]
                        })
                except Exception as e:
                    logger.debug(f"Error analyzing time difference: {prev_time_str} -> {curr_time_str} - {str(e)}")

        return all_timestamps, gaps
    except Exception as e:
        logger.error(f"Error analyzing timestamps: {str(e)}")
        return [], []


def analyze_nmea_quality(file_path: str) -> Optional[dict]:
    """Analyze GPS quality fields from GPRMC and GPGGA sentences."""
    rmc_total = 0
    rmc_invalid = 0
    low_fix = 0
    low_sats = 0
    high_hdop = 0
    issues = []

    try:
        with open(file_path, "r") as f:
            for idx, line in enumerate(f, start=1):
                stripped = line.strip()
                if stripped.startswith("$GPRMC"):
                    try:
                        msg = pynmea2.parse(stripped)
                        rmc_total += 1
                        reasons = []
                        if getattr(msg, "status", "A") == "V":
                            rmc_invalid += 1
                            reasons.append("invalid_rmc")
                        if reasons:
                            issues.append({"line_number": idx, "line": stripped, "issues": reasons})
                    except pynmea2.ParseError:
                        continue
                elif stripped.startswith("$GPGGA"):
                    try:
                        msg = pynmea2.parse(stripped)
                        reasons = []
                        gps_qual = getattr(msg, "gps_qual", None)
                        if gps_qual is not None and gps_qual != "" and int(gps_qual) == 0:
                            low_fix += 1
                            reasons.append("low_fix")
                        num_sats = getattr(msg, "num_sats", None)
                        if num_sats is not None and num_sats != "" and int(num_sats) < config.MIN_SATELLITES:
                            low_sats += 1
                            reasons.append("few_sats")
                        if getattr(msg, "horizontal_dil", None) not in (None, ""):
                            try:
                                hdop_val = float(msg.horizontal_dil)
                                if hdop_val > config.MAX_HDOP:
                                    high_hdop += 1
                                    reasons.append("high_hdop")
                            except ValueError:
                                pass
                        if reasons:
                            issues.append({"line_number": idx, "line": stripped, "issues": reasons})
                    except pynmea2.ParseError:
                        continue
    except Exception as e:
        logger.error(f"GPS quality analysis error: {e}")
        return None

    invalid_ratio = (rmc_invalid / rmc_total) if rmc_total else 0
    return {
        "invalid_ratio": invalid_ratio,
        "low_fix_count": low_fix,
        "low_sat_count": low_sats,
        "high_hdop_count": high_hdop,
        "total_points": rmc_total,
        "issues": issues,
    }


def extract_speed_altitude_data(file_path):
    """Extracts speed and altitude data from NMEA file."""
    all_timestamps, gaps = analyze_nmea_timestamps(file_path)

    speed_data = []
    altitude_data = []
    coords_data = []
    nmea_lines = {}
    first_timestamp_ms = None
    current_date = None
    gga_quality = {}
    rmc_index_by_ts = {}

    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            if line.startswith('$GPRMC'):
                try:
                    orig_line = line.strip()
                    msg = pynmea2.parse(orig_line)
                    msg.nmea_str = orig_line

                    if hasattr(msg, 'datestamp') and msg.datestamp:
                        current_date = msg.datestamp

                    if hasattr(msg, 'timestamp') and hasattr(msg, 'spd_over_grnd'):
                        # Skip points without valid speed (empty field in NMEA)
                        if msg.spd_over_grnd is None or msg.spd_over_grnd == '':
                            logger.debug(f"Skipped GPRMC without speed: {orig_line}")
                            continue
                        speed_kph = msg.spd_over_grnd * 1.852
                        timestamp_ms = convert_nmea_to_milliseconds(msg)

                        if timestamp_ms is None:
                            logger.debug(f"Failed to get timestamp_ms for line: {orig_line}")
                            continue

                        hours = msg.timestamp.hour
                        minutes = msg.timestamp.minute
                        seconds = msg.timestamp.second
                        milliseconds = int(msg.timestamp.microsecond / 1000)
                        timestamp_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

                        if first_timestamp_ms is None:
                            first_timestamp_ms = timestamp_ms

                        relative_ms = timestamp_ms - first_timestamp_ms if first_timestamp_ms is not None else 0
                        speed_data.append((timestamp_ms, relative_ms, timestamp_str, speed_kph, orig_line, None, None))
                        rmc_index_by_ts[timestamp_ms] = len(speed_data) - 1
                        nmea_lines[timestamp_str] = orig_line

                        if hasattr(msg, 'latitude') and hasattr(msg, 'longitude'):
                            if msg.latitude is not None and msg.longitude is not None:
                                coords_data.append((timestamp_ms, msg.latitude, msg.longitude))
                    else:
                        logger.debug(f"Skipped NMEA line due to missing timestamp or spd_over_grnd: {orig_line}")

                except pynmea2.ParseError as e:
                    logger.debug(f"NMEA line parse error: {line.strip()} - {str(e)}")
                    continue
                except Exception as e:
                    logger.debug(f"Unexpected error processing NMEA line: {line.strip()} - {str(e)}")
                    continue

            elif line.startswith('$GPGGA'):
                try:
                    orig_line = line.strip()
                    msg = pynmea2.parse(orig_line)
                    msg.nmea_str = orig_line

                    if current_date:
                        msg._date_from_rmc = current_date

                    if hasattr(msg, 'timestamp') and hasattr(msg, 'altitude'):
                        altitude_m = float(msg.altitude) if msg.altitude is not None else 0.0
                        timestamp_ms = convert_nmea_to_milliseconds(msg)

                        if timestamp_ms is None:
                            logger.debug(f"Failed to get timestamp_ms for GGA line: {orig_line}")
                            continue

                        hours = msg.timestamp.hour
                        minutes = msg.timestamp.minute
                        seconds = msg.timestamp.second
                        milliseconds = int(msg.timestamp.microsecond / 1000)
                        timestamp_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

                        if first_timestamp_ms is None:
                            first_timestamp_ms = timestamp_ms

                        relative_ms = timestamp_ms - first_timestamp_ms if first_timestamp_ms is not None else 0
                        num_sats = int(msg.num_sats) if getattr(msg, 'num_sats', None) not in (None, '') else None
                        hdop_val = float(msg.horizontal_dil) if getattr(msg, 'horizontal_dil', None) not in (None, '') else None
                        gga_quality[timestamp_ms] = (num_sats, hdop_val)
                        altitude_data.append((timestamp_ms, relative_ms, timestamp_str, altitude_m, orig_line))
                        nmea_lines[timestamp_str] = orig_line

                        if hasattr(msg, 'latitude') and hasattr(msg, 'longitude'):
                            if msg.latitude is not None and msg.longitude is not None:
                                coords_data.append((timestamp_ms, msg.latitude, msg.longitude))

                        if timestamp_ms in rmc_index_by_ts:
                            idx = rmc_index_by_ts[timestamp_ms]
                            existing = list(speed_data[idx])
                            if len(existing) < 7:
                                existing += [None] * (7 - len(existing))
                            existing[5] = num_sats
                            existing[6] = hdop_val
                            speed_data[idx] = tuple(existing)
                    else:
                        logger.debug(f"Skipped NMEA GGA line due to missing timestamp or altitude: {orig_line}")

                except pynmea2.ParseError as e:
                    logger.debug(f"NMEA GGA line parse error: {line.strip()} - {str(e)}")
                    continue
                except Exception as e:
                    logger.debug(f"Unexpected error processing NMEA GGA line: {line.strip()} - {str(e)}")
                    continue

        for idx, entry in enumerate(speed_data):
            if len(entry) < 7 or entry[5] is None or entry[6] is None:
                ts = entry[0]
                if ts in gga_quality:
                    num_s, hd = gga_quality[ts]
                    updated = list(entry)
                    if len(updated) < 7:
                        updated += [None] * (7 - len(updated))
                    updated[5] = num_s
                    updated[6] = hd
                    speed_data[idx] = tuple(updated)

        timestamp_milliseconds = [t[0] for t in speed_data] if speed_data else []
        gps_frequency = calculate_gps_frequency(timestamp_milliseconds)

        return speed_data, altitude_data, gps_frequency, nmea_lines, first_timestamp_ms, coords_data

    except Exception as e:
        logger.error(f"Error extracting speed and altitude data: {e}")
        return [], [], 0, {}, None, []


def extract_speed_data(file_path):
    """
    Backward compatibility with previous function version.
    """
    speed_data, _, gps_frequency, nmea_lines, first_timestamp_ms, _ = extract_speed_altitude_data(file_path)
    return speed_data, gps_frequency, nmea_lines, first_timestamp_ms


def calculate_gps_frequency(timestamp_milliseconds):
    """
    Calculates GPS frequency in Hz from timestamps in milliseconds.
    """
    if len(timestamp_milliseconds) < 2:
        return 0

    interval_sum = 0
    for i in range(1, len(timestamp_milliseconds)):
        time_diff = timestamp_milliseconds[i] - timestamp_milliseconds[i - 1]
        if 0 < time_diff < 5000:
            interval_sum += time_diff

    average_interval = interval_sum / (len(timestamp_milliseconds) - 1)
    gps_frequency = 1000 / average_interval if average_interval > 0 else 0
    return round(gps_frequency, 2)


def analyze_nmea_file(nmea_file_path):
    """
    Analyzes NMEA file and returns track coordinates.
    """
    _, _, _, _, _, coords_data = extract_speed_altitude_data(nmea_file_path)

    if not coords_data:
        return []

    coords = [(lat, lon) for _, lat, lon in coords_data]
    return coords


def detect_interpolation(file_path: str, gps_frequency: float = None) -> dict:
    """
    Detects signs of interpolated data in NMEA file.

    Data is considered interpolated if >=2 of 4 criteria are met:
    1. num_sats > INTERPOLATION_MAX_VALID_SATS (impossible satellite count)
    2. std(HDOP) < INTERPOLATION_MIN_HDOP_STD (HDOP nearly constant)
    3. std(intervals) < INTERPOLATION_MIN_INTERVAL_STD_MS (perfectly equal intervals)
    4. frequency > INTERPOLATION_MAX_GPS_FREQ (ultra-high frequency)

    Args:
        file_path: path to NMEA file
        gps_frequency: GPS frequency (if already computed), otherwise will be calculated

    Returns:
        dict: {
            'is_interpolated': bool,
            'reasons': list[str],
            'details': {
                'max_num_sats': int,
                'hdop_std': float,
                'interval_std_ms': float,
                'frequency': float
            }
        }
    """
    import numpy as np

    result = {
        'is_interpolated': False,
        'reasons': [],
        'details': {
            'max_num_sats': None,
            'hdop_std': None,
            'interval_std_ms': None,
            'frequency': gps_frequency
        }
    }

    num_sats_values = []
    hdop_values = []
    timestamps_ms = []

    try:
        with open(file_path, 'r') as f:
            for line in f:
                stripped = line.strip()

                # Collect num_sats and HDOP from GPGGA
                if stripped.startswith('$GPGGA'):
                    try:
                        msg = pynmea2.parse(stripped)
                        if getattr(msg, 'num_sats', None) not in (None, ''):
                            num_sats_values.append(int(msg.num_sats))
                        if getattr(msg, 'horizontal_dil', None) not in (None, ''):
                            hdop_values.append(float(msg.horizontal_dil))
                    except (pynmea2.ParseError, ValueError):
                        continue

                # Collect timestamps from GPRMC for interval calculation
                elif stripped.startswith('$GPRMC'):
                    try:
                        msg = pynmea2.parse(stripped)
                        if hasattr(msg, 'timestamp') and msg.timestamp:
                            ts = msg.timestamp
                            ts_ms = ts.hour * 3600000 + ts.minute * 60000 + ts.second * 1000 + ts.microsecond // 1000
                            timestamps_ms.append(ts_ms)
                    except (pynmea2.ParseError, ValueError):
                        continue
    except Exception as e:
        logger.warning(f"Error detecting interpolation: {e}")
        return result

    min_points = getattr(config, 'INTERPOLATION_MIN_POINTS', 100)

    # Insufficient data for analysis
    if len(timestamps_ms) < min_points:
        return result

    criteria_met = 0

    # Criterion 1: impossible satellite count
    if num_sats_values:
        max_sats = max(num_sats_values)
        result['details']['max_num_sats'] = max_sats
        max_valid_sats = getattr(config, 'INTERPOLATION_MAX_VALID_SATS', 100)
        if max_sats > max_valid_sats:
            criteria_met += 1
            result['reasons'].append(f"num_sats={max_sats}")

    # Criterion 2: HDOP nearly constant
    if len(hdop_values) >= min_points:
        hdop_std = np.std(hdop_values)
        result['details']['hdop_std'] = float(hdop_std)
        min_hdop_std = getattr(config, 'INTERPOLATION_MIN_HDOP_STD', 0.01)
        if hdop_std < min_hdop_std:
            criteria_met += 1
            result['reasons'].append("HDOP constant")

    # Criterion 3: perfectly equal intervals
    if len(timestamps_ms) >= min_points:
        intervals = np.diff(timestamps_ms)
        # Filter anomalous intervals (>5 sec) for std calculation
        valid_intervals = intervals[(intervals > 0) & (intervals < 5000)]
        if len(valid_intervals) > 0:
            interval_std = np.std(valid_intervals)
            result['details']['interval_std_ms'] = float(interval_std)
            min_interval_std = getattr(config, 'INTERPOLATION_MIN_INTERVAL_STD_MS', 1.0)
            if interval_std < min_interval_std:
                criteria_met += 1
                result['reasons'].append("perfect intervals")

    # Criterion 4: ultra-high frequency
    freq = gps_frequency
    if freq is None and len(timestamps_ms) >= 2:
        # Calculate frequency if not provided
        intervals = np.diff(timestamps_ms)
        valid_intervals = intervals[(intervals > 0) & (intervals < 5000)]
        if len(valid_intervals) > 0:
            avg_interval = np.mean(valid_intervals)
            freq = 1000 / avg_interval if avg_interval > 0 else 0

    if freq is not None:
        result['details']['frequency'] = float(freq)
        max_freq = getattr(config, 'INTERPOLATION_MAX_GPS_FREQ', 25.0)
        if freq > max_freq:
            criteria_met += 1
            result['reasons'].append(f"frequency {freq:.0f} Hz")

    # If >=2 criteria matched, data is interpolated
    if criteria_met >= 2:
        result['is_interpolated'] = True

    return result
