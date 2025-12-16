#!/usr/bin/env python3
# GPSDyno - GPS-based vehicle power calculator
# Copyright (C) 2024 GPSDyno Contributors
#
# Shared data-structure definitions and index constants used across the
# calculation pipeline. Centralising these avoids "magic" tuple indexes
# sprinkled through the code and makes the data flow easier to understand.

"""
Core tuple layouts used in the GPSDyno pipeline.

The parser currently represents NMEA-derived data as tuples for performance
and compactness. These constants document that layout so other modules can
refer to well‑named indices instead of bare numbers.

Speed sample tuple (``speed_data``)
-----------------------------------
Produced by ``parsers.nmea_handler.extract_speed_altitude_data`` and used
throughout the pipeline::

    (
        abs_ms,        # 0 – absolute timestamp in milliseconds from epoch
        rel_ms,        # 1 – relative timestamp in milliseconds from session start
        time_str,      # 2 – human‑readable time "HH:MM:SS.mmm"
        speed_kmh,     # 3 – speed over ground in km/h
        nmea_line,     # 4 – original $GPRMC NMEA sentence
        num_sats,      # 5 – satellites in view (int or None)
        hdop,          # 6 – horizontal dilution of precision (float or None)
    )

Altitude sample tuple (``altitude_data``)
-----------------------------------------
Also produced by ``extract_speed_altitude_data``::

    (
        abs_ms,        # 0 – absolute timestamp in milliseconds from epoch
        rel_ms,        # 1 – relative timestamp in milliseconds from session start
        time_str,      # 2 – human‑readable time "HH:MM:SS.mmm"
        altitude_m,    # 3 – altitude above sea level, metres
        nmea_line,     # 4 – original $GPGGA NMEA sentence
    )

Power vs speed tuple (``power_speed_data``)
-------------------------------------------
Produced by ``core.calculator.calculate_power`` and passed into statistics
and visualisation helpers::

    (
        speed_kmh,     # 0 – speed in km/h
        power_hp,      # 1 – instantaneous wheel power in HP
        is_valid,      # 2 – bool flag: True if point passed all filters
    )
"""

# Indices for elements of speed_data tuples
SPEED_ABS_MS = 0
SPEED_REL_MS = 1
SPEED_TIME_STR = 2
SPEED_SPEED_KMH = 3
SPEED_NMEA_LINE = 4
SPEED_NUM_SATS = 5
SPEED_HDOP = 6

# Indices for elements of altitude_data tuples
ALT_ABS_MS = 0
ALT_REL_MS = 1
ALT_TIME_STR = 2
ALT_ALTITUDE_M = 3
ALT_NMEA_LINE = 4

# Indices for elements of power_speed_data tuples
POWER_SPEED_SPEED_KMH = 0
POWER_SPEED_POWER_HP = 1
POWER_SPEED_IS_VALID = 2

# Unit conversion constants
KMH_TO_MS = 3.6  # km/h to m/s conversion factor
MS_TO_KMH = 1.0 / 3.6  # m/s to km/h conversion factor
MS_TO_S = 1000.0  # milliseconds to seconds conversion factor
PERCENT_TO_DECIMAL = 100.0  # percentage to decimal (e.g., 95 -> 0.95)

# Uncertainty calculation constants
HIGH_POWER_THRESHOLD_RATIO = 0.9  # Use points with power >= 90% of max for uncertainty
DEFAULT_ACCELERATION_MS2 = 1.0  # Default acceleration when no high-power data available
DEFAULT_GPS_FREQUENCY_HZ = 10.0  # Default GPS frequency when cannot be calculated
DEFAULT_HDOP_MEAN = 1.0  # Default HDOP mean when statistics unavailable
DEFAULT_CONSISTENCY_SCORE = 0.85  # Default consistency score when not computed
FALLBACK_MAX_POWER_SPEED_KMH = 150.0  # Fallback speed when no valid power data

# Physics constants
AERODYNAMIC_FORCE_COEFFICIENT = 0.5  # Coefficient for aerodynamic force: 0.5 * rho * Cd * A * v²


