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

"""Input format parsers: NMEA."""

from .nmea_handler import (
    convert_nmea_to_milliseconds,
    parse_nmea_file,
    analyze_nmea_timestamps,
    analyze_nmea_quality,
    extract_speed_altitude_data,
    extract_speed_data,
    calculate_gps_frequency,
    analyze_nmea_file,
)

__all__ = [
    # NMEA functions
    'convert_nmea_to_milliseconds',
    'parse_nmea_file',
    'analyze_nmea_timestamps',
    'analyze_nmea_quality',
    'extract_speed_altitude_data',
    'extract_speed_data',
    'calculate_gps_frequency',
    'analyze_nmea_file',
]
