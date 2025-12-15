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
Vehicle power calculation helper module.

Contains auxiliary functions for power calculation.
Main calculation logic is in calculator.py.
"""
import numpy as np
import os
import sys

# Add parent directory to path for config import
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from config import WINDOW_HALF_WIDTH
from .structures import (
    POWER_SPEED_SPEED_KMH,
    POWER_SPEED_POWER_HP,
    POWER_SPEED_IS_VALID,
)


def calculate_average_speed_for_percentile(power_speed_data, percentile):
    """
    Calculates average speed for points in a window around given percentile.

    Args:
        power_speed_data: List of tuples (speed_kmh, power_hp, is_valid)
        percentile: Target percentile (0-100)

    Returns:
        Average speed in km/h or None if no data
    """
    if not power_speed_data:
        return None

    filtered = [
        point
        for point in power_speed_data
        if len(point) <= POWER_SPEED_IS_VALID or point[POWER_SPEED_IS_VALID]
    ]
    if not filtered:
        return None

    # Sort by power value
    sorted_data = sorted(filtered, key=lambda x: x[POWER_SPEED_POWER_HP])
    n = len(sorted_data)
    if n == 0:
        return None

    start_percent = max(0, percentile - WINDOW_HALF_WIDTH)
    end_percent = min(100, percentile + WINDOW_HALF_WIDTH)

    start_index = int(np.floor(start_percent / 100 * n))
    end_index = int(np.ceil(end_percent / 100 * n))

    start_index = max(0, start_index)
    end_index = min(n, end_index)

    if start_index >= end_index:
        target_index = int(round(percentile / 100 * n)) - 1
        target_index = max(0, min(n - 1, target_index))
        return float(sorted_data[target_index][POWER_SPEED_SPEED_KMH])

    speeds_in_window = [
        point[POWER_SPEED_SPEED_KMH] for point in sorted_data[start_index:end_index]
    ]

    if not speeds_in_window:
        target_index = int(round(percentile / 100 * n)) - 1
        target_index = max(0, min(n - 1, target_index))
        return float(sorted_data[target_index][POWER_SPEED_SPEED_KMH])

    return float(np.mean(speeds_in_window))
