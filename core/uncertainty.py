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
Power uncertainty calculation module.

Accounts for uncertainty sources:
- GPS quality (HDOP, frequency)
- Vehicle mass (default ±20 kg)
- Consistency score as uncertainty multiplier

Note: Wind is not included in uncertainty since it's now accounted for
directly in the power calculation using wind direction and heading.

Outputs worst-case bounds.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from .. import config
except ImportError:
    import config

from .structures import KMH_TO_MS

# Use WATTS_TO_HP from config (avoid duplication)
WATTS_TO_HP = getattr(config, 'WATTS_TO_HP', 735.5)


# Watts to WHP conversion constant
WATTS_TO_HP = 735.5


@dataclass
class UncertaintyComponents:
    """Power uncertainty components (in WHP)."""
    mass_hp: float           # ±ΔP from mass
    gps_hp: float            # ±ΔP from GPS quality
    consistency_multiplier: float  # Multiplier from consistency score
    total_hp: float          # Total uncertainty (95% CI)


def calculate_wind_uncertainty(
    v_car_ms: float,
    wind_kph: float,
    rho: float,
    cd: float,
    frontal_area: float
) -> float:
    """
    Wind uncertainty calculation accounting for average angle.

    Applies angle coefficient (default 0.7 ≈ cos(45°)),
    assuming wind rarely blows strictly along driving direction.
    ΔP_wind = [P_aero(V+Vwind_eff) - P_aero(V-Vwind_eff)] / 2

    Args:
        v_car_ms: Car speed at peak power (m/s)
        wind_kph: Wind speed (km/h)
        rho: Air density (kg/m³)
        cd: Drag coefficient
        frontal_area: Frontal area (m²)

    Returns:
        Power uncertainty from wind (HP)
    """
    if wind_kph <= 0:
        return 0.0

    # Apply angle coefficient (0.7 ≈ cos(45°) — wind rarely strictly along)
    angle_factor = getattr(config, 'UNCERTAINTY_WIND_ANGLE_FACTOR', 0.7)
    v_wind_ms = wind_kph / KMH_TO_MS * angle_factor

    # Protection against negative effective speed with strong tailwind
    v_tail = max(v_car_ms - v_wind_ms, 0.1)

    # Power with headwind (worst case — more resistance)
    # Use aerodynamic force coefficient from structures
    from .structures import AERODYNAMIC_FORCE_COEFFICIENT
    p_headwind = AERODYNAMIC_FORCE_COEFFICIENT * rho * cd * frontal_area * (v_car_ms + v_wind_ms) ** 3

    # Power with tailwind (best case — less resistance)
    p_tailwind = AERODYNAMIC_FORCE_COEFFICIENT * rho * cd * frontal_area * v_tail ** 3

    # Uncertainty = half the range (in WHP)
    delta_p = (p_headwind - p_tailwind) / 2 / WATTS_TO_HP

    return abs(delta_p)


def calculate_mass_uncertainty(
    v_ms: float,
    a_ms2: float,
    crr: float,
    slope_avg: float,
    delta_mass_kg: Optional[float] = None
) -> float:
    """
    Mass uncertainty calculation.

    Mass affects three power components:
    - P_rolling = m × g × Crr × v
    - P_accel = m × a × v
    - P_slope = m × g × sin(θ) × v

    Args:
        v_ms: Speed at peak power (m/s)
        a_ms2: Average acceleration (m/s²)
        crr: Rolling resistance coefficient
        slope_avg: Average road slope (sin θ)
        delta_mass_kg: Mass uncertainty (kg), default from config

    Returns:
        Power uncertainty from mass (HP)
    """
    if delta_mass_kg is None:
        delta_mass_kg = getattr(config, 'UNCERTAINTY_MASS_KG', 20)

    g = getattr(config, 'GRAVITY', 9.81)

    # Partial derivatives ∂P/∂m × Δm for each component (in Watts)
    dp_rolling = g * crr * v_ms * delta_mass_kg
    dp_accel = abs(a_ms2) * v_ms * delta_mass_kg
    dp_slope = g * abs(slope_avg) * v_ms * delta_mass_kg

    # Quadratic sum (assume independent uncertainties)
    delta_p_watts = np.sqrt(dp_rolling**2 + dp_accel**2 + dp_slope**2)

    return delta_p_watts / WATTS_TO_HP


def calculate_gps_uncertainty(
    power_hp: float,
    hdop_mean: float,
    gps_frequency_hz: float
) -> float:
    """
    GPS quality uncertainty calculation.

    Uncertainty consists of two components:
    - From HDOP (horizontal dilution of precision) — percentage of power
    - From GPS frequency — fixed value by formula 3×(10-freq)^1.5

    Args:
        power_hp: Estimated power (HP)
        hdop_mean: Mean HDOP for session
        gps_frequency_hz: GPS frequency (Hz)

    Returns:
        Power uncertainty from GPS (HP)
    """
    base_percent = getattr(config, 'UNCERTAINTY_GPS_BASE_PERCENT', 0.02)
    hdop_factor = getattr(config, 'UNCERTAINTY_GPS_HDOP_FACTOR', 0.5)
    reference_freq = getattr(config, 'UNCERTAINTY_GPS_REFERENCE_FREQ', 10.0)
    freq_coefficient = getattr(config, 'UNCERTAINTY_GPS_FREQ_COEFFICIENT', 3.0)

    # HDOP uncertainty (percentage of power, baseline at HDOP=1.0)
    hdop_error = base_percent * (1 + hdop_factor * max(hdop_mean - 1.0, 0))
    hdop_error = max(hdop_error, base_percent)
    gps_hdop_hp = power_hp * hdop_error

    # Low GPS frequency uncertainty (fixed, independent of power)
    # Formula: 3 × (10 - freq)^1.5 gives ~80 WHP at 1 Hz, ~25 WHP at 5 Hz, 0 at ≥10 Hz
    if gps_frequency_hz >= reference_freq:
        gps_freq_hp = 0.0
    else:
        gps_freq_hp = freq_coefficient * (reference_freq - gps_frequency_hz) ** 1.5

    return gps_hdop_hp + gps_freq_hp


def get_consistency_multiplier(consistency_score: float) -> float:
    """
    Computes uncertainty multiplier based on consistency score.

    Consistency score shows result stability across segments:
    - score = 1.0 → data very stable → multiplier = 1.0
    - score = 0.5 → medium stability → multiplier = 1.5
    - score = 0.0 → data unstable → multiplier = 2.0

    Args:
        consistency_score: Stability score (0.0-1.0)

    Returns:
        Uncertainty multiplier (1.0-2.0)
    """
    max_multiplier = getattr(config, 'UNCERTAINTY_CONSISTENCY_MAX_MULTIPLIER', 2.0)

    # Clamp score within valid range
    score = max(0.0, min(1.0, consistency_score))

    # Linear interpolation: score=1 → 1.0, score=0 → max_multiplier
    multiplier = 1.0 + (max_multiplier - 1.0) * (1.0 - score)

    return multiplier


def calculate_total_uncertainty(
    v_car_ms: float,
    a_ms2: float,
    crr: float,
    slope_avg: float,
    power_hp: float,
    hdop_mean: float,
    gps_frequency_hz: float,
    consistency_score: float,
    delta_mass_kg: Optional[float] = None
) -> UncertaintyComponents:
    """
    Main function: computes all uncertainty components.

    Final formula (worst-case):
    total = √(mass² + gps²) × consistency_multiplier

    All components are worst-case (maximum deviations), not σ.
    Consistency multiplier increases uncertainty for unstable data.
    
    Note: Wind is not included in uncertainty since it's now accounted for
    directly in the power calculation using wind direction.

    Args:
        v_car_ms: Speed at peak power (m/s)
        a_ms2: Average acceleration (m/s²)
        crr: Rolling resistance coefficient
        slope_avg: Average road slope
        power_hp: Estimated power (HP)
        hdop_mean: Mean HDOP
        gps_frequency_hz: GPS frequency (Hz)
        consistency_score: Stability score (0.0-1.0)
        delta_mass_kg: Mass uncertainty (kg), optional

    Returns:
        UncertaintyComponents with components and total uncertainty
    """
    # Compute each component (all worst-case)
    mass_hp = calculate_mass_uncertainty(
        v_ms=v_car_ms,
        a_ms2=a_ms2,
        crr=crr,
        slope_avg=slope_avg,
        delta_mass_kg=delta_mass_kg
    )

    gps_hp = calculate_gps_uncertainty(
        power_hp=power_hp,
        hdop_mean=hdop_mean,
        gps_frequency_hz=gps_frequency_hz
    )

    consistency_multiplier = get_consistency_multiplier(consistency_score)

    # Quadratic sum of components (worst-case)
    base = np.sqrt(mass_hp**2 + gps_hp**2)

    # Total uncertainty with consistency (no ×2, since worst-case)
    total = base * consistency_multiplier

    return UncertaintyComponents(
        mass_hp=round(mass_hp, 1),
        gps_hp=round(gps_hp, 1),
        consistency_multiplier=round(consistency_multiplier, 2),
        total_hp=round(total, 1)
    )
