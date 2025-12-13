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
Configuration file for GPSDyno.
Contains all constants and settings for power calculation.

Note: External API settings (.env, Weather API, etc.) should be in scripts/
This module contains only calculation parameters and constants.
"""
import os

# Physical Constants
GRAVITY = 9.81  # Acceleration due to gravity, m/s²
AIR_DENSITY = 1.204  # Air density at sea level, kg/m³

# Unit Conversion
WATTS_TO_HP = 735.5  # Watts to horsepower (metric) conversion factor

# Psychrometric Constants (for air density calculation)
R_DRY_AIR = 287.058    # Specific gas constant for dry air, J/(kg·K)
R_VAPOR = 461.495      # Specific gas constant for water vapor, J/(kg·K)

# ============================================================
# Kalman Filter Parameters
# ============================================================

# Speed/Acceleration Kalman Filter (Constant Velocity model)
KALMAN_SPEED_Q = 0.5   # Process noise - higher = more responsive to changes
KALMAN_SPEED_R = 0.2   # Measurement noise - higher = more smoothing

# Altitude Kalman Filter (more aggressive smoothing for noisy GPS altitude)
KALMAN_ALTITUDE_Q = 0.5   # Process noise (increased for faster response to elevation changes)
KALMAN_ALTITUDE_R = 1.0   # Measurement noise (higher = trust model more than GPS)

# Adaptive Kalman Parameters
KALMAN_HDOP_FACTOR = 0.5           # Coefficient for HDOP influence on R: R = base_R * (1 + factor * hdop)
KALMAN_LOW_POWER_FILTER_PERCENTILE = 95  # Keep top N% points by power (95 = keep top-95%, discard bottom 5%)
KALMAN_MIN_DT = 1e-6               # Minimum time step to prevent division by zero
KALMAN_R_MIN_MULTIPLIER = 0.5      # Minimum R multiplier for adaptive Kalman
KALMAN_R_MAX_MULTIPLIER = 3.0      # Maximum R multiplier for adaptive Kalman

# Kalman Initialization
KALMAN_INIT_POINTS = 5             # Number of initial points for averaging (robustness against outliers)
KALMAN_INITIAL_COVARIANCE = 1.0    # Initial covariance matrix diagonal value
KALMAN_INITIAL_ACCELERATION = 0.0  # Initial acceleration estimate

# ============================================================
# Pre-Kalman Filtering (removes noisy points BEFORE Kalman filter)
# ============================================================
PRE_KALMAN_MIN_SATELLITES = 6        # Minimum sats to include in Kalman
PRE_KALMAN_MAX_HDOP_MULTIPLIER = 1.5 # hdop_threshold * multiplier for pre-filter
PRE_KALMAN_MIN_VALID_POINTS = 10     # Minimum valid points required
PRE_KALMAN_WARNING_THRESHOLD = 0.2   # Warn if >20% filtered pre-Kalman

# ============================================================
# Savitzky-Golay Filter Parameters
# ============================================================
SAVGOL_WINDOW_LENGTH = 11  # Window length (must be odd)
SAVGOL_WINDOW_DIVISOR = 5  # Dynamic window: min(WINDOW_LENGTH, len/DIVISOR)
SAVGOL_POLYORDER = 1       # Polynomial order for Savitzky-Golay filter

# ============================================================
# Slope Calculation Parameters
# ============================================================
MIN_DISTANCE_FOR_SLOPE = 0.5  # Minimum distance (m) for slope calculation
MAX_SAFE_SLOPE = 0.25         # Maximum safe slope (~14 degrees)

# ============================================================
# Data Processing Parameters
# ============================================================
GAP_THRESHOLD = 0.5      # Threshold for detecting gaps in data (seconds)
WINDOW_HALF_WIDTH = 0.5  # Window half-width for percentile smoothing (%)

# ============================================================
# GPS Quality Filtering (post-Kalman, for percentile calculation)
# ============================================================
HDOP_PERCENTILE = 75           # Use 75th percentile of HDOP as threshold
MAX_HDOP = 2.0                 # Maximum acceptable HDOP value
MIN_SATELLITES = 10            # Minimum satellites for valid point
SPEED_PERCENTILE_LOW = 95      # Keep top N% points by speed (95 = keep top-95%, discard bottom 5%)
SPEED_PERCENTILE_HIGH = 95     # Keep bottom N% points by speed (95 = keep bottom-95%, discard top 5%)

# HDOP defaults and bounds
HDOP_FALLBACK_VALUE = 0.7      # Default HDOP when missing
HDOP_MIN_BOUND = 0.5           # Minimum HDOP bound for clipping
HDOP_MAX_BOUND = 2.0           # Maximum HDOP bound for clipping

# ============================================================
# Power Estimation Methods (5 methods for comparison)
# ============================================================
ESTIMATION_MIN_POINTS = 20          # Minimum points for calculation

# Robust Mean parameters
ROBUST_MEAN_TRIM_PROPORTION = 0.1   # Proportion of data to trim from each side
ROBUST_MEAN_MIN_POINTS = 5          # Minimum points for robust mean calculation


# Mode KDE parameters
MODE_KDE_MIN_POINTS = 10            # Minimum points for KDE

# Peak Detection parameters
PEAK_PROMINENCE = 5                 # Minimum peak prominence (HP)
PEAK_MIN_WIDTH = 3                  # Minimum peak width (points)
PEAK_FILTER_PERCENTILE = 90         # Percentile for filtering high peaks

# Consistency score parameters (based on method agreement)
# CV (coefficient of variation) = std/mean of all method values
CONSISTENCY_CV_MIN = 0.02           # CV below this -> score = 1.0 (methods agree)
CONSISTENCY_CV_MAX = 0.15           # CV above this -> score = 0.0 (methods disagree)

# Legacy: segment-based consistency method parameters
CONSISTENCY_SEGMENTS = 5            # Number of segments to split into
CONSISTENCY_MIN_SEGMENT = 20        # Minimum points per segment
CONSISTENCY_MIN_SEGMENTS = 3        # Minimum valid segments for calculation

# ============================================================
# Warning Thresholds
# ============================================================
HIGH_WIND_SPEED_KPH = 30  # High wind speed caution (km/h)

# Weather defaults and detection
DEFAULT_WEATHER = {
    'temperature_c': 20.0,
    'pressure_hpa': 1013.25,
    'humidity_percent': 50.0,
    'wind_speed_kph': 0.0,
}

# Keywords indicating adverse weather in weather_data["conditions"]
BAD_WEATHER_KEYWORDS = [
    'rain', 'snow', 'дождь', 'снег', 'ливень', 'осадки', 'гроза',
    'гололед', 'thunderstorm', 'sleet', 'ice', 'showers'
]

# Minimum speed data points for calculation
MIN_SPEED_POINTS = 100  # Minimum points required for power calculation

# GPS Quality Thresholds
LOW_GPS_FREQUENCY_HZ = 5.0  # Low GPS frequency warning (Hz)
MEDIUM_GPS_FREQUENCY_HZ = 10.0  # Medium GPS frequency caution (Hz)

# ============================================================
# Visualization Parameters
# ============================================================
Y_MARGIN_FACTOR = 0.1  # Y-axis margin factor for plots
MAIN_LEGEND_LOCATION = 'lower right'
MARKER_SIZE = 8
ALPHA_VALID = 0.5
ALPHA_INVALID = 0.6
LEGEND_FONTSIZE = 11
ANNOTATION_FONTSIZE = 9

# Resource Paths
LOGO_PATH = os.path.join(os.path.dirname(__file__), 'logo.png')

# ============================================================
# Track Visualization Parameters (Pseudo-3D)
# ============================================================
TRACK_COLORMAP = 'viridis'           # Colormap for power visualization
TRACK_OFFSET_SCALE = 0.0006          # Altitude offset scale (in degrees)
TRACK_PILLAR_STEP = 20               # Vertical pillar every N points
TRACK_LINE_WIDTH = 4                 # Track line width
TRACK_SHADOW_ALPHA = 0.4             # Shadow transparency
TRACK_SHADOW_WIDTH = 5               # Shadow line width
TRACK_PILLAR_ALPHA = 0.4             # Pillar transparency

# Track peak labels
TRACK_PEAK_PROMINENCE = 30           # Min peak prominence (HP)
TRACK_PEAK_DISTANCE = 150            # Min distance between peaks (points)
TRACK_PEAK_MIN_POWER_RATIO = 0.7     # Min power as ratio of max (0.7 = 70% of max)
TRACK_PEAK_FONTSIZE = 10             # Peak label font size
TRACK_PEAK_COLOR = '#67001f'         # Peak label color (burgundy)

# ============================================================
# Uncertainty Estimation Parameters (worst-case bounds)
# ============================================================

# Wind angle factor (0.4 — assumes significant angle between wind and driving direction)
UNCERTAINTY_WIND_ANGLE_FACTOR = 0.4

# Mass uncertainty
UNCERTAINTY_MASS_KG = 20             # ±20 kg mass uncertainty

# GPS uncertainty
UNCERTAINTY_GPS_BASE_PERCENT = 0.02  # Base GPS uncertainty (2%)
UNCERTAINTY_GPS_HDOP_FACTOR = 0.5    # HDOP influence factor
UNCERTAINTY_GPS_REFERENCE_FREQ = 10.0   # Reference frequency (Hz), no freq penalty at or above this
UNCERTAINTY_GPS_FREQ_COEFFICIENT = 3.0  # Coefficient for freq formula: k × (10-freq)^1.5

# Consistency score influence
UNCERTAINTY_CONSISTENCY_MAX_MULTIPLIER = 3.0  # Max multiplier when score=0

# ============================================================
# Interpolation Detection Parameters
# ============================================================
INTERPOLATION_MAX_VALID_SATS = 100       # num_sats above this = interpolation
INTERPOLATION_MIN_HDOP_STD = 0.01        # std(HDOP) below this = interpolation (constant)
INTERPOLATION_MIN_INTERVAL_STD_MS = 1.0  # std(intervals) below this = interpolation
INTERPOLATION_MAX_GPS_FREQ = 25.0        # frequency above this = interpolation
INTERPOLATION_MIN_POINTS = 100           # minimum points for interpolation analysis
