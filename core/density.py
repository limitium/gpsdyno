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
Maximum power estimation module using multiple methods.

Implements 5 methods for comparison:
1. Robust Mean - trimmed mean of top 5%
2. Percentile 99 - 99th percentile of power values
3. Mode KDE - mode via kernel density estimation
4. Peak Detection - median of wide power peaks
5. Consistency - stability across session segments
"""
import numpy as np
from scipy.stats import gaussian_kde, trim_mean
from scipy.signal import find_peaks
try:
    from .. import config
except ImportError:
    import config


# Method aliases
METHOD_ALIASES = {
    'robust': 'robust_mean',
    'peak': 'peak_detection',
    'kde': 'mode_kde',
    'p99': 'percentile_99',
    'perc99': 'percentile_99',
}

# All available methods
ALL_METHODS = ['robust_mean', 'percentile_99', 'mode_kde', 'peak_detection', 'consistency']


def _normalize_methods(methods):
    """
    Normalizes method list: applies aliases and expands 'all'.

    Args:
        methods: list of methods or None

    Returns:
        list: normalized list of methods
    """
    if methods is None:
        return ['peak_detection']

    if isinstance(methods, str):
        methods = [m.strip() for m in methods.split(',')]

    normalized = []
    for m in methods:
        m = m.strip().lower()
        if m == 'all':
            return ALL_METHODS.copy()
        # Apply alias if exists
        m = METHOD_ALIASES.get(m, m)
        if m in ALL_METHODS and m not in normalized:
            normalized.append(m)

    return normalized if normalized else ['peak_detection']


def calculate_power_estimation(powers, time_series=None, methods=None):
    """
    Calculates maximum power using selected methods.

    Args:
        powers: array of power values (HP)
        time_series: optional, power time series for Peak Detection
        methods: list of methods to use (default ['peak_detection'])
                 Allowed: robust_mean, percentile_99, mode_kde, peak_detection, consistency, all

    Returns:
        dict: {
            'methods': {
                '<method_name>': {'value': float, ...},
                ...
            },
            'recommended': str,  # First method from list
            'recommended_value': float,  # Recommended method value
            'consistency_score': float,  # Based on CV of all methods (0.0-1.0)
            'consistency_computable': bool,  # True if CV was computed (>=3 methods)
            'methods_cv': float  # CV (std/mean) of method values, or None
        }
        or None if insufficient data
    """
    min_points = getattr(config, 'ESTIMATION_MIN_POINTS', 20)

    if len(powers) < min_points:
        return None

    powers = np.array(powers)
    selected_methods = _normalize_methods(methods)
    results = {'methods': {}}

    # ============================================================
    # STEP 1: ALWAYS compute ALL methods (for consistency score)
    # ============================================================
    all_method_values = {}  # Values only, for CV calculation
    all_method_data = {}    # Full data, for return if requested

    # Common top-5% calculation (used in robust_mean and mode_kde)
    top_95_threshold = np.percentile(powers, 95)
    top_5_percent = powers[powers >= top_95_threshold]

    # ----- ROBUST MEAN -----
    try:
        robust_min_points = getattr(config, 'ROBUST_MEAN_MIN_POINTS', 5)
        trim_proportion = getattr(config, 'ROBUST_MEAN_TRIM_PROPORTION', 0.1)

        if len(top_5_percent) >= robust_min_points:
            robust_value = float(trim_mean(top_5_percent, proportiontocut=trim_proportion))
            all_method_values['robust_mean'] = robust_value
            all_method_data['robust_mean'] = {
                'value': robust_value,
                'std': float(np.std(top_5_percent)),
                'count': len(top_5_percent)
            }
    except (ValueError, IndexError, RuntimeError):
        pass

    # ----- PERCENTILE 99 -----
    try:
        p99_value = float(np.percentile(powers, 99))
        all_method_values['percentile_99'] = p99_value
        all_method_data['percentile_99'] = {'value': p99_value}
    except (ValueError, IndexError, RuntimeError):
        pass

    # ----- MODE KDE -----
    try:
        kde_min_points = getattr(config, 'MODE_KDE_MIN_POINTS', 10)

        if len(top_5_percent) >= kde_min_points:
            kde = gaussian_kde(top_5_percent, bw_method='scott')
            x_range = np.linspace(top_5_percent.min(), top_5_percent.max(), 500)
            kde_values = kde(x_range)
            mode_idx = np.argmax(kde_values)
            mode_value = float(x_range[mode_idx])
            all_method_values['mode_kde'] = mode_value
            all_method_data['mode_kde'] = {
                'value': mode_value,
                'kde_max': float(kde_values[mode_idx])
            }
    except (ValueError, IndexError, RuntimeError):
        pass

    # ----- PEAK DETECTION -----
    try:
        signal = time_series if time_series is not None else powers
        prominence = getattr(config, 'PEAK_PROMINENCE', 5)
        min_width = getattr(config, 'PEAK_MIN_WIDTH', 3)
        peak_filter_percentile = getattr(config, 'PEAK_FILTER_PERCENTILE', 90)

        peaks, properties = find_peaks(signal, prominence=prominence, width=min_width)

        if len(peaks) > 0:
            wide_peaks_mask = properties['widths'] >= min_width
            valid_peak_indices = peaks[wide_peaks_mask]

            if len(valid_peak_indices) > 0:
                valid_peak_values = signal[valid_peak_indices]
                high_peaks = valid_peak_values[valid_peak_values >= np.percentile(powers, peak_filter_percentile)]

                if len(high_peaks) > 0:
                    peak_median = float(np.median(high_peaks))
                    all_method_values['peak_detection'] = peak_median
                    all_method_data['peak_detection'] = {
                        'value': peak_median,
                        'peak_count': len(high_peaks),
                        'all_peaks': len(valid_peak_indices)
                    }
    except (ValueError, IndexError, RuntimeError):
        pass

    # ----- CONSISTENCY (segment-based, legacy) -----
    try:
        n_segments = getattr(config, 'CONSISTENCY_SEGMENTS', 5)
        min_segment_size = getattr(config, 'CONSISTENCY_MIN_SEGMENT', 20)
        min_segments_required = getattr(config, 'CONSISTENCY_MIN_SEGMENTS', 3)

        segments = np.array_split(powers, n_segments)
        segment_99_values = [np.percentile(seg, 99) for seg in segments if len(seg) >= min_segment_size]

        if len(segment_99_values) >= min_segments_required:
            consistency_median = float(np.median(segment_99_values))
            consistency_std = float(np.std(segment_99_values))
            all_method_values['consistency'] = consistency_median
            all_method_data['consistency'] = {
                'value': consistency_median,
                'std': consistency_std,
                'segments_used': len(segment_99_values)
            }
    except (ValueError, IndexError, RuntimeError):
        pass

    # ============================================================
    # STEP 2: Compute NEW consistency score based on methods CV
    # ============================================================
    methods_cv = None
    consistency_score = 1.0  # Default: if cannot compute, assume stable
    consistency_computable = False

    if len(all_method_values) >= 3:
        values = list(all_method_values.values())
        mean_val = np.mean(values)
        if mean_val > 0:
            consistency_computable = True
            methods_cv = float(np.std(values) / mean_val)

            cv_min = getattr(config, 'CONSISTENCY_CV_MIN', 0.05)
            cv_max = getattr(config, 'CONSISTENCY_CV_MAX', 0.15)

            # Score: CV < cv_min → 1.0, CV > cv_max → 0.0
            if methods_cv <= cv_min:
                consistency_score = 1.0
            elif methods_cv >= cv_max:
                consistency_score = 0.0
            else:
                consistency_score = float(1.0 - (methods_cv - cv_min) / (cv_max - cv_min))

    results['consistency_score'] = consistency_score
    results['consistency_computable'] = consistency_computable
    results['methods_cv'] = methods_cv

    # ============================================================
    # STEP 3: Return only REQUESTED methods
    # ============================================================
    for method in selected_methods:
        if method in all_method_data:
            results['methods'][method] = all_method_data[method]
            # For consistency method add legacy score (segment-based)
            # Legacy threshold 0.1 = 10% relative_std for score=0
            if method == 'consistency' and 'std' in all_method_data[method]:
                seg_std = all_method_data[method]['std']
                seg_val = all_method_data[method]['value']
                if seg_val > 0:
                    relative_std = seg_std / seg_val
                    legacy_threshold = 0.1  # 10% relative std → score = 0
                    legacy_score = float(min(1.0, max(0, 1 - relative_std / legacy_threshold)))
                    results['methods'][method]['score'] = legacy_score

    # ===== RECOMMENDED METHOD SELECTION =====
    # Recommended = first requested method that was successfully computed
    results['recommended'] = None
    results['recommended_value'] = None

    for method in selected_methods:
        if method in results['methods']:
            results['recommended'] = method
            results['recommended_value'] = results['methods'][method].get('value')
            break

    return results


def plot_power_methods(ax, estimation_data, x_range, uncertainty=None):
    """
    Draws horizontal lines for each power estimation method.

    Args:
        ax: matplotlib axes
        estimation_data: result of calculate_power_estimation()
        x_range: tuple (x_min, x_max) for line positioning
        uncertainty: dict with uncertainty data (optional)
    """
    if not estimation_data or not estimation_data.get('methods'):
        return []

    methods = estimation_data['methods']
    x_min, x_max = x_range
    legend_items = []

    # Colors and styles for each method
    method_styles = {
        'robust_mean': {'color': '#e74c3c', 'linestyle': '-', 'linewidth': 2.5, 'label': 'Robust Mean'},
        'percentile_99': {'color': '#3498db', 'linestyle': '--', 'linewidth': 2, 'label': 'P99'},
        'mode_kde': {'color': '#2ecc71', 'linestyle': '-.', 'linewidth': 2, 'label': 'Mode (KDE)'},
        'peak_detection': {'color': '#9b59b6', 'linestyle': ':', 'linewidth': 2.5, 'label': 'Peak Detection'},
        'consistency': {'color': '#f39c12', 'linestyle': '-', 'linewidth': 2, 'label': 'Consistency'}
    }

    recommended = estimation_data.get('recommended')

    for method_name, data in methods.items():
        if data.get('value') is None:
            continue

        style = method_styles.get(method_name, {'color': 'gray', 'linestyle': '-', 'linewidth': 1})
        value = data['value']

        # If recommended method - make line thicker
        lw = style['linewidth'] * 1.5 if method_name == recommended else style['linewidth']

        # Draw horizontal line
        line = ax.axhline(
            y=value,
            color=style['color'],
            linestyle=style['linestyle'],
            linewidth=lw,
            alpha=0.8,
            zorder=15
        )

        # Format legend text
        label_text = f"{style['label']}: {value:.1f} WHP"

        if method_name == 'consistency' and 'score' in data:
            score_pct = data['score'] * 100
            label_text = f"{style['label']}: {value:.1f} WHP ({score_pct:.0f}%)"

        elif method_name == 'robust_mean' and 'std' in data:
            label_text = f"{style['label']}: {value:.1f} ± {data['std']:.1f} WHP"

        elif method_name == 'peak_detection' and 'peak_count' in data:
            label_text = f"{style['label']}: {value:.1f} WHP ({data['peak_count']} peaks)"

        # Add "recommended" marker and uncertainty
        if method_name == recommended:
            if uncertainty and uncertainty.get('total_hp'):
                total_unc = uncertainty['total_hp']
                label_text = f"★ {style['label']}: {value:.0f} ±{total_unc:.0f} WHP"
            else:
                label_text = f"★ {label_text}"

        legend_items.append((line, label_text))

    return legend_items
