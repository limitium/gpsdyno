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
Tests for GPSDyno power calculation using NMEA test files.

Each test file uses the weight extracted from the filename (last number)
and default car parameters (drag_coefficient: 0.30, frontal_area: 2.2, rolling_resistance: 0.015).

Tests use reference values captured from the current implementation to ensure
refactoring doesn't break functionality.
"""
import os
import re
import json
import pytest
import sys

# Add parent directory to path for imports
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from core.calculator import calculate_power
from parsers.nmea_handler import extract_speed_altitude_data, parse_nmea_file
import config


# Tolerance for floating point comparisons
FLOAT_TOLERANCE = 1e-6
POWER_TOLERANCE = 0.1  # 0.1 HP tolerance for power values


def extract_weight_from_filename(filename):
    """
    Extract weight (last number) from filename.
    
    Examples:
        "2105 - turbo - Grade4_1020.nmea" -> 1020
        "subaru_sti_mrw_1440.nmea" -> 1440
        "2108_delta_adm_970.nmea" -> 970
    """
    # Find all numbers in the filename
    numbers = re.findall(r'\d+', filename)
    if numbers:
        # Return the last number as weight
        return int(numbers[-1])
    return None


def get_default_car_info(weight):
    """
    Get default car info with specified weight.
    
    Uses default values from config:
    - drag_coefficient: 0.30
    - frontal_area: 2.2
    - rolling_resistance: 0.015
    """
    return {
        "name": f"Test Vehicle ({weight} kg)",
        "mass": weight,
        "drag_coefficient": 0.30,
        "frontal_area": 2.2,
        "rolling_resistance": 0.015,
    }


def get_test_files():
    """Get all NMEA test files from test/resources directory."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    resources_dir = os.path.join(test_dir, "resources")
    
    test_files = []
    for filename in sorted(os.listdir(resources_dir)):
        if filename.endswith('.nmea'):
            filepath = os.path.join(resources_dir, filename)
            weight = extract_weight_from_filename(filename)
            test_files.append((filepath, filename, weight))
    
    return test_files


def load_reference_values():
    """Load reference values from JSON file."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    ref_file = os.path.join(test_dir, "reference_values.json")
    
    if not os.path.exists(ref_file):
        pytest.skip(f"Reference values file not found: {ref_file}")
    
    with open(ref_file, 'r') as f:
        return json.load(f)


# Get all test files and reference values at module level
TEST_FILES = get_test_files()
REFERENCE_VALUES = load_reference_values()


@pytest.mark.parametrize("filepath,filename,weight", TEST_FILES)
def test_nmea_file_parsing(filepath, filename, weight):
    """Test that NMEA files can be parsed correctly."""
    assert os.path.exists(filepath), f"Test file {filename} does not exist"
    assert weight is not None, f"Could not extract weight from filename {filename}"
    
    # Test parse_nmea_file
    nmea_data = parse_nmea_file(filepath, return_dict=True)
    assert nmea_data is not None
    assert 'latitude' in nmea_data
    assert 'longitude' in nmea_data
    assert nmea_data['latitude'] is not None
    assert nmea_data['longitude'] is not None


@pytest.mark.parametrize("filepath,filename,weight", TEST_FILES)
def test_speed_altitude_extraction(filepath, filename, weight):
    """Test that speed and altitude data can be extracted from NMEA files."""
    assert os.path.exists(filepath), f"Test file {filename} does not exist"
    assert weight is not None, f"Could not extract weight from filename {filename}"
    
    # Test extract_speed_altitude_data
    speed_data, altitude_data, gps_frequency, nmea_lines, first_timestamp_ms, coords_data = extract_speed_altitude_data(filepath)
    
    assert speed_data is not None
    assert len(speed_data) > 0, f"No speed data extracted from {filename}"
    assert len(speed_data) >= config.MIN_SPEED_POINTS, f"Insufficient speed data points in {filename}: {len(speed_data)} < {config.MIN_SPEED_POINTS}"
    
    # Check against reference values
    if filename in REFERENCE_VALUES:
        ref = REFERENCE_VALUES[filename]
        assert len(speed_data) == ref['speed_data_points'], \
            f"Speed data points mismatch for {filename}: expected {ref['speed_data_points']}, got {len(speed_data)}"
        
        if altitude_data:
            assert len(altitude_data) == ref['altitude_data_points'], \
                f"Altitude data points mismatch for {filename}: expected {ref['altitude_data_points']}, got {len(altitude_data)}"
        
        if gps_frequency and ref.get('gps_frequency'):
            assert gps_frequency == pytest.approx(ref['gps_frequency'], abs=0.1), \
                f"GPS frequency mismatch for {filename}: expected {ref['gps_frequency']}, got {gps_frequency}"
    
    # Check speed data structure: (abs_ms, rel_ms, time_str, speed_kmh, nmea, sats, hdop)
    for point in speed_data[:10]:  # Check first 10 points
        assert len(point) >= 4, f"Invalid speed data point structure: {point}"
        assert point[3] >= 0, f"Invalid speed value: {point[3]}"


@pytest.mark.parametrize("filepath,filename,weight", TEST_FILES)
def test_power_calculation_peak_detection(filepath, filename, weight):
    """Test power calculation with peak_detection method against reference values."""
    assert os.path.exists(filepath), f"Test file {filename} does not exist"
    assert weight is not None, f"Could not extract weight from filename {filename}"
    
    if filename not in REFERENCE_VALUES:
        pytest.skip(f"No reference values for {filename}")
    
    ref = REFERENCE_VALUES[filename]
    
    # Extract data
    speed_data, altitude_data, gps_frequency, nmea_lines, first_timestamp_ms, coords_data = extract_speed_altitude_data(filepath)
    
    assert len(speed_data) >= config.MIN_SPEED_POINTS, f"Insufficient speed data: {len(speed_data)}"
    
    # Get car info with weight from filename
    car_info = get_default_car_info(weight)
    assert car_info['mass'] == ref['weight'], f"Weight mismatch for {filename}"
    
    # Use default weather
    weather_data = config.DEFAULT_WEATHER.copy()
    
    # Calculate power
    power_data = calculate_power(
        speed_data,
        car_info,
        weather_data,
        altitude_data,
        first_timestamp_ms,
        methods='peak_detection'
    )
    
    assert power_data is not None, f"Power calculation failed for {filename}"
    assert 'power_time' in power_data
    assert 'power_speed' in power_data
    assert 'power_estimation' in power_data
    
    # Check power estimation against reference
    if power_data['power_estimation']:
        estimation = power_data['power_estimation']
        ref_est = ref['peak_detection']
        
        assert 'recommended_value' in estimation
        recommended = estimation['recommended_value']
        assert recommended is not None
        assert recommended == pytest.approx(ref_est['recommended_value'], abs=POWER_TOLERANCE), \
            f"Recommended power mismatch for {filename}: expected {ref_est['recommended_value']}, got {recommended}"
        
        assert estimation.get('recommended') == ref_est['recommended_method'], \
            f"Recommended method mismatch for {filename}: expected {ref_est['recommended_method']}, got {estimation.get('recommended')}"
    
    # Check statistics against reference
    assert power_data['valid_points'] == ref_est['valid_points'], \
        f"Valid points mismatch for {filename}: expected {ref_est['valid_points']}, got {power_data['valid_points']}"
    
    assert power_data['total_points'] == ref_est['total_points'], \
        f"Total points mismatch for {filename}: expected {ref_est['total_points']}, got {power_data['total_points']}"
    
    assert power_data['filtered_ratio'] == pytest.approx(ref_est['filtered_ratio'], abs=FLOAT_TOLERANCE), \
        f"Filtered ratio mismatch for {filename}: expected {ref_est['filtered_ratio']}, got {power_data['filtered_ratio']}"
    
    # Check power_time sample
    if ref.get('power_time_sample'):
        power_time = power_data['power_time']
        ref_sample = ref['power_time_sample']
        
        if ref_sample['first']:
            first_point = power_time[0]
            assert first_point[0] == pytest.approx(ref_sample['first'][0], abs=0.01), \
                f"First time point mismatch for {filename}"
            assert first_point[1] == pytest.approx(ref_sample['first'][1], abs=POWER_TOLERANCE), \
                f"First power point mismatch for {filename}"
        
        max_power = max([p[1] for p in power_time])
        assert max_power == pytest.approx(ref_sample['max_power'], abs=POWER_TOLERANCE), \
            f"Max power mismatch for {filename}: expected {ref_sample['max_power']}, got {max_power}"
    
    # Check power_speed sample
    if ref.get('power_speed_sample'):
        power_speed = power_data['power_speed']
        ref_sample = ref['power_speed_sample']
        
        max_speed = max([p[0] for p in power_speed])
        assert max_speed == pytest.approx(ref_sample['max_speed'], abs=0.1), \
            f"Max speed mismatch for {filename}: expected {ref_sample['max_speed']}, got {max_speed}"
        
        valid_powers = [p[1] for p in power_speed if len(p) < 3 or p[2]]
        max_power_speed = max(valid_powers) if valid_powers else 0
        assert max_power_speed == pytest.approx(ref_sample['max_power'], abs=POWER_TOLERANCE), \
            f"Max power in speed data mismatch for {filename}: expected {ref_sample['max_power']}, got {max_power_speed}"


@pytest.mark.parametrize("filepath,filename,weight", TEST_FILES)
def test_power_calculation_all_methods(filepath, filename, weight):
    """Test power calculation with all estimation methods against reference values."""
    assert os.path.exists(filepath), f"Test file {filename} does not exist"
    assert weight is not None, f"Could not extract weight from filename {filename}"
    
    if filename not in REFERENCE_VALUES:
        pytest.skip(f"No reference values for {filename}")
    
    ref = REFERENCE_VALUES[filename]
    
    # Extract data
    speed_data, altitude_data, gps_frequency, nmea_lines, first_timestamp_ms, coords_data = extract_speed_altitude_data(filepath)
    
    if len(speed_data) < config.MIN_SPEED_POINTS:
        pytest.skip(f"Insufficient speed data: {len(speed_data)}")
    
    # Get car info with weight from filename
    car_info = get_default_car_info(weight)
    
    # Use default weather
    weather_data = config.DEFAULT_WEATHER.copy()
    
    # Calculate power with all methods
    power_data = calculate_power(
        speed_data,
        car_info,
        weather_data,
        altitude_data,
        first_timestamp_ms,
        methods='all'
    )
    
    assert power_data is not None, f"Power calculation failed for {filename}"
    assert power_data['power_estimation'] is not None
    
    # Check against reference values
    estimation = power_data['power_estimation']
    ref_methods = ref.get('all_methods', {})
    
    assert 'methods' in estimation, f"'methods' key missing in estimation for {filename}"
    methods_dict = estimation['methods']
    
    # Check consistency score
    if 'consistency_score' in ref_methods:
        assert estimation.get('consistency_score') == pytest.approx(ref_methods['consistency_score'], abs=FLOAT_TOLERANCE), \
            f"Consistency score mismatch for {filename}"
    
    # Check methods CV
    if ref_methods.get('methods_cv') is not None:
        assert estimation.get('methods_cv') == pytest.approx(ref_methods['methods_cv'], abs=FLOAT_TOLERANCE), \
            f"Methods CV mismatch for {filename}"
    
    # Check each method value
    for method_name in ['robust_mean', 'percentile_99', 'mode_kde', 'peak_detection', 'consistency']:
        if method_name in ref_methods:
            ref_method = ref_methods[method_name]
            assert method_name in methods_dict, f"Method {method_name} missing for {filename}"
            
            method_data = methods_dict[method_name]
            assert 'value' in method_data
            assert method_data['value'] == pytest.approx(ref_method['value'], abs=POWER_TOLERANCE), \
                f"Method {method_name} value mismatch for {filename}: expected {ref_method['value']}, got {method_data['value']}"
            
            # Check additional fields if present in reference
            if 'std' in ref_method and 'std' in method_data:
                assert method_data['std'] == pytest.approx(ref_method['std'], abs=0.1), \
                    f"Method {method_name} std mismatch for {filename}"
            
            if 'peak_count' in ref_method and 'peak_count' in method_data:
                assert method_data['peak_count'] == ref_method['peak_count'], \
                    f"Method {method_name} peak_count mismatch for {filename}"
            
            if 'segments_used' in ref_method and 'segments_used' in method_data:
                assert method_data['segments_used'] == ref_method['segments_used'], \
                    f"Method {method_name} segments_used mismatch for {filename}"


@pytest.mark.parametrize("filepath,filename,weight", TEST_FILES)
def test_power_calculation_statistics(filepath, filename, weight):
    """Test that power calculation returns proper statistics matching reference values."""
    assert os.path.exists(filepath), f"Test file {filename} does not exist"
    assert weight is not None, f"Could not extract weight from filename {filename}"
    
    if filename not in REFERENCE_VALUES:
        pytest.skip(f"No reference values for {filename}")
    
    ref = REFERENCE_VALUES[filename]
    
    # Extract data
    speed_data, altitude_data, gps_frequency, nmea_lines, first_timestamp_ms, coords_data = extract_speed_altitude_data(filepath)
    
    if len(speed_data) < config.MIN_SPEED_POINTS:
        pytest.skip(f"Insufficient speed data: {len(speed_data)}")
    
    # Get car info with weight from filename
    car_info = get_default_car_info(weight)
    
    # Use default weather
    weather_data = config.DEFAULT_WEATHER.copy()
    
    # Calculate power
    power_data = calculate_power(
        speed_data,
        car_info,
        weather_data,
        altitude_data,
        first_timestamp_ms,
        methods='peak_detection'
    )
    
    assert power_data is not None
    
    # Check statistics
    assert 'valid_points' in power_data
    assert 'total_points' in power_data
    assert power_data['valid_points'] > 0
    assert power_data['total_points'] > 0
    assert power_data['valid_points'] <= power_data['total_points']
    
    # Check HDOP statistics against reference
    if ref.get('hdop_statistics'):
        ref_hdop = ref['hdop_statistics']
        assert power_data.get('hdop_statistics') is not None, f"HDOP statistics missing for {filename}"
        
        hdop_stats = power_data['hdop_statistics']
        assert hdop_stats['threshold'] == pytest.approx(ref_hdop['threshold'], abs=0.01), \
            f"HDOP threshold mismatch for {filename}"
        assert hdop_stats['mean'] == pytest.approx(ref_hdop['mean'], abs=0.01), \
            f"HDOP mean mismatch for {filename}"
        assert hdop_stats['min'] == pytest.approx(ref_hdop['min'], abs=0.01), \
            f"HDOP min mismatch for {filename}"
        assert hdop_stats['max'] == pytest.approx(ref_hdop['max'], abs=0.01), \
            f"HDOP max mismatch for {filename}"
    elif power_data.get('hdop_statistics') is None:
        # Reference has None, so actual should also be None
        pass
    
    # Check pre-Kalman stats against reference
    if ref.get('pre_kalman_stats'):
        ref_pre_kalman = ref['pre_kalman_stats']
        assert power_data.get('pre_kalman_stats') is not None, f"Pre-Kalman stats missing for {filename}"
        
        pre_kalman = power_data['pre_kalman_stats']
        assert pre_kalman['filtered_ratio'] == pytest.approx(ref_pre_kalman['filtered_ratio'], abs=FLOAT_TOLERANCE), \
            f"Pre-Kalman filtered_ratio mismatch for {filename}"
        assert pre_kalman['filtered_count'] == ref_pre_kalman['filtered_count'], \
            f"Pre-Kalman filtered_count mismatch for {filename}"


@pytest.mark.parametrize("filepath,filename,weight", TEST_FILES)
def test_power_calculation_uncertainty(filepath, filename, weight):
    """Test that uncertainty calculation matches reference values."""
    assert os.path.exists(filepath), f"Test file {filename} does not exist"
    assert weight is not None, f"Could not extract weight from filename {filename}"
    
    if filename not in REFERENCE_VALUES:
        pytest.skip(f"No reference values for {filename}")
    
    ref = REFERENCE_VALUES[filename]
    
    # Extract data
    speed_data, altitude_data, gps_frequency, nmea_lines, first_timestamp_ms, coords_data = extract_speed_altitude_data(filepath)
    
    if len(speed_data) < config.MIN_SPEED_POINTS:
        pytest.skip(f"Insufficient speed data: {len(speed_data)}")
    
    # Get car info with weight from filename
    car_info = get_default_car_info(weight)
    
    # Use default weather
    weather_data = config.DEFAULT_WEATHER.copy()
    
    # Calculate power
    power_data = calculate_power(
        speed_data,
        car_info,
        weather_data,
        altitude_data,
        first_timestamp_ms,
        methods='peak_detection'
    )
    
    assert power_data is not None
    
    # Check uncertainty against reference
    if ref.get('uncertainty'):
        ref_unc = ref['uncertainty']
        assert power_data.get('uncertainty') is not None, f"Uncertainty missing for {filename}"
        
        uncertainty = power_data['uncertainty']
        assert uncertainty['total_hp'] == pytest.approx(ref_unc['total_hp'], abs=0.1), \
            f"Uncertainty total_hp mismatch for {filename}: expected {ref_unc['total_hp']}, got {uncertainty['total_hp']}"
        
        assert 'range' in uncertainty
        assert len(uncertainty['range']) == 2
        assert uncertainty['range'][0] == pytest.approx(ref_unc['range'][0], abs=0.1), \
            f"Uncertainty range[0] mismatch for {filename}"
        assert uncertainty['range'][1] == pytest.approx(ref_unc['range'][1], abs=0.1), \
            f"Uncertainty range[1] mismatch for {filename}"


def test_power_calculation_performance_100_runs():
    """Test power calculation performance with 100 runs against reference baseline."""
    import time
    
    longest_file = 'subaru_brz_gen2_mrw_1350.nmea'
    
    if longest_file not in REFERENCE_VALUES:
        pytest.skip(f"No reference values for {longest_file}")
    
    ref = REFERENCE_VALUES[longest_file]
    
    if 'performance_100_runs' not in ref:
        pytest.skip(f"No 100-run reference values for {longest_file}")
    
    # Find the file path
    test_dir = os.path.dirname(os.path.abspath(__file__))
    resources_dir = os.path.join(test_dir, "resources")
    filepath = os.path.join(resources_dir, longest_file)
    weight = extract_weight_from_filename(longest_file)
    
    assert os.path.exists(filepath), f"Test file {longest_file} does not exist"
    assert weight is not None, f"Could not extract weight from filename {longest_file}"
    
    # Extract data
    speed_data, altitude_data, gps_frequency, nmea_lines, first_timestamp_ms, coords_data = extract_speed_altitude_data(filepath)
    
    if len(speed_data) < config.MIN_SPEED_POINTS:
        pytest.skip(f"Insufficient speed data: {len(speed_data)}")
    
    # Get car info with weight from filename
    car_info = get_default_car_info(weight)
    weather_data = config.DEFAULT_WEATHER.copy()
    
    # Warm-up runs to stabilize performance (Python JIT, NumPy initialization, etc.)
    print(f"\nWarm-up: 5 iterations...")
    for _ in range(5):
        calculate_power(
            speed_data,
            car_info,
            weather_data,
            altitude_data,
            first_timestamp_ms,
            methods='peak_detection'
        )
        calculate_power(
            speed_data,
            car_info,
            weather_data,
            altitude_data,
            first_timestamp_ms,
            methods='all'
        )
    
    # Run 3 batches of 100 iterations and take median for stability
    import statistics
    
    peak_times = []
    all_times = []
    
    for batch in range(3):
        print(f"Batch {batch + 1}/3: Measuring peak_detection (100 runs)...")
        batch_peak_time = 0.0
        for i in range(100):
            start_time = time.perf_counter()
            power_data_peak = calculate_power(
                speed_data,
                car_info,
                weather_data,
                altitude_data,
                first_timestamp_ms,
                methods='peak_detection'
            )
            batch_peak_time += time.perf_counter() - start_time
            assert power_data_peak is not None
        peak_times.append(batch_peak_time)
        
        print(f"Batch {batch + 1}/3: Measuring all methods (100 runs)...")
        batch_all_time = 0.0
        for i in range(100):
            start_time = time.perf_counter()
            power_data_all = calculate_power(
                speed_data,
                car_info,
                weather_data,
                altitude_data,
                first_timestamp_ms,
                methods='all'
            )
            batch_all_time += time.perf_counter() - start_time
            assert power_data_all is not None
        all_times.append(batch_all_time)
    
    # Use median for more stable comparison (less affected by outliers)
    total_peak_time = statistics.median(peak_times)
    total_all_time = statistics.median(all_times)
    
    # Compare against reference
    ref_perf_100 = ref['performance_100_runs']
    ref_peak_total = ref_perf_100.get('peak_detection_total_seconds')
    ref_all_total = ref_perf_100.get('all_methods_total_seconds')
    
    # Print comparison
    print(f"\n{longest_file} (3 batches of 100 runs, using median):")
    print(f"  Points: {ref['speed_data_points']}")
    print(f"  Peak detection batches: {[f'{t:.3f}s' for t in peak_times]}, median: {total_peak_time:.3f}s")
    print(f"  All methods batches:    {[f'{t:.3f}s' for t in all_times]}, median: {total_all_time:.3f}s")
    if ref_peak_total:
        peak_diff = ((total_peak_time - ref_peak_total) / ref_peak_total) * 100
        peak_avg_current = total_peak_time / 100.0
        peak_avg_ref = ref_peak_total / 100.0
        print(f"  Peak detection total: {total_peak_time:.3f}s (ref: {ref_peak_total:.3f}s, diff: {peak_diff:+.1f}%)")
        print(f"  Peak detection avg:   {peak_avg_current:.3f}s (ref: {peak_avg_ref:.3f}s)")
    if ref_all_total:
        all_diff = ((total_all_time - ref_all_total) / ref_all_total) * 100
        all_avg_current = total_all_time / 100.0
        all_avg_ref = ref_all_total / 100.0
        print(f"  All methods total:    {total_all_time:.3f}s (ref: {ref_all_total:.3f}s, diff: {all_diff:+.1f}%)")
        print(f"  All methods avg:      {all_avg_current:.3f}s (ref: {all_avg_ref:.3f}s)")


def test_weight_extraction():
    """Test weight extraction from filenames."""
    test_cases = [
        ("2105 - turbo - Grade4_1020.nmea", 1020),
        ("2108_delta_adm_970.nmea", 970),
        ("dragy_bmw_e36_mrw_1300.nmea", 1300),
        ("lada_kalina_2_sport_mrw_1010.nmea", 1010),
        ("renault_megane_3_rs_mrw_2_1350.nmea", 1350),
        ("subaru_brz_gen2_mrw_1350.nmea", 1350),
        ("subaru_sti_mrw_1440.nmea", 1440),
    ]
    
    for filename, expected_weight in test_cases:
        weight = extract_weight_from_filename(filename)
        assert weight == expected_weight, f"Expected {expected_weight} for {filename}, got {weight}"


def test_default_car_info():
    """Test default car info generation."""
    weight = 1500
    car_info = get_default_car_info(weight)
    
    assert car_info['mass'] == 1500
    assert car_info['drag_coefficient'] == 0.30
    assert car_info['frontal_area'] == 2.2
    assert car_info['rolling_resistance'] == 0.015
    assert 'name' in car_info
