#!/usr/bin/env python3
"""
Script to capture reference power calculation values for test assertions.

This script runs power calculations on all test NMEA files and outputs
the results as a Python dictionary that can be used in tests.
"""
import os
import sys
import json
import re
import time

# Add parent directory to path for imports
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from core.calculator import calculate_power
from parsers.nmea_handler import extract_speed_altitude_data
import config


def extract_weight_from_filename(filename):
    """Extract weight (last number) from filename."""
    numbers = re.findall(r'\d+', filename)
    if numbers:
        return int(numbers[-1])
    return None


def get_default_car_info(weight):
    """Get default car info with specified weight."""
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


def capture_reference_values():
    """Capture reference values from all test files."""
    reference_values = {}
    test_files = get_test_files()
    
    for filepath, filename, weight in test_files:
        print(f"Processing {filename}...", file=sys.stderr)
        
        try:
            # Extract data
            speed_data, altitude_data, gps_frequency, nmea_lines, first_timestamp_ms, coords_data = extract_speed_altitude_data(filepath)
            
            if len(speed_data) < config.MIN_SPEED_POINTS:
                print(f"  Skipping {filename}: insufficient data ({len(speed_data)} points)", file=sys.stderr)
                continue
            
            # Get car info
            car_info = get_default_car_info(weight)
            weather_data = config.DEFAULT_WEATHER.copy()
            
            # For longest file, run 100 iterations for stable performance baseline
            longest_file = 'subaru_brz_gen2_mrw_1350.nmea'
            total_peak_time = None
            total_all_time = None
            
            if filename == longest_file:
                print(f"  Running performance baseline (3 batches of 100 iterations with warm-up)...", file=sys.stderr)
                
                # Warm-up runs to stabilize performance (Python JIT, NumPy initialization, etc.)
                print(f"    Warm-up: 5 iterations...", file=sys.stderr)
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
                peak_times = []
                all_times = []
                
                for batch in range(3):
                    print(f"    Batch {batch + 1}/3: Measuring peak_detection (100 runs)...", file=sys.stderr)
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
                    peak_times.append(batch_peak_time)
                    
                    print(f"    Batch {batch + 1}/3: Measuring all methods (100 runs)...", file=sys.stderr)
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
                    all_times.append(batch_all_time)
                
                # Use median for more stable reference (less affected by outliers)
                import statistics
                total_peak_time = statistics.median(peak_times)
                total_all_time = statistics.median(all_times)
                
                peak_detection_time = total_peak_time / 100.0  # Average for display
                all_methods_time = total_all_time / 100.0  # Average for display
                
                print(f"    Results: peak_detection batches={peak_times}, median={total_peak_time:.3f}s", file=sys.stderr)
                print(f"             all_methods batches={all_times}, median={total_all_time:.3f}s", file=sys.stderr)
            else:
                # For other files, just run once for data validation
                print(f"  Calculating peak_detection...", file=sys.stderr)
                start_time = time.perf_counter()
                power_data_peak = calculate_power(
                    speed_data,
                    car_info,
                    weather_data,
                    altitude_data,
                    first_timestamp_ms,
                    methods='peak_detection'
                )
                peak_detection_time = time.perf_counter() - start_time
                
                print(f"  Calculating all methods...", file=sys.stderr)
                start_time = time.perf_counter()
                power_data_all = calculate_power(
                    speed_data,
                    car_info,
                    weather_data,
                    altitude_data,
                    first_timestamp_ms,
                    methods='all'
                )
                all_methods_time = time.perf_counter() - start_time
            
            if not power_data_peak or not power_data_all:
                print(f"  Skipping {filename}: calculation failed", file=sys.stderr)
                continue
            
            # Extract key values
            ref = {
                'weight': weight,
                'speed_data_points': len(speed_data),
                'altitude_data_points': len(altitude_data) if altitude_data else 0,
                'gps_frequency': float(gps_frequency) if gps_frequency else None,
                
                # Performance timing (in seconds)
                'performance': {
                    'peak_detection_time_seconds': float(peak_detection_time),
                    'all_methods_time_seconds': float(all_methods_time),
                },
                
                # For longest file, store 100-run total time as reference
                'performance_100_runs': None,
                
                # Power calculation results (peak_detection)
                'peak_detection': {
                    'recommended_value': power_data_peak['power_estimation']['recommended_value'] if power_data_peak.get('power_estimation') else None,
                    'recommended_method': power_data_peak['power_estimation']['recommended'] if power_data_peak.get('power_estimation') else None,
                    'valid_points': power_data_peak.get('valid_points', 0),
                    'total_points': power_data_peak.get('total_points', 0),
                    'filtered_ratio': float(power_data_peak.get('filtered_ratio', 0)),
                },
                
                # Power calculation results (all methods)
                'all_methods': {},
                
                # Statistics
                'hdop_statistics': None,
                'pre_kalman_stats': None,
                'uncertainty': None,
            }
            
            # HDOP statistics
            if power_data_peak.get('hdop_statistics'):
                hdop = power_data_peak['hdop_statistics']
                ref['hdop_statistics'] = {
                    'threshold': float(hdop.get('threshold', 0)),
                    'mean': float(hdop.get('mean', 0)),
                    'min': float(hdop.get('min', 0)),
                    'max': float(hdop.get('max', 0)),
                    'median': float(hdop.get('median', 0)),
                }
            
            # Pre-Kalman stats
            if power_data_peak.get('pre_kalman_stats'):
                pre_kalman = power_data_peak['pre_kalman_stats']
                ref['pre_kalman_stats'] = {
                    'filtered_ratio': float(pre_kalman.get('filtered_ratio', 0)),
                    'filtered_count': pre_kalman.get('filtered_count', 0),
                    'valid_count': len(pre_kalman.get('valid_indices', [])),
                }
            
            # Uncertainty
            if power_data_peak.get('uncertainty'):
                unc = power_data_peak['uncertainty']
                ref['uncertainty'] = {
                    'total_hp': float(unc.get('total_hp', 0)),
                    'range': [float(x) for x in unc.get('range', [0, 0])],
                }
            
            # All methods results
            if power_data_all.get('power_estimation') and power_data_all['power_estimation'].get('methods'):
                methods_dict = power_data_all['power_estimation']['methods']
                ref['all_methods'] = {
                    'consistency_score': float(power_data_all['power_estimation'].get('consistency_score', 0)),
                    'methods_cv': float(power_data_all['power_estimation'].get('methods_cv', 0)) if power_data_all['power_estimation'].get('methods_cv') else None,
                }
                
                # Extract method values
                for method_name in ['robust_mean', 'percentile_99', 'mode_kde', 'peak_detection', 'consistency']:
                    if method_name in methods_dict:
                        method_data = methods_dict[method_name]
                        ref['all_methods'][method_name] = {
                            'value': float(method_data.get('value', 0)),
                        }
                        # Add additional fields if present
                        if 'std' in method_data:
                            ref['all_methods'][method_name]['std'] = float(method_data['std'])
                        if 'kde_max' in method_data:
                            ref['all_methods'][method_name]['kde_max'] = float(method_data['kde_max'])
                        if 'peak_count' in method_data:
                            ref['all_methods'][method_name]['peak_count'] = method_data['peak_count']
                        if 'segments_used' in method_data:
                            ref['all_methods'][method_name]['segments_used'] = method_data['segments_used']
            
            # Power time series sample (first, middle, last)
            if power_data_peak.get('power_time'):
                power_time = power_data_peak['power_time']
                ref['power_time_sample'] = {
                    'first': [float(x) for x in power_time[0]] if power_time else None,
                    'middle': [float(x) for x in power_time[len(power_time)//2]] if len(power_time) > 0 else None,
                    'last': [float(x) for x in power_time[-1]] if power_time else None,
                    'max_power': float(max([p[1] for p in power_time])) if power_time else None,
                }
            
            # Power speed sample
            if power_data_peak.get('power_speed'):
                power_speed = power_data_peak['power_speed']
                valid_powers = [p[1] for p in power_speed if len(p) < 3 or p[2]]
                ref['power_speed_sample'] = {
                    'max_speed': float(max([p[0] for p in power_speed])) if power_speed else None,
                    'max_power': float(max(valid_powers)) if valid_powers else None,
                    'valid_points': len(valid_powers),
                }
            
            # Store 100-run totals for longest file
            longest_file = 'subaru_brz_gen2_mrw_1350.nmea'
            if filename == longest_file and total_peak_time is not None and total_all_time is not None:
                ref['performance_100_runs'] = {
                    'peak_detection_total_seconds': float(total_peak_time),
                    'all_methods_total_seconds': float(total_all_time),
                }
                print(f"  ✓ Captured values for {filename}", file=sys.stderr)
                print(f"    Performance (100 runs): peak_detection_total={total_peak_time:.3f}s, all_methods_total={total_all_time:.3f}s", file=sys.stderr)
                print(f"    Performance (avg): peak_detection={peak_detection_time:.3f}s, all_methods={all_methods_time:.3f}s, points={len(speed_data)}", file=sys.stderr)
            elif filename == longest_file:
                print(f"  ⚠ Warning: {filename} is longest file but 100-run totals not set", file=sys.stderr)
            
            reference_values[filename] = ref
            if filename != longest_file:
                print(f"  ✓ Captured values for {filename}", file=sys.stderr)
                print(f"    Performance: peak_detection={peak_detection_time:.3f}s, all_methods={all_methods_time:.3f}s, points={len(speed_data)}", file=sys.stderr)
            
        except Exception as e:
            print(f"  ✗ Error processing {filename}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
    
    return reference_values


if __name__ == '__main__':
    ref_values = capture_reference_values()
    print(json.dumps(ref_values, indent=2, ensure_ascii=False))

