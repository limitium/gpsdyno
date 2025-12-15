#!/usr/bin/env python3
"""Show performance summary from reference values."""
import json
import os
import sys

test_dir = os.path.dirname(os.path.abspath(__file__))
ref_file = os.path.join(test_dir, "reference_values.json")

with open(ref_file, 'r') as f:
    ref = json.load(f)

print('=' * 80)
print('PERFORMANCE BENCHMARK SUMMARY')
print('=' * 80)
print()
print(f"{'File':<40} | {'Points':>8} | {'Peak (s)':>10} | {'All (s)':>10}")
print('-' * 80)

for filename in sorted(ref.keys()):
    data = ref[filename]
    points = data['speed_data_points']
    perf = data.get('performance', {})
    peak_time = perf.get('peak_detection_time_seconds', 0)
    all_time = perf.get('all_methods_time_seconds', 0)
    
    name = filename[:38].ljust(38)
    print(f'{name} | {points:8d} | {peak_time:10.3f} | {all_time:10.3f}')

print()
print('=' * 80)
print('REFERENCE FOR PERFORMANCE TESTING')
print('=' * 80)
longest_file = 'subaru_brz_gen2_mrw_1350.nmea'
if longest_file in ref:
    longest = ref[longest_file]
    perf = longest.get('performance', {})
    print(f'File: {longest_file}')
    print(f'Data points: {longest["speed_data_points"]}')
    print(f'Peak detection time: {perf.get("peak_detection_time_seconds", 0):.3f} seconds')
    print(f'All methods time: {perf.get("all_methods_time_seconds", 0):.3f} seconds')
    print()
    print('This file will be used for performance regression testing.')
    print('Performance test will fail if calculation takes >20% longer than reference.')

