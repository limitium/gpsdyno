#!/bin/bash
# Simple test runner script for GPSDyno tests
# Uses pyenv to manage Python version
#
# Usage:
#   ./test/run_tests.sh              # Run all tests
#   ./test/run_tests.sh --regular    # Run only regular tests (exclude performance)
#   ./test/run_tests.sh --performance # Run only performance test
#   ./test/run_tests.sh -k "peak"    # Run tests matching pattern

set -e

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Initialize pyenv if available
if command -v pyenv >/dev/null 2>&1; then
    eval "$(pyenv init -)"
    # Use pyenv's Python version for this project
    PYTHON_CMD=$(pyenv which python 2>/dev/null || echo "python3")
else
    PYTHON_CMD="python3"
fi

# Check if pytest is installed
if ! $PYTHON_CMD -m pytest --version >/dev/null 2>&1; then
    echo "pytest not found. Installing..."
    $PYTHON_CMD -m pip install pytest
fi

# Set matplotlib backend for headless operation
export MPLBACKEND=Agg

# Parse arguments
if [[ "$1" == "--regular" ]]; then
    echo "Running regular GPSDyno tests (excluding performance) with $($PYTHON_CMD --version)..."
    $PYTHON_CMD -m pytest test/test_calculator.py -v -k "not performance_100_runs" "${@:2}"
elif [[ "$1" == "--performance" ]]; then
    echo "Running performance test with $($PYTHON_CMD --version)..."
    $PYTHON_CMD -m pytest test/test_calculator.py::test_power_calculation_performance_100_runs -v -s "${@:2}"
else
    echo "Running all GPSDyno tests with $($PYTHON_CMD --version)..."
    $PYTHON_CMD -m pytest test/test_calculator.py -v "$@"
fi

