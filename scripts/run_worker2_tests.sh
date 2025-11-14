#!/bin/bash
# Worker 2 Test Execution Script
# Runs tests for evaluation module with coverage

set -e

echo "=========================================="
echo "Worker 2: Evaluation Module Tests"
echo "=========================================="

# Activate virtual environment
source venv/bin/activate

# Create coverage output directory
mkdir -p coverage_output
mkdir -p htmlcov/worker2

# Run tests with coverage
echo "Running tests with coverage..."
pytest \
    --cov=evaluation \
    --cov-report=term-missing \
    --cov-report=html:htmlcov/worker2 \
    --cov-report=json:coverage_output/worker2_coverage.json \
    tests/evaluation/ \
    -v \
    2>&1 | tee coverage_output/worker2_test_output.txt

# Generate summary
echo "Generating coverage summary..."
python scripts/worker2_coverage_summary.py

echo "=========================================="
echo "Worker 2 Tests Complete"
echo "=========================================="
echo "Coverage Report: htmlcov/worker2/index.html"
echo "JSON Report: coverage_output/worker2_coverage.json"
echo "Test Output: coverage_output/worker2_test_output.txt"
echo "Summary: coverage_output/worker2_summary.md"
