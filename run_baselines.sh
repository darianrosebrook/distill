#!/bin/bash

echo "=========================================="
echo "DISTILL TEST BASELINE SUITE"
echo "=========================================="
echo ""

# Unit Tests
echo "Running UNIT TESTS (all training tests)..."
pytest tests/training/ \
  --tb=short \
  -q \
  --timeout=120 \
  --cov=training \
  --cov-report=term-missing \
  2>&1 | tee baseline_unit_tests.log

UNIT_EXIT=$?

echo ""
echo "=========================================="
echo "Unit Tests Exit Code: $UNIT_EXIT"
echo "=========================================="
echo ""

# Integration Tests
echo "Running INTEGRATION TESTS (conversion, evaluation, e2e)..."
pytest tests/conversion tests/evaluation tests/e2e \
  --tb=short \
  -q \
  --timeout=180 \
  --cov=conversion,evaluation \
  --cov-report=term-missing \
  2>&1 | tee baseline_integration_tests.log

INTEGRATION_EXIT=$?

echo ""
echo "=========================================="
echo "Integration Tests Exit Code: $INTEGRATION_EXIT"
echo "=========================================="
echo ""

# Mutation Tests
echo "Running MUTATION TESTS (core training modules)..."
mutatest --target training/distill_kd.py \
  --output mutations_distill_kd.json \
  --mode line \
  2>&1 | tee baseline_mutation_tests.log

MUTATION_EXIT=$?

echo ""
echo "=========================================="
echo "Mutation Tests Exit Code: $MUTATION_EXIT"
echo "=========================================="
echo ""
echo "BASELINE SUMMARY"
echo "=========================================="
echo "Unit Tests: $UNIT_EXIT"
echo "Integration Tests: $INTEGRATION_EXIT"
echo "Mutation Tests: $MUTATION_EXIT"
echo "=========================================="
