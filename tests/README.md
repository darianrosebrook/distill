# Tests

Test suite for unit, integration, end-to-end, and property tests.

## Overview

This directory contains comprehensive tests for:
- **Unit Tests**: Individual component testing
- **Integration Tests**: Component interaction testing
- **E2E Tests**: End-to-end pipeline testing
- **Property Tests**: Invariant and property validation

## Test Structure

```
tests/
├── unit/              # Unit tests
├── integration/       # Integration tests
├── e2e/               # End-to-end tests
│   ├── test_code_mode.py
│   ├── test_latent_reasoning.py
│   └── test_token_reduction.py
├── training/          # Training tests
├── runtime/           # Runtime tests
├── models/            # Model tests
└── tokenizer/         # Tokenizer tests
```

## Running Tests

### All Tests
```bash
pytest
```

### Specific Test Categories
```bash
pytest tests/unit/           # Unit tests only
pytest tests/integration/    # Integration tests only
pytest tests/e2e/            # E2E tests only
```

### With Coverage
```bash
pytest --cov=. --cov-report=html
```

## Test Coverage Requirements

- **Unit Tests**: 80%+ line coverage, 90%+ branch coverage
- **Integration Tests**: Real database connections, actual API calls
- **E2E Tests**: Complete workflows from start to finish

## See Also

- [`docs/PIPELINE_CRITICAL_PATH_REVIEW.md`](../docs/PIPELINE_CRITICAL_PATH_REVIEW.md) - Pipeline review
- [`pytest.ini`](../pytest.ini) - Pytest configuration




















