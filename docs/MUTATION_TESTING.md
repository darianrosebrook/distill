# Mutation Testing Setup

## Overview

Mutation testing is configured for the distill project using `mutatest` (version 2.0.1). Mutation testing helps verify test quality by introducing small changes (mutations) to the code and checking if tests detect them.

## Installation

Mutation testing is included in the dev dependencies:

```bash
pip install -r requirements-dev.txt
# or
pip install -e ".[dev]"
```

## Configuration

### Configuration File

A configuration file is available at `.mutatest.yaml` with default settings and module targets.

### Mutation Score Targets

Based on CAWS policy tiers:

- **Critical Components** (Tier 1): 70%+ mutation score
  - `training/distill_kd.py`
  - `training/losses.py`
  - `training/input_validation.py`
  - `conversion/export_onnx.py`
  - `conversion/convert_coreml.py`

- **Standard Features** (Tier 2): 60%+ mutation score
  - `training/dataset.py`
  - `training/process_losses.py`
  - `models/teacher/teacher_client.py`

- **Utilities** (Tier 3): 50%+ mutation score
  - `training/utils.py`
  - `training/assertions.py`

## Usage

### Using the Script

The easiest way to run mutation testing is using the provided script:

```bash
# Test a single module
python scripts/run_mutation_testing.py --module training/losses.py

# Test with full mode (all mutations, slower but thorough)
python scripts/run_mutation_testing.py --module training/losses.py --mode f

# Test all critical modules
python scripts/run_mutation_testing.py --all-critical

# Test with custom settings
python scripts/run_mutation_testing.py --module training/losses.py \
    --mode s \
    -n 20 \
    -o mutation_report.rst \
    -x 5
```

### Using Makefile

```bash
# Test a specific module
make mutation-test MODULE=training/losses.py MODE=s N=10

# Test all critical modules
make mutation-test-critical
```

### Direct mutatest Command

```bash
# Basic usage
mutatest -s training/losses.py -t "pytest tests/training/test_losses.py -x" --mode s -n 10 --nocov

# Full mode (all mutations)
mutatest -s training/losses.py -t "pytest tests/training/test_losses.py -x" --mode f --nocov

# With output file
mutatest -s training/losses.py -t "pytest tests/training/test_losses.py -x" --mode s -n 10 -o report.rst --nocov
```

## Mutation Testing Modes

- **`f`** (full): Run all possible mutations (slowest but most thorough)
- **`s`** (survivor): Break on first survivor per location (default, balanced)
- **`d`** (detection): Break on first detection per location (faster)
- **`sd`** (survivor or detection): Break on first survivor or detection (fastest)

## Understanding Results

### Mutation Categories

- **SURVIVOR**: Mutation was not detected by tests (test quality issue)
- **DETECTED**: Mutation was caught by tests (good test quality)
- **TIMEOUT**: Mutation trial took too long (aborted)
- **UNKNOWN**: Mutation result unclear (may need investigation)

### Mutation Score

The mutation score is calculated as:

```
mutation_score = (detected_mutations / total_mutations) * 100
```

A higher score indicates better test quality. The target is:
- Critical: 70%+
- Standard: 60%+
- Utility: 50%+

## Known Issues

### mutatest 2.0.1 Compatibility

There is a known issue with mutatest 2.0.1 where it may fail with:

```
TypeError: Population must be a sequence.  For dicts or sets, use sorted(d).
```

This appears to be a bug in mutatest itself. Workarounds:

1. Use `--nocov` flag (already included in the script)
2. Try a different random seed: `-r <seed>`
3. Use a different number of locations: `-n <number>`
4. Consider downgrading to mutatest 0.19.0 if issues persist

### Coverage File Compatibility

mutatest may have issues with coverage file formats. The script uses `--nocov` by default to avoid these issues. This means mutation testing will test all code locations, not just covered ones.

## CI/CD Integration

To integrate mutation testing into CI/CD:

```yaml
# Example GitHub Actions step
- name: Mutation Testing
  run: |
    python scripts/run_mutation_testing.py --all-critical --mode s -n 10 -x 5
```

The `-x 5` flag will cause the step to fail if 5 or more survivors are found.

## Best Practices

1. **Start Small**: Begin with a small number of locations (`-n 5`) to verify setup
2. **Use Appropriate Mode**: Use `s` mode for regular testing, `f` mode for comprehensive analysis
3. **Focus on Critical Modules**: Prioritize mutation testing on critical business logic
4. **Review Survivors**: Investigate why mutations survived - may indicate missing test coverage
5. **Set Exception Thresholds**: Use `-x` flag to fail CI if too many survivors found

## Resources

- [mutatest Documentation](https://mutatest.readthedocs.io/)
- [Mutation Testing Explained](https://en.wikipedia.org/wiki/Mutation_testing)
- CAWS Policy: `.caws/policy.yaml` (mutation score targets)

