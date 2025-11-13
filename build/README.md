# Build

**Why it exists**: Transforms raw datasets into optimized formats that enable fast training and reliable evaluation - the critical link between data generation and model training.

**What's in it**: Dataset compression, validation, and packaging utilities that turn human-readable JSONL files into training-optimized binary formats.

**Key Features**:

- Dataset Packaging: Converts JSONL → compressed binary formats for 10x faster loading
- Integrity Validation: Checksum verification and schema compliance checking
- Performance Optimization: Compression algorithms (LZ4/ZSTD) tuned for training throughput
- CI/CD Integration: Automated dataset building in deployment pipelines

## Overview

The build directory handles the "last mile" of dataset preparation, ensuring that training data is optimized, validated, and ready for high-throughput model training.

## Key Components

### `pack_datasets.py`

**Dataset packaging and optimization** utilities:

- **Format Conversion**: JSONL ↔ optimized binary formats
- **Compression**: Efficient storage and loading
- **Validation**: Dataset integrity checks
- **Metadata**: Dataset statistics and provenance tracking

**Usage**:

```python
from build.pack_datasets import DatasetPacker

packer = DatasetPacker()
packer.pack(
    input_path="data/kd_mix.jsonl",
    output_path="data/kd_mix.packed",
    compression="lz4",
    validate=True
)
```

## Build Integration

### CI/CD Usage

Build utilities are integrated into the training pipeline:

```bash
# Package dataset for training
python -m build.pack_datasets \
  --input data/kd_mix.jsonl \
  --output data/kd_mix.packed \
  --compression lz4

# Validate dataset integrity
python -m build.pack_datasets \
  --validate data/kd_mix.packed
```

### Makefile Integration

```makefile
# Build target example
build-dataset:
	python -m build.pack_datasets \
	  --input $(DATASET) \
	  --output $(PACKED_DATASET) \
	  --validate
```

## Dataset Formats

### Supported Formats

- **JSONL**: Human-readable, uncompressed
- **Packed**: Binary format with compression
- **Sharded**: Multi-file datasets for distributed training

### Compression Options

- **LZ4**: Fast compression/decompression
- **ZSTD**: Better compression ratio
- **None**: Uncompressed for debugging

## Quality Assurance

### Validation Checks

- **Schema Compliance**: Dataset format validation
- **Data Integrity**: Checksum verification
- **Metadata Consistency**: Provenance tracking
- **Size Optimization**: Compression effectiveness

### Performance Metrics

- **Load Times**: Dataset loading performance
- **Memory Usage**: Runtime memory efficiency
- **Training Throughput**: Impact on training speed

## Integration Points

### With Scripts

- **Dataset Generation**: `scripts/make_kd_mix.py` creates raw datasets
- **Processing**: `scripts/extract_process_targets.py` adds supervision targets
- **Packaging**: `build/pack_datasets.py` optimizes for training

### With Training

- **Data Loading**: `training/dataset.py` loads packed datasets
- **Performance**: Optimized loading improves training throughput
- **Validation**: Runtime dataset integrity checks

### With Evaluation

- **Data Access**: Evaluation scripts use packed datasets
- **Consistency**: Same data format across training/evaluation

## Development

### Adding New Formats

1. Extend `DatasetPacker` class
2. Add compression algorithm support
3. Update validation logic
4. Add performance benchmarks

### Testing

```bash
# Test packaging
python -m pytest tests/test_dataset_packaging.py -v

# Benchmark loading
python -m build.benchmark_loading \
  --dataset data/kd_mix.packed \
  --batch-size 8
```

## See Also

- [`scripts/make_kd_mix.py`](../scripts/make_kd_mix.py) - Dataset generation
- [`training/dataset.py`](../training/dataset.py) - Dataset loading
- [`data/README.md`](../data/README.md) - Dataset directory documentation
- [`docs/CONTEXTUAL_DATASET_GENERATION.md`](../docs/CONTEXTUAL_DATASET_GENERATION.md) - Generation guide
