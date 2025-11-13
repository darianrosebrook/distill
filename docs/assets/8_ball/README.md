# 8-Ball Model Card Assets

This directory contains automatically generated SVG charts for the 8-Ball model card.

## Charts

- `latency_comparison.svg` - Inference speed comparison vs major LLMs
- `cost_comparison.svg` - Cost per inference comparison
- `throughput_comparison.svg` - Throughput per dollar comparison
- `efficiency_multipliers.svg` - Cost efficiency multipliers
- `billion_scale_savings.svg` - Cost savings at billion-scale
- `model_size_comparison.svg` - Model size comparison

## Generation

These charts are automatically generated when running `make 8-ball` via the script:
`scripts/generate_8ball_charts.py`

## Regeneration

To regenerate charts manually:
```bash
python scripts/generate_8ball_charts.py
```

