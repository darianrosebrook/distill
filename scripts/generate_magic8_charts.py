#!/usr/bin/env python3
"""
Generate SVG charts for Magic 8 Ball model card.

This script creates comparative performance charts showing how the Magic 8 Ball
model compares to major commercial LLMs on speed, cost, and efficiency metrics.
"""

from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_magic8_benchmarks(report_path='eval/reports/magic_8_ball_e2e.json'):
    """Load Magic 8 Ball benchmark data."""
    if os.path.exists(report_path):
        with open(report_path) as f:
            return json.load(f)
    return None


def generate_latency_chart(output_dir):
    """Generate inference latency comparison chart."""
    models = ['Magic 8 Ball', 'Claude-3 Haiku', 'GPT-3.5 Turbo', 'GPT-4']
    latencies_ms = [0.40, 300, 500, 2000]

    # Dark mode theme
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#2d2d2d')

    # Use log scale for better visualization
    # Mid blue-ish purple for Magic 8 Ball, grey for others
    # Purple for M8B, grey for others
    colors = ['#8B5CF6', '#6B7280', '#6B7280', '#6B7280']
    bars = ax.bar(models, latencies_ms, color=colors,
                  edgecolor='#ffffff', linewidth=1.5, alpha=0.8)

    # Add value labels on bars
    for bar, val in zip(bars, latencies_ms):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}ms',
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='white')

    ax.set_yscale('log')
    ax.set_ylabel('Inference Latency (ms, log scale)',
                  fontsize=12, fontweight='bold', color='white')
    ax.set_title('Inference Speed Comparison: Magic 8 Ball vs Major LLMs',
                 fontsize=14, fontweight='bold', pad=20, color='white')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', color='#666666')
    ax.set_axisbelow(True)

    # Highlight Magic 8 Ball with brighter color and glow effect
    bars[0].set_edgecolor('#8B5CF6')
    bars[0].set_linewidth(3)
    bars[0].set_alpha(1.0)

    # Set tick colors
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')

    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()

    output_path = output_dir / 'latency_comparison.svg'
    plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"✅ Generated: {output_path}")
    return output_path


def generate_cost_chart(output_dir):
    """Generate cost per inference comparison chart."""
    models = ['Magic 8 Ball', 'Claude-3 Haiku', 'GPT-3.5 Turbo', 'GPT-4']
    costs = [0.0000000042, 0.00025, 0.002, 0.030]

    # Dark mode theme
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#2d2d2d')

    # Purple for M8B, grey for others
    colors = ['#8B5CF6', '#6B7280', '#6B7280', '#6B7280']
    bars = ax.bar(models, costs, color=colors,
                  edgecolor='#ffffff', linewidth=1.5, alpha=0.8)

    # Add value labels
    for bar, val in zip(bars, costs):
        height = bar.get_height()
        if val < 0.001:
            label = f'${val:.2e}'
        else:
            label = f'${val:.4f}'
        ax.text(bar.get_x() + bar.get_width()/2., height,
                label,
                ha='center', va='bottom', fontsize=9, fontweight='bold', color='white')

    ax.set_yscale('log')
    ax.set_ylabel('Cost per Inference ($, log scale)',
                  fontsize=12, fontweight='bold', color='white')
    ax.set_title('Cost Efficiency: Magic 8 Ball vs Major LLMs',
                 fontsize=14, fontweight='bold', pad=20, color='white')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', color='#666666')
    ax.set_axisbelow(True)

    # Highlight Magic 8 Ball
    bars[0].set_edgecolor('#8B5CF6')
    bars[0].set_linewidth(3)
    bars[0].set_alpha(1.0)

    # Set tick colors
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')

    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()

    output_path = output_dir / 'cost_comparison.svg'
    plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"✅ Generated: {output_path}")
    return output_path


def generate_throughput_chart(output_dir):
    """Generate throughput per dollar comparison chart."""
    models = ['Magic 8 Ball', 'Claude-3 Haiku', 'GPT-3.5 Turbo', 'GPT-4 API']
    throughput = [605999318910, 10417, 500, 0.5]

    # Dark mode theme
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#2d2d2d')

    # Purple for M8B, grey for others
    colors = ['#8B5CF6', '#6B7280', '#6B7280', '#6B7280']
    bars = ax.bar(models, throughput, color=colors,
                  edgecolor='#ffffff', linewidth=1.5, alpha=0.8)

    # Add value labels with formatting
    labels = ['606B', '10K', '500', '0.5']
    for bar, label in zip(bars, labels):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                label,
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='white')

    ax.set_yscale('log')
    ax.set_ylabel('Throughput (inferences/sec per $, log scale)',
                  fontsize=12, fontweight='bold', color='white')
    ax.set_title('Throughput Efficiency: Magic 8 Ball vs Major LLMs',
                 fontsize=14, fontweight='bold', pad=20, color='white')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', color='#666666')
    ax.set_axisbelow(True)

    # Highlight Magic 8 Ball
    bars[0].set_edgecolor('#8B5CF6')
    bars[0].set_linewidth(3)
    bars[0].set_alpha(1.0)

    # Set tick colors
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')

    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()

    output_path = output_dir / 'throughput_comparison.svg'
    plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"✅ Generated: {output_path}")
    return output_path


def generate_efficiency_multipliers_chart(output_dir):
    """Generate cost efficiency multipliers chart."""
    models = ['Magic 8 Ball', 'Claude-3 Haiku', 'GPT-3.5 Turbo', 'GPT-4']
    multipliers = [1, 23122, 476190, 1387343]

    # Dark mode theme
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#2d2d2d')

    # Purple for M8B, grey for others
    colors = ['#8B5CF6', '#6B7280', '#6B7280', '#6B7280']
    bars = ax.bar(models, multipliers, color=colors,
                  edgecolor='#ffffff', linewidth=1.5, alpha=0.8)

    # Add value labels
    labels = ['1×', '23K×', '476K×', '1.4M×']
    for bar, label in zip(bars, labels):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                label,
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='white')

    ax.set_yscale('log')
    ax.set_ylabel('Cost Multiplier (Magic 8 Ball = 1×)',
                  fontsize=12, fontweight='bold', color='white')
    ax.set_title('Cost Efficiency Multipliers: How Much More Expensive?',
                 fontsize=14, fontweight='bold', pad=20, color='white')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', color='#666666')
    ax.set_axisbelow(True)

    # Highlight Magic 8 Ball
    bars[0].set_edgecolor('#8B5CF6')
    bars[0].set_linewidth(3)
    bars[0].set_alpha(1.0)

    # Set tick colors
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')

    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()

    output_path = output_dir / 'efficiency_multipliers.svg'
    plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"✅ Generated: {output_path}")
    return output_path


def generate_billion_scale_savings_chart(output_dir):
    """Generate billion-scale cost savings chart."""
    models = ['GPT-4', 'Claude-3 Opus', 'GPT-3.5 Turbo', 'Claude-3 Haiku']
    savings = [5759995.85, 7499995.85, 2239995.85, 95995.85]

    # Dark mode theme
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#2d2d2d')

    # Use different shades of grey for savings bars
    colors = ['#9CA3AF', '#6B7280', '#4B5563', '#374151']
    bars = ax.bar(models, savings, color=colors,
                  edgecolor='#ffffff', linewidth=1.5, alpha=0.8)

    # Add value labels in millions
    for bar, val in zip(bars, savings):
        height = bar.get_height()
        label = f'${val/1e6:.2f}M'
        ax.text(bar.get_x() + bar.get_width()/2., height,
                label,
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='white')

    ax.set_ylabel('Cost Savings vs Magic 8 Ball ($)',
                  fontsize=12, fontweight='bold', color='white')
    ax.set_title('Cost Savings: 1 Billion Inferences vs Magic 8 Ball',
                 fontsize=14, fontweight='bold', pad=20, color='white')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', color='#666666')
    ax.set_axisbelow(True)

    # Set tick colors
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')

    output_path = output_dir / 'billion_scale_savings.svg'
    plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"✅ Generated: {output_path}")
    return output_path


def generate_model_size_chart(output_dir):
    """Generate model size comparison chart."""
    models = ['Magic 8 Ball', 'Claude-3 Haiku', 'GPT-3.5 Turbo', 'GPT-4']
    sizes_mb = [1.24, 23000, 100000, 500000]

    # Dark mode theme
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#2d2d2d')

    colors = ['#8B5CF6', '#6B7280', '#6B7280', '#6B7280']  # Purple for M8B, grey for others
    bars = ax.bar(models, sizes_mb, color=colors, edgecolor='#ffffff', linewidth=1.5, alpha=0.8)

    # Add value labels
    labels = ['1.24 MB', '23 GB', '100 GB', '500 GB']
    for bar, label in zip(bars, labels):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                label,
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='white')

    ax.set_yscale('log')
    ax.set_ylabel('Model Size (MB, log scale)', fontsize=12, fontweight='bold', color='white')
    ax.set_title('Model Size Comparison: Magic 8 Ball vs Major LLMs',
                 fontsize=14, fontweight='bold', pad=20, color='white')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', color='#666666')
    ax.set_axisbelow(True)

    # Highlight Magic 8 Ball
    bars[0].set_edgecolor('#8B5CF6')
    bars[0].set_linewidth(3)
    bars[0].set_alpha(1.0)

    # Set tick colors
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')

    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()

    output_path = output_dir / 'model_size_comparison.svg'
    plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"✅ Generated: {output_path}")
    return output_path


def main():
    """Generate all Magic 8 Ball model card charts."""
    # Create output directory
    output_dir = project_root / 'docs' / 'assets' / 'magic_8_ball'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🎨 Generating Magic 8 Ball Model Card Charts...")
    print("=" * 60)

    # Generate all charts
    charts = {
        'latency': generate_latency_chart(output_dir),
        'cost': generate_cost_chart(output_dir),
        'throughput': generate_throughput_chart(output_dir),
        'efficiency_multipliers': generate_efficiency_multipliers_chart(output_dir),
        'billion_scale_savings': generate_billion_scale_savings_chart(output_dir),
        'model_size': generate_model_size_chart(output_dir),
    }

    # Save chart metadata
    metadata = {
        'generated_at': datetime.now().isoformat(),
        'charts': {k: str(v.relative_to(project_root)) for k, v in charts.items()}
    }

    metadata_path = output_dir / 'charts_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print()
    print("=" * 60)
    print(f"✅ Generated {len(charts)} charts in {output_dir}")
    print(f"📊 Charts available at: {output_dir.relative_to(project_root)}")

    return charts


if __name__ == '__main__':
    main()
