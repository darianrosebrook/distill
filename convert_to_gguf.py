#!/usr/bin/env python3
"""
Convert Magic 8 Ball (or any toy) model to GGUF format for Ollama.

This script converts PyTorch checkpoints to GGUF format compatible with llama.cpp
and creates an Ollama modelfile for easy deployment.

Usage:
    python convert_to_gguf.py --checkpoint /tmp/magic_8_ball.ckpt --out magic-8-ball.gguf
    ollama create magic-8-ball -f Modelfile.magic-8-ball
    ollama run magic-8-ball "Will this work?"

Author: @darianrosebrook
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.student.architectures.gqa_transformer import StudentLM, ModelCfg


def check_llama_cpp():
    """Check if llama.cpp conversion tools are available."""
    # Try different possible binary names
    possible_names = [
        'llama-convert',
        'convert-llama-gguf',
        'convert',
        'llama-convert-gguf',
    ]
    
    for name in possible_names:
        try:
            result = subprocess.run(
                ['which', name],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            continue
    
    # Try common installation paths
    common_paths = [
        '/opt/homebrew/bin/llama-convert',
        '/usr/local/bin/llama-convert',
        '/opt/homebrew/Cellar/llama.cpp/*/bin/llama-convert',
        '~/.local/bin/llama-convert',
    ]
    
    for path_pattern in common_paths:
        expanded = Path(path_pattern).expanduser()
        # Handle glob patterns
        if '*' in str(expanded):
            import glob
            matches = glob.glob(str(expanded))
            if matches:
                return matches[0]
        elif expanded.exists():
            return str(expanded)
    
    # Try finding via brew
    try:
        result = subprocess.run(
            ['brew', '--prefix', 'llama.cpp'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            brew_prefix = result.stdout.strip()
            # Check multiple possible binary locations
            possible_bins = [
                Path(brew_prefix) / 'bin' / 'llama-convert',
                Path(brew_prefix) / 'bin' / 'convert',
                Path(brew_prefix) / 'bin' / 'convert-llama-gguf',
            ]
            for bin_path in possible_bins:
                if bin_path.exists():
                    return str(bin_path)
    except Exception:
        pass
    
    # Try finding via brew's intel Python (if user mentioned it)
    # Check common intel Python locations
    intel_python_paths = [
        '/usr/local/bin/python3',  # Common intel Python location
        '/opt/homebrew/opt/python@3.11/bin/python3',  # Intel Python via homebrew
    ]
    
    for python_path in intel_python_paths:
        if Path(python_path).exists():
            # Try to find llama.cpp tools via that Python
            try:
                result = subprocess.run(
                    [python_path, '-m', 'llama_cpp.convert'],
                    capture_output=True,
                    text=True
                )
                # If module exists, we can use it
                if result.returncode != 127:  # 127 = command not found
                    return f"{python_path} -m llama_cpp.convert"
            except Exception:
                continue
    
    return None


def export_to_huggingface_format(checkpoint_path: str, output_dir: Path):
    """Export PyTorch checkpoint to HuggingFace-compatible format."""
    print(f"📦 Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']['arch']
    
    # Create model config
    cfg = ModelCfg(
        d_model=config['d_model'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        n_kv_heads=config.get('n_kv_heads', config['n_heads']),
        d_head=config['d_model'] // config['n_heads'],
        vocab_size=config['vocab_size'],
        rope_theta=config.get('rope_theta', 10000.0),
        rope_scaling=config.get('rope_scaling', 'dynamic'),
        dropout=config.get('dropout', 0.0),
    )
    
    # Create model and load weights
    model = StudentLM(cfg)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded: {config['d_model']}d, {config['n_layers']} layers")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model state dict
    torch.save(model.state_dict(), output_dir / 'pytorch_model.bin')
    
    # Create config.json (simplified HuggingFace format)
    hf_config = {
        'architectures': ['StudentLM'],
        'model_type': 'llama',  # Use llama as base type for compatibility
        'vocab_size': config['vocab_size'],
        'hidden_size': config['d_model'],
        'num_hidden_layers': config['n_layers'],
        'num_attention_heads': config['n_heads'],
        'num_key_value_heads': config.get('n_kv_heads', config['n_heads']),
        'intermediate_size': config['d_model'] * 4,  # Estimate for SwiGLU
        'max_position_embeddings': config.get('max_seq_len', 2048),
        'rope_theta': config.get('rope_theta', 10000.0),
        'rope_scaling': config.get('rope_scaling', 'dynamic'),
        'torch_dtype': 'float32',
    }
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(hf_config, f, indent=2)
    
    print(f"✅ Exported to HuggingFace format: {output_dir}")
    return output_dir


def convert_to_gguf(hf_dir: Path, output_gguf: Path, llama_convert_path: str = None):
    """Convert HuggingFace format to GGUF using llama.cpp."""
    print(f"🔄 Converting to GGUF format...")
    
    # Find llama-convert if not provided
    if llama_convert_path is None:
        llama_convert_path = check_llama_cpp()
    
    if llama_convert_path is None:
        print("⚠️  llama.cpp conversion tool not found!")
        print()
        print("📋 To complete conversion, you have two options:")
        print()
        print("Option 1: Install llama.cpp via Homebrew")
        print("   brew install llama.cpp")
        print("   Then run this script again, or manually:")
        print(f"   llama-convert {hf_dir} --outfile {output_gguf} --outtype f16")
        print()
        print("Option 2: Use Python-based conversion (if available)")
        print("   pip install llama-cpp-python")
        print("   Then use their conversion utilities")
        print()
        print("Option 3: Manual conversion with llama.cpp source")
        print("   git clone https://github.com/ggerganov/llama.cpp.git")
        print("   cd llama.cpp")
        print("   make")
        print(f"   ./convert-hf-to-gguf.py {hf_dir} --outfile {output_gguf} --outtype f16")
        print()
        print(f"✅ HuggingFace format exported to: {hf_dir}")
        print("   You can convert this manually when llama.cpp is available.")
        return False
    
    print(f"   Using: {llama_convert_path}")
    
    # Handle Python-based conversion differently
    if 'python' in llama_convert_path or '-m' in llama_convert_path:
        # Python module-based conversion
        cmd = llama_convert_path.split() + [
            str(hf_dir),
            '--outfile', str(output_gguf),
            '--outtype', 'f16',
        ]
    else:
        # Binary-based conversion
        cmd = [
            llama_convert_path,
            str(hf_dir),
            '--outfile', str(output_gguf),
            '--outtype', 'f16',  # FP16 for smaller size
        ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        if output_gguf.exists():
            print(f"✅ GGUF conversion complete: {output_gguf}")
            print(f"   File size: {output_gguf.stat().st_size / (1024*1024):.2f} MB")
            return True
        else:
            print("⚠️  Conversion command succeeded but output file not found")
            print(f"   Expected: {output_gguf}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"❌ Conversion failed: {e}")
        if e.stdout:
            print(f"   stdout: {e.stdout}")
        if e.stderr:
            print(f"   stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"❌ Conversion tool not found: {llama_convert_path}")
        print("   Please verify llama.cpp is installed correctly")
        return False


def create_ollama_modelfile(gguf_path: Path, model_name: str, output_modelfile: Path):
    """Create an Ollama Modelfile for the GGUF model."""
    print(f"📝 Creating Ollama Modelfile...")
    
    modelfile_content = f"""FROM {gguf_path.absolute()}

# Magic 8 Ball Model - Hyper-optimized for M1 Mac
# Parameters: ~623K
# Training cost: $0.00003
# Performance: 1,090 tokens/sec, 1.22ms TTFT

PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER top_k 40

# System prompt for mystical responses
SYSTEM \"\"\"You are a Magic 8 Ball fortune-telling device. 
Respond to yes/no questions with mystical wisdom using classic Magic 8 Ball responses like:
- It is certain
- Outlook good
- Reply hazy, try again
- Very doubtful
Keep responses brief and mystical.\"\"\"

TEMPLATE \"\"\"{{{{ .System }}}}

{{{{ .Prompt }}}}\"\"\"
"""
    
    with open(output_modelfile, 'w') as f:
        f.write(modelfile_content)
    
    print(f"✅ Modelfile created: {output_modelfile}")
    print()
    print("📋 Next steps:")
    print(f"   1. ollama create {model_name} -f {output_modelfile}")
    print(f"   2. ollama run {model_name} \"Will this work?\"")
    print()


def main():
    ap = argparse.ArgumentParser(description="Convert Magic 8 Ball model to GGUF for Ollama")
    ap.add_argument('--checkpoint', required=True, help='Input PyTorch checkpoint path')
    ap.add_argument('--out', '--output', dest='output', required=True, help='Output GGUF file path')
    ap.add_argument('--model-name', default='magic-8-ball', help='Ollama model name')
    ap.add_argument('--llama-convert', help='Path to llama-convert binary (auto-detected if not provided)')
    ap.add_argument('--skip-gguf', action='store_true', help='Skip GGUF conversion (just create modelfile)')
    ap.add_argument('--skip-modelfile', action='store_true', help='Skip modelfile creation')
    args = ap.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    output_gguf = Path(args.output)
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    print("🎱 Magic 8 Ball → GGUF → Ollama Conversion 🎱")
    print("=" * 60)
    
    # Step 1: Export to HuggingFace format
    hf_dir = output_gguf.parent / f"{output_gguf.stem}_hf"
    export_to_huggingface_format(str(checkpoint_path), hf_dir)
    print()
    
    # Step 2: Convert to GGUF
    if not args.skip_gguf:
        success = convert_to_gguf(hf_dir, output_gguf, args.llama_convert)
        if not success:
            print("⚠️  GGUF conversion skipped or failed. You can manually convert later.")
            print()
    else:
        print("⏭️  Skipping GGUF conversion (--skip-gguf)")
        print()
    
    # Step 3: Create Ollama modelfile
    if not args.skip_modelfile and output_gguf.exists():
        modelfile_path = output_gguf.parent / f"Modelfile.{args.model_name}"
        create_ollama_modelfile(output_gguf, args.model_name, modelfile_path)
    elif args.skip_modelfile:
        print("⏭️  Skipping modelfile creation (--skip-modelfile)")
    
    print("🎉 Conversion pipeline complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
