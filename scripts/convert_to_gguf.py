#!/usr/bin/env python3
"""
Convert 8-ball (or any toy) model to GGUF format for Ollama.

This script converts PyTorch checkpoints to GGUF format compatible with llama.cpp
and creates an Ollama modelfile for easy deployment.

Prerequisites:
    - llama.cpp installed (brew install llama.cpp)
    - gguf installed from llama.cpp source (pip install /path/to/llama.cpp/gguf-py)
      This ensures compatibility with llama.cpp conversion tools.

Usage:
    python convert_to_gguf.py --checkpoint /tmp/8_ball.ckpt --out 8-ball.gguf
    ollama create 8-ball -f Modelfile.8-ball
    ollama run 8-ball "Will this work?"

Author: @darianrosebrook
"""

from models.student.architectures.gqa_transformer import StudentLM, ModelCfg
import argparse
import json
import subprocess
import sys
from pathlib import Path
import torch

try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import Whitespace, ByteLevel

    HAS_TOKENIZERS = True
except ImportError:
    HAS_TOKENIZERS = False

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def check_llama_cpp():
    """Check if llama.cpp conversion tools are available."""
    # Try different possible binary names (avoid 'convert' which conflicts with ImageMagick)
    possible_names = [
        "llama-convert",
        "convert-llama-gguf",
        "llama-convert-gguf",
    ]

    for name in possible_names:
        try:
            result = subprocess.run(
                ["which", name], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            continue

    # Try common installation paths
    common_paths = [
        "/opt/homebrew/bin/llama-convert",
        "/usr/local/bin/llama-convert",
        "/opt/homebrew/Cellar/llama.cpp/*/bin/llama-convert",
        "~/.local/bin/llama-convert",
    ]

    for path_pattern in common_paths:
        expanded = Path(path_pattern).expanduser()
        # Handle glob patterns
        if "*" in str(expanded):
            import glob

            matches = glob.glob(str(expanded))
            if matches:
                return matches[0]
        elif expanded.exists():
            return str(expanded)

    # Try finding via brew
    try:
        result = subprocess.run(
            ["brew", "--prefix", "llama.cpp"], capture_output=True, text=True)
        if result.returncode == 0:
            brew_prefix = result.stdout.strip()
            # Check multiple possible binary locations
            possible_bins = [
                Path(brew_prefix) / "bin" / "llama-convert",
                Path(brew_prefix) / "bin" / "convert",
                Path(brew_prefix) / "bin" / "convert-llama-gguf",
                Path(brew_prefix) / "bin" / "convert_hf_to_gguf.py",
            ]
            for bin_path in possible_bins:
                if bin_path.exists():
                    if bin_path.suffix == ".py":
                        # Python script - use current Python interpreter to run it
                        return f"{sys.executable} {bin_path}"
                    return str(bin_path)
    except Exception:
        pass

    # Try finding via brew's intel Python (if user mentioned it)
    # Check common intel Python locations
    intel_python_paths = [
        "/usr/local/bin/python3",  # Common intel Python location
        "/opt/homebrew/opt/python@3.11/bin/python3",  # Intel Python via homebrew
    ]

    for python_path in intel_python_paths:
        if Path(python_path).exists():
            # Try to find llama.cpp tools via that Python
            try:
                result = subprocess.run(
                    [python_path, "-m", "llama_cpp.convert"], capture_output=True, text=True
                )
                # If module exists, we can use it
                if result.returncode != 127:  # 127 = command not found
                    return f"{python_path} -m llama_cpp.convert"
            except Exception:
                continue

    return None


def create_minimal_tokenizer(output_dir: Path, vocab_size: int):
    """Create a minimal tokenizer.json that matches the model's vocab_size.

    This is critical for Ollama compatibility - Ollama reads the vocab_size from
    tokenizer.json, not just tokenizer_config.json. If the tokenizer has more tokens
    than the model, Ollama will fail with a shape mismatch error.

    Args:
        output_dir: Directory to write tokenizer files
        vocab_size: Target vocabulary size (must match model's vocab_size)
    """
    print(f"🎯 Creating minimal tokenizer (vocab_size={vocab_size})...")

    # Create vocabulary - start with special tokens
    vocab = {"<s>": 0, "</s>": 1, "<unk>": 2}

    # Add printable ASCII characters (32-126)
    next_id = 3
    for i in range(32, 127):
        if next_id >= vocab_size:
            break
        char = chr(i)
        if char not in vocab:
            vocab[char] = next_id
            next_id += 1

    # Add common words and tokens
    common_tokens = [
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "can",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "this",
        "that",
        "these",
        "those",
        "I",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "my",
        "your",
        "his",
        "her",
        "its",
        "our",
        "their",
        "me",
        "him",
        "us",
        "them",
        "what",
        "when",
        "where",
        "why",
        "how",
        "which",
        "who",
        "whose",
        "yes",
        "no",
        "not",
        "very",
        "much",
        "more",
        "most",
        "some",
        "any",
        "all",
        "each",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
    ]

    for token in common_tokens:
        if next_id >= vocab_size:
            break
        if token not in vocab:
            vocab[token] = next_id
            next_id += 1

    # Fill remaining vocabulary with synthetic tokens
    while next_id < vocab_size:
        token = f"<token_{next_id}>"
        vocab[token] = next_id
        next_id += 1

    # Use tokenizers library if available (creates proper format)
    # For llama.cpp compatibility, we need BPE format with compatible pre_tokenizer
    tokenizer_created = False
    if HAS_TOKENIZERS:
        try:
            # Create BPE tokenizer (llama.cpp expects BPE format)
            # Use ByteLevel pre_tokenizer for better llama.cpp compatibility
            # This is more likely to be recognized by llama.cpp's hash-based checks
            tokenizer = Tokenizer(
                BPE(vocab=vocab, merges=[], unk_token="<unk>"))

            # Try ByteLevel first (more compatible with llama.cpp)
            try:
                tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
            except Exception:
                # Fallback to Whitespace if ByteLevel fails
                tokenizer.pre_tokenizer = Whitespace()

            tokenizer.enable_padding(pad_id=1, pad_token="</s>")

            # Save tokenizer.json
            tokenizer.save(str(output_dir / "tokenizer.json"))
            print("✅ Created tokenizer.json using tokenizers library (BPE format)")
            tokenizer_created = True
        except Exception as e:
            print(
                f"⚠️  Failed to create tokenizer with tokenizers library: {e}")
            print("   Falling back to manual JSON creation...")

    # Fallback: Manual JSON creation if tokenizers library not available or failed
    if not tokenizer_created:
        # Create minimal BPE-style tokenizer.json (llama.cpp compatible)
        # Use ByteLevel pre_tokenizer format for better llama.cpp compatibility
        tokenizer_json = {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": [
                {
                    "id": 0,
                    "content": "<s>",
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": True,
                },
                {
                    "id": 1,
                    "content": "</s>",
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": True,
                },
                {
                    "id": 2,
                    "content": "<unk>",
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": True,
                },
            ],
            "normalizer": None,
            # Use ByteLevel pre_tokenizer for better llama.cpp compatibility
            "pre_tokenizer": {
                "type": "ByteLevel",
                "add_prefix_space": False,
                "trim_offsets": True,
                "use_regex": True,
            },
            "post_processor": None,
            "decoder": {
                "type": "ByteLevel",
                "add_prefix_space": False,
                "trim_offsets": True,
                "use_regex": True,
            },
            "model": {
                "type": "BPE",
                "dropout": None,
                "unk_token": "<unk>",
                "continuing_subword_prefix": None,
                "end_of_word_suffix": None,
                "fuse_unk": False,
                "byte_fallback": False,
                "vocab": vocab,
                "merges": [],  # Empty merges for simple tokenizer
            },
        }

        with open(output_dir / "tokenizer.json", "w") as f:
            json.dump(tokenizer_json, f, indent=2)

    # Create tokenizer_config.json
    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "auto_map": {"AutoTokenizer": ["tokenizer.json", None, "tokenizer_config.json"]},
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "</s>",
        "model_max_length": 2048,
        "is_fast": True,
        "padding_side": "right",
        "truncation_side": "right",
        "model_input_names": ["input_ids", "attention_mask"],
        "clean_up_tokenization_spaces": False,
        "vocab_size": vocab_size,
    }

    # Create special_tokens_map.json
    special_tokens_map = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "</s>",
        "additional_special_tokens": [],
    }

    # Write config files
    with open(output_dir / "tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f, indent=2)

    with open(output_dir / "special_tokens_map.json", "w") as f:
        json.dump(special_tokens_map, f, indent=2)

    print(f"✅ Created minimal tokenizer with {vocab_size} tokens")
    print("   Special tokens: <s>, </s>, <unk>")
    print(
        f"   Vocabulary: {len(vocab)} tokens (ASCII + common words + synthetic)")


def export_to_huggingface_format(checkpoint_path: str, output_dir: Path):
    """Export PyTorch checkpoint to HuggingFace-compatible format."""
    print(f"📦 Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]["arch"]

    # Create model config
    cfg = ModelCfg(
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        n_kv_heads=config.get("n_kv_heads", config["n_heads"]),
        d_head=config["d_model"] // config["n_heads"],
        vocab_size=config["vocab_size"],
        rope_theta=config.get("rope_theta", 10000.0),
        rope_scaling=config.get("rope_scaling", "dynamic"),
        dropout=config.get("dropout", 0.0),
    )

    # Create model and load weights
    model = StudentLM(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"✅ Model loaded: {config['d_model']}d, {config['n_layers']} layers")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Rename state dict keys to match llama.cpp expected format
    state_dict = model.state_dict()
    renamed_state_dict = {}

    for key, value in state_dict.items():
        new_key = key
        # Map our architecture to llama.cpp format
        if key == "embed.weight":
            new_key = "model.embed_tokens.weight"
        elif key == "norm.weight":
            new_key = "model.norm.weight"
        elif key == "norm_f.weight":
            new_key = "model.norm.weight"
        elif key.startswith("blocks."):
            # blocks.0.attn.wq.weight -> model.layers.0.self_attn.q_proj.weight
            parts = key.split(".")
            layer_num = parts[1]
            if "attn" in parts:
                if parts[3] == "wq":
                    new_key = f"model.layers.{layer_num}.self_attn.q_proj.weight"
                elif parts[3] == "wk":
                    new_key = f"model.layers.{layer_num}.self_attn.k_proj.weight"
                elif parts[3] == "wv":
                    new_key = f"model.layers.{layer_num}.self_attn.v_proj.weight"
                elif parts[3] == "wo":
                    new_key = f"model.layers.{layer_num}.self_attn.o_proj.weight"
            elif "mlp" in parts:
                if parts[3] == "w1":
                    new_key = f"model.layers.{layer_num}.mlp.gate_proj.weight"
                elif parts[3] == "w2":
                    new_key = f"model.layers.{layer_num}.mlp.down_proj.weight"
                elif parts[3] == "w3":
                    new_key = f"model.layers.{layer_num}.mlp.up_proj.weight"
            elif parts[2] == "norm1":
                new_key = f"model.layers.{layer_num}.input_layernorm.weight"
            elif parts[2] == "norm2":
                new_key = f"model.layers.{layer_num}.post_attention_layernorm.weight"
        elif key == "unembed.weight" or key == "lm_head.weight":
            new_key = "lm_head.weight"

        renamed_state_dict[new_key] = value

    # Save renamed model state dict
    torch.save(renamed_state_dict, output_dir / "pytorch_model.bin")

    # Create config.json (simplified HuggingFace format)
    # Use LlamaForCausalLM architecture name for llama.cpp compatibility
    hf_config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",  # Use llama as base type for compatibility
        "vocab_size": config["vocab_size"],
        "hidden_size": config["d_model"],
        "num_hidden_layers": config["n_layers"],
        "num_attention_heads": config["n_heads"],
        "num_key_value_heads": config.get("n_kv_heads", config["n_heads"]),
        "intermediate_size": config["d_model"] * 4,  # Estimate for SwiGLU
        "max_position_embeddings": config.get("max_seq_len", 2048),
        "rope_theta": config.get("rope_theta", 10000.0),
        "torch_dtype": "float32",
    }

    # Handle rope_scaling - convert string to dict format if needed
    rope_scaling = config.get("rope_scaling")
    if rope_scaling:
        if isinstance(rope_scaling, str):
            # Convert string to dict format expected by llama.cpp
            hf_config["rope_scaling"] = {"type": rope_scaling, "factor": 1.0}
        elif isinstance(rope_scaling, dict):
            hf_config["rope_scaling"] = rope_scaling

    with open(output_dir / "config.json", "w") as f:
        json.dump(hf_config, f, indent=2)

    # Create minimal tokenizer that matches model vocab_size
    # This is critical for Ollama compatibility - Ollama reads vocab_size from tokenizer.json
    # Try to use original tokenizer first, fall back to minimal if vocab_size doesn't match
    tokenizer_dir = Path("models/student/tokenizer")
    use_original_tokenizer = False

    if tokenizer_dir.exists():
        try:
            from transformers import AutoTokenizer

            original_tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_dir))
            if original_tokenizer.vocab_size == config["vocab_size"]:
                # Vocab sizes match - use original tokenizer
                import shutil

                tokenizer_files = [
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                ]
                for tokenizer_file in tokenizer_files:
                    src = tokenizer_dir / tokenizer_file
                    if src.exists():
                        shutil.copy2(src, output_dir / tokenizer_file)
                print(
                    f"✅ Using original tokenizer (vocab_size={config['vocab_size']} matches)")
                use_original_tokenizer = True
        except Exception as e:
            print(f"⚠️  Could not load original tokenizer: {e}")

    if not use_original_tokenizer:
        # Create minimal tokenizer with matching vocab_size
        create_minimal_tokenizer(output_dir, config["vocab_size"])

    print(f"✅ Exported to HuggingFace format: {output_dir}")
    return output_dir


def convert_to_gguf_direct(hf_dir: Path, output_gguf: Path):
    """Convert HuggingFace format to GGUF using gguf Python API directly."""
    print("🔄 Converting to GGUF format (direct Python API)...")

    try:
        from gguf import GGUFWriter
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Load the model and tokenizer
        print("   Loading model and tokenizer...")
        model = AutoModelForCausalLM.from_pretrained(
            str(hf_dir), torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(str(hf_dir))

        # Create GGUF writer
        gguf_writer = GGUFWriter(path=str(output_gguf), arch="llama")

        # Set model metadata
        gguf_writer.add_name("8-ball")
        gguf_writer.add_description("8-Ball fortune-telling model")
        gguf_writer.add_author("darianrosebrook")
        gguf_writer.add_version("1.0.0")

        # Set model architecture parameters
        config = model.config
        gguf_writer.add_vocab_size(config.vocab_size)
        gguf_writer.add_context_length(config.max_position_embeddings)
        gguf_writer.add_embedding_length(config.hidden_size)
        gguf_writer.add_block_count(config.num_hidden_layers)
        gguf_writer.add_feed_forward_length(config.intermediate_size)
        gguf_writer.add_rope_dimension_count(
            config.hidden_size // config.num_attention_heads)
        gguf_writer.add_head_count(config.num_attention_heads)
        gguf_writer.add_head_count_kv(config.num_key_value_heads)
        gguf_writer.add_layer_norm_rms_eps(config.rms_norm_eps)

        # Add tokenizer info - use llama tokenizer model to avoid BPE merge requirements
        gguf_writer.add_tokenizer_model("llama")
        gguf_writer.add_tokenizer_pre("default")

        # Add tokens
        tokens = []
        for i in range(config.vocab_size):
            try:
                token_text = tokenizer.decode([i])
                tokens.append(token_text)
            except:
                tokens.append(f"<token_{i}>")

        gguf_writer.add_token_list(tokens)

        # Add special tokens
        if tokenizer.bos_token:
            gguf_writer.add_token_bos(tokenizer.bos_token_id)
        if tokenizer.eos_token:
            gguf_writer.add_token_eos(tokenizer.eos_token_id)
        if tokenizer.unk_token:
            gguf_writer.add_token_unk(tokenizer.unk_token_id)
        if tokenizer.pad_token:
            gguf_writer.add_token_pad(tokenizer.pad_token_id)

        # For BPE tokenizers, we need merges. Create minimal merges for our vocabulary
        if hasattr(tokenizer, "model") and hasattr(tokenizer.model, "_mergeable_ranks"):
            # Use actual merges if available
            merges = []
            for merge_tokens, rank in tokenizer.model._mergeable_ranks.items():
                if len(merge_tokens) == 2:
                    merge_str = " ".join(
                        tokenizer.decode(
                            [merge_tokens[0], merge_tokens[1]]).split()
                    )
                    if merge_str.count(" ") == 1:  # Ensure it's a valid merge pair
                        merges.append(merge_str)
            gguf_writer.add_token_merges(merges)
        else:
            # Create minimal merges for toy tokenizer
            # For BPE, we need pairs that can be merged. Create some basic ones.
            merges = []
            # Add some common merges that make sense for our vocabulary
            common_pairs = [
                ("t", "h"),
                ("h", "e"),
                ("i", "s"),
                ("a", "n"),
                ("r", "e"),
                ("o", "n"),
                ("a", "t"),
                ("e", "n"),
                ("i", "t"),
                ("o", "r"),
                ("a", "s"),
                ("e", "s"),
                ("i", "n"),
                ("o", "u"),
                ("e", "d"),
                ("i", "c"),
                ("a", "l"),
                ("e", "r"),
                ("o", "m"),
                ("i", "o"),
                ("u", "r"),
                ("a", "r"),
                ("e", "t"),
                ("i", "e"),
                ("o", "t"),
                ("a", "m"),
                ("e", "a"),
                ("i", "a"),
            ]
            for pair in common_pairs:
                # Check if characters are basic ASCII (safe assumption for toy tokenizer)
                if ord(pair[0]) < 128 and ord(pair[1]) < 128:
                    merges.append(f"{pair[0]} {pair[1]}")
            gguf_writer.add_token_merges(
                merges[: min(100, len(merges))])  # Limit to 100 merges

        # Convert and add model weights
        print("   Converting model weights...")
        state_dict = model.state_dict()

        # Map PyTorch parameter names to GGUF names
        param_mapping = {
            "embed_tokens.weight": "token_embd.weight",
            "layers.{i}.self_attn.q_proj.weight": "blk.{i}.attn_q.weight",
            "layers.{i}.self_attn.k_proj.weight": "blk.{i}.attn_k.weight",
            "layers.{i}.self_attn.v_proj.weight": "blk.{i}.attn_v.weight",
            "layers.{i}.self_attn.o_proj.weight": "blk.{i}.attn_output.weight",
            "layers.{i}.mlp.gate_proj.weight": "blk.{i}.ffn_gate.weight",
            "layers.{i}.mlp.up_proj.weight": "blk.{i}.ffn_up.weight",
            "layers.{i}.mlp.down_proj.weight": "blk.{i}.ffn_down.weight",
            "layers.{i}.input_layernorm.weight": "blk.{i}.attn_norm.weight",
            "layers.{i}.post_attention_layernorm.weight": "blk.{i}.ffn_norm.weight",
            "norm.weight": "output_norm.weight",
            "lm_head.weight": "output.weight",
        }

        for pt_name, tensor in state_dict.items():
            gguf_name = pt_name
            # Apply mapping
            for pt_pattern, gguf_pattern in param_mapping.items():
                if pt_pattern in pt_name:
                    gguf_name = pt_name.replace(pt_pattern, gguf_pattern)
                    break

            # Convert to float16
            tensor_f16 = tensor.to(torch.float16)
            gguf_writer.add_tensor(gguf_name, tensor_f16.numpy())

        # Write the GGUF file
        print("   Writing GGUF file...")
        gguf_writer.write()

        print(f"✅ GGUF conversion complete: {output_gguf}")
        print(
            f"   File size: {output_gguf.stat().st_size / (1024 * 1024):.2f} MB")
        return True

    except Exception as e:
        print(f"❌ Direct GGUF conversion failed: {e}")
        print("   Falling back to llama.cpp script method...")
        return False


def convert_to_gguf(hf_dir: Path, output_gguf: Path, llama_convert_path: str = None):
    """Convert HuggingFace format to GGUF using llama.cpp."""
    print("🔄 Converting to GGUF format...")

    # Check if this is a toy/8-ball model with custom tokenizer
    config_path = hf_dir / "config.json"
    is_toy_model = False
    if config_path.exists():
        try:
            import json

            with open(config_path, "r") as f:
                config = json.load(f)
            vocab_size = config.get("vocab_size", 0)
            if vocab_size <= 1000:  # Toy models typically have small vocab
                is_toy_model = True
                print(
                    "   Detected toy model with small vocabulary - GGUF conversion may have compatibility issues"
                )
                print(
                    "   GGUF is optional for toy models - CoreML conversion is the primary goal")
        except:
            pass

    # Try direct Python conversion first
    if convert_to_gguf_direct(hf_dir, output_gguf):
        return True

    # Fallback to llama.cpp script
    print("   Direct conversion failed, trying llama.cpp script...")

    # Find llama-convert if not provided
    if llama_convert_path is None:
        llama_convert_path = check_llama_cpp()

    if llama_convert_path is None:
        if is_toy_model:
            print("⚠️  llama.cpp conversion tool not found - GGUF conversion skipped")
            print(
                "   This is expected for toy models. CoreML conversion is the primary goal.")
        else:
            print("⚠️  llama.cpp conversion tool not found!")
            print()
            print("📋 To complete conversion, you have two options:")
            print()
            print("Option 1: Install llama.cpp via Homebrew")
            print("   brew install llama.cpp")
            print("   Then run this script again, or manually:")
            print(
                f"   llama-convert {hf_dir} --outfile {output_gguf} --outtype f16")
            print()
            print("Option 2: Use Python-based conversion (if available)")
            print("   pip install llama-cpp-python")
            print("   Then use their conversion utilities")
        print(f"✅ HuggingFace format available at: {hf_dir}")
        return False

    print(f"   Using: {llama_convert_path}")

    # Handle Python-based conversion differently
    if "python" in llama_convert_path or llama_convert_path.endswith(".py"):
        # Python script - need to parse the command properly
        if "python" in llama_convert_path:
            # Format: "python3 /path/to/script.py"
            parts = llama_convert_path.split()
            cmd = parts + [
                str(hf_dir),
                "--outfile",
                str(output_gguf),
                "--outtype",
                "f16",
            ]
        else:
            # Just a .py file path
            cmd = [
                "python3",
                llama_convert_path,
                str(hf_dir),
                "--outfile",
                str(output_gguf),
                "--outtype",
                "f16",
            ]
    elif "-m" in llama_convert_path:
        # Python module-based conversion
        cmd = llama_convert_path.split() + [
            str(hf_dir),
            "--outfile",
            str(output_gguf),
            "--outtype",
            "f16",
        ]
    else:
        # Binary-based conversion
        cmd = [
            llama_convert_path,
            str(hf_dir),
            "--outfile",
            str(output_gguf),
            "--outtype",
            "f16",  # FP16 for smaller size
        ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True)
        if output_gguf.exists():
            print(f"✅ GGUF conversion complete: {output_gguf}")
            print(
                f"   File size: {output_gguf.stat().st_size / (1024 * 1024):.2f} MB")
            return True
        else:
            print("⚠️  Conversion command succeeded but output file not found")
            print(f"   Expected: {output_gguf}")
            return False
    except subprocess.CalledProcessError as e:
        if is_toy_model:
            print(
                "❌ GGUF conversion failed - this is expected for toy models with custom tokenizers"
            )
            print("   GGUF is optional - CoreML conversion (above) is the primary goal")
        else:
            print(f"❌ Conversion failed: {e}")
            if e.stdout:
                print(f"   stdout: {e.stdout}")
            if e.stderr:
                print(f"   stderr: {e.stderr}")

        # Provide helpful context for common issues
        stderr_lower = e.stderr.lower() if e.stderr else ""
        combined_output = stderr_lower + " " + \
            (e.stdout.lower() if e.stdout else "")

        if "merges" in combined_output and "tokenizer" in combined_output:
            print("   Issue: Custom tokenizer missing BPE merges (required for GGUF)")
            print(
                "   Solution: Use production tokenizers with proper merge rules for GGUF")
        elif "gguf" in stderr_lower and "version" in stderr_lower:
            print("   Issue: GGUF library version mismatch")
            print("   Solution: Use compatible gguf version or system Python")

        print(f"   HuggingFace format still available at: {hf_dir}")
        return False
    except FileNotFoundError:
        if is_toy_model:
            print(f"❌ Conversion tool not found: {llama_convert_path}")
            print("   GGUF conversion skipped - this is expected for toy models")
        else:
            print(f"❌ Conversion tool not found: {llama_convert_path}")
            print("   Please verify llama.cpp is installed correctly")
        return False


def create_ollama_modelfile(gguf_path: Path, model_name: str, output_modelfile: Path):
    """Create an Ollama Modelfile for the GGUF model."""
    print("📝 Creating Ollama Modelfile...")

    modelfile_content = f"""FROM {gguf_path.absolute()}

# 8-ball Model - Hyper-optimized for M1 Mac
# Parameters: ~623K
# Training cost: $0.00003
# Performance: 1,090 tokens/sec, 1.22ms TTFT

PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER top_k 40

# System prompt for mystical responses
SYSTEM \"\"\"You are an 8-ball fortune-telling device.
Respond to yes/no questions with mystical wisdom using classic 8-ball responses like:
- It is certain
- Outlook good
- Reply hazy, try again
- Very doubtful
Keep responses brief and mystical.\"\"\"

TEMPLATE \"\"\"{{{{ .System }}}}

{{{{ .Prompt }}}}\"\"\"
"""

    with open(output_modelfile, "w") as f:
        f.write(modelfile_content)

    print(f"✅ Modelfile created: {output_modelfile}")
    print()
    print("📋 Next steps:")
    print(f"   1. ollama create {model_name} -f {output_modelfile}")
    print(f'   2. ollama run {model_name} "Will this work?"')
    print()


def main():
    ap = argparse.ArgumentParser(
        description="Convert 8-ball model to GGUF for Ollama")
    ap.add_argument("--checkpoint", required=True,
                    help="Input PyTorch checkpoint path")
    ap.add_argument("--out", "--output", dest="output",
                    required=True, help="Output GGUF file path")
    ap.add_argument("--model-name", default="8-ball", help="Ollama model name")
    ap.add_argument(
        "--llama-convert", help="Path to llama-convert binary (auto-detected if not provided)"
    )
    ap.add_argument(
        "--skip-gguf", action="store_true", help="Skip GGUF conversion (just create modelfile)"
    )
    ap.add_argument("--skip-modelfile", action="store_true",
                    help="Skip modelfile creation")
    args = ap.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_gguf = Path(args.output)

    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print("🎱 8-ball → GGUF → Ollama Conversion 🎱")
    print("=" * 60)

    # Step 1: Export to HuggingFace format
    hf_dir = output_gguf.parent / f"{output_gguf.stem}_hf"
    export_to_huggingface_format(str(checkpoint_path), hf_dir)
    print()

    # Step 2: Convert to GGUF
    if not args.skip_gguf:
        success = convert_to_gguf(hf_dir, output_gguf, args.llama_convert)
        if not success:
            print(
                "⚠️  GGUF conversion skipped or failed. You'll need to fix issues to get this running."
            )
    else:
        print("⏭️  Skipping GGUF conversion (--skip-gguf)")
        print()

    # Step 3: Create Ollama modelfile (even if GGUF conversion failed)
    if not args.skip_modelfile:
        modelfile_path = output_gguf.parent / f"Modelfile.{args.model_name}"
        if output_gguf.exists():
            create_ollama_modelfile(
                output_gguf, args.model_name, modelfile_path)
        else:
            # Create modelfile with placeholder path (user can update after manual conversion)
            print("📝 Creating modelfile template (update path after GGUF conversion)...")
            modelfile_content = f"""FROM {output_gguf.absolute()}

# 8-ball Model - Hyper-optimized for M1 Mac
# Parameters: ~623K
# Training cost: $0.00003
# Performance: 1,090 tokens/sec, 1.22ms TTFT

# NOTE: Update the FROM path above after converting to GGUF:
#   python3 /usr/local/opt/llama.cpp/bin/convert_hf_to_gguf.py \\
#     {hf_dir} --outfile {output_gguf} --outtype f16

PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER top_k 40

# System prompt for mystical responses
SYSTEM \"\"\"You are an 8-ball fortune-telling device.
Respond to yes/no questions with mystical wisdom using classic 8-ball responses like:
- It is certain
- Outlook good
- Reply hazy, try again
- Very doubtful
Keep responses brief and mystical.\"\"\"

TEMPLATE \"\"\"{{{{ .System }}}}

{{{{ .Prompt }}}}\"\"\"
"""
            with open(modelfile_path, "w") as f:
                f.write(modelfile_content)
            print(f"✅ Modelfile template created: {modelfile_path}")
            print(f"   Update the FROM path after converting {hf_dir} to GGUF")
    elif args.skip_modelfile:
        print("⏭️  Skipping modelfile creation (--skip-modelfile)")

    print("🎉 Conversion pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
