#!/usr/bin/env python3
"""
Download required models and tokenizers to external/ folder.

This script downloads:
1. Tokenizer (vocab_size=32000) - Llama-2 or Mistral compatible
2. Optional: Teacher model for local testing (if not using HTTP endpoint)

All downloads go to external/ folder which is git-ignored.
"""
import argparse
import sys
from pathlib import Path

try:
    from transformers import AutoTokenizer
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("ERROR: transformers library not available. Install with: pip install transformers")
    sys.exit(1)


def download_tokenizer(model_id: str, output_dir: Path, verify_vocab_size: int = 32000):
    """Download tokenizer from HuggingFace."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[download_models] Downloading tokenizer: {model_id}")
    print(f"[download_models] Output directory: {output_dir}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Verify vocab size
        vocab_size = len(tokenizer)
        if vocab_size != verify_vocab_size:
            print(f"WARNING: Tokenizer vocab_size={vocab_size} does not match expected {verify_vocab_size}")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Aborted.")
                return False
        
        # Save tokenizer
        tokenizer.save_pretrained(str(output_dir))
        print(f"✅ Tokenizer saved to {output_dir}")
        print(f"   Vocab size: {vocab_size}")
        print(f"   Model type: {type(tokenizer).__name__}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to download tokenizer: {e}")
        return False


def main():
    ap = argparse.ArgumentParser(
        description="Download required models and tokenizers to external/ folder"
    )
    ap.add_argument(
        '--tokenizer',
        default='meta-llama/Llama-2-7b-hf',
        help='Tokenizer model ID from HuggingFace (default: meta-llama/Llama-2-7b-hf)'
    )
    ap.add_argument(
        '--output-dir',
        default='external/models',
        help='Output directory for downloads (default: external/models)'
    )
    ap.add_argument(
        '--vocab-size',
        type=int,
        default=32000,
        help='Expected vocab size for verification (default: 32000)'
    )
    ap.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing tokenizer, do not download'
    )
    
    args = ap.parse_args()
    
    output_dir = Path(args.output_dir)
    tokenizer_dir = output_dir / 'tokenizer'
    
    print("=" * 60)
    print("Model Download Script")
    print("=" * 60)
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Output directory: {output_dir}")
    print(f"Expected vocab size: {args.vocab_size}")
    print("=" * 60)
    
    if args.verify_only:
        # Check if tokenizer exists at expected location
        expected_path = Path('models/student/tokenizer')
        if expected_path.exists():
            print(f"[verify] Checking existing tokenizer at: {expected_path}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(str(expected_path))
                vocab_size = len(tokenizer)
                print(f"✅ Tokenizer found: vocab_size={vocab_size}")
                if vocab_size == args.vocab_size:
                    print(f"✅ Vocab size matches expected ({args.vocab_size})")
                else:
                    print(f"⚠️  Vocab size mismatch: expected {args.vocab_size}, got {vocab_size}")
                return 0
            except Exception as e:
                print(f"❌ Failed to load tokenizer: {e}")
                return 1
        else:
            print(f"❌ Tokenizer not found at: {expected_path}")
            return 1
    else:
        # Download tokenizer
        success = download_tokenizer(args.tokenizer, tokenizer_dir, args.vocab_size)
        
        if success:
            print("\n" + "=" * 60)
            print("✅ Download complete!")
            print("=" * 60)
            print(f"\nTokenizer location: {tokenizer_dir}")
            print("\nTo use this tokenizer, update your config:")
            print("  io:")
            print(f"    tokenizer_path: \"{tokenizer_dir.absolute()}\"")
            print("\nOr create a symlink:")
            print(f"  ln -s {tokenizer_dir.absolute()} models/student/tokenizer")
            return 0
        else:
            print("\n❌ Download failed!")
            return 1


if __name__ == '__main__':
    sys.exit(main())


