# ARM Python Setup for Apple Silicon

This project requires ARM-native Python for proper PyTorch MPS (Metal Performance Shaders) support on Apple Silicon Macs.

## Current Setup

- **Python Version**: 3.11.13 (ARM Homebrew)
- **PyTorch**: 2.9.1 with MPS support enabled
- **Virtual Environment**: `venv/` (ARM-native)
- **Installation**: `/opt/homebrew/bin/python3.11 -m venv venv`

## Why ARM Python?

- **MPS Support**: Apple Silicon GPUs require ARM-native PyTorch with MPS backend
- **Performance**: ARM Python avoids x86_64 emulation overhead
- **Compatibility**: CoreML tools and other ML libraries work better with ARM Python
- **CUDA Issues**: x86 Python causes CUDA-related errors on Apple Silicon

## Verification

To verify the setup is correct:

```bash
source venv/bin/activate
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"
# Should print: MPS: True
```

## For Future Agents

**ALWAYS use ARM Python (`/opt/homebrew/bin/python3.11`) for this project.**

The README.md and .python-version file have been updated to reflect this requirement.
















