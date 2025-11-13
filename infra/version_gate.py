"""
Version gates: Check Python, macOS, and dependency versions before proceeding.

Version Gate Strategy:
----------------------

This module enforces version requirements for production model export and conversion.
The gates ensure compatibility with TorchScript export and CoreML conversion, which
have strict Python version requirements (3.10 or 3.11).

Production Models:
    - Version gates MUST be enforced for all production model exports/conversions
    - Ensures compatibility with PyTorch TorchScript and CoreML toolchains
    - Prevents deployment of models that may fail in production environments

Toy Models:
    - Version gates can be bypassed using `--toy` flag in export/conversion scripts
    - Allows faster testing with any Python version (including 3.13+)
    - Bypass is explicit and intentional - never bypass for production models

Bypass Mechanism:
    - Export scripts (`conversion/export_pytorch.py`) accept `--toy` flag
    - Conversion scripts (`conversion/convert_coreml.py`) accept `--toy` flag
    - When `--toy` is present, version checks are skipped entirely
    - This allows toy model testing without Python version constraints

Why Version Gates Exist:
    - TorchScript export requires Python 3.10 or 3.11 (PyTorch limitation)
    - CoreML conversion has compatibility requirements with specific PyTorch versions
    - Production deployments need predictable, tested environments
    - Version mismatches can cause silent failures or incorrect model behavior

When to Bypass:
    - ONLY for toy/test models during development
    - NEVER for production model exports
    - Use `--toy` flag explicitly when testing pipeline with toy checkpoints
    - Document any bypass usage in commit messages or PR notes

See Also:
    - `docs/DEPLOYMENT.md` - Full deployment guide with version requirements
    - `docs/PRODUCTION_PIPELINE_CHECKLIST.md` - Pre-deployment checklist
    - `conversion/export_pytorch.py` - Export implementation with bypass support
    - `conversion/convert_coreml.py` - CoreML conversion with bypass support
"""

import sys
import platform
from typing import Tuple, Optional


def check_python_version() -> Tuple[int, int]:
    """Require Python 3.10 or 3.11.

    Returns:
        Tuple of (major, minor) version numbers.

    Raises:
        RuntimeError: If Python version is not supported.
    """
    major, minor = sys.version_info[:2]
    if (major, minor) not in [(3, 10), (3, 11)]:
        raise RuntimeError(
            f"Python {major}.{minor} detected. "
            "This project requires Python 3.10 or 3.11. "
            f"Current: {sys.version}"
        )
    return major, minor


def check_macos_version() -> Optional[str]:
    """Check macOS version (optional, warn only).

    Returns:
        macOS version string, or None if not available.
    """
    if platform.system() != "Darwin":
        return None

    try:
        macos_version = platform.mac_ver()[0]
        major = int(macos_version.split(".")[0])
        if major < 13:
            print(f"WARNING: macOS {macos_version} detected. CoreML requires macOS 13+")
        return macos_version
    except Exception:
        return None


def check_coremltools() -> str:
    """Check coremltools version.

    Returns:
        Version string of coremltools.

    Raises:
        RuntimeError: If coremltools is not installed or version is too old.
    """
    try:
        import coremltools as ct

        version = ct.__version__
        major = int(version.split(".")[0])
        if major < 9:
            raise RuntimeError(
                f"coremltools {version} detected. This project requires coremltools >= 9.0"
            )
        return version
    except ImportError:
        raise RuntimeError("coremltools not installed. Install with: pip install coremltools>=9.0")


def check_pytorch() -> str:
    """Check PyTorch version.

    Returns:
        Version string of PyTorch.

    Raises:
        RuntimeError: If PyTorch is not installed or version is too old.
    """
    try:
        import torch

        version = torch.__version__
        # Parse version: "2.3.0" -> (2, 3)
        parts = version.split(".")
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        if major < 2 or (major == 2 and minor < 0):
            raise RuntimeError(
                f"PyTorch {version} detected. "
                "This project requires PyTorch >= 2.0. "
                "For best compatibility, use PyTorch >= 2.3.0"
            )
        return version
    except ImportError:
        raise RuntimeError("PyTorch not installed. Install with: pip install torch>=2.0")


def check_transformers() -> str:
    """Check HuggingFace Transformers version.

    Returns:
        Version string of transformers.

    Raises:
        RuntimeError: If transformers is not installed or version is too old.
    """
    try:
        import transformers

        version = transformers.__version__
        # Parse version: "4.43.0" -> (4, 43)
        parts = version.split(".")
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        if major < 4 or (major == 4 and minor < 43):
            raise RuntimeError(
                f"transformers {version} detected. "
                "This project requires transformers >= 4.43. "
                f"Current: {version}"
            )
        return version
    except ImportError:
        raise RuntimeError(
            "transformers not installed. Install with: pip install transformers>=4.43"
        )


def check_accelerate() -> str:
    """Check HuggingFace Accelerate version.

    Returns:
        Version string of accelerate.

    Raises:
        RuntimeError: If accelerate is not installed or version is too old.
    """
    try:
        import accelerate

        version = accelerate.__version__
        # Parse version: "0.33.0" -> (0, 33)
        parts = version.split(".")
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        if major < 0 or (major == 0 and minor < 33):
            raise RuntimeError(
                f"accelerate {version} detected. "
                "This project requires accelerate >= 0.33. "
                f"Current: {version}"
            )
        return version
    except ImportError:
        raise RuntimeError("accelerate not installed. Install with: pip install accelerate>=0.33")


def check_onnxruntime(required: bool = False) -> Optional[str]:
    """Check if onnxruntime is available (optional for smoke tests).

    Args:
        required: If True, raise error if not available.

    Returns:
        Version string if available, None otherwise.

    Raises:
        RuntimeError: If required=True and onnxruntime is not installed.
    """
    try:
        import onnxruntime as ort

        return ort.__version__
    except ImportError:
        if required:
            raise RuntimeError(
                "onnxruntime not installed. "
                "For parity tests, install with: pip install onnxruntime or onnxruntime-silicon"
            )
        return None


def check_training_versions() -> dict:
    """Check versions required for training.

    Returns:
        Dictionary with check results.

    Raises:
        RuntimeError: If any required dependency is missing or version is incompatible.
    """
    results = {
        "python": check_python_version(),
        "pytorch": check_pytorch(),
        "transformers": check_transformers(),
        "accelerate": check_accelerate(),
    }
    return results


def check_export_versions() -> dict:
    """Check versions required for PyTorch export.

    Returns:
        Dictionary with check results.

    Raises:
        RuntimeError: If any required dependency is missing or version is incompatible.
    """
    results = {
        "python": check_python_version(),
        "pytorch": check_pytorch(),
    }
    return results


def check_coreml_versions() -> dict:
    """Check versions required for CoreML conversion.

    Returns:
        Dictionary with check results.

    Raises:
        RuntimeError: If any required dependency is missing or version is incompatible.
    """
    results = {
        "python": check_python_version(),
        "macos": check_macos_version(),
        "coremltools": check_coremltools(),
        "pytorch": check_pytorch(),
    }
    return results


def check_all(skip_ort: bool = False) -> dict:
    """Run all checks. Returns dict of results.

    Args:
        skip_ort: If True, skip onnxruntime check.

    Returns:
        Dictionary with check results.
    """
    results = {
        "python": check_python_version(),
        "macos": check_macos_version(),
        "coremltools": check_coremltools(),
        "pytorch": check_pytorch(),
        "transformers": check_transformers(),
        "accelerate": check_accelerate(),
        "onnxruntime": check_onnxruntime(required=not skip_ort),
    }
    return results


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-ort", action="store_true", help="Skip onnxruntime check")
    ap.add_argument("--verbose", action="store_true", help="Print all versions")
    args = ap.parse_args()

    try:
        results = check_all(skip_ort=args.skip_ort)
        if args.verbose:
            print("Version checks passed:")
            for key, value in results.items():
                print(f"  {key}: {value}")
        print("✅ All version checks passed")
        sys.exit(0)
    except RuntimeError as e:
        print(f"❌ Version check failed: {e}")
        sys.exit(1)
