"""
Version gates: Check Python, macOS, and dependency versions before proceeding.
"""
import sys
import platform
from pathlib import Path


def check_python_version():
    """Require Python 3.10 or 3.11."""
    major, minor = sys.version_info[:2]
    if (major, minor) not in [(3, 10), (3, 11)]:
        raise RuntimeError(
            f"Python {major}.{minor} detected. "
            "This project requires Python 3.10 or 3.11. "
            f"Current: {sys.version}"
        )
    return major, minor


def check_macos_version():
    """Check macOS version (optional, warn only)."""
    if platform.system() != "Darwin":
        return None
    
    try:
        macos_version = platform.mac_ver()[0]
        major = int(macos_version.split('.')[0])
        if major < 13:
            print(f"WARNING: macOS {macos_version} detected. CoreML requires macOS 13+")
        return macos_version
    except Exception:
        return None


def check_coremltools():
    """Check coremltools version."""
    try:
        import coremltools as ct
        version = ct.__version__
        major = int(version.split('.')[0])
        if major < 9:
            raise RuntimeError(
                f"coremltools {version} detected. "
                "This project requires coremltools >= 9.0"
            )
        return version
    except ImportError:
        raise RuntimeError("coremltools not installed. Install with: pip install coremltools>=9.0")


def check_onnxruntime(required=False):
    """Check if onnxruntime is available (optional for smoke tests)."""
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


def check_all(skip_ort=False):
    """Run all checks. Returns dict of results."""
    results = {
        "python": check_python_version(),
        "macos": check_macos_version(),
        "coremltools": check_coremltools(),
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

