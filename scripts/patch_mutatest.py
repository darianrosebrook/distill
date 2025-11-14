#!/usr/bin/env python3
"""
Patch mutatest to fix the random.sample issue with sets in Python 3.11.

This patches the bug in mutatest 2.0.1 where random.sample is called on a set,
which doesn't work in Python 3.11+.

Usage:
    python scripts/patch_mutatest.py
    # Then run mutation testing as normal
"""

import sys
from pathlib import Path


def patch_mutatest():
    """Patch mutatest run.py to fix random.sample(set) issue."""
    try:
        import mutatest
        mutatest_path = Path(mutatest.__file__).parent
        run_py = mutatest_path / "run.py"
        
        if not run_py.exists():
            print(f"❌ mutatest run.py not found at {run_py}")
            return False
        
        # Read the file
        with open(run_py, "r") as f:
            content = f.read()
        
        # Check if already patched
        if "list(mutant_operations)" in content:
            print("✅ mutatest already patched")
            return True
        
        # Find the problematic line
        old_line = "            current_mutation = random.sample(mutant_operations, k=1)[0]"
        new_line = "            current_mutation = random.sample(list(mutant_operations), k=1)[0]"
        
        if old_line not in content:
            print(f"❌ Could not find line to patch in {run_py}")
            print("   The bug may have been fixed or the code changed")
            return False
        
        # Apply the patch
        patched_content = content.replace(old_line, new_line)
        
        # Write back
        with open(run_py, "w") as f:
            f.write(patched_content)
        
        print(f"✅ Successfully patched {run_py}")
        print(f"   Changed: random.sample(mutant_operations, k=1)")
        print(f"   To:      random.sample(list(mutant_operations), k=1)")
        return True
        
    except ImportError:
        print("❌ mutatest not installed")
        return False
    except Exception as e:
        print(f"❌ Error patching mutatest: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_patch():
    """Verify the patch was applied correctly."""
    try:
        import mutatest
        mutatest_path = Path(mutatest.__file__).parent
        run_py = mutatest_path / "run.py"
        
        with open(run_py, "r") as f:
            content = f.read()
        
        if "list(mutant_operations)" in content:
            print("✅ Patch verified: list() conversion is present")
            return True
        else:
            print("❌ Patch verification failed: list() conversion not found")
            return False
            
    except Exception as e:
        print(f"❌ Error verifying patch: {e}")
        return False


if __name__ == "__main__":
    print("🔧 Patching mutatest for Python 3.11 compatibility...")
    print("=" * 60)
    
    if patch_mutatest():
        print("\n" + "=" * 60)
        if verify_patch():
            print("\n✅ Patch applied successfully!")
            print("   You can now run mutation testing:")
            print("   python scripts/run_mutation_testing.py --module training/losses.py")
            sys.exit(0)
        else:
            print("\n⚠️  Patch applied but verification failed")
            sys.exit(1)
    else:
        print("\n❌ Failed to apply patch")
        sys.exit(1)

