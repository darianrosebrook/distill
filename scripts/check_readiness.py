#!/usr/bin/env python3
"""
Check repository readiness for KD dataset generation.

Usage:
    python3 scripts/check_readiness.py
"""
import sys
from pathlib import Path


def check_python_version():
    """Check Python version >= 3.10."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (need 3.10+)")
        return False


def check_dependencies():
    """Check required dependencies."""
    results = {}
    
    try:
        import requests
        print(f"✅ requests {requests.__version__}")
        results['requests'] = True
    except ImportError:
        print("❌ requests not installed (run: pip install -e .)")
        results['requests'] = False
    
    try:
        from urllib3.util.retry import Retry
        import urllib3
        print(f"✅ urllib3 {urllib3.__version__}")
        results['urllib3'] = True
    except ImportError:
        print("❌ urllib3 not installed (run: pip install -e .)")
        results['urllib3'] = False
    
    return all(results.values())


def check_api_key():
    """Check if API key is configured."""
    env_file = Path(".env.local")
    
    if not env_file.exists():
        print("❌ .env.local not found")
        print("   Create with: echo 'KIMI_API_KEY=your-key' > .env.local")
        return False
    
    try:
        with open(env_file, 'r') as f:
            content = f.read()
            if "KIMI_API_KEY" in content:
                # Check if it's not just a placeholder
                for line in content.split('\n'):
                    if line.startswith("KIMI_API_KEY="):
                        key = line.split("=", 1)[1].strip().strip('"\'')
                        if key and key != "your-key-here" and len(key) > 10:
                            print("✅ API key found in .env.local")
                            return True
                        else:
                            print("⚠️ API key in .env.local appears to be placeholder")
                            return False
        print("❌ KIMI_API_KEY not found in .env.local")
        return False
    except Exception as e:
        print(f"❌ Error reading .env.local: {e}")
        return False


def check_gitignore():
    """Check if .env.local is in .gitignore."""
    gitignore = Path(".gitignore")
    if not gitignore.exists():
        print("⚠️ .gitignore not found")
        return False
    
    try:
        with open(gitignore, 'r') as f:
            content = f.read()
            if ".env.local" in content:
                print("✅ .env.local is in .gitignore")
                return True
            else:
                print("⚠️ .env.local not in .gitignore (should be ignored)")
                return False
    except Exception:
        print("⚠️ Could not read .gitignore")
        return False


def check_code_imports():
    """Check if code modules can be imported."""
    results = {}
    
    try:
        from models.teacher.teacher_client import TeacherClient
        print("✅ TeacherClient imports successfully")
        results['teacher_client'] = True
    except Exception as e:
        print(f"❌ TeacherClient import failed: {e}")
        results['teacher_client'] = False
    
    try:
        from scripts.prompt_sources import get_prompt_mix
        print("✅ prompt_sources imports successfully")
        results['prompt_sources'] = True
    except Exception as e:
        print(f"❌ prompt_sources import failed: {e}")
        results['prompt_sources'] = False
    
    return all(results.values())


def check_directories():
    """Check if required directories exist."""
    data_dir = Path("data")
    if data_dir.exists():
        print("✅ data/ directory exists")
        return True
    else:
        print("⚠️ data/ directory doesn't exist (will be created automatically)")
        return True  # Not a blocker


def check_script_permissions():
    """Check if daily script is executable."""
    script = Path("scripts/make_kd_mix_daily.sh")
    if script.exists():
        import os
        if os.access(script, os.X_OK):
            print("✅ Daily script is executable")
            return True
        else:
            print("⚠️ Daily script not executable (run: chmod +x scripts/make_kd_mix_daily.sh)")
            return False
    else:
        print("⚠️ Daily script not found")
        return False


def test_api_connection():
    """Test API connection (optional)."""
    print("\n--- Testing API Connection (optional) ---")
    
    try:
        from models.teacher.teacher_client import TeacherClient
        
        client = TeacherClient.from_endpoint("https://api.kimi.com/v1")
        
        print("Checking API health...")
        if client.health_check():
            print("✅ API health check passed")
            
            tier = client.get_tier()
            if tier:
                print(f"✅ Detected API tier: {tier.value}")
                tier_limits = client.get_tier_limits()
                if tier_limits:
                    print(f"   Rate limits: {tier_limits.rpm} RPM, {tier_limits.tpm:,} TPM")
            else:
                print("⚠️ Tier detection failed (may default to FREE)")
            
            return True
        else:
            print("❌ API health check failed")
            print("   Check: API key, network connectivity, $1 recharge completed")
            return False
    except Exception as e:
        print(f"⚠️ API connection test failed: {e}")
        print("   This is optional - you can still proceed if dependencies are installed")
        return False


def main():
    """Run all readiness checks."""
    print("=" * 50)
    print("Repository Readiness Check")
    print("=" * 50)
    print()
    
    checks = {
        "Python Version": check_python_version,
        "Dependencies": check_dependencies,
        "API Key Configuration": check_api_key,
        ".gitignore": check_gitignore,
        "Code Imports": check_code_imports,
        "Directories": check_directories,
        "Script Permissions": check_script_permissions,
    }
    
    results = {}
    for name, check_func in checks.items():
        print(f"\n--- {name} ---")
        results[name] = check_func()
    
    # Optional API test
    if results.get("API Key Configuration"):
        test_api_connection()
    
    # Summary
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    
    critical = [
        "Python Version",
        "Dependencies",
        "API Key Configuration",
        "Code Imports",
    ]
    
    critical_passed = all(results.get(k, False) for k in critical)
    
    if critical_passed:
        print("✅ All critical checks passed!")
        print("\nYou're ready to generate datasets:")
        print("  python3 -m scripts.make_kd_mix --out data/kd_mix.jsonl --teacher https://api.kimi.com/v1 --total 10 --delay 20")
    else:
        print("❌ Some critical checks failed")
        print("\nFix the issues above before proceeding.")
        print("\nQuick fixes:")
        if not results.get("Dependencies"):
            print("  pip install -e .")
        if not results.get("API Key Configuration"):
            print("  echo 'KIMI_API_KEY=your-key' > .env.local")
    
    return 0 if critical_passed else 1


if __name__ == "__main__":
    sys.exit(main())



