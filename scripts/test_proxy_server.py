"""
Test proxy server with Kimi API to capture tool-use traces.

Usage:
    # Terminal 1: Start proxy server
    python -m capture.proxy_server \
        --upstream https://api.moonshot.ai/v1 \
        --bind 127.0.0.1 \
        --port 8081 \
        --out capture/raw

    # Terminal 2: Run this test
    python -m scripts.test_proxy_server
"""
import os
import sys
import json
from pathlib import Path
from typing import Optional

import requests

from models.teacher.teacher_client import TeacherClient


def load_api_key() -> Optional[str]:
    """Load API key from environment or .env.local."""
    api_key = os.getenv("MOONSHOT_API_KEY") or os.getenv("KIMI_API_KEY")
    if api_key:
        return api_key
    
    env_file = Path(".env.local")
    if env_file.exists():
        try:
            with open(env_file, 'r') as f:
                for line in f:
                    if line.startswith("MOONSHOT_API_KEY="):
                        return line.split("=", 1)[1].strip().strip('"\'')
                    if line.startswith("KIMI_API_KEY="):
                        return line.split("=", 1)[1].strip().strip('"\'')
        except Exception as e:
            print(f"[test_proxy_server] ERROR: Failed to load .env.local: {e}")
    
    return None


def check_proxy_running(proxy_url: str) -> bool:
    """Check if proxy server is running."""
    try:
        response = requests.get(f"{proxy_url}/health", timeout=2)
        return response.status_code < 500
    except requests.exceptions.RequestException:
        return False


def test_proxy_capture(proxy_url: str, api_key: str) -> bool:
    """Test that proxy captures API calls correctly."""
    print(f"\n[test_proxy_server] Testing proxy capture at {proxy_url}...")
    
    # Initialize client pointing to proxy
    client = TeacherClient.from_endpoint(
        proxy_url,
        api_key=api_key,
        max_retries=3,
        retry_backoff_factor=2.0
    )
    
    # Make a test API call through proxy
    test_prompt = "What is 2+2? Answer briefly."
    print("[test_proxy_server] Making API call through proxy...")
    print(f"  Prompt: {test_prompt}")
    
    try:
        results = client.sample(
            [test_prompt],
            temperature=1.0,
            top_p=0.95,
            max_tokens=50,
            return_logits=False,
        )
        
        if results and not results[0].get("error"):
            print("  ✅ API call successful")
            print(f"  Response: {results[0]['text'][:100]}...")
            
            # Check if capture files were created
            capture_dir = Path("capture/raw")
            if capture_dir.exists():
                trace_files = list(capture_dir.glob("*.jsonl"))
                list(capture_dir.glob("*.meta.json"))
                
                if trace_files:
                    latest_trace = max(trace_files, key=lambda p: p.stat().st_mtime)
                    latest_meta = latest_trace.with_suffix(".meta.json")
                    
                    print("\n[test_proxy_server] ✅ Capture files created:")
                    print(f"  Trace file: {latest_trace}")
                    print(f"  Meta file: {latest_meta}")
                    
                    # Verify trace file has content
                    with open(latest_trace, 'r') as f:
                        lines = [line.strip() for line in f if line.strip()]
                        print(f"  Trace lines: {len(lines)}")
                        if lines:
                            try:
                                first_line = json.loads(lines[0])
                                print(f"  First line keys: {list(first_line.keys())}")
                            except Exception as e:
                                print(f"  ⚠️ Failed to parse first line: {e}")
                    
                    # Verify meta file
                    if latest_meta.exists():
                        with open(latest_meta, 'r') as f:
                            meta = json.load(f)
                            print("\n[test_proxy_server] Meta file contents:")
                            print(f"  Trace ID: {meta.get('trace_id')}")
                            print(f"  Method: {meta.get('request', {}).get('method')}")
                            print(f"  Body length: {meta.get('request', {}).get('body_len')}")
                            print(f"  Timestamps: {meta.get('timestamps')}")
                    
                    return True
                else:
                    print(f"\n[test_proxy_server] ⚠️ No trace files found in {capture_dir}")
                    return False
            else:
                print(f"\n[test_proxy_server] ⚠️ Capture directory not found: {capture_dir}")
                return False
        else:
            error_msg = results[0].get("error", "Unknown error") if results else "No response"
            print(f"  ❌ API call failed: {error_msg}")
            return False
            
    except Exception as e:
        print(f"  ❌ Exception during API call: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    proxy_url = os.getenv("PROXY_URL", "http://127.0.0.1:8081")
    
    print("[test_proxy_server] Testing proxy server with Kimi API")
    print(f"[test_proxy_server] Proxy URL: {proxy_url}")
    
    # Check if proxy is running
    print("\n[test_proxy_server] Checking if proxy server is running...")
    if not check_proxy_running(proxy_url):
        print(f"  ❌ Proxy server not running at {proxy_url}")
        print("\n[test_proxy_server] Please start the proxy server first:")
        print("  python -m capture.proxy_server \\")
        print("    --upstream https://api.moonshot.ai/v1 \\")
        print("    --bind 127.0.0.1 \\")
        print("    --port 8081 \\")
        print("    --out capture/raw")
        return 1
    
    print("  ✅ Proxy server is running")
    
    # Load API key
    api_key = load_api_key()
    if not api_key:
        print("\n[test_proxy_server] ERROR: MOONSHOT_API_KEY or KIMI_API_KEY not found")
        print("[test_proxy_server] Please set MOONSHOT_API_KEY in .env.local")
        return 1
    
    print("  ✅ API key loaded")
    
    # Test proxy capture
    success = test_proxy_capture(proxy_url, api_key)
    
    if success:
        print("\n[test_proxy_server] ✅ All tests passed!")
        return 0
    else:
        print("\n[test_proxy_server] ❌ Tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())

