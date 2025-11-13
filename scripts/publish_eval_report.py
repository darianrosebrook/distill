"""Publish evaluation report to Supabase dashboard."""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict


def publish_report(
    report_path: str,
    supabase_url: str | None = None,
    supabase_key: str | None = None,
) -> None:
    """
    Publish evaluation report to Supabase.
    
    Args:
        report_path: Path to evaluation report JSON
        supabase_url: Supabase REST URL (from env if not provided)
        supabase_key: Supabase service key (from env if not provided)
    """
    supabase_url = supabase_url or os.environ.get("SUPABASE_REST_URL")
    supabase_key = supabase_key or os.environ.get("SUPABASE_SERVICE_KEY")
    
    if not supabase_url or not supabase_key:
        print("⚠️ Supabase credentials not found. Set SUPABASE_REST_URL and SUPABASE_SERVICE_KEY")
        return
    
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    
    # Handle merged reports (multiple shards)
    if "reports" in report:
        reports = report["reports"]
        checkpoint = report.get("checkpoint", "unknown")
    else:
        reports = [report]
        checkpoint = report.get("checkpoint") or report.get("header", {}).get("model_fingerprint", "unknown")
    
    # Extract metadata from first report
    first_report = reports[0]
    header = first_report.get("header", {})
    
    # Determine gates_ok (all shards must pass)
    gates_ok = all(r.get("gates_ok", False) for r in reports)
    
    payload = {
        "checkpoint": checkpoint,
        "report": report,
        "gates_ok": gates_ok,
        "dataset_sha256": header.get("dataset_sha256"),
        "tokenizer_fingerprint": header.get("tokenizer_fingerprint"),
        "tool_registry_sha256": header.get("tool_registry_sha256"),
    }
    
    try:
        import requests
        
        response = requests.post(
            f"{supabase_url}/eval_reports",
            headers={
                "apikey": supabase_key,
                "Authorization": f"Bearer {supabase_key}",
                "Content-Type": "application/json",
                "Prefer": "return=representation",
            },
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        
        print(f"✅ Published evaluation report for checkpoint: {checkpoint}")
        print(f"   Gates OK: {gates_ok}")
        print(f"   Dataset SHA256: {header.get('dataset_sha256', 'N/A')}")
        
    except ImportError:
        print("❌ requests library not available. Install with: pip install requests")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to publish report: {e}")
        sys.exit(1)


def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.publish_eval_report <report.json>")
        sys.exit(1)
    
    report_path = sys.argv[1]
    
    if not os.path.exists(report_path):
        print(f"❌ Report not found: {report_path}")
        sys.exit(1)
    
    publish_report(report_path)


if __name__ == "__main__":
    main()

