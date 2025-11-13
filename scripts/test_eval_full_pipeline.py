"""Full pipeline test for evaluation harness - tests complete flow without requiring a real model."""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Test the complete evaluation pipeline components
def test_dataset_loading():
    """Test loading and parsing dataset."""
    print("\n[PIPELINE] Testing dataset loading...")
    try:
        dataset_path = Path("data/contextual_final.jsonl")
        if not dataset_path.exists():
            print(f"⚠️  Dataset not found: {dataset_path}")
            return False
        
        items = []
        with open(dataset_path, "r") as f:
            for i, line in enumerate(f):
                if i >= 5:  # Just test first 5 items
                    break
                if line.strip():
                    item = json.loads(line)
                    items.append(item)
        
        print(f"✅ Loaded {len(items)} items from dataset")
        print(f"   Sample item keys: {list(items[0].keys()) if items else 'none'}")
        
        # Check required fields
        if items:
            sample = items[0]
            required = ["prompt", "metadata"]
            missing = [k for k in required if k not in sample]
            if missing:
                print(f"❌ Missing required fields: {missing}")
                return False
            print("✅ All required fields present")
        
        return True
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fixture_matching():
    """Test fixture matching with real dataset queries."""
    print("\n[PIPELINE] Testing fixture matching with dataset queries...")
    try:
        from eval.tool_broker.broker import ToolBroker
        
        broker = ToolBroker("eval/tool_broker/fixtures")
        
        # Load a few items and test fixture matching
        dataset_path = Path("data/contextual_final.jsonl")
        if not dataset_path.exists():
            print(f"⚠️  Dataset not found: {dataset_path}")
            return False
        
        matches = 0
        misses = 0
        
        with open(dataset_path, "r") as f:
            for i, line in enumerate(f):
                if i >= 10:  # Test first 10 items
                    break
                if not line.strip():
                    continue
                
                item = json.loads(line)
                call_sequence = item.get("metadata", {}).get("call_sequence", [])
                
                for call in call_sequence:
                    tool_name = call.get("name", "")
                    args = call.get("arguments", {})
                    
                    result = broker.call(tool_name, args)
                    if result.get("error") == "fixture_miss":
                        misses += 1
                        print(f"   ⚠️  Miss: {tool_name}({list(args.keys())})")
                    else:
                        matches += 1
        
        hit_rate = matches / (matches + misses) if (matches + misses) > 0 else 0
        print(f"✅ Fixture matching: {matches} hits, {misses} misses (hit rate: {hit_rate:.1%})")
        
        if hit_rate < 0.5:
            print("⚠️  Low hit rate - may need more fixtures")
        
        return True
    except Exception as e:
        print(f"❌ Fixture matching failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scorer_integration():
    """Test scorer with sample data."""
    print("\n[PIPELINE] Testing scorer integration...")
    try:
        from eval.scoring.scorer import score_item
        
        # Create a sample item
        sample_item = {
            "prompt": "Read the file README.md",
            "metadata": {
                "call_sequence": [
                    {"name": "read_file", "arguments": {"path": "README.md"}}
                ],
                "expected_behaviour": "normal",
            }
        }
        
        # Mock model output with tool call
        model_output = """I'll read the file.
TOOL_CALL: {"name":"read_file","arguments":{"path":"README.md"}}
Integration: The file contains project documentation."""
        
        tool_trace = [
            {
                "name": "read_file",
                "arguments": {"path": "README.md"},
                "result": {"ok": True, "content": "# Project\n\nDocumentation here."}
            }
        ]
        
        scores = score_item(
            item=sample_item,
            model_output=model_output,
            tool_trace=tool_trace,
            integration_span_cap=3,
        )
        
        print("✅ Scorer executed successfully")
        print(f"   Scores keys: {list(scores.keys())[:5]}...")
        print(f"   Integration F1 eligible: {scores.get('integration_f1_eligible')}")
        
        return True
    except Exception as e:
        print(f"❌ Scorer integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_report_generation():
    """Test report generation from sample results."""
    print("\n[PIPELINE] Testing report generation...")
    try:
        from eval.reports.summarize import summarize_results
        
        # Create sample results
        results = [
            {
                "sample_id": "test_1",
                "model_output": "Test output",
                "tool_trace": [
                    {"name": "read_file", "arguments": {"path": "test.txt"}, "result": {"ok": True}}
                ],
                "scores": {
                    "integration_f1_eligible": True,
                    "integration_f1_lax": 0.9,
                    "integration_f1_strict": 0.8,
                    "integration_spans_count": 1,
                    "multi_call_parity_ok": True,
                    "json_args_valid": True,
                    "controls_with_integration": 0,
                    "privacy_ok": True,
                },
                "runner_fingerprint": {"runner_type": "TestRunner"},
                "model_fingerprint": {"model": "test"},
            }
        ]
        
        report = summarize_results(
            results=results,
            report_version="1.0.0",
            dataset_header=None,
            dataset_sha256="test_hash",
            tool_registry_sha256="test_registry_hash",
            tokenizer_fingerprint=None,
            config={"runner": "test", "model": "test", "seed": 42},
            wall_time_sec=1.0,
            gates_overrides={"min_eligible_for_gates": 1},
        )
        
        print("✅ Report generated successfully")
        print(f"   Report keys: {list(report.keys())}")
        print(f"   Summary keys: {list(report.get('summary', {}).keys())[:5]}...")
        print(f"   Gates OK: {report.get('gates_ok')}")
        
        return True
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sharding_logic():
    """Test sharding logic with real dataset."""
    print("\n[PIPELINE] Testing sharding logic...")
    try:
        from eval.cli import select_shard, stable_shard
        
        # Load dataset items
        dataset_path = Path("data/contextual_final.jsonl")
        if not dataset_path.exists():
            print(f"⚠️  Dataset not found: {dataset_path}")
            return False
        
        items = []
        with open(dataset_path, "r") as f:
            for i, line in enumerate(f):
                if i >= 20:  # Test with 20 items
                    break
                if line.strip():
                    item = json.loads(line)
                    items.append(item)
        
        # Test sharding
        num_shards = 4
        shard_results = []
        for shard_idx in range(num_shards):
            shard_items = select_shard(items, shard_idx, num_shards)
            shard_results.append(len(shard_items))
        
        total_sharded = sum(shard_results)
        print(f"✅ Sharding test: {len(items)} items → {shard_results} per shard (total: {total_sharded})")
        
        if total_sharded != len(items):
            print(f"❌ Shard count mismatch: {total_sharded} != {len(items)}")
            return False
        
        # Test stable sharding (same items should go to same shard)
        sample_id = items[0].get("metadata", {}).get("sample_id", "test")
        shard1 = stable_shard(sample_id, num_shards)
        shard2 = stable_shard(sample_id, num_shards)
        
        if shard1 != shard2:
            print(f"❌ Stable sharding failed: {shard1} != {shard2}")
            return False
        
        print(f"✅ Stable sharding verified (sample_id '{sample_id}' → shard {shard1})")
        
        return True
    except Exception as e:
        print(f"❌ Sharding logic failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run full pipeline tests."""
    print("=" * 70)
    print("Evaluation Harness Full Pipeline Test")
    print("=" * 70)
    
    results = []
    
    results.append(("Dataset Loading", test_dataset_loading()))
    results.append(("Fixture Matching", test_fixture_matching()))
    results.append(("Scorer Integration", test_scorer_integration()))
    results.append(("Report Generation", test_report_generation()))
    results.append(("Sharding Logic", test_sharding_logic()))
    
    # Summary
    print("\n" + "=" * 70)
    print("Pipeline Test Summary")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All pipeline tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

