#!/usr/bin/env python3
"""
Simple broker smoke test for CI.

Tests that the basic CAWS evaluation infrastructure can be imported.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    try:
        from eval.scoring.scorer import _evaluate_caws_compliance
        from eval.tool_broker.broker import ToolBroker

        # Verify imports are available (basic smoke test)
        assert callable(_evaluate_caws_compliance)
        assert ToolBroker is not None

        print("✅ Successfully imported CAWS scorer and tool broker")
        print("✅ Broker smoke test passed")
        return 0
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
