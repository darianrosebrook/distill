"""
Module alias for evaluation.8ball_eval to support test patching.

Tests patch 'evaluation.eightball_eval' but the actual module is 'evaluation.8ball_eval'.
This module provides an alias for compatibility.
"""
import importlib
import sys

# Import the actual 8ball_eval module (note: module name with number requires importlib)
_8ball_module = importlib.import_module("evaluation.8ball_eval")

# Re-export everything from 8ball_eval
for name in dir(_8ball_module):
    if not name.startswith("_"):
        setattr(sys.modules[__name__], name, getattr(_8ball_module, name))

