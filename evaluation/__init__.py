"""
Evaluation module for model evaluation and testing.

This module contains various evaluation frameworks for different model types
and evaluation scenarios in the distillation pipeline.
"""
# @author: @darianrosebrook

# Create alias for 8ball_eval module so tests can patch evaluation.eightball_eval
import importlib
eightball_eval = importlib.import_module("evaluation.8ball_eval")
