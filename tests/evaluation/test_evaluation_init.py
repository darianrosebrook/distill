"""
Tests for evaluation/__init__.py - Evaluation module initialization.

Tests the import aliases and module setup.
"""
# @author: @darianrosebrook


class TestEvaluationInit:
    """Test evaluation module initialization."""

    def test_evaluation_module_import(self):
        """Test that evaluation module can be imported."""
        import evaluation
        assert evaluation is not None

    def test_eightball_eval_import_alias(self):
        """Test that eightball_eval is properly aliased."""
        import evaluation
        import importlib
        direct_import = importlib.import_module("evaluation.8ball_eval")

        # The aliased module should be the same as direct import
        assert evaluation.eightball_eval is direct_import

        # Should have the expected module name
        assert evaluation.eightball_eval.__name__ == "evaluation.8ball_eval"

    def test_evaluation_module_has_eightball_eval(self):
        """Test that evaluation module exposes eightball_eval."""
        import evaluation

        # Should have the eightball_eval attribute
        assert hasattr(evaluation, "eightball_eval")

        # Should be a module
        assert hasattr(evaluation.eightball_eval, "__name__")
