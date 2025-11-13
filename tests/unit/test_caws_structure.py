"""
Unit tests for Priority 5: CAWS Structure Scoring.

Tests:
1. caws_structure_score() computes scores correctly
2. Structure elements are detected
3. Structure loss penalizes lower student scores
"""
from training.caws_structure import caws_structure_score, extract_caws_structure_elements, batch_caws_structure_score
from training.losses import caws_structure_loss


class TestCAWSStructureScore:
    """Test CAWS structure scoring."""

    def test_score_with_all_headers(self):
        """Test score with all required headers."""
        text = """
        Working Spec:
        - ID: FEAT-001
        - Title: Test Feature
        
        Invariants:
        - No side effects
        
        Acceptance:
        - Feature works correctly
        """
        
        score = caws_structure_score(text)
        
        # Should have high score (all headers present)
        assert score >= 0.5, f"Expected score >= 0.5 for text with all headers, got {score}"

    def test_score_with_code_blocks(self):
        """Test score with code blocks."""
        text = """
        Working Spec:
        - ID: FEAT-001
        
        Code:
        ```python
        def test():
            pass
        ```
        """
        
        score = caws_structure_score(text)
        
        # Should have bonus for code blocks
        assert score >= 0.3, f"Expected score >= 0.3 for text with code blocks, got {score}"

    def test_score_empty_text(self):
        """Test score with empty text."""
        score = caws_structure_score("")
        
        assert score == 0.0, f"Expected score 0.0 for empty text, got {score}"

    def test_score_with_json(self):
        """Test score with JSON structure."""
        text = """
        {
            "id": "FEAT-001",
            "title": "Test"
        }
        """
        
        score = caws_structure_score(text)
        
        # JSON alone without CAWS headers should get low score (structure bonus only)
        # JSON structure bonus is ~0.15, field score ~0.2-0.5, total ~0.07-0.15
        assert score >= 0.05, f"Expected score >= 0.05 for text with JSON, got {score}"
        assert score < 0.3, f"Expected score < 0.3 for JSON without CAWS headers, got {score}"

    def test_extract_structure_elements(self):
        """Test structure element extraction."""
        text = """
        Working Spec:
        - ID: FEAT-001
        
        Invariants:
        - No side effects
        
        ```python
        def test():
            pass
        ```
        """
        
        elements = extract_caws_structure_elements(text)
        
        assert elements["has_working_spec"] is True
        assert elements["has_invariants"] is True
        assert elements["has_code_blocks"] is True
        assert elements["has_acceptance"] is False  # Not present

    def test_batch_scoring(self):
        """Test batch structure scoring."""
        texts = [
            "Working Spec: Test",
            "Invariants: None",
            "Acceptance: Pass",
        ]
        
        result = batch_caws_structure_score(texts)
        
        assert "mean_score" in result
        assert "min_score" in result
        assert "max_score" in result
        assert "scores" in result
        assert len(result["scores"]) == 3
        assert 0.0 <= result["mean_score"] <= 1.0


class TestCAWSStructureLoss:
    """Test CAWS structure loss function."""

    def test_loss_when_student_lower(self):
        """Test loss when student score is lower than teacher."""
        teacher_score = 0.8
        student_score = 0.5
        
        loss = caws_structure_loss(teacher_score, student_score)
        
        # Should penalize difference
        expected_loss = teacher_score - student_score  # 0.3
        assert abs(loss.item() - expected_loss) < 0.01, \
            f"Expected loss ~{expected_loss}, got {loss.item()}"

    def test_loss_when_student_higher(self):
        """Test loss when student score is higher than teacher."""
        teacher_score = 0.5
        student_score = 0.8
        
        loss = caws_structure_loss(teacher_score, student_score)
        
        # Should not penalize (student is better)
        assert loss.item() == 0.0, f"Expected loss 0.0 when student > teacher, got {loss.item()}"

    def test_loss_when_equal(self):
        """Test loss when scores are equal."""
        teacher_score = 0.7
        student_score = 0.7
        
        loss = caws_structure_loss(teacher_score, student_score)
        
        assert loss.item() == 0.0, f"Expected loss 0.0 when scores equal, got {loss.item()}"

    def test_loss_requires_grad(self):
        """Test that loss tensor requires gradients."""
        teacher_score = 0.8
        student_score = 0.5
        
        loss = caws_structure_loss(teacher_score, student_score)
        
        assert loss.requires_grad, "Loss tensor should require gradients"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

