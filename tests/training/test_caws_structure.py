"""
Unit tests for training/caws_structure.py

Tests CAWS structure scoring and element extraction functionality.
"""

import pytest
from training.caws_structure import (
    caws_structure_score,
    extract_caws_structure_elements,
    batch_caws_structure_score,
)


class TestCawsStructureScore:
    """Test CAWS structure scoring functionality."""

    def test_caws_structure_score_empty_text(self):
        """Test scoring empty text."""
        assert caws_structure_score("") == 0.0
        assert caws_structure_score("   ") == 0.0
        assert caws_structure_score(None) == 0.0

    def test_caws_structure_score_no_structure(self):
        """Test scoring text with no CAWS structure."""
        text = "This is just some random text without any structure."
        score = caws_structure_score(text)
        assert 0.0 < score < 0.3  # Should have low but non-zero score for content

    def test_caws_structure_score_partial_headers(self):
        """Test scoring with partial CAWS headers."""
        text = """
        Working Spec
        ============
        Some content here.
        """
        score = caws_structure_score(text)
        assert 0.1 < score < 0.5  # Should have moderate score

    def test_caws_structure_score_all_headers(self):
        """Test scoring with all required CAWS headers."""
        text = """
        Working Spec
        ============
        This is the working spec.

        Invariants
        ==========
        These are the invariants.

        Acceptance
        ==========
        These are acceptance criteria.
        """
        score = caws_structure_score(text)
        assert 0.3 < score < 0.7  # Should have good score

    def test_caws_structure_score_with_code_blocks(self):
        """Test scoring with code blocks (bonus)."""
        text = """
        Working Spec
        ============
        Some spec here.

        ```python
        def example():
            return True
        ```

        Invariants
        ==========
        Some invariants.
        """
        score = caws_structure_score(text)
        # Should be higher than without code blocks
        baseline = caws_structure_score("""
        Working Spec
        ============
        Some spec here.

        Invariants
        ==========
        Some invariants.
        """)
        assert score > baseline

    def test_caws_structure_score_with_json(self):
        """Test scoring with JSON structures."""
        text = """
        Working Spec
        ============
        {"key": "value", "array": [1, 2, 3]}

        Invariants
        ==========
        Some invariants.
        """
        score = caws_structure_score(text)
        baseline = caws_structure_score("""
        Working Spec
        ============
        Some text content.

        Invariants
        ==========
        Some invariants.
        """)
        assert score > baseline

    def test_caws_structure_score_with_lists(self):
        """Test scoring with list structures."""
        text = """
        Working Spec
        ============
        - Item 1
        - Item 2
        1. Numbered item
        2. Another numbered item

        Invariants
        ==========
        Some invariants.
        """
        score = caws_structure_score(text)
        baseline = caws_structure_score("""
        Working Spec
        ============
        Some content.

        Invariants
        ==========
        Some invariants.
        """)
        assert score > baseline

    def test_caws_structure_score_with_markdown_headers(self):
        """Test scoring with markdown headers."""
        text = """
        # Working Spec

        Content here.

        ## Subheader

        # Invariants

        More content.

        # Acceptance

        Final content.
        """
        score = caws_structure_score(text)
        assert 0.4 < score < 0.7  # Should have moderate to good score

    def test_caws_structure_score_content_length_impact(self):
        """Test that longer content influences scoring (field_score component)."""
        # Test that very short content gets lower field score
        short_text = "Working Spec\n===========\nHi.\n\nInvariants\n==========\nHi."
        short_score = caws_structure_score(short_text)

        # Test that substantial content gets higher field score
        long_text = "Working Spec\n===========\n" + "This is substantial content. " * 30 + "\n\nInvariants\n==========\n" + "This is also substantial. " * 20
        long_score = caws_structure_score(long_text)

        # Long content should score higher due to field_score component
        assert long_score > short_score

    def test_caws_structure_score_perfect_match(self):
        """Test scoring with all elements present."""
        text = """
        # Working Spec

        This is a comprehensive working spec with lots of detail.
        It includes many words and comprehensive content.

        ```python
        def example_function():
            return "structured code"
        ```

        {"json": "structure", "complex": {"nested": "object"}}

        - Bullet point 1
        - Bullet point 2
        1. Numbered item 1
        2. Numbered item 2

        ## Invariants

        These are the system invariants with substantial detail.
        The content continues with meaningful information.

        ## Acceptance

        These are the acceptance criteria with comprehensive details.
        The documentation is thorough and well-structured.
        """

        score = caws_structure_score(text)
        assert 0.6 < score <= 1.0  # Should be high but capped at 1.0

    def test_caws_structure_score_case_insensitive_headers(self):
        """Test that header matching is case insensitive."""
        text = """
        working spec
        ============
        Content.

        INVARIANTS
        ==========
        More content.

        Acceptance
        ==========
        Final content.
        """
        score = caws_structure_score(text)
        assert score > 0.3  # Should recognize headers regardless of case


class TestExtractCawsStructureElements:
    """Test CAWS structure element extraction."""

    def test_extract_caws_structure_elements_empty_text(self):
        """Test extraction from empty text."""
        result = extract_caws_structure_elements("")
        expected = {
            "has_working_spec": False,
            "has_invariants": False,
            "has_acceptance": False,
            "has_code_blocks": False,
            "has_json": False,
            "has_lists": False,
            "has_markdown": False,
        }
        assert result == expected

    def test_extract_caws_structure_elements_none_text(self):
        """Test extraction from None text."""
        result = extract_caws_structure_elements(None)
        expected = {
            "has_working_spec": False,
            "has_invariants": False,
            "has_acceptance": False,
            "has_code_blocks": False,
            "has_json": False,
            "has_lists": False,
            "has_markdown": False,
        }
        assert result == expected

    def test_extract_caws_structure_elements_all_elements(self):
        """Test extraction with all structure elements present."""
        text = """# Working Spec

This has working spec content.

```python
code block here
```

{"json": "content"}

- Bullet list
1. Numbered list

## Invariants

Invariant content.

## Acceptance

Acceptance content."""

        result = extract_caws_structure_elements(text)
        expected = {
            "has_working_spec": True,
            "has_invariants": True,
            "has_acceptance": True,
            "has_code_blocks": True,
            "has_json": True,
            "has_lists": True,
            "has_markdown": True,
        }
        assert result == expected

    def test_extract_caws_structure_elements_partial_elements(self):
        """Test extraction with only some elements present."""
        text = """
        working spec
        ============
        Content here.

        Some regular text without other elements.
        """

        result = extract_caws_structure_elements(text)
        expected = {
            "has_working_spec": True,
            "has_invariants": False,
            "has_acceptance": False,
            "has_code_blocks": False,
            "has_json": False,
            "has_lists": False,
            "has_markdown": False,
        }
        assert result == expected

    def test_extract_caws_structure_elements_case_insensitive(self):
        """Test that header detection is case insensitive."""
        text = """
        WORKING SPEC
        ============
        Content.

        invariants
        ==========
        More content.

        ACCEPTANCE
        ==========
        Final content.
        """

        result = extract_caws_structure_elements(text)
        assert result["has_working_spec"] is True
        assert result["has_invariants"] is True
        assert result["has_acceptance"] is True

    def test_extract_caws_structure_elements_complex_patterns(self):
        """Test extraction with complex patterns."""
        text = """# Working Spec

Content with `{"json": "in backticks"}` that shouldn't count.

```json
{
  "proper": "json",
  "structure": ["array", "elements"]
}
```

- List item 1
- List item 2
  - Nested item (shouldn't count as new list)
1. Numbered item 1
2. Numbered item 2"""

        result = extract_caws_structure_elements(text)
        assert result["has_working_spec"] is True
        assert result["has_code_blocks"] is True
        assert result["has_json"] is True  # Should detect JSON in code block
        assert result["has_lists"] is True
        assert result["has_markdown"] is True


class TestBatchCawsStructureScore:
    """Test batch CAWS structure scoring."""

    def test_batch_caws_structure_score_empty_list(self):
        """Test batch scoring with empty list."""
        result = batch_caws_structure_score([])
        expected = {
            "mean_score": 0.0,
            "min_score": 0.0,
            "max_score": 0.0,
            "scores": [],
        }
        assert result == expected

    def test_batch_caws_structure_score_single_item(self):
        """Test batch scoring with single item."""
        texts = ["Working Spec\n===========\nContent."]
        result = batch_caws_structure_score(texts)

        assert len(result["scores"]) == 1
        assert result["mean_score"] == result["scores"][0]
        assert result["min_score"] == result["scores"][0]
        assert result["max_score"] == result["scores"][0]
        assert result["mean_score"] > 0.0

    def test_batch_caws_structure_score_multiple_items(self):
        """Test batch scoring with multiple items."""
        texts = [
            "",  # Empty
            "Working Spec\n===========\nContent.",  # Partial
            """
            Working Spec
            ===========
            Content.

            Invariants
            ==========
            More content.

            Acceptance
            ==========
            Final content.
            """,  # Complete
        ]

        result = batch_caws_structure_score(texts)

        assert len(result["scores"]) == 3
        assert result["scores"][0] == 0.0  # Empty text
        assert result["scores"][1] > 0.0   # Partial structure
        assert result["scores"][2] > result["scores"][1]  # Complete structure

        # Check statistics
        assert result["min_score"] == 0.0
        assert result["max_score"] == result["scores"][2]
        assert result["mean_score"] == sum(result["scores"]) / 3

    def test_batch_caws_structure_score_varied_quality(self):
        """Test batch scoring with varied quality texts."""
        texts = [
            "Poor quality text with no structure.",
            """
            Working Spec
            ===========
            Some content.
            """,
            """
            Working Spec
            ===========
            Good content with code.

            ```python
            def example():
                return True
            ```

            Invariants
            ==========
            System invariants.
            """,
        ]

        result = batch_caws_structure_score(texts)

        # Scores should be strictly increasing
        assert result["scores"][0] < result["scores"][1] < result["scores"][2]

        # Statistics should be reasonable
        assert 0.0 < result["mean_score"] < 1.0
        assert result["min_score"] == result["scores"][0]
        assert result["max_score"] == result["scores"][2]

    def test_batch_caws_structure_score_all_identical(self):
        """Test batch scoring when all texts are identical."""
        text = """
        Working Spec
        ===========
        Content.

        Invariants
        ==========
        Invariants.
        """

        texts = [text] * 5
        result = batch_caws_structure_score(texts)

        assert len(result["scores"]) == 5
        assert all(score == result["scores"][0] for score in result["scores"])
        assert result["min_score"] == result["max_score"] == result["mean_score"]
