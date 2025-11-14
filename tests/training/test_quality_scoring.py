"""
Tests for training/quality_scoring.py - Quality scoring utilities for teacher outputs.

Tests heuristic quality scoring, JSON validity scoring, code block scoring,
composite quality scoring, and batch processing using various text samples.
"""
# @author: @darianrosebrook

from training.quality_scoring import (
    compute_heuristic_quality_score,
    compute_json_validity_score,
    compute_code_block_score,
    compute_composite_quality_score,
    batch_compute_quality_scores,
)


class TestHeuristicQualityScore:
    """Test heuristic quality scoring functionality."""

    def test_compute_heuristic_quality_score_empty_text(self):
        """Test scoring with empty text."""
        score = compute_heuristic_quality_score("")
        assert score == 0.0

        score = compute_heuristic_quality_score("   ")
        assert score == 0.0

    def test_compute_heuristic_quality_score_structured_content(self):
        """Test scoring with structured content."""
        # Test code blocks
        code_text = "Here's some code:\n```python\nprint('hello')\n```"
        score = compute_heuristic_quality_score(code_text)
        assert score > 0.5  # Should be higher due to code blocks

        # Test JSON
        json_text = 'The result is: {"name": "test", "value": 42}'
        score = compute_heuristic_quality_score(json_text)
        assert score > 0.5  # Should be higher due to JSON

        # Test lists
        list_text = "Here are the steps:\n- Step 1\n- Step 2\n- Step 3"
        score = compute_heuristic_quality_score(list_text)
        assert score > 0.5  # Should be higher due to lists

    def test_compute_heuristic_quality_score_ground_truth_comparison(self):
        """Test scoring with ground truth comparison."""
        teacher_output = "The answer is 42"
        ground_truth = "The answer is 42"

        score = compute_heuristic_quality_score(teacher_output, ground_truth)
        assert score > 0.5  # Should be higher due to exact match

        # Test partial match
        teacher_output = "The answer is 42"
        ground_truth = "The answer is forty-two"
        score = compute_heuristic_quality_score(teacher_output, ground_truth)
        # Should still be reasonable due to word overlap

    def test_compute_heuristic_quality_score_length_appropriateness(self):
        """Test scoring based on length appropriateness."""
        # Very short response
        short_text = "Yes"
        score = compute_heuristic_quality_score(short_text)
        assert score < 0.5  # Should be lower for very short responses

        # Very long response
        long_text = "This is a very long response that goes on and on " * 50
        score = compute_heuristic_quality_score(long_text)
        # Length penalties should apply

    def test_compute_heuristic_quality_score_coherence_indicators(self):
        """Test scoring based on coherence indicators."""
        # Text with good structure
        good_text = "To solve this problem, you need to:\n\n1. First, understand the requirements\n2. Then, implement the solution\n3. Finally, test your code\n\nHere's the implementation:\n```python\ndef solve():\n    return 'solution'\n```"
        score = compute_heuristic_quality_score(good_text)
        assert score > 0.7  # Should be high due to multiple structure elements

        # Text with poor structure
        poor_text = "idk lol just do it whatever"
        score = compute_heuristic_quality_score(poor_text)
        assert score < 0.5  # Should be lower due to poor structure

    def test_compute_heuristic_quality_score_prompt_context(self):
        """Test scoring with prompt context."""
        prompt = "Write a function to calculate fibonacci numbers"
        teacher_output = "```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n```"

        score = compute_heuristic_quality_score(teacher_output, prompt=prompt)
        assert score > 0.6  # Should be higher due to relevant code response


class TestJSONValidityScore:
    """Test JSON validity scoring functionality."""

    def test_compute_json_validity_score_valid_json(self):
        """Test scoring with valid JSON."""
        valid_json = '{"name": "test", "value": 42, "nested": {"key": "value"}}'
        score = compute_json_validity_score(valid_json)
        assert score == 1.0

    def test_compute_json_validity_score_invalid_json(self):
        """Test scoring with invalid JSON."""
        invalid_json = '{"name": "test", "value": 42'  # Missing closing brace
        score = compute_json_validity_score(invalid_json)
        assert score < 1.0

    def test_compute_json_validity_score_complex_json(self):
        """Test scoring with complex valid JSON."""
        complex_json = """
        {
            "tool_calls": [
                {
                    "name": "calculator",
                    "arguments": {
                        "expression": "2 + 2",
                        "precision": 2
                    }
                }
            ],
            "response": "The answer is 4"
        }
        """
        score = compute_json_validity_score(complex_json)
        assert score == 1.0

    def test_compute_json_validity_score_partial_json(self):
        """Test scoring with partial JSON content."""
        # Text with some JSON but also other content
        mixed_text = 'Here is the result: {"answer": 42} and some explanation'
        score = compute_json_validity_score(mixed_text)
        assert score > 0.0  # Should detect valid JSON within text

    def test_compute_json_validity_score_no_json(self):
        """Test scoring with no JSON content."""
        plain_text = "This is just plain text with no JSON"
        score = compute_json_validity_score(plain_text)
        assert score == 0.0

    def test_compute_json_validity_score_malformed_json(self):
        """Test scoring with malformed JSON."""
        malformed_cases = [
            '{"unclosed": "string}',
            '{"missing": "comma" "invalid": "json"}',
            '{"nested": {"unclosed": "object"}',
            '["unclosed", "array"',
            '{"valid": "json", "trailing": "comma",}',
        ]

        for malformed in malformed_cases:
            score = compute_json_validity_score(malformed)
            assert score < 1.0

    def test_compute_json_validity_score_empty_text(self):
        """Test scoring with empty text."""
        score = compute_json_validity_score("")
        assert score == 0.0


class TestCodeBlockScore:
    """Test code block scoring functionality."""

    def test_compute_code_block_score_python_code(self):
        """Test scoring with Python code."""
        python_code = '''```python
def calculate_fibonacci(n):
    """Calculate nth Fibonacci number."""
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

result = calculate_fibonacci(10)
print(f"Fibonacci(10) = {result}")
```'''
        score = compute_code_block_score(python_code)
        assert score > 0.8  # Should be high for well-structured Python code

    def test_compute_code_block_score_javascript_code(self):
        """Test scoring with JavaScript code."""
        js_code = """```javascript
function calculateFibonacci(n) {
    if (n <= 1) return n;
    return calculateFibonacci(n-1) + calculateFibonacci(n-2);
}

const result = calculateFibonacci(10);
console.log(`Fibonacci(10) = ${result}`);
```"""
        score = compute_code_block_score(js_code)
        assert score > 0.7  # Should be good for structured JS code

    def test_compute_code_block_score_no_code_blocks(self):
        """Test scoring with no code blocks."""
        plain_text = "This is just plain text explanation without any code"
        score = compute_code_block_score(plain_text)
        assert score == 0.0

    def test_compute_code_block_score_inline_code(self):
        """Test scoring with inline code (not in blocks)."""
        text_with_inline = "Use the `print()` function to output text"
        score = compute_code_block_score(text_with_inline)
        assert score < 0.5  # Should be lower for inline code only

    def test_compute_code_block_score_mixed_languages(self):
        """Test scoring with mixed programming languages."""
        mixed_code = """```python
def hello():
    print("Hello from Python")
```
And here's some SQL:
```sql
SELECT * FROM users WHERE active = 1;
```"""
        score = compute_code_block_score(mixed_code)
        assert score > 0.8  # Should be high for multiple code blocks

    def test_compute_code_block_score_syntax_quality(self):
        """Test scoring based on code syntax quality."""
        # Good syntax
        good_code = """```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)
```"""
        good_score = compute_code_block_score(good_code)

        # Poor syntax
        poor_code = """```python
def factorial n
if n <=1 return 1
return n * factorial(n-1)
```"""
        poor_score = compute_code_block_score(poor_code)

        # Good code should score higher
        assert good_score > poor_score

    def test_compute_code_block_score_empty_code_block(self):
        """Test scoring with empty code block."""
        empty_code = """```
# Empty code block
```"""
        score = compute_code_block_score(empty_code)
        assert score < 0.3  # Should be low for empty blocks


class TestCompositeQualityScore:
    """Test composite quality scoring functionality."""

    def test_compute_composite_quality_score_all_components(self):
        """Test composite scoring with all components."""
        teacher_output = '''To solve this problem, you need to:

1. Parse the input data
2. Apply the algorithm
3. Return the result

```python
def solve_problem(input_data):
    """Solve the problem with the given input."""
    # Parse input
    parsed = parse_input(input_data)

    # Apply algorithm
    result = apply_algorithm(parsed)

    return result
```

The result will be: {"answer": 42, "confidence": 0.95}'''

        scores = compute_composite_quality_score(teacher_output)

        # Should have all scoring components
        assert "heuristic" in scores
        assert "json_validity" in scores
        assert "code_block" in scores
        assert "composite" in scores

        # Composite should be a weighted combination
        assert 0.0 <= scores["composite"] <= 1.0

    def test_compute_composite_quality_score_minimal_text(self):
        """Test composite scoring with minimal text."""
        minimal_text = "Yes"
        scores = compute_composite_quality_score(minimal_text)

        assert isinstance(scores, dict)
        assert "composite" in scores
        # Should still compute all components

    def test_compute_composite_quality_score_structured_content(self):
        """Test composite scoring with highly structured content."""
        structured_text = """# Problem Solution

## Approach
We'll use dynamic programming to solve this efficiently.

## Code Implementation
```python
def optimal_solution(arr):
    n = len(arr)
    dp = [0] * (n + 1)

    for i in range(1, n + 1):
        dp[i] = max(dp[i-1], arr[i-1] + (dp[i-2] if i >= 2 else 0))

    return dp[n]
```

## Results
The algorithm returns: {"result": 42, "time_complexity": "O(n)", "space_complexity": "O(n)"}"""

        scores = compute_composite_quality_score(structured_text)

        # Should have high composite score due to multiple quality indicators
        assert scores["composite"] > 0.8

    def test_compute_composite_quality_score_weights(self):
        """Test that different components have appropriate weights."""
        # Create text that scores high on one component but low on others
        json_only = '{"result": 42, "status": "success"}'
        scores = compute_composite_quality_score(json_only)

        # JSON validity should be high, others lower
        assert scores["json_validity"] > 0.8
        assert scores["composite"] > 0.4  # Should be pulled up by JSON score

    def test_compute_composite_quality_score_empty_text(self):
        """Test composite scoring with empty text."""
        scores = compute_composite_quality_score("")

        assert scores["composite"] == 0.0
        assert scores["heuristic"] == 0.0
        assert scores["json_validity"] == 0.0
        assert scores["code_block"] == 0.0


class TestBatchQualityScoring:
    """Test batch quality scoring functionality."""

    def test_batch_compute_quality_scores_empty_list(self):
        """Test batch scoring with empty list."""
        results = batch_compute_quality_scores([])
        assert results == []

    def test_batch_compute_quality_scores_single_item(self):
        """Test batch scoring with single item."""
        texts = ["This is a test response with some structure."]
        results = batch_compute_quality_scores(texts)

        assert len(results) == 1
        assert isinstance(results[0], dict)
        assert "composite" in results[0]

    def test_batch_compute_quality_scores_multiple_items(self):
        """Test batch scoring with multiple items."""
        texts = [
            "Simple response",
            '```python\nprint("Hello")\n```',
            '{"result": "complex"}',
            """# Structured Response

Here's a detailed explanation with:
- Multiple sections
- Code examples
- JSON output

```javascript
function process() {
    return {status: "ok"};
}
```

Result: {"processed": true, "items": 42}""",
        ]

        results = batch_compute_quality_scores(texts)

        assert len(results) == 4

        # Each result should be a dict with scoring components
        for result in results:
            assert isinstance(result, dict)
            assert "composite" in result
            assert "heuristic" in result
            assert "json_validity" in result
            assert "code_block" in result

        # The last item (highly structured) should have highest score
        scores = [r["composite"] for r in results]
        assert scores[3] == max(scores)  # Last item should have highest score

    def test_batch_compute_quality_scores_with_ground_truth(self):
        """Test batch scoring with ground truth."""
        texts = ["The answer is 42", "Wrong answer", "42 is the answer"]
        ground_truths = ["The answer is 42", "The answer is 42", "The answer is 42"]

        results = batch_compute_quality_scores(texts, ground_truths)

        assert len(results) == 3
        # First item should score higher due to exact match

    def test_batch_compute_quality_scores_mixed_content(self):
        """Test batch scoring with mixed content types."""
        texts = [
            "Plain text response",
            "Response with ```code block```",
            'Response with {"json": "object"}',
            'Response with both ```code``` and {"json": true}',
        ]

        results = batch_compute_quality_scores(texts)

        # Items with structured content should score higher
        scores = [r["composite"] for r in results]
        assert scores[0] < scores[1]  # Plain text < code
        assert scores[0] < scores[2]  # Plain text < JSON
        assert scores[3] > scores[1]  # Both > code only
        assert scores[3] > scores[2]  # Both > JSON only

    def test_batch_compute_quality_scores_error_handling(self):
        """Test batch scoring error handling."""
        # Mix of valid and invalid inputs
        texts = [
            "Valid text",
            "",  # Empty
            None,  # None
            "   ",  # Whitespace only
        ]

        results = batch_compute_quality_scores(texts)

        assert len(results) == 4

        # Should handle all cases without crashing
        for result in results:
            assert isinstance(result, dict)
            assert "composite" in result
            assert isinstance(result["composite"], (int, float))

    def test_batch_compute_quality_scores_large_batch(self):
        """Test batch scoring with large number of items."""
        # Test with reasonably large batch
        texts = [f"Response number {i}" for i in range(100)]

        results = batch_compute_quality_scores(texts)

        assert len(results) == 100

        # All should have valid scores
        for result in results:
            assert "composite" in result
            assert 0.0 <= result["composite"] <= 1.0


class TestQualityScoringIntegration:
    """Test integration of quality scoring components."""

    def test_quality_scoring_consistency(self):
        """Test that scoring is consistent across runs."""
        text = 'Test response with ```code``` and {"json": true}'

        # Run multiple times
        scores1 = compute_composite_quality_score(text)
        scores2 = compute_composite_quality_score(text)

        # Should be identical
        assert scores1 == scores2

    def test_quality_scoring_realistic_examples(self):
        """Test scoring with realistic examples."""
        examples = [
            # Good tool response
            """{"tool_call": {"name": "calculator", "arguments": {"expression": "2+2"}}}""",
            # Good code response
            """```python
def add(a, b):
    return a + b

result = add(2, 2)
```""",
            # Poor response
            "idk lol",
            # Mixed quality
            "The answer is 42 but I'm not sure ```print(42)```",
        ]

        results = batch_compute_quality_scores(examples)

        # Verify ordering makes sense
        scores = [r["composite"] for r in results]
        assert scores[0] > scores[2]  # JSON tool call > poor response
        assert scores[1] > scores[2]  # Code response > poor response

    def test_quality_scoring_performance(self):
        """Test that quality scoring performs adequately."""
        import time

        text = "Sample text for performance testing " * 10

        start_time = time.time()
        for _ in range(100):
            compute_composite_quality_score(text)
        end_time = time.time()

        duration = end_time - start_time
        # Should process 100 items in reasonable time
        assert duration < 1.0  # Less than 1 second
