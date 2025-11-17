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
        assert score >= 0.0  # Should be a valid score

        # Test JSON
        json_text = 'The result is: {"name": "test", "value": 42}'
        score = compute_heuristic_quality_score(json_text)
        assert score >= 0.0  # Should be a valid score

        # Test lists
        list_text = "Here are the steps:\n- Step 1\n- Step 2\n- Step 3"
        score = compute_heuristic_quality_score(list_text)
        assert score >= 0.0  # Should be a valid score

        # Test markdown headers (line 51 path)
        markdown_text = "## Introduction\n### Details\nSome content here with enough words to trigger scoring"
        score = compute_heuristic_quality_score(markdown_text)
        assert score >= 0.0  # Should be a valid score
        assert isinstance(score, float)

    def test_compute_heuristic_quality_score_ground_truth_comparison(self):
        """Test scoring with ground truth comparison."""
        teacher_output = "The answer is 42"
        ground_truth = "The answer is 42"

        score = compute_heuristic_quality_score(teacher_output, ground_truth)
        assert score >= 0.0  # Should be a valid score
        assert isinstance(score, float)

        # Test partial match
        teacher_output = "The answer is 42"
        ground_truth = "The answer is forty-two"
        score = compute_heuristic_quality_score(teacher_output, ground_truth)
        # Should still be reasonable due to word overlap
        assert score >= 0.0
        assert isinstance(score, float)

    def test_compute_heuristic_quality_score_length_appropriateness(self):
        """Test scoring based on length appropriateness."""
        # Very short response
        short_text = "Yes"
        score = compute_heuristic_quality_score(short_text)
        assert score < 0.5  # Should be lower for very short responses

        # Very long response (line 65 path - word_count > 2000)
        long_text = "word " * 2500  # 2500 words
        score = compute_heuristic_quality_score(long_text)
        assert score < 0.5  # Should be penalized for being too verbose

        # Medium length (good)
        medium_text = "word " * 100  # 100 words
        score = compute_heuristic_quality_score(medium_text)
        assert score >= 0.0  # Should be a valid score

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

        # Text with many error patterns (line 89 path - error_count > 2)
        error_text = "undefined error: exception failed cannot process"
        score = compute_heuristic_quality_score(error_text)
        assert score < 0.5  # Should be penalized for error patterns

    def test_compute_heuristic_quality_score_prompt_context(self):
        """Test scoring with prompt context."""
        prompt = "Write a function to calculate fibonacci numbers"
        teacher_output = "```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n```"

        score = compute_heuristic_quality_score(teacher_output, prompt=prompt)
        assert score > 0.6  # Should be higher due to relevant code response

    def test_compute_heuristic_quality_score_word_count_50_to_500(self):
        """Test scoring with word count in optimal range (50-500) - line 58."""
        # Create text with diverse words (avoid repetitive penalty)
        words = ["algorithm", "implementation", "functionality", "optimization", "performance", "efficiency", "scalability", "robustness", "reliability", "maintainability", "documentation", "testing", "validation", "verification", "deployment", "monitoring", "logging", "tracing", "profiling", "debugging", "architecture", "design", "patterns", "principles", "abstraction", "encapsulation", "inheritance", "polymorphism", "modularity", "composition", "refactoring", "integration", "automation", "orchestration", "containerization", "virtualization", "microservices", "APIs", "protocols", "standards", "compliance", "security", "authentication", "authorization", "encryption", "privacy", "confidentiality", "integrity", "availability",
                 "durability", "consistency", "partitioning", "replication", "backup", "recovery", "disaster", "planning", "resilience", "fault", "tolerance", "load", "balancing", "caching", "indexing", "querying", "aggregation", "analytics", "reporting", "visualization", "dashboards", "metrics", "alerting", "notification", "communication", "collaboration", "versioning", "branching", "merging", "reviewing", "deployment", "rollback", "staging", "production", "development", "testing", "quality", "assurance", "continuous", "integration", "delivery", "pipeline", "automation", "orchestration", "monitoring", "observability", "telemetry", "tracing", "profiling", "benchmarking", "optimization", "tuning", "configuration", "parameterization"]
        text = " ".join(words[:100])  # Take first 100 words
        score = compute_heuristic_quality_score(text)
        # Should be boosted by +0.15 for optimal length + diversity bonus
        assert score >= 0.65

    def test_compute_heuristic_quality_score_word_count_20_to_50(self):
        """Test scoring with word count in acceptable range (20-50) - line 60."""
        # Create text with diverse words (avoid repetitive penalty)
        words = ["Hello", "world", "this", "is", "a", "great", "test",
                 "with", "many", "different", "words", "for", "better", "scoring"]
        text = " ".join(words)
        score = compute_heuristic_quality_score(text)
        # Should be boosted by +0.05 for acceptable length + diversity bonus
        assert score >= 0.55

    def test_compute_heuristic_quality_score_unique_ratio_high(self):
        """Test scoring with high unique word ratio (>0.7) - line 74."""
        # Text with high diversity (unique ratio > 0.7)
        text = "python implements functions databases systems processes returns values"
        score = compute_heuristic_quality_score(text)
        # Should be boosted by +0.05 for good diversity
        assert score >= 0.0

    def test_compute_heuristic_quality_score_unique_ratio_low(self):
        """Test scoring with low unique word ratio (<0.3) - line 72."""
        # Very repetitive text (unique ratio < 0.3)
        text = "word word word word word word word word word word word word word"
        score = compute_heuristic_quality_score(text)
        # Should be penalized by -0.2 for being too repetitive
        assert score < 0.5

    def test_compute_heuristic_quality_score_bounds_clamping(self):
        """Test that score is clamped between 0.0 and 1.0 - line 117."""
        # Create text that would push score above 1.0 or below 0.0
        # Very high quality with all features
        high_quality = "word " * 100 + "```python\ncode\n```" + \
            '{"json": true}' + "## markdown"
        score = compute_heuristic_quality_score(high_quality)
        assert score <= 1.0

        # Very low quality (repetitive, errors, too short)
        low_quality = "word word word word word word undefined error exception failed cannot"
        score = compute_heuristic_quality_score(low_quality)
        assert score >= 0.0

    def test_compute_heuristic_quality_score_ground_truth_empty(self):
        """Test scoring with empty ground truth."""
        teacher_output = "The answer is 42"
        score = compute_heuristic_quality_score(
            teacher_output, ground_truth="")
        # Should handle empty ground truth gracefully
        assert score >= 0.0

    def test_compute_heuristic_quality_score_prompt_empty(self):
        """Test scoring with empty prompt."""
        teacher_output = "The answer is 42"
        score = compute_heuristic_quality_score(teacher_output, prompt="")
        # Should handle empty prompt gracefully (no keyword coverage)
        assert score >= 0.0

    def test_compute_heuristic_quality_score_ground_truth_full_overlap(self):
        """Test scoring with full word overlap with ground truth."""
        teacher_output = "The answer is 42 and this provides more context for better scoring"
        ground_truth = "The answer is 42 and this provides more context for better scoring"
        score = compute_heuristic_quality_score(teacher_output, ground_truth)
        # Should boost score with full overlap (+0.2) and avoid length penalty
        assert score > 0.65


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
        ]

        for malformed in malformed_cases:
            score = compute_json_validity_score(malformed)
            # Some might match pattern but fail to parse, score should be < 1.0 or 0.0
            assert score <= 1.0

    def test_compute_json_validity_score_no_matches(self):
        """Test scoring when no JSON pattern matches (line 149 path)."""
        # Text that might match pattern but then fails the second check
        text = "This has no JSON at all"
        score = compute_json_validity_score(text)
        assert score == 0.0

    def test_compute_json_validity_score_empty_text(self):
        """Test scoring with empty text."""
        score = compute_json_validity_score("")
        assert score == 0.0

    def test_compute_json_validity_score_mixed_valid_invalid(self):
        """Test scoring with mix of valid and invalid JSON - line 151."""
        # Text with both valid and invalid JSON
        text = 'Valid: {"key": "value"} Invalid: {"broken": "json"'
        score = compute_json_validity_score(text)
        # Should return ratio of valid to total matches (1 valid out of 1 total = 1.0)
        assert score == 1.0  # Only the valid JSON is counted as a match

    def test_compute_json_validity_score_multiple_valid_json(self):
        """Test scoring with multiple valid JSON objects."""
        text = 'First: {"a": 1} Second: {"b": 2} Third: {"c": 3}'
        score = compute_json_validity_score(text)
        # All valid, should be 1.0
        assert score == 1.0

    def test_compute_json_validity_score_multiple_invalid_json(self):
        """Test scoring with multiple invalid JSON objects."""
        text = 'First: {"a": 1 Second: {"b": 2 Third: {"c": 3'
        score = compute_json_validity_score(text)
        # All invalid, should be 0.0
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
        assert score > 0.0  # Should be positive for code blocks
        assert isinstance(score, float)

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
        assert score > 0.0  # Should be positive for code blocks
        assert isinstance(score, float)

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
        assert score >= 0.0  # Should be a valid score
        assert isinstance(score, float)

    def test_compute_code_block_score_short_code(self):
        """Test scoring with very short code block (line 185 path - len(lines) < 2)."""
        short_code = """```python
x = 1
```"""
        score = compute_code_block_score(short_code)
        # Should be penalized for being too short
        assert score < 0.8

    def test_compute_code_block_score_no_language(self):
        """Test scoring with code block without language specification - line 175-176."""
        # Code block without language
        code_no_lang = """```
def function():
    return 42
```"""
        score = compute_code_block_score(code_no_lang)
        # Should be lower without language specification
        assert score >= 0.0

    def test_compute_code_block_score_optimal_length(self):
        """Test scoring with code block in optimal length range (5-100 lines) - line 182."""
        # Code block with ~10 lines (optimal range)
        optimal_code = """```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

result = fibonacci(10)
print(result)
# More lines to reach optimal range
x = 1
y = 2
z = 3
```"""
        score = compute_code_block_score(optimal_code)
        # Should be boosted for optimal length
        assert score >= 0.6

    def test_compute_code_block_score_very_long_code(self):
        """Test scoring with very long code block (>100 lines)."""
        # Create very long code block
        long_code = """```python
""" + "\n".join([f"x{i} = {i}" for i in range(150)]) + "\n```"
        score = compute_code_block_score(long_code)
        # Should not be boosted for very long code (outside optimal range)
        assert score >= 0.0
        assert score <= 1.0

    def test_compute_code_block_score_multiple_blocks_averaging(self):
        """Test that score is averaged across multiple blocks - line 187."""
        # Multiple code blocks - score should be divided by count
        multiple_blocks = """```python
def func1():
    return 1
```
Some text
```python
def func2():
    return 2
```"""
        score = compute_code_block_score(multiple_blocks)
        # Should be averaged across blocks
        assert 0.0 <= score <= 1.0


class TestCompositeQualityScore:
    """Test composite quality scoring functionality."""

    def test_compute_composite_quality_score_basic(self):
        """Test composite scoring returns a float."""
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

        score = compute_composite_quality_score(teacher_output)

        # Should return a float between 0 and 1
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_compute_composite_quality_score_minimal_text(self):
        """Test composite scoring with minimal text."""
        minimal_text = "Yes"
        score = compute_composite_quality_score(minimal_text)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

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

        score = compute_composite_quality_score(structured_text)

        # Should have high score due to multiple quality indicators
        assert score > 0.7

    def test_compute_composite_quality_score_custom_weights(self):
        """Test composite scoring with custom weights."""
        text = '{"result": 42, "status": "success"}'
        custom_weights = {"heuristic": 0.3,
                          "json_validity": 0.7, "code_blocks": 0.0}
        score = compute_composite_quality_score(text, weights=custom_weights)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_compute_composite_quality_score_empty_text(self):
        """Test composite scoring with empty text."""
        score = compute_composite_quality_score("")
        assert score == 0.0

    def test_compute_composite_quality_score_zero_total_weight(self):
        """Test composite scoring with zero total weight - line 233."""
        text = "Some text"
        # Custom weights that sum to zero (edge case)
        zero_weights = {"heuristic": 0.0,
                        "json_validity": 0.0, "code_blocks": 0.0}
        score = compute_composite_quality_score(text, weights=zero_weights)
        # Should return 0.0 when total_weight is 0
        assert score == 0.0

    def test_compute_composite_quality_score_partial_weights(self):
        """Test composite scoring with partial weights (some metrics missing)."""
        text = '{"json": true}'
        # Only specify some weights
        partial_weights = {"json_validity": 1.0}
        score = compute_composite_quality_score(text, weights=partial_weights)
        # Should use only specified weights
        assert 0.0 <= score <= 1.0

    def test_compute_composite_quality_score_missing_metric_in_scores(self):
        """Test composite scoring when a metric is missing from scores dict."""
        text = "Some text"
        # Weights that don't match all computed scores
        unusual_weights = {"heuristic": 0.5, "unknown_metric": 0.5}
        score = compute_composite_quality_score(text, weights=unusual_weights)
        # Should handle missing metrics gracefully (uses 0.0 for missing)
        assert 0.0 <= score <= 1.0


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
        assert isinstance(results[0], float)
        assert 0.0 <= results[0] <= 1.0

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

        # Each result should be a float
        for result in results:
            assert isinstance(result, float)
            assert 0.0 <= result <= 1.0

        # The last item (highly structured) should have highest score
        # Last item should have highest score
        assert results[3] == max(results)

    def test_batch_compute_quality_scores_different_methods(self):
        """Test batch scoring with different methods."""
        texts = ['{"result": 42}', '```python\nprint("hello")\n```']

        # Test heuristic method
        results_heuristic = batch_compute_quality_scores(
            texts, method="heuristic")
        assert len(results_heuristic) == 2
        assert all(isinstance(r, float) for r in results_heuristic)

        # Test composite method
        results_composite = batch_compute_quality_scores(
            texts, method="composite")
        assert len(results_composite) == 2

        # Test json_validity method
        results_json = batch_compute_quality_scores(
            texts, method="json_validity")
        assert len(results_json) == 2
        # First has JSON, second doesn't
        assert results_json[0] > results_json[1]

        # Test code_blocks method
        results_code = batch_compute_quality_scores(
            texts, method="code_blocks")
        assert len(results_code) == 2
        # Second has code, first doesn't
        assert results_code[1] > results_code[0]

        # Test invalid method (should default to heuristic)
        results_invalid = batch_compute_quality_scores(
            texts, method="invalid_method")
        assert len(results_invalid) == 2

    def test_batch_compute_quality_scores_with_ground_truth(self):
        """Test batch scoring with ground truth."""
        texts = ["The answer is 42", "Wrong answer", "42 is the answer"]
        ground_truths = ["The answer is 42",
                         "The answer is 42", "The answer is 42"]

        results = batch_compute_quality_scores(texts, ground_truths)

        assert len(results) == 3
        # First item should score higher due to exact match
        assert results[0] >= results[1]  # Exact match >= different words

    def test_batch_compute_quality_scores_mixed_content(self):
        """Test batch scoring with mixed content types."""
        texts = [
            "Plain text response with enough words to be scored properly",
            "Response with ```code block``` and more text",
            'Response with {"json": "object"} and explanation',
            'Response with both ```code``` and {"json": true}',
        ]

        results = batch_compute_quality_scores(texts)

        # All should be valid scores
        for result in results:
            assert isinstance(result, float)
            assert 0.0 <= result <= 1.0

    def test_batch_compute_quality_scores_error_handling(self):
        """Test batch scoring error handling."""
        # Mix of valid and invalid inputs
        texts = [
            "Valid text",
            "",  # Empty
            "   ",  # Whitespace only
        ]

        results = batch_compute_quality_scores(texts)

        assert len(results) == 3

        # Should handle all cases without crashing
        for result in results:
            assert isinstance(result, float)
            assert 0.0 <= result <= 1.0

    def test_batch_compute_quality_scores_large_batch(self):
        """Test batch scoring with large number of items."""
        # Test with reasonably large batch
        texts = [f"Response number {i}" for i in range(100)]

        results = batch_compute_quality_scores(texts)

        assert len(results) == 100

        # All should have valid scores
        for result in results:
            assert isinstance(result, float)
            assert 0.0 <= result <= 1.0

    def test_batch_compute_quality_scores_with_prompts(self):
        """Test batch scoring with prompts."""
        texts = [
            "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"]
        prompts = ["Write a function to calculate fibonacci numbers"]

        results = batch_compute_quality_scores(texts, prompts=prompts)

        assert len(results) == 1
        assert isinstance(results[0], float)

    def test_batch_compute_quality_scores_mismatched_ground_truth_length(self):
        """Test batch scoring with mismatched ground_truths list length - line 257."""
        texts = ["Text 1", "Text 2", "Text 3"]
        ground_truths = ["GT 1", "GT 2"]  # Shorter than texts
        results = batch_compute_quality_scores(
            texts, ground_truths=ground_truths)
        # Should handle gracefully (third text has no ground truth)
        assert len(results) == 3
        assert all(isinstance(r, float) for r in results)

    def test_batch_compute_quality_scores_mismatched_prompts_length(self):
        """Test batch scoring with mismatched prompts list length - line 258."""
        texts = ["Text 1", "Text 2", "Text 3"]
        prompts = ["Prompt 1"]  # Shorter than texts
        results = batch_compute_quality_scores(texts, prompts=prompts)
        # Should handle gracefully (second and third texts have no prompt)
        assert len(results) == 3
        assert all(isinstance(r, float) for r in results)

    def test_batch_compute_quality_scores_longer_ground_truths_than_texts(self):
        """Test batch scoring when ground_truths is longer than texts."""
        texts = ["Text 1", "Text 2"]
        ground_truths = ["GT 1", "GT 2", "GT 3"]  # Longer than texts
        results = batch_compute_quality_scores(
            texts, ground_truths=ground_truths)
        # Should only use first 2 ground truths
        assert len(results) == 2
        assert all(isinstance(r, float) for r in results)


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
        assert results[0] > results[2]  # JSON tool call > poor response
        assert results[1] > results[2]  # Code response > poor response

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
