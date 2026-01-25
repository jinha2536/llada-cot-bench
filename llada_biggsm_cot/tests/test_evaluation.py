"""Unit tests for LLaDA CoT benchmark."""

import pytest

from llada_cot.evaluation import (
    extract_hash_answer,
    is_correct,
    to_decimal,
    compute_metrics,
)
from llada_cot.prompts import PromptMethod, build_prompt, get_available_methods


class TestExtractHashAnswer:
    """Tests for answer extraction."""
    
    def test_standard_format(self):
        assert extract_hash_answer("The answer is #### 42.") == "42"
        assert extract_hash_answer("Therefore, #### -3.14") == "-3.14"
        assert extract_hash_answer("####100") == "100"
    
    def test_with_spacing(self):
        assert extract_hash_answer("####   123") == "123"
        assert extract_hash_answer("####\n456") == "456"
    
    def test_fallback_last_number(self):
        # When no #### format, should return last number
        assert extract_hash_answer("The result is 42") == "42"
        assert extract_hash_answer("First 10, then 20, finally 30") == "30"
    
    def test_none_input(self):
        assert extract_hash_answer(None) is None
    
    def test_no_numbers(self):
        assert extract_hash_answer("No numbers here") is None


class TestToDecimal:
    """Tests for decimal conversion."""
    
    def test_integers(self):
        assert to_decimal("42") == 42
        assert to_decimal("-17") == -17
    
    def test_floats(self):
        assert to_decimal("3.14") == pytest.approx(3.14)
        assert to_decimal("-0.5") == pytest.approx(-0.5)
    
    def test_with_commas(self):
        assert to_decimal("1,234") == 1234
        assert to_decimal("1,234,567.89") == pytest.approx(1234567.89)
    
    def test_none(self):
        assert to_decimal(None) is None
    
    def test_invalid(self):
        assert to_decimal("abc") is None
        assert to_decimal("") is None


class TestIsCorrect:
    """Tests for answer comparison."""
    
    def test_exact_match(self):
        assert is_correct("42", "42") is True
        assert is_correct("3.14", "3.14") is True
    
    def test_numeric_equivalence(self):
        assert is_correct("42", "42.0") is True
        assert is_correct("42.00", "42") is True
    
    def test_wrong_answer(self):
        assert is_correct("42", "43") is False
        assert is_correct("3.14", "3.15") is False
    
    def test_none_values(self):
        assert is_correct(None, "42") is False
        assert is_correct("42", None) is False
        assert is_correct(None, None) is False


class TestComputeMetrics:
    """Tests for aggregate metrics computation."""
    
    def test_all_correct(self):
        results = [
            {"correct": True, "pred": "42"},
            {"correct": True, "pred": "100"},
        ]
        metrics = compute_metrics(results)
        assert metrics["accuracy"] == 1.0
        assert metrics["parse_rate"] == 1.0
        assert metrics["count"] == 2
    
    def test_partial_correct(self):
        results = [
            {"correct": True, "pred": "42"},
            {"correct": False, "pred": "wrong"},
            {"correct": False, "pred": None},
        ]
        metrics = compute_metrics(results)
        assert metrics["accuracy"] == pytest.approx(1/3)
        assert metrics["parse_rate"] == pytest.approx(2/3)
    
    def test_empty_results(self):
        metrics = compute_metrics([])
        assert metrics["accuracy"] == 0.0
        assert metrics["count"] == 0


class TestBuildPrompt:
    """Tests for prompt building."""
    
    def test_zero_cot(self):
        prompt = build_prompt("Zero-CoT", "What is 2+2?")
        assert "Let's think step by step" in prompt
        assert "What is 2+2?" in prompt
        assert "#### <number>" in prompt
    
    def test_complex_cot(self):
        prompt = build_prompt("Complex-CoT", "Solve x=5")
        assert "thoroughly" in prompt
        assert "detail" in prompt
    
    def test_marp(self):
        prompt = build_prompt("MARP", "Calculate 10*10")
        assert "parallel" in prompt
        assert "up to 5" in prompt
    
    def test_diff_marp(self):
        prompt = build_prompt("Diff-MARP", "Find the sum")
        assert "parallel" in prompt
        assert "simple" in prompt
    
    def test_enum_method(self):
        prompt = build_prompt(PromptMethod.ZERO_COT, "Test")
        assert "step by step" in prompt
    
    def test_invalid_method(self):
        with pytest.raises(ValueError, match="Unknown method"):
            build_prompt("InvalidMethod", "Test")


class TestGetAvailableMethods:
    """Tests for method listing."""
    
    def test_returns_all_methods(self):
        methods = get_available_methods()
        assert "Zero-CoT" in methods
        assert "Complex-CoT" in methods
        assert "MARP" in methods
        assert "Diff-MARP" in methods
        assert len(methods) == 4
