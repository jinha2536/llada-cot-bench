"""Evaluation utilities for answer extraction and comparison."""
import re
from decimal import Decimal, InvalidOperation
from typing import Optional, List

from .config import DatasetType


def extract_hash_answer_strict(text: str | None) -> str | None:
    """
    Extract the answer following the #### marker (STRICT - no fallback).
    
    Use this for trace analysis where we need to know exactly when
    the #### pattern appears.
    """
    if text is None:
        return None
    
    match = re.search(r"####\s*([-+]?\d+(?:[.,]\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")
    
    return None


def extract_hash_answer(text: str | None) -> str | None:
    """
    Extract the answer following the #### marker (with fallback).
    
    Use this for final answer extraction where we want to be lenient.
    """
    if text is None:
        return None
    
    # First try the explicit #### format
    match = re.search(r"####\s*([-+]?\d+(?:[.,]\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")
    
    # Fallback: extract the last number in the text
    numbers = re.findall(r"[-+]?\d+(?:[.,]\d+)?", text.replace(",", ""))
    return numbers[-1] if numbers else None


def extract_countdown_answer(text: str | None) -> str | None:
    """Extract equation from Countdown answer.
    
    Looks for #### pattern or <answer> tags.
    """
    if text is None:
        return None
    
    # Try #### format
    match = re.search(r"####\s*([^\n]+)", text)
    if match:
        return _clean_equation(match.group(1))
    
    # Try <answer> tags (TinyZero format)
    match = re.search(r"<answer>\s*([^<]+)\s*</answer>", text)
    if match:
        return _clean_equation(match.group(1))
    
    # Try "answer is" format
    match = re.search(r"(?:answer|equation|result)\s*(?:is|:)\s*([^\n]+)", 
                     text, re.IGNORECASE)
    if match:
        return _clean_equation(match.group(1))
    
    # Last resort: find equation-like pattern
    match = re.search(r"(\d+\s*[\+\-\*/\(\)]\s*[\d\+\-\*/\(\)\s]+)", text)
    if match:
        return _clean_equation(match.group(1))
    
    return None


def _clean_equation(eq: str) -> str:
    """Clean up equation string."""
    eq = re.sub(r"[<>\[\]]", "", eq)
    eq = eq.replace("=", "").strip()
    eq = re.sub(r"\s*=?\s*\d+\s*$", "", eq)
    return eq.strip()


def to_decimal(s: str | None) -> Decimal | None:
    """Convert a string to Decimal, handling commas."""
    if s is None:
        return None
    
    s = s.replace(",", "").strip()
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def is_correct(pred: str | None, gold: str | None) -> bool:
    """Check if predicted answer matches gold answer numerically."""
    pred_dec = to_decimal(pred)
    gold_dec = to_decimal(gold)
    
    if pred_dec is None or gold_dec is None:
        return False
    
    return pred_dec == gold_dec


def is_correct_countdown(
    equation: str | None, 
    target: int, 
    available_numbers: List[int]
) -> bool:
    """Check if Countdown equation is correct.
    
    Verifies:
    1. Equation evaluates to target
    2. Only uses available numbers (each at most once)
    """
    if equation is None:
        return False
    
    try:
        # Evaluate the equation
        result = _safe_eval(equation)
        if result is None:
            return False
        
        # Check if result equals target
        if abs(result - target) > 1e-6:
            return False
        
        # Check number usage
        used_numbers = _extract_numbers(equation)
        available = available_numbers.copy()
        for num in used_numbers:
            if num in available:
                available.remove(num)
            else:
                return False  # Used unavailable number or used twice
        
        return True
        
    except Exception:
        return False


def _safe_eval(equation: str) -> Optional[float]:
    """Safely evaluate a math equation."""
    # Only allow digits, operators, parentheses, and spaces
    if not re.match(r'^[\d\+\-\*/\(\)\.\s]+$', equation):
        return None
    
    try:
        result = eval(equation, {"__builtins__": {}}, {})
        return float(result)
    except:
        return None


def _extract_numbers(equation: str) -> List[int]:
    """Extract all numbers used in an equation."""
    numbers = re.findall(r'\d+', equation)
    return [int(n) for n in numbers]


def compute_metrics(results: list[dict]) -> dict[str, float]:
    """Compute aggregate metrics from a list of result dicts."""
    if not results:
        return {"accuracy": 0.0, "parse_rate": 0.0, "count": 0}
    
    n = len(results)
    correct = sum(1 for r in results if r.get("correct", False))
    parsed = sum(1 for r in results if r.get("pred") is not None)
    
    return {
        "accuracy": correct / n,
        "parse_rate": parsed / n,
        "count": n,
    }


def extract_and_evaluate(
    gen_text: str,
    gold_answer,
    dataset_type: DatasetType,
    numbers: List[int] = None,
    target: int = None,
) -> tuple[str | None, bool]:
    """Extract answer and evaluate correctness based on dataset type.
    
    Returns:
        Tuple of (predicted_answer, is_correct)
    """
    if dataset_type == DatasetType.COUNTDOWN:
        pred = extract_countdown_answer(gen_text)
        correct = is_correct_countdown(pred, target, numbers) if pred else False
    else:
        # BigGSM and MATH
        pred = extract_hash_answer(gen_text)
        correct = is_correct(pred, str(gold_answer))
    
    return pred, correct
