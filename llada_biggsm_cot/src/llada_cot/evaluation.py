"""Evaluation utilities for answer extraction and comparison."""

import re
from decimal import Decimal, InvalidOperation


def extract_hash_answer_strict(text: str | None) -> str | None:
    """
    Extract the answer following the #### marker (STRICT - no fallback).
    
    Use this for trace analysis where we need to know exactly when
    the #### pattern appears.
    
    Args:
        text: The generated text to parse.
        
    Returns:
        The extracted number as a string, or None if #### pattern not found.
        
    Examples:
        >>> extract_hash_answer_strict("The answer is #### 42.")
        '42'
        >>> extract_hash_answer_strict("The answer is 42")  # No ####
        None
    """
    if text is None:
        return None
    
    # Only match explicit #### format
    match = re.search(r"####\s*([-+]?\d+(?:[.,]\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")
    
    return None


def extract_hash_answer(text: str | None) -> str | None:
    """
    Extract the answer following the #### marker (with fallback).
    
    Use this for final answer extraction where we want to be lenient.
    
    Args:
        text: The generated text to parse.
        
    Returns:
        The extracted number as a string, or None if not found.
        
    Examples:
        >>> extract_hash_answer("The answer is #### 42.")
        '42'
        >>> extract_hash_answer("Therefore, #### -3.14")
        '-3.14'
        >>> extract_hash_answer("No hash but answer is 42")
        '42'
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


def to_decimal(s: str | None) -> Decimal | None:
    """
    Convert a string to Decimal, handling commas.
    
    Args:
        s: String representation of a number.
        
    Returns:
        Decimal value or None if conversion fails.
    """
    if s is None:
        return None
    
    s = s.replace(",", "").strip()
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def is_correct(pred: str | None, gold: str | None) -> bool:
    """
    Check if predicted answer matches gold answer numerically.
    
    Args:
        pred: Predicted answer string.
        gold: Gold answer string.
        
    Returns:
        True if answers match numerically.
        
    Examples:
        >>> is_correct("42", "42.0")
        True
        >>> is_correct("42", "43")
        False
        >>> is_correct(None, "42")
        False
    """
    pred_dec = to_decimal(pred)
    gold_dec = to_decimal(gold)
    
    if pred_dec is None or gold_dec is None:
        return False
    
    return pred_dec == gold_dec


def compute_metrics(results: list[dict]) -> dict[str, float]:
    """
    Compute aggregate metrics from a list of result dicts.
    
    Args:
        results: List of dicts with 'correct' and 'pred' keys.
        
    Returns:
        Dict with accuracy, parse_rate, and count.
    """
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
