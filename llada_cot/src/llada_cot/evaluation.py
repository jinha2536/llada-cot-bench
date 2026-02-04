"""Evaluation utilities for answer extraction and comparison."""
import re
from decimal import Decimal, InvalidOperation
from typing import Optional, List

from .config import DatasetType

# Try to import math-verify for better MATH grading
# Falls back to built-in logic if not available
_MATH_VERIFY_AVAILABLE = False
try:
    from math_verify import verify, parse
    _MATH_VERIFY_AVAILABLE = True
except ImportError:
    pass


def grade_math_answer_with_verify(pred: str | None, gold: str | None) -> bool:
    """Grade MATH answer using math-verify library (if available).
    
    Uses HuggingFace's official math-verify for symbolic comparison.
    """
    if not _MATH_VERIFY_AVAILABLE:
        return None  # Signal to use fallback
    
    if pred is None or gold is None:
        return False
    
    try:
        # Parse both answers to sympy expressions
        pred_parsed = parse(pred)
        gold_parsed = parse(gold)
        
        if pred_parsed is None or gold_parsed is None:
            return None  # Fallback to built-in
        
        # Use math-verify's comparison
        return verify(pred_parsed, gold_parsed)
    except Exception:
        return None  # Fallback to built-in


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


def extract_math_answer(text: str | None) -> str | None:
    """Extract answer from MATH dataset responses.
    
    Handles various formats:
    - \\boxed{answer}
    - #### answer
    - The answer is: answer
    - Final answer: answer
    """
    if text is None:
        return None
    
    # 1. Try \boxed{} pattern (most common in MATH)
    # Handle nested braces
    boxed_match = re.search(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if boxed_match:
        return boxed_match.group(1).strip()
    
    # 2. Try #### pattern
    hash_match = re.search(r"####\s*(.+?)(?:\n|$)", text)
    if hash_match:
        return hash_match.group(1).strip()
    
    # 3. Try "answer is" patterns
    answer_patterns = [
        r"(?:final\s+)?answer\s*(?:is|:)\s*\$?([^\$\n]+)\$?",
        r"(?:therefore|thus|hence)[,\s]+(?:the\s+)?(?:answer\s+)?(?:is\s+)?\$?([^\$\n]+)\$?",
        r"=\s*\$?\\boxed\{([^}]+)\}",
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # 4. Try to find the last $...$ expression
    dollar_matches = re.findall(r"\$([^\$]+)\$", text)
    if dollar_matches:
        return dollar_matches[-1].strip()
    
    return None


def normalize_math_answer(answer: str | None) -> str:
    """Normalize MATH answer for comparison.
    
    Handles LaTeX formatting differences.
    """
    if answer is None:
        return ""
    
    s = answer.strip()
    
    # Remove common LaTeX wrappers
    s = re.sub(r"^\$+|\$+$", "", s)  # Remove $ delimiters
    s = re.sub(r"^\\left\s*|\\right\s*$", "", s)  # Remove \left \right
    
    # Normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()
    
    # Remove \text{} wrapper
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    
    # Normalize common LaTeX
    s = s.replace("\\%", "%")
    s = s.replace("\\$", "$")
    s = s.replace("\\,", " ")
    s = s.replace("\\ ", " ")
    s = s.replace("\\!", "")
    s = s.replace("\\;", " ")
    s = s.replace("\\:", " ")
    
    # Normalize fractions: \frac{a}{b} -> (a)/(b)
    s = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\dfrac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\tfrac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)", s)
    
    # Normalize sqrt
    s = re.sub(r"\\sqrt\{([^}]*)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\sqrt\[([^\]]*)\]\{([^}]*)\}", r"(\2)^(1/\1)", s)
    
    # Normalize degrees
    s = s.replace("^\\circ", "°")
    s = s.replace("^{\\circ}", "°")
    s = s.replace("\\circ", "°")
    s = s.replace("\\degree", "°")
    
    # Normalize pi
    s = s.replace("\\pi", "π")
    
    # Normalize common trig functions (remove backslash)
    for func in ["sin", "cos", "tan", "cot", "sec", "csc", 
                 "arcsin", "arccos", "arctan", "log", "ln", "exp"]:
        s = s.replace(f"\\{func}", func)
    
    # Normalize infinity
    s = s.replace("\\infty", "∞")
    s = s.replace("\\infinity", "∞")
    
    # Remove remaining backslashes (careful - some are needed)
    s = re.sub(r"\\([a-zA-Z]+)", r"\1", s)
    
    # Normalize spacing around operators
    s = re.sub(r"\s*([+\-*/=<>])\s*", r" \1 ", s)
    s = re.sub(r"\s+", " ", s).strip()
    
    # Lowercase for comparison
    s = s.lower()
    
    return s


def is_correct_math(pred: str | None, gold: str | None) -> bool:
    """Check if MATH answer is correct.
    
    Tries multiple comparison strategies:
    1. math-verify library (if available) - symbolic comparison
    2. Exact match after normalization
    3. Numeric comparison (if both are numbers)
    4. Simplified string comparison
    """
    if pred is None or gold is None:
        return False
    
    # 1. Try math-verify first (best accuracy)
    if _MATH_VERIFY_AVAILABLE:
        result = grade_math_answer_with_verify(pred, gold)
        if result is not None:  # None means fallback needed
            return result
    
    # Fallback: built-in comparison
    # Normalize both
    pred_norm = normalize_math_answer(pred)
    gold_norm = normalize_math_answer(gold)
    
    # 2. Exact match after normalization
    if pred_norm == gold_norm:
        return True
    
    # 3. Try numeric comparison
    try:
        # Extract numbers
        pred_num = extract_number(pred_norm)
        gold_num = extract_number(gold_norm)
        if pred_num is not None and gold_num is not None:
            if abs(pred_num - gold_num) < 1e-6:
                return True
    except:
        pass
    
    # 4. Remove all spaces and compare
    pred_compact = re.sub(r"\s+", "", pred_norm)
    gold_compact = re.sub(r"\s+", "", gold_norm)
    if pred_compact == gold_compact:
        return True
    
    # 5. Check if one contains the other (for partial matches)
    # e.g., pred="42" and gold="42 degrees"
    if pred_compact in gold_compact or gold_compact in pred_compact:
        # Only if the contained string is substantial
        if len(pred_compact) > 0 and len(gold_compact) > 0:
            ratio = min(len(pred_compact), len(gold_compact)) / max(len(pred_compact), len(gold_compact))
            if ratio > 0.8:  # At least 80% overlap
                return True
    
    return False


def extract_number(s: str) -> Optional[float]:
    """Try to extract a single number from a string."""
    # Remove common non-numeric suffixes
    s = re.sub(r"°|degrees?|%", "", s)
    s = s.strip()
    
    # Try to parse as float
    try:
        return float(s)
    except ValueError:
        pass
    
    # Try to find a number in the string
    match = re.search(r"[-+]?\d*\.?\d+", s)
    if match:
        try:
            return float(match.group())
        except ValueError:
            pass
    
    return None


def extract_countdown_answer(text: str | None) -> str | None:
    """Extract equation from Countdown answer.
    
    Priority:
    1. #### pattern (most reliable)
    2. <answer> tags
    3. "answer is" pattern (last occurrence)
    4. Last equation-like pattern (fallback)
    """
    if text is None:
        return None
    
    # 1. Try #### format (most reliable - explicitly marked answer)
    match = re.search(r"####\s*([^\n]+)", text)
    if match:
        return _clean_countdown_equation(match.group(1))
    
    # 2. Try <answer> tags (TinyZero format)
    match = re.search(r"<answer>\s*([^<]+)\s*</answer>", text)
    if match:
        return _clean_countdown_equation(match.group(1))
    
    # 3. Try "answer is" format - use LAST occurrence to avoid intermediate steps
    matches = list(re.finditer(
        r"(?:final\s+)?(?:answer|equation|result)\s*(?:is|:)\s*:?\s*([^\n]+)", 
        text, re.IGNORECASE
    ))
    if matches:
        return _clean_countdown_equation(matches[-1].group(1))
    
    # 4. Last resort: find the LAST equation-like pattern
    # This helps avoid picking up intermediate calculations
    equation_pattern = r"(\(?\d+\s*[\+\-\*/]\s*[\d\+\-\*/\(\)\s]+\)?)"
    matches = list(re.finditer(equation_pattern, text))
    if matches:
        return _clean_countdown_equation(matches[-1].group(1))
    
    return None


def _clean_countdown_equation(eq: str) -> str:
    """Clean up Countdown equation string.
    
    Removes:
    - Brackets [ ]
    - Trailing "= number" patterns
    - Extra whitespace
    """
    eq = eq.strip()
    
    # Remove brackets (sometimes model outputs [expression])
    eq = re.sub(r"[\[\]]", "", eq)
    
    # Remove trailing "= result" (e.g., "44 + 19 + 35 = 98" -> "44 + 19 + 35")
    eq = re.sub(r"\s*=\s*\d+\.?\d*\s*$", "", eq)
    
    # Remove any text after the equation (e.g., "44+19+35 which equals 98")
    eq = re.sub(r"\s+(?:which|that|equals|is).*$", "", eq, flags=re.IGNORECASE)
    
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
    """Check if Countdown equation is correct (ToT standard scoring).
    
    Verifies:
    1. Equation evaluates to target
    2. Uses ALL available numbers exactly once (no more, no less)
    3. Only uses +, -, *, / operators
    """
    if equation is None:
        return False
    
    try:
        # Clean equation
        equation = equation.strip()
        
        # Only allow digits, operators, parentheses, and spaces
        if not re.match(r'^[\d\+\-\*/\(\)\.\s]+$', equation):
            return False
        
        # Use sympy for accurate calculation (avoids float precision issues)
        try:
            import sympy
            result = sympy.sympify(equation)
        except ImportError:
            # Fallback to eval if sympy not available
            result = eval(equation, {"__builtins__": {}}, {})
        
        # Check if result equals target
        if abs(float(result) - target) > 1e-6:
            return False
        
        # Extract numbers used in equation
        used_numbers = [int(n) for n in re.findall(r'\d+', equation)]
        
        # ToT standard: ALL numbers must be used exactly once
        # Compare sorted lists to check exact match
        if sorted(used_numbers) != sorted(available_numbers):
            return False
        
        return True
        
    except Exception:
        return False


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
    elif dataset_type == DatasetType.MATH:
        # MATH dataset: LaTeX answers
        pred = extract_math_answer(gen_text)
        correct = is_correct_math(pred, str(gold_answer))
    else:
        # BigGSM: numeric answers
        pred = extract_hash_answer(gen_text)
        correct = is_correct(pred, str(gold_answer))
    
    return pred, correct
