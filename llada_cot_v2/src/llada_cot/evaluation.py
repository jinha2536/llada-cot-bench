"""Evaluation utilities for answer extraction and comparison."""
import re
from decimal import Decimal, InvalidOperation
from typing import Optional, List

from .config import DatasetType

# Try to import math-verify for better MATH grading
_MATH_VERIFY_AVAILABLE = False
try:
    from math_verify import verify, parse
    _MATH_VERIFY_AVAILABLE = True
except ImportError:
    pass


def grade_math_answer_with_verify(pred: str | None, gold: str | None) -> bool | None:
    """Grade MATH answer using math-verify library (if available)."""
    if not _MATH_VERIFY_AVAILABLE:
        return None
    if pred is None or gold is None:
        return False
    try:
        pred_parsed = parse(pred)
        gold_parsed = parse(gold)
        if pred_parsed is None or gold_parsed is None:
            return None
        return verify(pred_parsed, gold_parsed)
    except Exception:
        return None


def extract_hash_answer_strict(text: str | None) -> str | None:
    """Extract the answer following #### (STRICT — no fallback).

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
    """Extract the answer following #### (with last-number fallback)."""
    if text is None:
        return None
    match = re.search(r"####\s*([-+]?\d+(?:[.,]\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")
    numbers = re.findall(r"[-+]?\d+(?:[.,]\d+)?", text.replace(",", ""))
    return numbers[-1] if numbers else None


def extract_math_answer(text: str | None) -> str | None:
    """Extract answer from MATH dataset responses."""
    if text is None:
        return None

    # 1. \\boxed{}
    boxed_match = re.search(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if boxed_match:
        return boxed_match.group(1).strip()

    # 2. ####
    hash_match = re.search(r"####\s*(.+?)(?:\n|$)", text)
    if hash_match:
        return hash_match.group(1).strip()

    # 3. "answer is" patterns
    answer_patterns = [
        r"(?:final\s+)?answer\s*(?:is|:)\s*\$?([^\$\n]+)\$?",
        r"(?:therefore|thus|hence)[,\s]+(?:the\s+)?(?:answer\s+)?(?:is\s+)?\$?([^\$\n]+)\$?",
        r"=\s*\$?\\boxed\{([^}]+)\}",
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # 4. Last $...$ expression
    dollar_matches = re.findall(r"\$([^\$]+)\$", text)
    if dollar_matches:
        return dollar_matches[-1].strip()

    return None


def normalize_math_answer(answer: str | None) -> str:
    """Normalize MATH answer for comparison."""
    if answer is None:
        return ""
    s = answer.strip()
    s = re.sub(r"^\$+|\$+$", "", s)
    s = re.sub(r"^\\left\s*|\\right\s*$", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = s.replace("\\%", "%")
    s = s.replace("\\$", "$")
    s = s.replace("\\,", " ")
    s = s.replace("\\ ", " ")
    s = s.replace("\\!", "")
    s = s.replace("\\;", " ")
    s = s.replace("\\:", " ")
    s = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\dfrac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\tfrac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\sqrt\{([^}]*)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\sqrt\[([^\]]*)\]\{([^}]*)\}", r"(\2)^(1/\1)", s)
    s = s.replace("^\\circ", "°")
    s = s.replace("^{\\circ}", "°")
    s = s.replace("\\circ", "°")
    s = s.replace("\\degree", "°")
    s = s.replace("\\pi", "π")
    for func in ["sin", "cos", "tan", "cot", "sec", "csc",
                 "arcsin", "arccos", "arctan", "log", "ln", "exp"]:
        s = s.replace(f"\\{func}", func)
    s = s.replace("\\infty", "∞")
    s = s.replace("\\infinity", "∞")
    s = re.sub(r"\\([a-zA-Z]+)", r"\1", s)
    s = re.sub(r"\s*([+\-*/=<>])\s*", r" \1 ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.lower()
    return s


def is_correct_math(pred: str | None, gold: str | None) -> bool:
    """Check if MATH answer is correct."""
    if pred is None or gold is None:
        return False

    if _MATH_VERIFY_AVAILABLE:
        result = grade_math_answer_with_verify(pred, gold)
        if result is not None:
            return result

    pred_norm = normalize_math_answer(pred)
    gold_norm = normalize_math_answer(gold)

    if pred_norm == gold_norm:
        return True

    try:
        pred_num = extract_number(pred_norm)
        gold_num = extract_number(gold_norm)
        if pred_num is not None and gold_num is not None:
            if abs(pred_num - gold_num) < 1e-6:
                return True
    except Exception:
        pass

    pred_compact = re.sub(r"\s+", "", pred_norm)
    gold_compact = re.sub(r"\s+", "", gold_norm)
    if pred_compact == gold_compact:
        return True

    if pred_compact in gold_compact or gold_compact in pred_compact:
        if len(pred_compact) > 0 and len(gold_compact) > 0:
            ratio = min(len(pred_compact), len(gold_compact)) / max(len(pred_compact), len(gold_compact))
            if ratio > 0.8:
                return True

    return False


def extract_number(s: str) -> Optional[float]:
    """Try to extract a single number from a string."""
    s = re.sub(r"°|degrees?|%", "", s).strip()
    try:
        return float(s)
    except ValueError:
        pass
    match = re.search(r"[-+]?\d*\.?\d+", s)
    if match:
        try:
            return float(match.group())
        except ValueError:
            pass
    return None


def extract_countdown_answer(text: str | None) -> str | None:
    """Extract equation from Countdown answer."""
    if text is None:
        return None

    match = re.search(r"####\s*([^\n]+)", text)
    if match:
        return _clean_countdown_equation(match.group(1))

    match = re.search(r"<answer>\s*([^<]+)\s*</answer>", text)
    if match:
        return _clean_countdown_equation(match.group(1))

    matches = list(re.finditer(
        r"(?:final\s+)?(?:answer|equation|result)\s*(?:is|:)\s*:?\s*([^\n]+)",
        text, re.IGNORECASE
    ))
    if matches:
        return _clean_countdown_equation(matches[-1].group(1))

    equation_pattern = r"(\(?\d+\s*[\+\-\*/]\s*[\d\+\-\*/\(\)\s]+\)?)"
    matches = list(re.finditer(equation_pattern, text))
    if matches:
        return _clean_countdown_equation(matches[-1].group(1))

    return None


def _clean_countdown_equation(eq: str) -> str:
    eq = eq.strip()
    eq = re.sub(r"[\[\]]", "", eq)
    eq = re.sub(r"\s*=\s*\d+\.?\d*\s*$", "", eq)
    eq = re.sub(r"\s+(?:which|that|equals|is).*$", "", eq, flags=re.IGNORECASE)
    return eq.strip()


def to_decimal(s: str | None) -> Decimal | None:
    if s is None:
        return None
    s = s.replace(",", "").strip()
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def is_correct(pred: str | None, gold: str | None) -> bool:
    pred_dec = to_decimal(pred)
    gold_dec = to_decimal(gold)
    if pred_dec is None or gold_dec is None:
        return False
    return pred_dec == gold_dec


def is_correct_countdown(
    equation: str | None,
    target: int,
    available_numbers: List[int],
) -> bool:
    """Check if Countdown equation is correct (ToT standard scoring)."""
    if equation is None:
        return False
    try:
        equation = equation.strip()
        if not re.match(r'^[\d\+\-\*/\(\)\.\s]+$', equation):
            return False
        try:
            import sympy
            result = sympy.sympify(equation)
        except ImportError:
            result = eval(equation, {"__builtins__": {}}, {})
        if abs(float(result) - target) > 1e-6:
            return False
        used_numbers = [int(n) for n in re.findall(r'\d+', equation)]
        if sorted(used_numbers) != sorted(available_numbers):
            return False
        return True
    except Exception:
        return False


def compute_metrics(results: list[dict]) -> dict[str, float]:
    if not results:
        return {"accuracy": 0.0, "parse_rate": 0.0, "count": 0}
    n = len(results)
    correct = sum(1 for r in results if r.get("correct", False))
    parsed = sum(1 for r in results if r.get("pred") is not None)
    return {"accuracy": correct / n, "parse_rate": parsed / n, "count": n}


def extract_and_evaluate(
    gen_text: str,
    gold_answer,
    dataset_type: DatasetType,
    numbers: List[int] | None = None,
    target: int | None = None,
) -> tuple[str | None, bool]:
    if dataset_type == DatasetType.COUNTDOWN:
        pred = extract_countdown_answer(gen_text)
        correct = is_correct_countdown(pred, target, numbers) if pred else False
    elif dataset_type == DatasetType.MATH:
        pred = extract_math_answer(gen_text)
        correct = is_correct_math(pred, str(gold_answer))
    else:
        pred = extract_hash_answer(gen_text)
        correct = is_correct(pred, str(gold_answer))
    return pred, correct
