"""Detailed analysis utilities for benchmark results."""

import re
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from .trace import GenerationTrace, TraceStep

logger = logging.getLogger(__name__)


@dataclass
class ReasoningAnalysis:
    """Analysis of a reasoning path."""
    
    # Length metrics
    char_count: int
    word_count: int
    token_count: int  # Approximate
    
    # Structure metrics
    step_count: int  # Number of reasoning steps detected
    equation_count: int  # Number of equations/calculations
    
    # Answer position
    answer_position_ratio: float | None  # Where #### appears (0=start, 1=end)
    
    # Patterns detected
    has_step_markers: bool  # "Step 1", "First", etc.
    has_therefore: bool  # Conclusion markers
    operation_counts: dict[str, int]  # +, -, *, /, =


@dataclass 
class DigitFixOrder:
    """Analysis of when each digit in the answer was fixed."""
    
    answer_str: str
    digit_positions: list[int]  # Token positions of each digit
    digit_fix_steps: list[int | None]  # Step when each digit was fixed
    fix_order: str  # "left-to-right", "right-to-left", "simultaneous", "mixed"
    analysis_success: bool  # Whether analysis succeeded
    failure_reason: str | None  # Why analysis failed (if applicable)
    
    def to_dict(self) -> dict:
        return {
            "answer": self.answer_str,
            "digit_positions": self.digit_positions,
            "digit_fix_steps": self.digit_fix_steps,
            "fix_order": self.fix_order,
            "analysis_success": self.analysis_success,
            "failure_reason": self.failure_reason,
        }


def analyze_reasoning(text: str, token_ids: list[int] | None = None) -> ReasoningAnalysis:
    """
    Analyze the reasoning path in generated text.
    
    Args:
        text: Generated text.
        token_ids: Optional token IDs for accurate token count.
        
    Returns:
        ReasoningAnalysis with various metrics.
    """
    if not text:
        return ReasoningAnalysis(
            char_count=0, word_count=0, token_count=0,
            step_count=0, equation_count=0,
            answer_position_ratio=None,
            has_step_markers=False, has_therefore=False,
            operation_counts={"+": 0, "-": 0, "*": 0, "/": 0, "=": 0},
        )
    
    # Length metrics
    char_count = len(text)
    word_count = len(text.split())
    token_count = len(token_ids) if token_ids else int(word_count * 1.3)
    
    # Step detection patterns
    step_patterns = [
        r"[Ss]tep\s*\d+",           # Step 1, step 2
        r"[Ff]irst(?:ly)?[,:]",     # First, Firstly
        r"[Ss]econd(?:ly)?[,:]",    # Second, Secondly
        r"[Tt]hird(?:ly)?[,:]",     # Third, Thirdly
        r"[Nn]ext[,:]",             # Next,
        r"[Tt]hen[,:]",             # Then,
        r"\d+\)\s",                 # 1) 2) 3)
        r"\d+\.\s+[A-Z]",           # 1. Something
    ]
    
    step_count = 0
    for pattern in step_patterns:
        step_count += len(re.findall(pattern, text))
    
    has_step_markers = step_count > 0
    
    # Equation detection
    equation_patterns = [
        r"\d+\s*[+\-*/×÷]\s*\d+\s*=",  # 5 + 3 =
        r"=\s*\d+",                      # = 8
        r"\d+\s*[+\-*/×÷]\s*\d+",       # 5 + 3
    ]
    equation_count = sum(len(re.findall(p, text)) for p in equation_patterns)
    
    # Conclusion markers
    conclusion_patterns = [
        r"[Tt]herefore",
        r"[Tt]hus",
        r"[Ss]o,?\s+the",
        r"[Hh]ence",
        r"[Ii]n conclusion",
        r"[Ff]inal(?:ly)?",
        r"[Tt]he answer is",
    ]
    has_therefore = any(re.search(p, text) for p in conclusion_patterns)
    
    # Operation counts
    operation_counts = {
        "+": len(re.findall(r"\+", text)),
        "-": len(re.findall(r"(?<!\w)-(?=\d)", text)),  # Minus, not negative
        "*": len(re.findall(r"[*×]", text)),
        "/": len(re.findall(r"[/÷]", text)),
        "=": len(re.findall(r"=", text)),
    }
    
    # Answer position
    answer_match = re.search(r"####", text)
    if answer_match:
        answer_position_ratio = answer_match.start() / len(text)
    else:
        answer_position_ratio = None
    
    return ReasoningAnalysis(
        char_count=char_count,
        word_count=word_count,
        token_count=int(token_count),
        step_count=step_count,
        equation_count=equation_count,
        answer_position_ratio=answer_position_ratio,
        has_step_markers=has_step_markers,
        has_therefore=has_therefore,
        operation_counts=operation_counts,
    )


def find_answer_token_positions(
    tokenizer,
    gen_ids: list[int],
    answer_str: str,
) -> tuple[list[tuple[int, str]], str | None]:
    """
    Find the token positions corresponding to the answer digits.
    
    Returns:
        Tuple of (list of (position, digit_char) tuples, failure_reason or None)
    """
    if not answer_str or not gen_ids:
        return [], "empty_input"
    
    # Decode full text to find #### position
    full_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    
    # Try multiple patterns to find the answer
    patterns = [
        r"####\s*" + re.escape(answer_str) + r"(?:\s|$|\.)",  # Exact match with boundary
        r"####\s*" + re.escape(answer_str),                    # Exact match
        r"####\s*([-+]?\d+(?:[.,]\d+)?)",                       # Any number after ####
    ]
    
    hash_match = None
    matched_answer = answer_str
    
    for pattern in patterns:
        hash_match = re.search(pattern, full_text)
        if hash_match:
            if hash_match.lastindex:  # If there's a group
                matched_answer = hash_match.group(1).replace(",", "")
            break
    
    if not hash_match:
        return [], "no_hash_pattern"
    
    # Build character to token position map by decoding each token
    char_to_token = []
    cumulative_text = ""
    
    for tok_idx, tid in enumerate(gen_ids):
        tok_str = tokenizer.decode([tid], skip_special_tokens=True)
        for char in tok_str:
            char_to_token.append(tok_idx)
        cumulative_text += tok_str
    
    # Verify cumulative text matches
    if len(char_to_token) != len(full_text):
        # Fallback: try without skip_special_tokens
        char_to_token = []
        for tok_idx, tid in enumerate(gen_ids):
            tok_str = tokenizer.decode([tid], skip_special_tokens=False)
            # Remove special token markers
            tok_str = re.sub(r"<\|.*?\|>", "", tok_str)
            for char in tok_str:
                char_to_token.append(tok_idx)
    
    # Find the start position of the matched answer in the text
    answer_match = re.search(r"####\s*(" + re.escape(matched_answer) + r")", full_text)
    if not answer_match:
        return [], "answer_not_in_text"
    
    answer_start_char = answer_match.start(1)
    
    # Map each digit to its token position
    digit_positions = []
    for i, char in enumerate(matched_answer):
        if char.isdigit() or char in ".-+":
            char_idx = answer_start_char + i
            if char_idx < len(char_to_token):
                tok_pos = char_to_token[char_idx]
                digit_positions.append((tok_pos, char))
            else:
                return digit_positions, f"char_idx_out_of_range_{char_idx}_vs_{len(char_to_token)}"
    
    if not digit_positions:
        return [], "no_digits_found"
    
    return digit_positions, None


def analyze_digit_fix_order(
    trace: GenerationTrace,
    tokenizer,
    gen_ids: list[int],
    answer_str: str,
) -> DigitFixOrder:
    """
    Analyze when each digit of the answer was fixed during generation.
    
    Always returns a DigitFixOrder, with analysis_success=False if failed.
    """
    # Handle edge cases
    if not answer_str:
        return DigitFixOrder(
            answer_str="",
            digit_positions=[],
            digit_fix_steps=[],
            fix_order="unknown",
            analysis_success=False,
            failure_reason="no_answer_provided",
        )
    
    if not trace.steps:
        return DigitFixOrder(
            answer_str=answer_str,
            digit_positions=[],
            digit_fix_steps=[],
            fix_order="unknown",
            analysis_success=False,
            failure_reason="no_trace_steps",
        )
    
    gen_length = trace.meta.gen_length
    
    # Find token positions for answer digits
    digit_info, failure_reason = find_answer_token_positions(tokenizer, gen_ids, answer_str)
    
    if not digit_info:
        return DigitFixOrder(
            answer_str=answer_str,
            digit_positions=[],
            digit_fix_steps=[],
            fix_order="unknown",
            analysis_success=False,
            failure_reason=failure_reason or "token_mapping_failed",
        )
    
    digit_positions = [pos for pos, _ in digit_info]
    digit_chars = [char for _, char in digit_info]
    
    # Validate positions are within trace range
    out_of_range = [p for p in digit_positions if p >= gen_length]
    if out_of_range:
        return DigitFixOrder(
            answer_str=answer_str,
            digit_positions=digit_positions,
            digit_fix_steps=[],
            fix_order="unknown",
            analysis_success=False,
            failure_reason=f"positions_out_of_range_{out_of_range}_gen_length_{gen_length}",
        )
    
    # For each digit position, find when it was first fixed
    # by looking at fixed_map transitions (0 -> 1)
    digit_fix_steps = []
    
    for pos in digit_positions:
        fix_step = None
        prev_fixed = 0
        
        for step in trace.steps:
            if pos < len(step.fixed_map):
                current_fixed = step.fixed_map[pos]
                # Detect transition from 0 to 1
                if prev_fixed == 0 and current_fixed == 1:
                    fix_step = step.global_step
                    break
                prev_fixed = current_fixed
        
        digit_fix_steps.append(fix_step)
    
    # Determine fix order pattern
    valid_steps = [s for s in digit_fix_steps if s is not None]
    
    if len(valid_steps) == 0:
        fix_order = "unknown"
    elif len(valid_steps) == 1:
        fix_order = "single_digit"
    elif len(set(valid_steps)) == 1:
        # All same step
        fix_order = "simultaneous"
    else:
        # Check ordering
        increasing = all(valid_steps[i] <= valid_steps[i+1] for i in range(len(valid_steps)-1))
        strictly_increasing = all(valid_steps[i] < valid_steps[i+1] for i in range(len(valid_steps)-1))
        decreasing = all(valid_steps[i] >= valid_steps[i+1] for i in range(len(valid_steps)-1))
        strictly_decreasing = all(valid_steps[i] > valid_steps[i+1] for i in range(len(valid_steps)-1))
        
        if strictly_increasing:
            fix_order = "left-to-right"
        elif strictly_decreasing:
            fix_order = "right-to-left"
        elif increasing or decreasing:
            fix_order = "mostly-sequential"
        else:
            fix_order = "mixed"
    
    return DigitFixOrder(
        answer_str=answer_str,
        digit_positions=digit_positions,
        digit_fix_steps=digit_fix_steps,
        fix_order=fix_order,
        analysis_success=True,
        failure_reason=None,
    )


def aggregate_reasoning_stats(analyses: list[ReasoningAnalysis]) -> dict[str, Any]:
    """Aggregate reasoning analysis across multiple examples."""
    if not analyses:
        return {}
    
    valid_positions = [a.answer_position_ratio for a in analyses if a.answer_position_ratio is not None]
    
    return {
        "avg_char_count": float(np.mean([a.char_count for a in analyses])),
        "avg_word_count": float(np.mean([a.word_count for a in analyses])),
        "avg_token_count": float(np.mean([a.token_count for a in analyses])),
        "avg_step_count": float(np.mean([a.step_count for a in analyses])),
        "avg_equation_count": float(np.mean([a.equation_count for a in analyses])),
        "avg_answer_position": float(np.mean(valid_positions)) if valid_positions else None,
        "pct_with_step_markers": float(np.mean([a.has_step_markers for a in analyses])),
        "pct_with_conclusion": float(np.mean([a.has_therefore for a in analyses])),
        "total_operations": {
            op: sum(a.operation_counts.get(op, 0) for a in analyses)
            for op in ["+", "-", "*", "/", "="]
        },
        "n_samples": len(analyses),
    }


def _compute_digit_fix_substats(analyses: list[DigitFixOrder]) -> dict[str, Any]:
    """Compute digit fix statistics for a subset of analyses."""
    if not analyses:
        return {}
    
    # Only consider analysis-successful items
    successful = [d for d in analyses if d.analysis_success]
    
    if not successful:
        return {"n_samples": len(analyses), "n_analysis_success": 0}
    
    fix_order_counts = {}
    for d in successful:
        fix_order_counts[d.fix_order] = fix_order_counts.get(d.fix_order, 0) + 1
    
    first_digit_steps = [
        d.digit_fix_steps[0] for d in successful 
        if d.digit_fix_steps and d.digit_fix_steps[0] is not None
    ]
    last_digit_steps = [
        d.digit_fix_steps[-1] for d in successful 
        if d.digit_fix_steps and d.digit_fix_steps[-1] is not None
    ]
    
    # Compute spread (last - first) as simultaneity indicator
    # Small spread = more simultaneous, large spread = more sequential
    spreads = []
    for d in successful:
        if d.digit_fix_steps and len(d.digit_fix_steps) >= 2:
            first = d.digit_fix_steps[0]
            last = d.digit_fix_steps[-1]
            if first is not None and last is not None:
                spreads.append(last - first)
    
    return {
        "n_samples": len(analyses),
        "n_analysis_success": len(successful),
        "fix_order_distribution": fix_order_counts,
        "avg_first_digit_step": float(np.mean(first_digit_steps)) if first_digit_steps else None,
        "avg_last_digit_step": float(np.mean(last_digit_steps)) if last_digit_steps else None,
        "std_first_digit_step": float(np.std(first_digit_steps)) if len(first_digit_steps) > 1 else None,
        "std_last_digit_step": float(np.std(last_digit_steps)) if len(last_digit_steps) > 1 else None,
        "avg_digit_spread": float(np.mean(spreads)) if spreads else None,
        "std_digit_spread": float(np.std(spreads)) if len(spreads) > 1 else None,
    }


def aggregate_digit_fix_stats(
    digit_analyses: list[DigitFixOrder],
    correctness: list[bool] = None,
) -> dict[str, Any]:
    """Aggregate digit fix order statistics, optionally split by correctness.
    
    Args:
        digit_analyses: List of DigitFixOrder results
        correctness: Optional list of bool indicating if each sample was correct
        
    Returns:
        Dict with overall stats and optionally correct/incorrect breakdowns
    """
    if not digit_analyses:
        return {}
    
    # Separate successful and failed analyses (analysis success, not answer correctness)
    analysis_successful = [d for d in digit_analyses if d.analysis_success]
    analysis_failed = [d for d in digit_analyses if not d.analysis_success]
    
    # Count failure reasons
    failure_reasons = {}
    for d in analysis_failed:
        reason = d.failure_reason or "unknown"
        failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
    
    # Overall stats
    overall = _compute_digit_fix_substats(digit_analyses)
    overall["n_analysis_failed"] = len(analysis_failed)
    overall["failure_reasons"] = failure_reasons if failure_reasons else None
    
    result = {"overall": overall}
    
    # Split by correctness if provided
    if correctness is not None and len(correctness) == len(digit_analyses):
        correct_analyses = [d for d, c in zip(digit_analyses, correctness) if c]
        incorrect_analyses = [d for d, c in zip(digit_analyses, correctness) if not c]
        
        result["correct"] = _compute_digit_fix_substats(correct_analyses)
        result["incorrect"] = _compute_digit_fix_substats(incorrect_analyses)
        
        # Compute deltas for easy comparison
        if (result["correct"].get("avg_first_digit_step") is not None and 
            result["incorrect"].get("avg_first_digit_step") is not None):
            result["delta"] = {
                "first_digit_step_diff": (
                    result["incorrect"]["avg_first_digit_step"] - 
                    result["correct"]["avg_first_digit_step"]
                ),
                "last_digit_step_diff": (
                    (result["incorrect"].get("avg_last_digit_step") or 0) - 
                    (result["correct"].get("avg_last_digit_step") or 0)
                ),
                "spread_diff": (
                    (result["incorrect"].get("avg_digit_spread") or 0) - 
                    (result["correct"].get("avg_digit_spread") or 0)
                ),
            }
    
    return result
