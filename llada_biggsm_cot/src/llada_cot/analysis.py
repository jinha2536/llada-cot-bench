"""Detailed analysis utilities for benchmark results."""

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from .trace import GenerationTrace, TraceStep


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
    
    def to_dict(self) -> dict:
        return {
            "answer": self.answer_str,
            "digit_positions": self.digit_positions,
            "digit_fix_steps": self.digit_fix_steps,
            "fix_order": self.fix_order,
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
    token_count = len(token_ids) if token_ids else word_count * 1.3  # Rough estimate
    
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
) -> list[tuple[int, str]]:
    """
    Find the token positions corresponding to the answer digits.
    
    Returns:
        List of (position, digit_char) tuples.
    """
    if not answer_str:
        return []
    
    # Decode each token individually
    token_strs = [tokenizer.decode([tid]) for tid in gen_ids]
    
    # Find where #### appears
    full_text = "".join(token_strs)
    hash_match = re.search(r"####\s*", full_text)
    
    if not hash_match:
        return []
    
    # Find character position after ####
    answer_start_char = hash_match.end()
    
    # Map character positions to token positions
    char_to_token = []
    char_idx = 0
    for tok_idx, tok_str in enumerate(token_strs):
        for _ in tok_str:
            char_to_token.append(tok_idx)
            char_idx += 1
    
    # Find positions of each digit in the answer
    digit_positions = []
    for i, char in enumerate(answer_str):
        if char.isdigit() or char in ".-":
            char_pos = answer_start_char + i
            if char_pos < len(char_to_token):
                tok_pos = char_to_token[char_pos]
                digit_positions.append((tok_pos, char))
    
    return digit_positions


def analyze_digit_fix_order(
    trace: GenerationTrace,
    tokenizer,
    gen_ids: list[int],
    answer_str: str,
) -> DigitFixOrder | None:
    """
    Analyze when each digit of the answer was fixed during generation.
    
    Args:
        trace: Generation trace.
        tokenizer: Tokenizer for decoding.
        gen_ids: Generated token IDs.
        answer_str: The extracted answer string (e.g., "42").
        
    Returns:
        DigitFixOrder analysis or None if answer not found.
    """
    if not answer_str or not trace.steps:
        return None
    
    # Find answer digit positions in the generated sequence
    digit_info = find_answer_token_positions(tokenizer, gen_ids, answer_str)
    
    if not digit_info:
        return None
    
    digit_positions = [pos for pos, _ in digit_info]
    digit_chars = [char for _, char in digit_info]
    
    # Find when each position was fixed
    digit_fix_steps = []
    for pos in digit_positions:
        fix_step = None
        for step in trace.steps:
            if pos < len(step.transfer_map) and step.transfer_map[pos] == 1:
                fix_step = step.global_step
                break
        digit_fix_steps.append(fix_step)
    
    # Determine fix order pattern
    valid_steps = [s for s in digit_fix_steps if s is not None]
    
    if len(valid_steps) == 0:
        fix_order = "unknown"
    elif len(valid_steps) == 1:
        fix_order = "single_digit"
    elif all(s == valid_steps[0] for s in valid_steps):
        fix_order = "simultaneous"
    else:
        # Check if monotonically increasing (left-to-right)
        increasing = all(valid_steps[i] <= valid_steps[i+1] for i in range(len(valid_steps)-1))
        decreasing = all(valid_steps[i] >= valid_steps[i+1] for i in range(len(valid_steps)-1))
        
        if increasing and not decreasing:
            fix_order = "left-to-right"
        elif decreasing and not increasing:
            fix_order = "right-to-left"
        else:
            fix_order = "mixed"
    
    return DigitFixOrder(
        answer_str=answer_str,
        digit_positions=digit_positions,
        digit_fix_steps=digit_fix_steps,
        fix_order=fix_order,
    )


def aggregate_reasoning_stats(analyses: list[ReasoningAnalysis]) -> dict[str, Any]:
    """Aggregate reasoning analysis across multiple examples."""
    if not analyses:
        return {}
    
    return {
        "avg_char_count": np.mean([a.char_count for a in analyses]),
        "avg_word_count": np.mean([a.word_count for a in analyses]),
        "avg_token_count": np.mean([a.token_count for a in analyses]),
        "avg_step_count": np.mean([a.step_count for a in analyses]),
        "avg_equation_count": np.mean([a.equation_count for a in analyses]),
        "avg_answer_position": np.mean([a.answer_position_ratio for a in analyses if a.answer_position_ratio]),
        "pct_with_step_markers": np.mean([a.has_step_markers for a in analyses]),
        "pct_with_conclusion": np.mean([a.has_therefore for a in analyses]),
        "total_operations": {
            op: sum(a.operation_counts[op] for a in analyses)
            for op in ["+", "-", "*", "/", "="]
        },
    }


def aggregate_digit_fix_stats(digit_analyses: list[DigitFixOrder]) -> dict[str, Any]:
    """Aggregate digit fix order statistics."""
    valid = [d for d in digit_analyses if d is not None]
    
    if not valid:
        return {}
    
    fix_order_counts = {}
    for d in valid:
        fix_order_counts[d.fix_order] = fix_order_counts.get(d.fix_order, 0) + 1
    
    # Average steps to fix first vs last digit
    first_digit_steps = [d.digit_fix_steps[0] for d in valid if d.digit_fix_steps and d.digit_fix_steps[0] is not None]
    last_digit_steps = [d.digit_fix_steps[-1] for d in valid if d.digit_fix_steps and d.digit_fix_steps[-1] is not None]
    
    return {
        "fix_order_distribution": fix_order_counts,
        "avg_first_digit_step": np.mean(first_digit_steps) if first_digit_steps else None,
        "avg_last_digit_step": np.mean(last_digit_steps) if last_digit_steps else None,
        "n_samples": len(valid),
    }
