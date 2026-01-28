"""CoT prompting strategies for diffusion language models."""

from enum import Enum


class PromptMethod(str, Enum):
    """Available prompting methods."""
    
    DIRECT = "Direct"  # No CoT, just the question
    ZERO_COT = "Zero-CoT"
    COMPLEX_COT = "Complex-CoT"
    MARP = "MARP"
    DIFF_MARP = "Diff-MARP"


# Template components
_ANSWER_FORMAT = "\n\nProvide the final answer in the format: #### <number>."

_TEMPLATES: dict[PromptMethod, str] = {
    PromptMethod.DIRECT: (
        "Question: {question}{suffix}"
    ),
    PromptMethod.ZERO_COT: (
        "Question: {question}\n"
        "Let's think step by step:{suffix}"
    ),
    PromptMethod.COMPLEX_COT: (
        "You should think about the following question as thoroughly "
        "and in as much detail as possible.\n"
        "Question: {question}{suffix}"
    ),
    PromptMethod.MARP: (
        "Reason step by step, but process operations in parallel.\n"
        "• At each step, you may perform multiple simple operations (up to 5).\n"
        "• Each operation must remain basic and not involve excessive complexity.\n"
        "• If you choose to perform more operations in a single step, "
        "then each operation must be correspondingly smaller in scope.\n"
        "Question: {question}{suffix}"
    ),
    PromptMethod.DIFF_MARP: (
        "Reasoning in parallel.\n"
        "• At each step, you may carry out several small operations at the same time.\n"
        "• Keep each operation simple.\n"
        "Question: {question}{suffix}"
    ),
}


def build_prompt(method: str | PromptMethod, question: str) -> str:
    """
    Build a prompt for the given CoT method and question.
    
    Args:
        method: The prompting method to use.
        question: The math question to solve.
        
    Returns:
        The formatted prompt string.
        
    Raises:
        ValueError: If the method is not recognized.
        
    Examples:
        >>> prompt = build_prompt("Zero-CoT", "What is 2 + 2?")
        >>> "Let's think step by step" in prompt
        True
    """
    if isinstance(method, str):
        try:
            method = PromptMethod(method)
        except ValueError:
            valid = [m.value for m in PromptMethod]
            raise ValueError(f"Unknown method: {method}. Valid: {valid}")
    
    template = _TEMPLATES[method]
    return template.format(question=question, suffix=_ANSWER_FORMAT)


def get_available_methods() -> list[str]:
    """Return list of available prompting method names."""
    return [m.value for m in PromptMethod]
