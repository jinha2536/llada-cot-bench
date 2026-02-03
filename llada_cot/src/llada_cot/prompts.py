"""CoT prompting strategies for different datasets."""
from enum import Enum
from typing import Optional

from .config import DatasetType


class PromptMethod(str, Enum):
    DIRECT = "Direct"
    ZERO_COT = "Zero-CoT"
    COMPLEX_COT = "Complex-CoT"
    MARP = "MARP"
    DIFF_MARP = "Diff-MARP"


# =============================================================================
# BigGSM (Grade School Math - numeric answers)
# Original paper prompts for arithmetic word problems
# =============================================================================

BIGGSM_PROMPTS = {
    PromptMethod.DIRECT: """\
Question: {question}

Provide the final answer in the format: #### <number>.""",

    PromptMethod.ZERO_COT: """\
Question: {question}

Let's think step by step.

Provide the final answer in the format: #### <number>.""",

    PromptMethod.COMPLEX_COT: """\
You should think about the following question as thoroughly and in as much detail as possible.

Question: {question}

Provide the final answer in the format: #### <number>.""",

    PromptMethod.MARP: """\
Reason step by step, but process operations in parallel.
- At each step, you may perform multiple simple operations (up to 5).
- Each operation must remain basic and not involve excessive complexity.
- If you choose to perform more operations in a single step, then each operation must be correspondingly smaller in scope.

Question: {question}

Provide the final answer in the format: #### <number>.""",

    PromptMethod.DIFF_MARP: """\
Reasoning in parallel. In each step, do as many basic operations as you can, up to 5.
Any single operation cannot be too complex.
If you use more operations in a step, the maximum allowed size for any operation decreases.

Question: {question}

#### <number>""",
}


# =============================================================================
# MATH-500 (Competition Math - LaTeX answers)
# Adapted prompts without arithmetic-specific constraints + \boxed{} format
# =============================================================================

MATH_ANSWER_FORMAT = """

Put your final answer in \\boxed{}."""

MATH500_PROMPTS = {
    PromptMethod.DIRECT: """\
Question: {question}

Put your final answer in \\boxed{}.""",

    PromptMethod.ZERO_COT: """\
Question: {question}

Let's think step by step.""" + MATH_ANSWER_FORMAT,

    PromptMethod.COMPLEX_COT: """\
You should think about the following question as thoroughly and in as much detail as possible.

Question: {question}""" + MATH_ANSWER_FORMAT,

    PromptMethod.MARP: """\
Reason step by step, but process operations in parallel.
- At each step, you may perform multiple simple operations.
- Each operation must remain basic and not involve excessive complexity.
- If you choose to perform more operations in a single step, then each operation must be correspondingly smaller in scope.

Question: {question}""" + MATH_ANSWER_FORMAT,

    PromptMethod.DIFF_MARP: """\
Reasoning in parallel. In each step, do as many basic operations as you can.
Any single operation cannot be too complex.
If you use more operations in a step, the maximum allowed size for any operation decreases.

Question: {question}

\\boxed{answer}""",
}


# =============================================================================
# Countdown Number Game
# Same prompt structure as BigGSM/MATH for consistency
# =============================================================================

COUNTDOWN_PROMPTS = {
    PromptMethod.DIRECT: """\
Using the numbers {numbers}, create an equation that equals {target}.
You can use +, -, *, / and each number at most once.

Provide the final answer in the format: #### <equation>""",

    PromptMethod.ZERO_COT: """\
Using the numbers {numbers}, create an equation that equals {target}.
You can use +, -, *, / and each number at most once.

Let's think step by step.

Provide the final answer in the format: #### <equation>""",

    PromptMethod.COMPLEX_COT: """\
You should think about the following question as thoroughly and in as much detail as possible.

Using the numbers {numbers}, create an equation that equals {target}.
You can use +, -, *, / and each number at most once.

Provide the final answer in the format: #### <equation>""",

    PromptMethod.MARP: """\
Reason step by step, but process operations in parallel.
- At each step, you may perform multiple simple operations (up to 5).
- Each operation must remain basic and not involve excessive complexity.
- If you choose to perform more operations in a single step, then each operation must be correspondingly smaller in scope.

Using the numbers {numbers}, create an equation that equals {target}.
You can use +, -, *, / and each number at most once.

Provide the final answer in the format: #### <equation>""",

    PromptMethod.DIFF_MARP: """\
Reasoning in parallel. In each step, do as many basic operations as you can, up to 5.
Any single operation cannot be too complex.
If you use more operations in a step, the maximum allowed size for any operation decreases.

Using the numbers {numbers}, create an equation that equals {target}.
You can use +, -, *, / and each number at most once.

#### <equation>""",
}


def build_prompt(
    method: str | PromptMethod,
    question: str,
    dataset_type: DatasetType = DatasetType.BIGGSM,
    numbers: Optional[list] = None,
    target: Optional[int] = None,
) -> str:
    """
    Build a prompt for the given method and dataset.
    
    Args:
        method: Prompting method name or enum
        question: The question text (for math problems)
        dataset_type: Type of dataset
        numbers: List of numbers (for Countdown)
        target: Target number (for Countdown)
        
    Returns:
        Formatted prompt string
        
    Raises:
        ValueError: If method is unknown
    """
    # Convert string to enum
    if isinstance(method, str):
        try:
            method = PromptMethod(method)
        except ValueError:
            raise ValueError(f"Unknown method: {method}. Available: {get_available_methods()}")
    
    # Select prompt template based on dataset
    if dataset_type == DatasetType.COUNTDOWN:
        templates = COUNTDOWN_PROMPTS
        if numbers is None or target is None:
            raise ValueError("Countdown requires 'numbers' and 'target' arguments")
        numbers_str = str(numbers) if isinstance(numbers, list) else numbers
        return templates[method].format(numbers=numbers_str, target=target)
    elif dataset_type == DatasetType.MATH:
        templates = MATH500_PROMPTS
        return templates[method].format(question=question)
    else:
        templates = BIGGSM_PROMPTS
        return templates[method].format(question=question)


def get_available_methods() -> list[str]:
    """Get list of available prompting method names."""
    return [m.value for m in PromptMethod]
