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
# =============================================================================

COUNTDOWN_PROMPTS = {
    PromptMethod.DIRECT: """\
Question: Using the numbers {numbers}, create an expression that equals {target}. Each number can be used at most once. Use +, -, *, / only.

Provide the final answer in the format: #### <expression>""",

    PromptMethod.ZERO_COT: """\
Question: Using the numbers {numbers}, create an expression that equals {target}. Each number can be used at most once. Use +, -, *, / only.

Let's think step by step.

Provide the final answer in the format: #### <expression>""",

    PromptMethod.COMPLEX_COT: """\
You should think about the following question as thoroughly and in as much detail as possible.

Question: Using the numbers {numbers}, create an expression that equals {target}. Each number can be used at most once. Use +, -, *, / only.

Provide the final answer in the format: #### <expression>""",

    PromptMethod.MARP: """\
Reason step by step, but process operations in parallel.
- At each step, you may perform multiple simple operations (up to 5).
- Each operation must remain basic and not involve excessive complexity.
- If you choose to perform more operations in a single step, then each operation must be correspondingly smaller in scope.

Question: Using the numbers {numbers}, create an expression that equals {target}. Each number can be used at most once. Use +, -, *, / only.

Provide the final answer in the format: #### <expression>""",

    PromptMethod.DIFF_MARP: """\
Reasoning in parallel. In each step, do as many basic operations as you can, up to 5.
Any single operation cannot be too complex.
If you use more operations in a step, the maximum allowed size for any operation decreases.

Question: Using the numbers {numbers}, create an expression that equals {target}. Each number can be used at most once. Use +, -, *, / only.

#### <expression>""",
}


def build_prompt(
    method: str | PromptMethod,
    question: str,
    dataset_type: DatasetType = DatasetType.BIGGSM,
    numbers: Optional[list] = None,
    target: Optional[int] = None,
) -> str:
    """Build a prompt for the given method and dataset."""
    if isinstance(method, str):
        try:
            method = PromptMethod(method)
        except ValueError:
            raise ValueError(f"Unknown method: {method}. Available: {get_available_methods()}")

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
    return [m.value for m in PromptMethod]
