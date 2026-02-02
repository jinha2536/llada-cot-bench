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
# Math Word Problems (BigGSM, MATH)
# =============================================================================

MATH_PROMPTS = {
    PromptMethod.DIRECT: """\
Question: {question}

Provide the final answer in the format: #### <number>.""",

    PromptMethod.ZERO_COT: """\
Question: {question}

Let's think step by step and solve this problem.

Provide the final answer in the format: #### <number>.""",

    PromptMethod.COMPLEX_COT: """\
Question: {question}

Let's think about this problem carefully. Think as thoroughly and in as much detail as possible. \
Consider all relevant information and show your complete reasoning process.

Provide the final answer in the format: #### <number>.""",

    PromptMethod.MARP: """\
Question: {question}

Solve this problem using multi-step parallel reasoning:
- At each step, you may perform up to 5 arithmetic operations simultaneously
- Clearly show which operations can be computed in parallel
- Use intermediate results for subsequent parallel steps

Format your work as:
Step 1: [parallel operations]
Step 2: [parallel operations using results from Step 1]
...

Provide the final answer in the format: #### <number>.""",

    PromptMethod.DIFF_MARP: """\
Question: {question}

Solve step by step with simple, parallel computations where possible:

Step 1: [computations]
Step 2: [computations]
...

#### <number>""",
}


# =============================================================================
# Countdown Number Game
# =============================================================================

COUNTDOWN_PROMPTS = {
    PromptMethod.DIRECT: """\
Using the numbers {numbers}, create an equation that equals {target}.
You can use basic arithmetic operations (+, -, *, /) and each number can only be used once.

Provide the final answer in the format: #### <equation>""",

    PromptMethod.ZERO_COT: """\
Using the numbers {numbers}, create an equation that equals {target}.
You can use basic arithmetic operations (+, -, *, /) and each number can only be used once.

Let's think step by step and try different combinations.

Provide the final answer in the format: #### <equation>""",

    PromptMethod.COMPLEX_COT: """\
Using the numbers {numbers}, create an equation that equals {target}.
You can use basic arithmetic operations (+, -, *, /) and each number can only be used once.

Let's systematically explore different combinations:
1. First, consider what operations might get us close to {target}
2. Try multiplying larger numbers first
3. Consider division to create useful intermediate values
4. Check if addition/subtraction can bridge remaining gaps

Show your exploration process thoroughly.

Provide the final answer in the format: #### <equation>""",

    PromptMethod.MARP: """\
Using the numbers {numbers}, create an equation that equals {target}.
You can use basic arithmetic operations (+, -, *, /) and each number can only be used once.

Explore multiple paths in parallel:
- Path A: Start with multiplication
- Path B: Start with division  
- Path C: Start with largest numbers
- Path D: Factor {target} first

For each path, show intermediate results. Select the path that reaches {target}.

Provide the final answer in the format: #### <equation>""",

    PromptMethod.DIFF_MARP: """\
Using the numbers {numbers}, create an equation that equals {target}.
Operations: +, -, *, /. Each number used once.

Try parallel combinations:
A: [combination 1]
B: [combination 2]
C: [combination 3]

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
    else:
        # BigGSM and MATH use the same prompts
        templates = MATH_PROMPTS
        return templates[method].format(question=question)


def get_available_methods() -> list[str]:
    """Get list of available prompting method names."""
    return [m.value for m in PromptMethod]
