"""MATH dataset loader."""
import re
import random
from typing import List, Tuple, Optional

from datasets import load_dataset

from . import DatasetExample


def load_math(
    n_eval: int, 
    seed: int,
    levels: List[int] = None,
    subjects: List[str] = None,
) -> Tuple[List[DatasetExample], str]:
    """Load MATH dataset.
    
    Args:
        n_eval: Number of examples to load
        seed: Random seed for shuffling
        levels: Difficulty levels to include (1-5)
        subjects: Subjects to include (None = all)
        
    Returns:
        Tuple of (list of examples, dataset name)
    """
    levels = levels or [3, 4, 5]
    
    # Try different MATH dataset sources
    try:
        ds = load_dataset("lighteval/MATH", split="test", trust_remote_code=True)
    except Exception:
        try:
            ds = load_dataset("hendrycks/competition_math", split="test")
        except Exception:
            ds = load_dataset("EleutherAI/hendrycks_math", "all", split="test")
    
    # Filter by level
    level_strs = [f"Level {l}" for l in levels]
    filtered = [ex for ex in ds if ex.get("level") in level_strs]
    
    # Filter by subject if specified
    if subjects:
        filtered = [ex for ex in filtered if ex.get("type") in subjects]
    
    # Shuffle and select
    random.seed(seed)
    random.shuffle(filtered)
    selected = filtered[:min(n_eval, len(filtered))]
    
    examples = []
    for idx, item in enumerate(selected):
        gold = _extract_boxed_answer(item.get("solution", ""))
        
        examples.append(DatasetExample(
            idx=idx,
            question=item["problem"],
            gold_answer=gold,
            raw_data=dict(item),
        ))
    
    level_str = ",".join(map(str, levels))
    name = f"MATH (Level {level_str})"
    print(f"Loaded {len(examples)} examples from {name}")
    return examples, name


def _extract_boxed_answer(solution: str) -> str | None:
    """Extract \\boxed{answer} from MATH solution."""
    # Handle nested braces
    match = re.search(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", solution)
    if match:
        answer = match.group(1).strip()
        return _normalize_math_answer(answer)
    return None


def _normalize_math_answer(answer: str) -> str:
    """Normalize MATH answer for comparison."""
    answer = answer.replace("\\$", "")
    answer = answer.replace("$", "")
    answer = answer.replace("\\%", "%")
    answer = re.sub(r"\\text\{([^}]*)\}", r"\1", answer)
    answer = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", answer)
    answer = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)", answer)
    answer = answer.replace("\\", "")
    answer = answer.strip()
    return answer
