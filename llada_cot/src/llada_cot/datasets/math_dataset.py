"""MATH500 dataset loader."""
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
    """Load MATH500 dataset from HuggingFaceH4/MATH-500.
    
    Args:
        n_eval: Number of examples to load
        seed: Random seed for shuffling
        levels: Difficulty levels to include (1-5)
        subjects: Subjects to include (None = all)
        
    Returns:
        Tuple of (list of examples, dataset name)
    """
    levels = levels or [3, 4, 5]
    
    # Load MATH500 dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    
    # Convert to list for filtering
    all_examples = list(ds)
    
    # Filter by level (level is stored as string like "Level 3" or int)
    filtered = []
    for ex in all_examples:
        level = ex.get("level")
        # Handle both "Level 3" string format and integer format
        if isinstance(level, str):
            level_num = int(level.replace("Level ", "")) if "Level" in level else int(level)
        else:
            level_num = int(level)
        
        if level_num in levels:
            filtered.append(ex)
    
    # Filter by subject if specified
    if subjects:
        filtered = [ex for ex in filtered if ex.get("subject") in subjects]
    
    # Shuffle and select
    random.seed(seed)
    random.shuffle(filtered)
    selected = filtered[:min(n_eval, len(filtered))]
    
    examples = []
    for idx, item in enumerate(selected):
        # MATH500 has 'answer' field already extracted
        gold = item.get("answer")
        if gold is None:
            # Fallback: extract from solution
            gold = _extract_boxed_answer(item.get("solution", ""))
        
        examples.append(DatasetExample(
            idx=idx,
            question=item["problem"],
            gold_answer=gold,
            raw_data=dict(item),
        ))
    
    level_str = ",".join(map(str, levels))
    name = f"MATH500 (Level {level_str})"
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
