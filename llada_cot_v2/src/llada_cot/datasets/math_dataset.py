"""MATH500 dataset loader."""
import re
import random
from typing import List, Tuple, Optional

from datasets import load_dataset

from . import DatasetExample


def load_math(
    n_eval: int, seed: int,
    levels: List[int] = None,
    subjects: List[str] = None,
) -> Tuple[List[DatasetExample], str]:
    levels = levels or [3, 4, 5]
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    all_examples = list(ds)

    filtered = []
    for ex in all_examples:
        level = ex.get("level")
        if isinstance(level, str):
            level_num = int(level.replace("Level ", "")) if "Level" in level else int(level)
        else:
            level_num = int(level)
        if level_num in levels:
            filtered.append(ex)

    if subjects:
        filtered = [ex for ex in filtered if ex.get("subject") in subjects]

    random.seed(seed)
    random.shuffle(filtered)
    selected = filtered[:min(n_eval, len(filtered))]

    examples = []
    for idx, item in enumerate(selected):
        gold = item.get("answer")
        if gold is None:
            gold = _extract_boxed_answer(item.get("solution", ""))
        examples.append(DatasetExample(
            idx=idx, question=item["problem"],
            gold_answer=gold, raw_data=dict(item),
        ))

    level_str = ",".join(map(str, levels))
    name = f"MATH500 (Level {level_str})"
    print(f"Loaded {len(examples)} examples from {name}")
    return examples, name


def _extract_boxed_answer(solution: str) -> str | None:
    match = re.search(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", solution)
    if match:
        answer = match.group(1).strip()
        return _normalize_math_answer(answer)
    return None


def _normalize_math_answer(answer: str) -> str:
    answer = answer.replace("\\$", "").replace("$", "")
    answer = answer.replace("\\%", "%")
    answer = re.sub(r"\\text\{([^}]*)\}", r"\1", answer)
    answer = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", answer)
    answer = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)", answer)
    answer = answer.replace("\\", "")
    return answer.strip()
