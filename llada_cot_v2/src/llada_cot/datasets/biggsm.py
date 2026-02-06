"""BigGSM dataset loader."""
import re
from typing import List, Tuple

from datasets import load_dataset

from . import DatasetExample


def load_biggsm(n_eval: int, seed: int) -> Tuple[List[DatasetExample], str]:
    ds = load_dataset("BigSAMa/BigGSM", split="test")
    ds = ds.shuffle(seed=seed).select(range(min(n_eval, len(ds))))

    examples = []
    for idx, item in enumerate(ds):
        gold = _extract_gold_answer(item.get("solution", ""))
        examples.append(DatasetExample(
            idx=idx, question=item["question"],
            gold_answer=gold, raw_data=dict(item),
        ))

    print(f"Loaded {len(examples)} examples from BigGSM")
    return examples, "BigGSM"


def _extract_gold_answer(solution: str) -> str | None:
    match = re.search(r"####\s*([^\n]+)", solution)
    if match:
        return match.group(1).strip().replace(",", "")
    return None
