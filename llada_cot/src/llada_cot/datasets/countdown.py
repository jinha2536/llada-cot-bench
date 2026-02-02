"""Countdown numbers game dataset loader."""
import random
from typing import List, Tuple

from datasets import load_dataset

from . import DatasetExample


def load_countdown(
    n_eval: int,
    seed: int,
    num_count: int = 4,
) -> Tuple[List[DatasetExample], str]:
    """Load Countdown dataset.
    
    Args:
        n_eval: Number of examples to load
        seed: Random seed for shuffling
        num_count: Number of starting numbers (3-6)
        
    Returns:
        Tuple of (list of examples, dataset name)
    """
    # Select dataset based on number count
    if num_count <= 4:
        dataset_name = "Jiayi-Pan/Countdown-Tasks-3to4"
    else:
        dataset_name = "Jiayi-Pan/Countdown-Tasks-5to6"
    
    try:
        ds = load_dataset(dataset_name, split="train")
    except Exception:
        # Fallback
        ds = load_dataset("alexjackson17/countdown-numbers-6-gr", split="train")
    
    # Filter by number count
    filtered = []
    for ex in ds:
        nums = ex.get("nums", ex.get("starting", []))
        if isinstance(nums, str):
            nums = eval(nums)
        if len(nums) == num_count:
            filtered.append(ex)
    
    # If filtering removed everything, use all
    if not filtered:
        filtered = list(ds)
    
    # Shuffle and select
    random.seed(seed)
    random.shuffle(filtered)
    selected = filtered[:min(n_eval, len(filtered))]
    
    examples = []
    for idx, item in enumerate(selected):
        nums = item.get("nums", item.get("starting", []))
        if isinstance(nums, str):
            nums = eval(nums)
        target = item.get("target", item.get("closest", 0))
        
        examples.append(DatasetExample(
            idx=idx,
            question=f"Numbers: {nums}, Target: {target}",
            gold_answer=target,
            raw_data=dict(item),
            numbers=list(nums),
            target=int(target),
        ))
    
    name = f"Countdown ({num_count} numbers)"
    print(f"Loaded {len(examples)} examples from {name}")
    return examples, name
