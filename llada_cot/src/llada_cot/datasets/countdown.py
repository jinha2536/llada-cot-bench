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
    """Load Countdown dataset from Jiayi-Pan/Countdown-Tasks-3to4.
    
    Args:
        n_eval: Number of examples to load
        seed: Random seed for shuffling
        num_count: Number of starting numbers (3 or 4 for this dataset)
        
    Returns:
        Tuple of (list of examples, dataset name)
    """
    # Load dataset - only has train split
    ds = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
    
    # Convert to list for filtering
    all_examples = list(ds)
    
    # Filter by number count if specified
    if num_count in [3, 4]:
        filtered = [ex for ex in all_examples if len(ex["nums"]) == num_count]
    else:
        filtered = all_examples
    
    # If filtering removed too many, use all
    if len(filtered) < n_eval:
        print(f"Warning: Only {len(filtered)} examples with {num_count} numbers, using all available")
        filtered = all_examples
    
    # Shuffle and select
    random.seed(seed)
    random.shuffle(filtered)
    selected = filtered[:min(n_eval, len(filtered))]
    
    examples = []
    for idx, item in enumerate(selected):
        nums = item["nums"]  # Already a list
        target = item["target"]  # Already an int
        
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
