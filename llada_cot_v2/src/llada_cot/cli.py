"""CLI entry point for LLaDA CoT Benchmark."""
import argparse
import logging
from pathlib import Path

from .benchmark import run_benchmark


def main():
    parser = argparse.ArgumentParser(description="LLaDA CoT Benchmark")
    parser.add_argument("--dataset", type=str, default="biggsm",
                       choices=["biggsm", "math", "countdown"])
    parser.add_argument("--model", type=str, default="llada",
                       choices=["llada", "qwen3", "ling"])
    parser.add_argument("--n-eval", type=int, default=80)
    parser.add_argument("--methods", type=str, nargs="+",
                       default=["Direct", "Zero-CoT", "Complex-CoT", "MARP", "Diff-MARP"])
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--math-levels", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--countdown-nums", type=int, default=4)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    summary = run_benchmark(
        dataset=args.dataset,
        model=args.model,
        n_eval=args.n_eval,
        methods=args.methods,
        output_dir=args.output_dir,
        use_wandb=args.wandb,
        math_levels=args.math_levels,
        countdown_num_count=args.countdown_nums,
    )

    return summary


if __name__ == "__main__":
    main()
