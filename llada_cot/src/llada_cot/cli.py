#!/usr/bin/env python3
"""Command-line interface for the benchmark."""
import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(
        description="LLaDA CoT Benchmark - Compare prompting strategies across models and datasets"
    )
    
    # Dataset options
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="biggsm",
        choices=["biggsm", "math", "countdown"],
        help="Dataset to evaluate on (default: biggsm)"
    )
    parser.add_argument(
        "--n-eval", "-n",
        type=int,
        default=80,
        help="Number of examples to evaluate (default: 80)"
    )
    
    # MATH specific
    parser.add_argument(
        "--math-levels",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="MATH difficulty levels (default: 1 2 3 4 5)"
    )
    
    # Countdown specific
    parser.add_argument(
        "--countdown-nums",
        type=int,
        default=4,
        help="Number of starting numbers for Countdown (default: 4)"
    )
    
    # Model options
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="llada",
        choices=["llada", "qwen3"],
        help="Model to use (default: llada)"
    )
    
    # Method options
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["Direct", "Zero-CoT", "Complex-CoT", "MARP", "Diff-MARP"],
        help="Prompting methods to test"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="outputs",
        help="Output directory (default: outputs)"
    )
    
    # W&B options
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable W&B logging"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="llada-cot-bench",
        help="W&B project name"
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="W&B entity (username or team)"
    )
    
    args = parser.parse_args()
    
    # Import here to avoid slow startup
    from llada_cot import run_benchmark
    
    print("=" * 60)
    print("LLaDA CoT Benchmark")
    print("=" * 60)
    print(f"Dataset:  {args.dataset}")
    print(f"Model:    {args.model}")
    print(f"N eval:   {args.n_eval}")
    print(f"Methods:  {args.methods}")
    print("=" * 60)
    
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
    
    print("\nBenchmark complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
