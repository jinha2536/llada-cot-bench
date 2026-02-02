#!/usr/bin/env python3
"""
Colab Quickstart Script for LLaDA CoT Benchmark v0.3

Usage in Colab:
    !git clone https://github.com/YOUR_USERNAME/llada-cot-bench.git
    %cd llada-cot-bench
    !pip install -q -e .
    !python scripts/colab_quickstart.py --dataset math --model llada --n-eval 50
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description="LLaDA CoT Benchmark")
    
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="biggsm",
        choices=["biggsm", "math", "countdown"],
        help="Dataset: biggsm (easy), math (hard), countdown (search)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="llada",
        choices=["llada", "ling"],
        help="Model: llada (diffusion), ling (autoregressive)"
    )
    parser.add_argument(
        "--n-eval", "-n",
        type=int,
        default=50,
        help="Number of examples (default: 50)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: only Direct and Zero-CoT"
    )
    parser.add_argument(
        "--math-levels",
        type=str,
        default="3,4,5",
        help="MATH levels (comma-separated)"
    )
    parser.add_argument(
        "--countdown-nums",
        type=int,
        default=4,
        help="Countdown starting numbers"
    )
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="llada-cot-bench")
    parser.add_argument("--wandb-entity", type=str, default=None)
    
    args = parser.parse_args()
    
    methods = ["Direct", "Zero-CoT"] if args.quick else [
        "Direct", "Zero-CoT", "Complex-CoT", "MARP", "Diff-MARP"
    ]
    math_levels = [int(x) for x in args.math_levels.split(",")]
    
    print("=" * 60)
    print("LLaDA CoT Benchmark v0.3")
    print("=" * 60)
    print(f"Dataset:  {args.dataset}")
    print(f"Model:    {args.model}")
    print(f"N eval:   {args.n_eval}")
    print(f"Methods:  {methods}")
    print("=" * 60)
    
    from llada_cot import run_benchmark
    
    summary = run_benchmark(
        dataset=args.dataset,
        model=args.model,
        n_eval=args.n_eval,
        methods=methods,
        use_wandb=args.wandb,
        math_levels=math_levels,
        countdown_num_count=args.countdown_nums,
    )
    
    print("\n✓ Benchmark complete!")
    return summary


if __name__ == "__main__":
    main()
