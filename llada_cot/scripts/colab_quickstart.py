#!/usr/bin/env python3
"""
Colab Quickstart Script for LLaDA CoT Benchmark v0.3

Usage in Colab:
    # Cell 1: Mount Drive
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Cell 2: Install
    !git clone https://github.com/YOUR_USERNAME/llada-cot-bench.git
    %cd llada-cot-bench
    !pip install -q -e ".[wandb]"
    
    # Cell 3: W&B login
    import wandb
    wandb.login()
    
    # Cell 4: Run
    !python scripts/colab_quickstart.py --save-to-drive --wandb
"""
import argparse
import shutil
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="LLaDA CoT Benchmark")
    
    parser.add_argument("--dataset", "-d", type=str, default="biggsm",
                       choices=["biggsm", "math", "countdown"])
    parser.add_argument("--model", "-m", type=str, default="llada",
                       choices=["llada", "qwen3"])
    parser.add_argument("--n-eval", "-n", type=int, default=80)
    parser.add_argument("--math-levels", type=str, default="1,2,3,4,5")
    parser.add_argument("--countdown-nums", type=int, default=4)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="llada-cot-bench")
    parser.add_argument("--save-to-drive", action="store_true")
    parser.add_argument("--drive-folder", type=str, default="llada_cot_results")
    
    args = parser.parse_args()
    
    methods = ["Direct", "Zero-CoT", "Complex-CoT", "MARP", "Diff-MARP"]
    math_levels = [int(x) for x in args.math_levels.split(",")]
    
    print("=" * 60)
    print("LLaDA CoT Benchmark v0.3")
    print("=" * 60)
    print(f"Dataset:  {args.dataset}")
    print(f"Model:    {args.model}")
    print(f"N eval:   {args.n_eval}")
    print(f"W&B:      {args.wandb}")
    print("=" * 60)
    
    from llada_cot import run_benchmark
    
    summary = run_benchmark(
        dataset=args.dataset,
        model=args.model,
        n_eval=args.n_eval,
        methods=methods,
        output_dir="outputs",
        use_wandb=args.wandb,
        math_levels=math_levels,
        countdown_num_count=args.countdown_nums,
    )
    
    # Save to Google Drive
    if args.save_to_drive:
        drive_base = Path("/content/drive/MyDrive") / args.drive_folder
        if drive_base.parent.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = drive_base / f"{args.dataset}_{args.model}_{timestamp}"
            run_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy all outputs
            for f in Path("outputs").rglob("*"):
                if f.is_file():
                    dest = run_dir / f.relative_to("outputs")
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dest)
            
            print(f"\n✓ Results saved to: {run_dir}")
        else:
            print("\n⚠ Google Drive not mounted. Run: drive.mount('/content/drive')")
    
    print("\n✓ Benchmark complete!")
    return summary


if __name__ == "__main__":
    main()
