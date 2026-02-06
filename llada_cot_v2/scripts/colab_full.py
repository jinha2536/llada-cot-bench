#!/usr/bin/env python3
"""
LLaDA CoT Benchmark - Full Colab Script with Google Drive Integration

=== HOW TO USE IN COLAB ===

# Cell 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 2: Clone and Install
!git clone https://github.com/YOUR_USERNAME/llada-cot-bench.git
%cd llada-cot-bench
!pip install -q -e ".[wandb]"

# Cell 3: Run Benchmark
!python scripts/colab_full.py \
    --dataset biggsm \
    --model llada \
    --n-eval 80 \
    --save-to-drive \
    --drive-folder "llada_cot_results"

# Cell 4 (Optional): Run comparison
!python scripts/colab_full.py \
    --dataset math \
    --model llada \
    --n-eval 100 \
    --math-levels 4 5 \
    --save-to-drive \
    --wandb
"""

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path


def setup_drive_output(drive_folder: str) -> Path:
    """Setup Google Drive output directory."""
    drive_base = Path("/content/drive/MyDrive")
    
    if not drive_base.exists():
        print("⚠️  Google Drive not mounted!")
        print("   Run: from google.colab import drive; drive.mount('/content/drive')")
        return None
    
    output_dir = drive_base / drive_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Drive output directory: {output_dir}")
    return output_dir


def copy_to_drive(local_dir: Path, drive_dir: Path, run_name: str) -> None:
    """Copy results to Google Drive."""
    if drive_dir is None:
        return
    
    # Create run-specific folder
    run_dir = drive_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy all files
    for f in local_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, run_dir / f.name)
            print(f"  → {f.name}")
    
    # Copy figures subfolder
    figures_dir = local_dir / "figures"
    if figures_dir.exists():
        drive_figures = run_dir / "figures"
        drive_figures.mkdir(exist_ok=True)
        for f in figures_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, drive_figures / f.name)
                print(f"  → figures/{f.name}")
    
    print(f"\n✓ Results saved to: {run_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="LLaDA CoT Benchmark with Google Drive support"
    )
    
    # Dataset options
    parser.add_argument("--dataset", "-d", type=str, default="biggsm",
                       choices=["biggsm", "math", "countdown"])
    parser.add_argument("--n-eval", "-n", type=int, default=80)
    parser.add_argument("--math-levels", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--countdown-nums", type=int, default=4)
    
    # Model options
    parser.add_argument("--model", "-m", type=str, default="llada",
                       choices=["llada", "ling"])
    
    # Method options
    parser.add_argument("--methods", type=str, nargs="+",
                       default=["Direct", "Zero-CoT", "Complex-CoT", "MARP", "Diff-MARP"])
    parser.add_argument("--quick", action="store_true",
                       help="Quick mode: Direct + Zero-CoT only")
    
    # Output options
    parser.add_argument("--output-dir", "-o", type=str, default="outputs")
    parser.add_argument("--save-to-drive", action="store_true",
                       help="Save results to Google Drive")
    parser.add_argument("--drive-folder", type=str, default="llada_cot_results",
                       help="Folder name in Google Drive (default: llada_cot_results)")
    
    # W&B options
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="llada-cot-bench")
    parser.add_argument("--wandb-entity", type=str, default=None)
    
    args = parser.parse_args()
    
    # Quick mode
    if args.quick:
        args.methods = ["Direct", "Zero-CoT"]
    
    # Setup Drive if requested
    drive_dir = None
    if args.save_to_drive:
        drive_dir = setup_drive_output(args.drive_folder)
    
    # Generate run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.dataset}_{args.model}_{timestamp}"
    
    print("=" * 60)
    print("LLaDA CoT Benchmark v0.3")
    print("=" * 60)
    print(f"Dataset:    {args.dataset}")
    print(f"Model:      {args.model}")
    print(f"N eval:     {args.n_eval}")
    print(f"Methods:    {args.methods}")
    print(f"Run name:   {run_name}")
    if args.wandb:
        print(f"W&B:        {args.wandb_project}")
    print("=" * 60)
    
    # Import and run
    from llada_cot import run_benchmark
    
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
    
    # Copy to Drive
    if args.save_to_drive and drive_dir:
        print("\nCopying to Google Drive...")
        copy_to_drive(Path(args.output_dir), drive_dir, run_name)
    
    print("\n" + "=" * 60)
    print("✓ Benchmark complete!")
    print("=" * 60)
    
    return summary


if __name__ == "__main__":
    main()
