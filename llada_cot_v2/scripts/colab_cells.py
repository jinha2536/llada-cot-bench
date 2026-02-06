# =============================================================================
# LLaDA CoT Benchmark - Colab Notebook Cells
# Copy each cell into your Colab notebook
# =============================================================================

# %%
# === CELL 1: Mount Google Drive ===
from google.colab import drive
drive.mount('/content/drive')

# %%
# === CELL 2: Clone Repository and Install ===
!git clone https://github.com/YOUR_USERNAME/llada-cot-bench.git
%cd llada-cot-bench
!pip install -q -e ".[wandb]"

# %%
# === CELL 3: Quick Test (Direct + Zero-CoT, 20 samples) ===
!python scripts/colab_full.py \
    --dataset biggsm \
    --model llada \
    --n-eval 20 \
    --quick \
    --save-to-drive

# %%
# === CELL 4: Full BigGSM Benchmark (LLaDA) ===
!python scripts/colab_full.py \
    --dataset biggsm \
    --model llada \
    --n-eval 80 \
    --save-to-drive \
    --drive-folder "llada_cot_results"

# %%
# === CELL 5: MATH Level 4-5 (Hard problems - CoT should help) ===
!python scripts/colab_full.py \
    --dataset math \
    --model llada \
    --n-eval 100 \
    --math-levels 4 5 \
    --save-to-drive

# %%
# === CELL 6: Countdown (Search-based - diffusion advantage?) ===
!python scripts/colab_full.py \
    --dataset countdown \
    --model llada \
    --n-eval 100 \
    --countdown-nums 4 \
    --save-to-drive

# %%
# === CELL 7: Ling Baseline (Autoregressive comparison) ===
!python scripts/colab_full.py \
    --dataset biggsm \
    --model ling \
    --n-eval 80 \
    --save-to-drive

# %%
# === CELL 8: With W&B Logging ===
# First login to W&B:
import wandb
wandb.login()

!python scripts/colab_full.py \
    --dataset math \
    --model llada \
    --n-eval 100 \
    --math-levels 4 5 \
    --save-to-drive \
    --wandb \
    --wandb-project "llada-cot-bench" \
    --wandb-entity "YOUR_WANDB_USERNAME"

# %%
# === CELL 9: Python API (Alternative to CLI) ===
from llada_cot import run_benchmark
import shutil
from pathlib import Path

# Run benchmark
summary = run_benchmark(
    dataset="biggsm",
    model="llada",
    n_eval=50,
    methods=["Direct", "Zero-CoT", "MARP"],
    output_dir="outputs",
    use_wandb=False,
)

# Display summary
print(summary)

# Copy to Drive manually
drive_path = Path("/content/drive/MyDrive/llada_cot_results/manual_run")
drive_path.mkdir(parents=True, exist_ok=True)
for f in Path("outputs").glob("*"):
    if f.is_file():
        shutil.copy2(f, drive_path)
print(f"Results saved to: {drive_path}")

# %%
# === CELL 10: View Results ===
import pandas as pd
from pathlib import Path

# List all runs in Drive
drive_results = Path("/content/drive/MyDrive/llada_cot_results")
if drive_results.exists():
    print("Available runs:")
    for run_dir in sorted(drive_results.iterdir()):
        print(f"  - {run_dir.name}")

# Load and display a specific summary
# summary_file = drive_results / "biggsm_llada_YYYYMMDD_HHMMSS" / "summary_biggsm_llada.csv"
# df = pd.read_csv(summary_file)
# print(df)

# %%
# === CELL 11: Compare LLaDA vs Ling ===
import pandas as pd
import matplotlib.pyplot as plt

drive_base = Path("/content/drive/MyDrive/llada_cot_results")

# Find latest runs for each model (adjust folder names)
# llada_summary = pd.read_csv(drive_base / "biggsm_llada_xxx/summary_biggsm_llada.csv")
# ling_summary = pd.read_csv(drive_base / "biggsm_ling_xxx/summary_biggsm_ling.csv")

# Example comparison plot
# fig, ax = plt.subplots(figsize=(10, 6))
# x = range(len(llada_summary))
# width = 0.35
# ax.bar([i - width/2 for i in x], llada_summary["accuracy"], width, label="LLaDA")
# ax.bar([i + width/2 for i in x], ling_summary["accuracy"], width, label="Ling")
# ax.set_xticks(x)
# ax.set_xticklabels(llada_summary["method"])
# ax.legend()
# ax.set_ylabel("Accuracy")
# ax.set_title("LLaDA vs Ling: Accuracy by Method")
# plt.savefig(drive_base / "comparison_plot.png", dpi=150)
# plt.show()

# %%
# === CELL 12: Analyze Digit Fix Patterns (Correct vs Incorrect) ===
import json

drive_base = Path("/content/drive/MyDrive/llada_cot_results")

# Load digit fix summary (adjust path)
# digit_fix_path = drive_base / "biggsm_llada_xxx/digit_fix_summary.json"
# with open(digit_fix_path) as f:
#     digit_fix = json.load(f)

# for method, stats in digit_fix.items():
#     print(f"\n=== {method} ===")
#     if "correct" in stats and "incorrect" in stats:
#         print(f"  Correct samples:")
#         print(f"    avg_first_digit_step: {stats['correct'].get('avg_first_digit_step')}")
#         print(f"    avg_digit_spread: {stats['correct'].get('avg_digit_spread')}")
#         print(f"  Incorrect samples:")
#         print(f"    avg_first_digit_step: {stats['incorrect'].get('avg_first_digit_step')}")
#         print(f"    avg_digit_spread: {stats['incorrect'].get('avg_digit_spread')}")
#     if "delta" in stats:
#         print(f"  Delta (incorrect - correct):")
#         print(f"    first_digit_step_diff: {stats['delta'].get('first_digit_step_diff')}")
#         print(f"    spread_diff: {stats['delta'].get('spread_diff')}")
