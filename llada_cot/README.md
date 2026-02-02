# LLaDA CoT Benchmark v0.3

Benchmark Chain-of-Thought prompting strategies on **diffusion** (LLaDA) vs **autoregressive** (Ling) language models across multiple datasets.

## Features

- **Multi-dataset**: BigGSM (easy), MATH (hard), Countdown (search-based)
- **Multi-model**: LLaDA 2.0 (diffusion), Ling (autoregressive)
- **5 CoT methods**: Direct, Zero-CoT, Complex-CoT, MARP, Diff-MARP
- **Trace analysis**: Visualize when answer tokens are fixed (diffusion models)
- **W&B integration**: Comprehensive experiment tracking

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/llada-cot-bench.git
cd llada-cot-bench
pip install -e ".[wandb]"
```

## Quick Start

### Command Line

```bash
# BigGSM with LLaDA (default - easy, expect Direct to win)
llada-bench --dataset biggsm --model llada --n-eval 80

# MATH Level 4-5 with LLaDA (hard - CoT should help)
llada-bench --dataset math --model llada --n-eval 100 --math-levels 4 5

# Countdown with LLaDA (search-based - diffusion advantage?)
llada-bench --dataset countdown --model llada --n-eval 100 --countdown-nums 4

# Compare with Ling (autoregressive baseline)
llada-bench --dataset math --model ling --n-eval 100 --math-levels 4 5
```

### Python API

```python
from llada_cot import run_benchmark

summary = run_benchmark(
    dataset="math",      # "biggsm", "math", "countdown"
    model="llada",       # "llada", "ling"
    n_eval=100,
    methods=["Direct", "Zero-CoT", "MARP"],
    math_levels=[4, 5],  # MATH specific
)
print(summary)
```

### Google Colab

```python
# Cell 1: Install
!git clone https://github.com/YOUR_USERNAME/llada-cot-bench.git
%cd llada-cot-bench
!pip install -q -e .

# Cell 2: Run
!python scripts/colab_quickstart.py --dataset math --model llada --n-eval 50

# Or Python API
from llada_cot import run_benchmark
summary = run_benchmark(dataset="countdown", model="llada", n_eval=100)
```

## Datasets

| Dataset | Difficulty | Task Type | Expected Direct Acc |
|---------|------------|-----------|---------------------|
| BigGSM | Easy | Word problems | ~80%+ |
| MATH (L3-5) | Hard | Competition math | ~30-50% |
| Countdown | Medium | Number search | ~30-40% |

## Prompting Methods

| Method | Description | Diffusion-optimized? |
|--------|-------------|---------------------|
| Direct | No reasoning, just answer | Baseline |
| Zero-CoT | "Let's think step by step" | No |
| Complex-CoT | Detailed, thorough reasoning | No |
| MARP | Multi-step parallel reasoning | Yes |
| Diff-MARP | Simplified MARP for diffusion | Yes |

## Expected Results

### If task is too easy (BigGSM with LLaDA):
```
Direct > Zero-CoT > MARP  
(longer = worse due to diffusion length penalty)
```

### If task is appropriately hard (MATH, Countdown):
```
MARP ≈ Diff-MARP > Zero-CoT > Direct  
(CoT actually helps, parallel reasoning may help more)
```

### Diffusion vs Autoregressive Comparison:

| Hypothesis | How to Test |
|------------|-------------|
| Diffusion has length penalty | Compare Direct accuracy: should be similar across models |
| Parallel prompts help diffusion | MARP gap: (MARP - Zero-CoT) larger for LLaDA than Ling |
| Diffusion better at search | Countdown: LLaDA with MARP > Ling with MARP |

## Output Files

```
outputs/
├── results_{dataset}_{model}.csv      # Detailed predictions
├── summary_{dataset}_{model}.csv      # Accuracy by method
├── config_{dataset}_{model}.json      # Experiment config
├── stability_summary.json             # Answer stability stats
├── digit_fix_summary.json             # Digit fix order analysis
└── figures/
    ├── accuracy_by_method.png
    ├── avg_fix_order_heatmap.png
    └── answer_stability_stats.png
```

## W&B Integration

```python
from llada_cot import run_benchmark

summary = run_benchmark(
    dataset="math",
    model="llada",
    n_eval=100,
    use_wandb=True,
    # Set your W&B entity
)
```

Or via CLI:
```bash
llada-bench --dataset math --model llada --wandb --wandb-entity YOUR_USERNAME
```

## Project Structure

```
src/llada_cot/
├── config.py          # All configuration dataclasses
├── prompts.py         # CoT prompts for all datasets
├── datasets/          # Dataset loaders
│   ├── biggsm.py
│   ├── math_dataset.py
│   └── countdown.py
├── models/            # Model wrappers
│   ├── llada.py       # LLaDA with trace support
│   └── ling.py        # Ling (AR baseline)
├── trace.py           # Diffusion trace analysis
├── analysis.py        # Reasoning analysis
├── evaluation.py      # Answer extraction & comparison
└── benchmark.py       # Main benchmark runner
```

## Citation

```bibtex
@software{llada_cot_bench,
  title = {LLaDA CoT Benchmark},
  author = {Kim, Dueun},
  year = {2026},
  url = {https://github.com/YOUR_USERNAME/llada-cot-bench}
}
```

## License

MIT
