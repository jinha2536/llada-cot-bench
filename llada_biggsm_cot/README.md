# LLaDA CoT Benchmark

Chain-of-Thought prompting evaluation for **LLaDA 2.0** on **BigGSM** math problems.

## Overview

This project benchmarks different CoT (Chain-of-Thought) prompting strategies on diffusion language models, specifically LLaDA 2.0. It includes:

- **4 prompting methods**: Zero-CoT, Complex-CoT, MARP, Diff-MARP
- **Denoising trace analysis**: Visualize how tokens transition from masked→fixed during generation
- **Answer stability tracking**: Detect when the final answer first appears and stabilizes
- **W&B integration**: Optional experiment tracking

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/llada-cot-bench.git
cd llada-cot-bench

# Install in development mode
pip install -e ".[dev,wandb]"
```

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0 with CUDA
- ~16GB GPU VRAM for LLaDA2.0-mini-CAP

## Quick Start

### Command Line

```bash
# Basic run with defaults
llada-bench

# Custom configuration
llada-bench --n-eval 100 --methods Zero-CoT MARP --wandb

# Using YAML config
llada-bench --config configs/default.yaml
```

### Python API

```python
from llada_cot import LLaDABenchmark, ExperimentConfig

config = ExperimentConfig(
    methods=["Zero-CoT", "Diff-MARP"],
    dataset=DatasetConfig(n_eval=50),
)

benchmark = LLaDABenchmark(config)
summary = benchmark.run()
print(summary)
```

### Google Colab

```python
!pip install -q git+https://github.com/your-org/llada-cot-bench.git

from llada_cot import LLaDABenchmark, ExperimentConfig

config = ExperimentConfig()
benchmark = LLaDABenchmark(config)
results = benchmark.run()
```

## Prompting Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| **Zero-CoT** | "Let's think step by step" | Baseline |
| **Complex-CoT** | Encourages thorough, detailed reasoning | Complex multi-step problems |
| **MARP** | Multi-step parallel reasoning (up to 5 ops/step) | Diffusion-optimized parallelism |
| **Diff-MARP** | Simplified MARP for diffusion LMs | Best for LLaDA |

## Trace Analysis

The trace module captures LLaDA's denoising dynamics:

```python
from llada_cot import generate_with_trace, analyze_answer_stability

final_seq, trace = generate_with_trace(
    model, tokenizer, input_ids,
    gen_length=512, steps=32, threshold=0.95
)

stability = analyze_answer_stability(trace)
print(f"Answer first seen at step {stability.first_seen_step}")
print(f"Answer stabilized at step {stability.first_stable_step}")
```

### Heatmap Outputs

The benchmark generates two types of heatmaps:

- **Fixed Map**: Shows which positions are masked (0) vs fixed (1) at each step
- **Transfer Map**: Shows which positions were newly fixed at each step

These visualizations help understand:
- How fast the model converges to the final answer
- Whether there are "late corrections" that could indicate instability
- The relationship between answer position and stability

## Project Structure

```
llada-cot-bench/
├── src/llada_cot/
│   ├── __init__.py       # Public API
│   ├── benchmark.py      # Main benchmark runner
│   ├── cli.py            # Command-line interface
│   ├── config.py         # Configuration dataclasses
│   ├── evaluation.py     # Answer extraction & metrics
│   ├── prompts.py        # CoT prompt templates
│   └── trace.py          # Denoising trace analysis
├── configs/
│   └── default.yaml      # Example configuration
├── tests/
│   └── test_evaluation.py
├── pyproject.toml
└── README.md
```

## Configuration

All settings can be configured via YAML:

```yaml
model:
  model_id: "inclusionAI/LLaDA2.0-mini-CAP"
  torch_dtype: "auto"

generation:
  gen_length: 512
  steps: 32
  block_length: 32

dataset:
  n_eval: 60
  seed: 42

trace:
  enabled: true
  n_examples: 3
  threshold: 0.95
```

## Research Context

This benchmark is designed to answer:

1. **Do parallel reasoning prompts (MARP/Diff-MARP) improve diffusion LM performance?**
   - Hypothesis: Diffusion models can leverage parallel token generation, so prompts encouraging parallel reasoning may align better with the generation process.

2. **How early does the answer stabilize during denoising?**
   - Related to the "Rainbow Padding" observation that early termination issues exist in instruction-tuned diffusion LMs.

3. **What is the relationship between answer stability and correctness?**
   - Early stability might indicate confident (correct) answers, while late corrections might indicate uncertainty.

## Citation

If you use this code, please cite:

```bibtex
@software{llada_cot_bench,
  title = {LLaDA CoT Benchmark},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/your-org/llada-cot-bench}
}
```

## License

MIT License
