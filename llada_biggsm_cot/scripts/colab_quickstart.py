#!/usr/bin/env python3
"""
Quick start script for Google Colab.

Run this entire cell in Colab:

```python
!git clone https://github.com/your-org/llada-cot-bench.git
%cd llada-cot-bench
!pip install -q -e ".[wandb]"
!python scripts/colab_quickstart.py
```

Or copy this file's content directly into a Colab cell.
"""

import sys
from pathlib import Path

# Add src to path if running directly
if Path("src").exists():
    sys.path.insert(0, str(Path("src").resolve()))

def main():
    import torch
    
    # Check GPU
    if not torch.cuda.is_available():
        print("⚠️  No GPU detected! Please enable GPU runtime:")
        print("   Runtime → Change runtime type → T4 GPU")
        return
    
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # W&B login
    try:
        import wandb
        if not wandb.api.api_key:
            print("\n🔑 W&B 로그인이 필요합니다:")
            wandb.login()
        print(f"✅ W&B logged in")
    except Exception as e:
        print(f"⚠️  W&B login failed: {e}")
        print("   Continuing without W&B logging...")
    
    # Import after GPU check
    from llada_cot import LLaDABenchmark, ExperimentConfig
    from llada_cot.config import DatasetConfig, GenerationConfig, TraceConfig
    
    # Configure experiment
    config = ExperimentConfig(
        # Use fewer examples for quick test
        dataset=DatasetConfig(
            n_eval=20,  # Increase to 60+ for full benchmark
            seed=42,
        ),
        generation=GenerationConfig(
            gen_length=512,
            steps=32,
            block_length=32,
            temperature=0.0,
        ),
        # Evaluate all methods
        methods=["Zero-CoT", "Complex-CoT", "MARP", "Diff-MARP"],
        # Trace analysis (runs for all methods)
        trace=TraceConfig(
            enabled=True,
            n_examples=2,  # Per method, so 2 * 4 = 8 traces total
            threshold=0.95,
        ),
        # Output
        output_dir=Path("outputs"),
        figures_dir=Path("figures"),
        # W&B logging
        use_wandb=True,
        wandb_project="llada-biggsm-cot",
        wandb_entity=None,  # None = 자동으로 로그인된 계정 사용
    )
    
    print("\n" + "=" * 50)
    print("STARTING BENCHMARK")
    print("=" * 50)
    print(f"Model: {config.model.model_id}")
    print(f"Methods: {config.methods}")
    print(f"Examples: {config.dataset.n_eval}")
    print("=" * 50 + "\n")
    
    # Run benchmark
    benchmark = LLaDABenchmark(config)
    summary = benchmark.run()
    
    # Display results
    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print(summary.to_string(index=False))
    
    # Show output files
    print("\n📁 Output files:")
    for f in Path("outputs").glob("*"):
        print(f"   {f}")
    
    print("\n📊 Figures:")
    for f in Path("figures").glob("*.png"):
        print(f"   {f}")
    
    # Display heatmaps in Colab
    try:
        from IPython.display import display, Image
        for png in sorted(Path("figures").glob("*.png"))[:4]:
            print(f"\n{png.name}:")
            display(Image(filename=str(png), width=800))
    except ImportError:
        pass


if __name__ == "__main__":
    main()
