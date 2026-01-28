"""Command-line interface for LLaDA CoT benchmark."""

import argparse
import logging
import random
import sys
from pathlib import Path

import torch
import yaml

from .benchmark import LLaDABenchmark
from .config import (
    DatasetConfig,
    ExperimentConfig,
    GenerationConfig,
    ModelConfig,
    TraceConfig,
)
from .prompts import get_available_methods

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config_from_yaml(path: Path) -> ExperimentConfig:
    """Load configuration from YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    
    return ExperimentConfig(
        model=ModelConfig(**data.get("model", {})),
        generation=GenerationConfig(**data.get("generation", {})),
        trace=TraceConfig(**data.get("trace", {})),
        dataset=DatasetConfig(**data.get("dataset", {})),
        methods=data.get("methods", get_available_methods()),
        output_dir=Path(data.get("output_dir", "outputs")),
        figures_dir=Path(data.get("figures_dir", "figures")),
        use_wandb=data.get("use_wandb", False),
        wandb_project=data.get("wandb_project", "llada2-bigGSM-cot"),
        wandb_entity=data.get("wandb_entity"),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description="LLaDA 2.0 CoT Benchmark on BigGSM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Config file (takes precedence)
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to YAML config file (overrides all other args)",
    )
    
    # Model args
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--model-id",
        default="inclusionAI/LLaDA2.0-mini-CAP",
        help="HuggingFace model ID",
    )
    model_group.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
        help="Model dtype",
    )
    
    # Generation args
    gen_group = parser.add_argument_group("Generation")
    gen_group.add_argument("--gen-length", type=int, default=512)
    gen_group.add_argument("--steps", type=int, default=32)
    gen_group.add_argument("--block-length", type=int, default=32)
    gen_group.add_argument("--temperature", type=float, default=0.0)
    
    # Dataset args
    data_group = parser.add_argument_group("Dataset")
    data_group.add_argument(
        "--dataset-id",
        default="LightChen2333/BigGSM",
    )
    data_group.add_argument("--split", default="test")
    data_group.add_argument("--n-eval", type=int, default=100)
    data_group.add_argument("--seed", type=int, default=42)
    
    # Method args
    parser.add_argument(
        "--methods",
        nargs="+",
        default=get_available_methods(),
        help="CoT methods to evaluate",
    )
    
    # Trace args
    trace_group = parser.add_argument_group("Trace Analysis")
    trace_group.add_argument("--trace-threshold", type=float, default=0.95)
    trace_group.add_argument("--no-trace", action="store_true",
                            help="Disable trace (faster but no detailed analysis)")
    
    # Output args
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    
    # Logging args
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="llada2-bigGSM-cot")
    parser.add_argument("--wandb-entity", default=None)
    
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """Build config from command-line arguments."""
    return ExperimentConfig(
        model=ModelConfig(
            model_id=args.model_id,
            torch_dtype=args.dtype,
        ),
        generation=GenerationConfig(
            gen_length=args.gen_length,
            steps=args.steps,
            block_length=args.block_length,
            temperature=args.temperature,
        ),
        trace=TraceConfig(
            enabled=not args.no_trace,
            threshold=args.trace_threshold,
        ),
        dataset=DatasetConfig(
            dataset_id=args.dataset_id,
            split=args.split,
            n_eval=args.n_eval,
            seed=args.seed,
        ),
        methods=args.methods,
        output_dir=args.output_dir,
        figures_dir=args.figures_dir,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )


def main() -> int:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()
    
    # Load config
    if args.config:
        logger.info(f"Loading config from {args.config}")
        config = load_config_from_yaml(args.config)
    else:
        config = config_from_args(args)
    
    # Validate environment
    if not torch.cuda.is_available():
        logger.error("CUDA not available. Please use a GPU runtime.")
        return 1
    
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Set seed
    set_seed(config.dataset.seed)
    
    # Run benchmark
    benchmark = LLaDABenchmark(config)
    summary = benchmark.run()
    
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    print(summary.to_string(index=False))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
