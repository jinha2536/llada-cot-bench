"""LLaDA CoT Benchmark - Multi-dataset, multi-model CoT evaluation."""
from .config import (
    ExperimentConfig,
    DatasetConfig,
    ModelConfig,
    GenerationConfig,
    TraceConfig,
    DatasetType,
    ModelType,
)
from .prompts import PromptMethod, build_prompt, get_available_methods
from .datasets import create_dataset, DatasetExample
from .models import create_model
from .benchmark import Benchmark, run_benchmark

__version__ = "0.3.0"

__all__ = [
    # Config
    "ExperimentConfig",
    "DatasetConfig",
    "ModelConfig",
    "GenerationConfig",
    "TraceConfig",
    "DatasetType",
    "ModelType",
    # Prompts
    "PromptMethod",
    "build_prompt",
    "get_available_methods",
    # Datasets
    "create_dataset",
    "DatasetExample",
    # Models
    "create_model",
    # Benchmark
    "Benchmark",
    "run_benchmark",
]
