"""LLaDA CoT Benchmark - Chain-of-Thought prompting evaluation for diffusion LMs."""

from .benchmark import LLaDABenchmark
from .config import (
    DatasetConfig,
    ExperimentConfig,
    GenerationConfig,
    ModelConfig,
    TraceConfig,
)
from .evaluation import compute_metrics, extract_hash_answer, is_correct
from .prompts import PromptMethod, build_prompt, get_available_methods
from .trace import (
    AnswerStability,
    GenerationTrace,
    TraceMeta,
    TraceStep,
    analyze_answer_stability,
    generate_with_trace,
    save_trace_heatmaps,
)

__version__ = "0.1.0"

__all__ = [
    # Main class
    "LLaDABenchmark",
    # Config
    "ExperimentConfig",
    "ModelConfig",
    "GenerationConfig",
    "TraceConfig",
    "DatasetConfig",
    # Prompts
    "PromptMethod",
    "build_prompt",
    "get_available_methods",
    # Evaluation
    "extract_hash_answer",
    "is_correct",
    "compute_metrics",
    # Trace
    "GenerationTrace",
    "TraceMeta",
    "TraceStep",
    "AnswerStability",
    "generate_with_trace",
    "analyze_answer_stability",
    "save_trace_heatmaps",
]
