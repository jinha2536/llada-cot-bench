"""LLaDA CoT Benchmark - Chain-of-Thought prompting evaluation for diffusion LMs."""

from .analysis import (
    DigitFixOrder,
    ReasoningAnalysis,
    aggregate_digit_fix_stats,
    aggregate_reasoning_stats,
    analyze_digit_fix_order,
    analyze_reasoning,
)
from .benchmark import LLaDABenchmark
from .config import (
    DatasetConfig,
    ExperimentConfig,
    GenerationConfig,
    ModelConfig,
    TraceConfig,
)
from .evaluation import (
    compute_metrics,
    extract_hash_answer,
    extract_hash_answer_strict,
    is_correct,
)
from .prompts import PromptMethod, build_prompt, get_available_methods
from .trace import (
    AnswerStability,
    GenerationTrace,
    TraceMeta,
    TraceStep,
    analyze_answer_stability,
    compute_fix_order,
    compute_stability_summary,
    generate_with_trace,
    plot_answer_stability_stats,
    plot_average_fix_order_heatmap,
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
    "extract_hash_answer_strict",
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
    "compute_fix_order",
    "plot_average_fix_order_heatmap",
    "plot_answer_stability_stats",
    "compute_stability_summary",
    # Analysis
    "ReasoningAnalysis",
    "DigitFixOrder",
    "analyze_reasoning",
    "analyze_digit_fix_order",
    "aggregate_reasoning_stats",
    "aggregate_digit_fix_stats",
]
