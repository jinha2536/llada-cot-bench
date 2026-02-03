"""LLaDA CoT Benchmark - Multi-dataset, multi-model CoT evaluation."""

# ============================================================
# CRITICAL: Apply transformers compatibility patches FIRST
# LLaDA 2.0's custom code uses deprecated transformers imports.
# This must run before any model loading happens.
# ============================================================
def _apply_transformers_patches():
    """Patch transformers for LLaDA 2.0 compatibility.
    
    LLaDA's modeling code imports is_torch_fx_available which was
    removed in transformers >= 4.40. We patch it to use is_torch_available
    since torch.fx is part of core PyTorch.
    """
    try:
        import transformers.utils.import_utils as import_utils
        # Fix: is_torch_fx_available was removed in transformers >= 4.40
        # torch.fx is available whenever torch is available (PyTorch >= 1.8)
        if not hasattr(import_utils, 'is_torch_fx_available'):
            # Use existing is_torch_available since torch.fx is part of PyTorch
            import_utils.is_torch_fx_available = import_utils.is_torch_available
    except Exception:
        pass

_apply_transformers_patches()
# ============================================================

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
