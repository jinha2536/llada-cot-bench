"""Configuration dataclasses for the benchmark."""
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional


class DatasetType(str, Enum):
    BIGGSM = "biggsm"
    MATH = "math"
    COUNTDOWN = "countdown"


class ModelType(str, Enum):
    LLADA = "llada"
    QWEN3 = "qwen3"


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    type: DatasetType = DatasetType.BIGGSM
    n_eval: int = 80
    seed: int = 42
    # MATH specific - default to all levels for fair comparison
    math_levels: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    math_subjects: Optional[List[str]] = None  # None = all subjects
    # Countdown specific
    countdown_num_count: int = 4  # 3, 4, 5, 6 numbers


@dataclass
class ModelConfig:
    """Model configuration."""
    type: ModelType = ModelType.LLADA
    device_map: str = "auto"
    torch_dtype: str = "auto"
    trust_remote_code: bool = True
    # LLaDA specific (16B MoE, 1.4B active)
    llada_model_id: str = "inclusionAI/LLaDA2.0-mini"
    # Qwen3 specific (AR baseline)
    qwen3_model_id: str = "Qwen/Qwen3-4B"
    qwen3_max_new_tokens: int = 1024
    qwen3_temperature: float = 0.0
    qwen3_do_sample: bool = False


@dataclass
class GenerationConfig:
    """Generation parameters (primarily for LLaDA)."""
    gen_length: int = 1024
    block_length: int = 32
    steps: int = 64
    temperature: float = 0.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    eos_early_stop: bool = True


@dataclass
class TraceConfig:
    """Trace analysis configuration."""
    enabled: bool = True
    threshold: float = 0.95
    save_heatmaps: bool = True


@dataclass
class ExperimentConfig:
    """Main experiment configuration."""
    # Sub-configs
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    trace: TraceConfig = field(default_factory=TraceConfig)
    
    # Methods to test
    methods: List[str] = field(default_factory=lambda: [
        "Direct", "Zero-CoT", "Complex-CoT", "MARP", "Diff-MARP"
    ])
    
    # Output paths
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    figures_dir: Path = field(default_factory=lambda: Path("outputs/figures"))
    
    # W&B
    use_wandb: bool = False
    wandb_project: str = "llada-cot-bench"
    wandb_entity: Optional[str] = None
    
    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.figures_dir = Path(self.figures_dir)
