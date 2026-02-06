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
    LING = "ling"


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    type: DatasetType = DatasetType.BIGGSM
    n_eval: int = 80
    seed: int = 42
    math_levels: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    math_subjects: Optional[List[str]] = None
    countdown_num_count: int = 4


@dataclass
class ModelConfig:
    """Model configuration."""
    type: ModelType = ModelType.LLADA
    device_map: str = "auto"
    torch_dtype: str = "auto"
    trust_remote_code: bool = True
    # LLaDA specific (16B MoE diffusion, 1.4B active)
    llada_model_id: str = "inclusionAI/LLaDA2.0-mini-CAP"
    # Qwen3 specific (AR baseline)
    qwen3_model_id: str = "Qwen/Qwen3-4B"
    qwen3_max_new_tokens: int = 1024
    qwen3_temperature: float = 0.0
    qwen3_do_sample: bool = False
    # Ling specific (16B MoE AR, 1.4B active — same scale as LLaDA)
    ling_model_id: str = "inclusionAI/Ling-mini-2.0"
    ling_max_new_tokens: int = 1024
    ling_temperature: float = 0.0
    ling_do_sample: bool = False


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
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    trace: TraceConfig = field(default_factory=TraceConfig)

    methods: List[str] = field(default_factory=lambda: [
        "Direct", "Zero-CoT", "Complex-CoT", "MARP", "Diff-MARP"
    ])

    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    figures_dir: Path = field(default_factory=lambda: Path("outputs/figures"))

    use_wandb: bool = False
    wandb_project: str = "llada-cot-bench"
    wandb_entity: Optional[str] = None

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.figures_dir = Path(self.figures_dir)
