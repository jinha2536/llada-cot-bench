"""Configuration dataclasses for the benchmark."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class ModelConfig:
    """Model and tokenizer configuration."""
    
    model_id: str = "inclusionAI/LLaDA2.0-mini-CAP"
    torch_dtype: Literal["auto", "bfloat16", "float16", "float32"] = "auto"
    device_map: str = "auto"
    trust_remote_code: bool = True


@dataclass
class GenerationConfig:
    """Generation hyperparameters for LLaDA."""
    
    gen_length: int = 512
    steps: int = 32
    block_length: int = 32
    temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = None
    eos_early_stop: bool = True


@dataclass
class TraceConfig:
    """Configuration for denoising trace analysis."""
    
    enabled: bool = True
    threshold: float = 0.95
    save_heatmaps: bool = True


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    
    dataset_id: str = "LightChen2333/BigGSM"
    split: str = "test"
    n_eval: int = 30  # All samples are benchmarked AND traced
    seed: int = 42


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    
    # Sub-configs
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    trace: TraceConfig = field(default_factory=TraceConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    
    # Experiment settings
    methods: list[str] = field(
        default_factory=lambda: ["Zero-CoT", "Complex-CoT", "MARP", "Diff-MARP"]
    )
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    figures_dir: Path = field(default_factory=lambda: Path("figures"))
    
    # Logging
    use_wandb: bool = False
    wandb_project: str = "llada2-bigGSM-cot"
    wandb_entity: str | None = None
    
    def __post_init__(self) -> None:
        """Convert string paths to Path objects."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.figures_dir, str):
            self.figures_dir = Path(self.figures_dir)
