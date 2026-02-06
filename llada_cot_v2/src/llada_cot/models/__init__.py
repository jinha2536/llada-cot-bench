"""Model implementations."""
from ..config import ModelConfig, ModelType


def create_model(config: ModelConfig):
    """Factory function to create a model.

    Args:
        config: ModelConfig specifying which model to create

    Returns:
        Model wrapper instance (not yet loaded)
    """
    if config.type == ModelType.LLADA:
        from .llada import LLaDAModel
        return LLaDAModel(config)

    elif config.type == ModelType.QWEN3:
        from .qwen3 import Qwen3Model
        return Qwen3Model(config)

    elif config.type == ModelType.LING:
        from .ling import LingModel
        return LingModel(config)

    else:
        raise ValueError(f"Unknown model type: {config.type}")


__all__ = ["create_model"]
