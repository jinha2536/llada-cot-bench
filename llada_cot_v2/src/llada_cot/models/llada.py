"""LLaDA 2.0 diffusion language model wrapper.

Follows the official HuggingFace model card usage exactly:
  https://huggingface.co/inclusionAI/LLaDA2.0-mini-CAP

Key points from the official guide:
  - Use AutoModelForCausalLM with trust_remote_code=True
  - Call model.to(torch.bfloat16) *after* from_pretrained
  - model.generate() accepts: inputs, gen_length, block_length, steps,
    temperature, eos_early_stop
  - Recommended: Temperature=0.0, block_length=32, steps=32
"""
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import ModelConfig, GenerationConfig


class LLaDAModel:
    """LLaDA 2.0 diffusion language model."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model_id = config.llada_model_id
        self.model = None
        self.tokenizer = None
        self._device = None

    @property
    def name(self) -> str:
        return f"LLaDA ({self.model_id.split('/')[-1]})"

    @property
    def device(self):
        return self._device

    def load(self) -> None:
        """Load model and tokenizer following official model card."""
        print(f"Loading {self.model_id}...")

        # Official loading pattern: device_map first, then .to(bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            device_map=self.config.device_map,
        )
        self.model = self.model.to(torch.bfloat16)
        self.model.eval()

        # Resolve the device the model ended up on
        try:
            self._device = next(self.model.parameters()).device
        except StopIteration:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loaded {self.name} on {self._device}")

    def apply_chat_template(self, prompt: str) -> torch.Tensor:
        """Apply chat template and tokenize (official API)."""
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        gen_config: GenerationConfig,
    ) -> tuple[str, list[int], float]:
        """Generate using the official model.generate() API.

        Returns:
            (generated_text, token_ids, latency_seconds)
        """
        input_ids = self.apply_chat_template(prompt).to(self.device)
        prompt_len = input_ids.shape[1]

        t0 = time.time()

        output = self.model.generate(
            inputs=input_ids,
            gen_length=gen_config.gen_length,
            steps=gen_config.steps,
            block_length=gen_config.block_length,
            temperature=gen_config.temperature,
            eos_early_stop=gen_config.eos_early_stop,
        )

        latency = time.time() - t0

        gen_ids = output[0, prompt_len:].tolist()
        gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        return gen_text, gen_ids, latency

    @torch.inference_mode()
    def generate_with_trace(
        self,
        prompt: str,
        gen_config: GenerationConfig,
        trace_config,
    ):
        """Generate with denoising trace for analysis.

        Returns:
            (generated_text, token_ids, latency, trace)
        """
        from ..trace import generate_with_trace as trace_generate

        input_ids = self.apply_chat_template(prompt).to(self.device)
        prompt_len = input_ids.shape[1]

        t0 = time.time()

        final_seq, trace = trace_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            input_ids=input_ids,
            gen_length=gen_config.gen_length,
            block_length=gen_config.block_length,
            steps=gen_config.steps,
            temperature=gen_config.temperature,
            threshold=trace_config.threshold,
            eos_early_stop=gen_config.eos_early_stop,
        )

        latency = time.time() - t0

        gen_ids = final_seq[0, prompt_len:].detach().cpu().tolist()
        gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        return gen_text, gen_ids, latency, trace
