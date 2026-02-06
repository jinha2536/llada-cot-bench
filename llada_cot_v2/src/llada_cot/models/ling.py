"""Ling autoregressive language model wrapper using pure transformers.

Ling-mini-2.0 is an MoE model (16B total, 1.4B active) from InclusionAI.
It serves as an autoregressive baseline at the same parameter scale as
LLaDA2.0-mini.

Official HuggingFace model card:
  https://huggingface.co/inclusionAI/Ling-mini-2.0

Pure-transformers usage (from model card):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        "inclusionAI/Ling-mini-2.0",
        dtype="auto", device_map="auto", trust_remote_code=True,
    )
"""
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import ModelConfig, GenerationConfig


class LingModel:
    """Ling autoregressive MoE model via pure transformers.

    Ling-mini-2.0: 16B total params, 1.4B active (same scale as LLaDA2.0-mini)
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model_id = config.ling_model_id
        self.max_new_tokens = config.ling_max_new_tokens
        self.temperature = config.ling_temperature
        self.do_sample = config.ling_do_sample
        self.model = None
        self.tokenizer = None
        self._device = None

    @property
    def name(self) -> str:
        return f"Ling ({self.model_id.split('/')[-1]})"

    @property
    def device(self):
        return self._device

    def load(self) -> None:
        """Load model and tokenizer using pure transformers."""
        print(f"Loading {self.model_id} via transformers...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )

        # Official model card pattern for pure-transformers inference
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map=self.config.device_map,
            trust_remote_code=True,
        )
        self.model.eval()

        try:
            self._device = next(self.model.parameters()).device
        except StopIteration:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loaded {self.name} on {self._device}")

    def _format_prompt(self, prompt: str) -> str:
        """Format prompt using chat template."""
        messages = [
            {"role": "system", "content": "You are Ling, an assistant created by inclusionAI"},
            {"role": "user", "content": prompt},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        gen_config: GenerationConfig,
    ) -> tuple[str, list[int], float]:
        """Standard autoregressive generation via transformers."""
        formatted = self._format_prompt(prompt)
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[1]

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        if self.do_sample and self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature

        t0 = time.time()
        output_ids = self.model.generate(**inputs, **gen_kwargs)
        latency = time.time() - t0

        gen_ids = output_ids[0, prompt_len:].tolist()
        gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        return gen_text, gen_ids, latency

    @torch.inference_mode()
    def generate_with_trace(
        self,
        prompt: str,
        gen_config: GenerationConfig,
        trace_config,
    ):
        """Generate with pseudo-trace for AR model.

        AR models fix tokens left-to-right, so the trace is sequential.
        """
        from ..trace import GenerationTrace, TraceMeta, TraceStep

        formatted = self._format_prompt(prompt)
        prompt_tokens = self.tokenizer.encode(formatted)
        prompt_len = len(prompt_tokens)

        gen_text, gen_ids, latency = self.generate(prompt, gen_config)
        gen_length = len(gen_ids)

        eos_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else -1

        trace = GenerationTrace(
            meta=TraceMeta(
                prompt_len=prompt_len,
                gen_length=gen_length,
                block_length=1,
                steps=gen_length,
                threshold=1.0,
                mask_id=-1,
                eos_id=eos_id,
            )
        )

        for step in range(gen_length):
            fixed_map = [1 if i <= step else 0 for i in range(gen_length)]
            transfer_map = [1 if i == step else 0 for i in range(gen_length)]

            trace.steps.append(TraceStep(
                global_step=step,
                block=step,
                step_in_block=0,
                fixed_map=fixed_map,
                transfer_map=transfer_map,
                parsed_answer=None,
            ))

        return gen_text, gen_ids, latency, trace
