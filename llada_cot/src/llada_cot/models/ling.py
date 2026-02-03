"""Ling autoregressive language model wrapper."""
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import ModelConfig, GenerationConfig


class LingModel:
    """Ling autoregressive MoE model from InclusionAI.
    
    This serves as an autoregressive baseline for comparison with LLaDA.
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
        """Load model and tokenizer."""
        print(f"Loading {self.model_id}...")
        
        # Load tokenizer first
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=self.config.trust_remote_code,
        )
        
        # Determine dtype
        if self.config.torch_dtype == "auto":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = getattr(torch, self.config.torch_dtype)
        
        # Load model - Ling-mini-2.0 uses standard HF API
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                device_map=self.config.device_map,
                trust_remote_code=self.config.trust_remote_code,
            ).eval()
        except TypeError:
            # Some models use 'dtype' instead of 'torch_dtype'
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                dtype=dtype,
                device_map=self.config.device_map,
                trust_remote_code=self.config.trust_remote_code,
            ).eval()
        
        # Get device
        try:
            self._device = next(self.model.parameters()).device
        except StopIteration:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loaded {self.name} on {self._device}, dtype={dtype}")
    
    def apply_chat_template(self, prompt: str) -> torch.Tensor:
        """Apply Ling's chat template."""
        messages = [
            {"role": "system", "content": "You are Ling, an assistant created by inclusionAI"},
            {"role": "user", "content": prompt}
        ]
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return self.tokenizer(text, return_tensors="pt")["input_ids"]
        except Exception:
            # Fallback: direct tokenization
            return self.tokenizer(prompt, return_tensors="pt")["input_ids"]
    
    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        gen_config: GenerationConfig,
    ) -> tuple[str, list[int], float]:
        """Standard autoregressive generation.
        
        Returns:
            Tuple of (generated_text, token_ids, latency_seconds)
        """
        import time
        
        input_ids = self.apply_chat_template(prompt).to(self.device)
        prompt_len = input_ids.shape[1]
        
        t0 = time.time()
        
        # Build generation kwargs
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
        }
        
        # Add temperature only if sampling
        if self.do_sample and self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature
        
        # Set pad token if needed
        if self.tokenizer.pad_token_id is not None:
            gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        elif self.tokenizer.eos_token_id is not None:
            gen_kwargs["pad_token_id"] = self.tokenizer.eos_token_id
        
        output_ids = self.model.generate(input_ids, **gen_kwargs)
        
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
        
        For autoregressive models, we create a "trace" showing sequential
        token generation (position n generated at step n).
        
        Returns:
            Tuple of (generated_text, token_ids, latency, trace)
        """
        import time
        from ..trace import GenerationTrace, TraceMeta, TraceStep
        
        input_ids = self.apply_chat_template(prompt).to(self.device)
        prompt_len = input_ids.shape[1]
        
        t0 = time.time()
        
        # Build generation kwargs
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
        }
        
        if self.do_sample and self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature
        
        if self.tokenizer.pad_token_id is not None:
            gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        elif self.tokenizer.eos_token_id is not None:
            gen_kwargs["pad_token_id"] = self.tokenizer.eos_token_id
        
        output_ids = self.model.generate(input_ids, **gen_kwargs)
        
        latency = time.time() - t0
        
        gen_ids = output_ids[0, prompt_len:].tolist()
        gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        gen_length = len(gen_ids)
        
        # Create pseudo-trace for AR model
        # Each step fixes exactly one position (sequential L-to-R)
        eos_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else -1
        
        trace = GenerationTrace(
            meta=TraceMeta(
                prompt_len=int(prompt_len),
                gen_length=int(gen_length),
                block_length=1,  # AR = block size 1
                steps=gen_length,
                threshold=1.0,
                mask_id=-1,
                eos_id=eos_id,
            )
        )
        
        for step in range(gen_length):
            # In AR, positions 0..step are fixed at step
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
