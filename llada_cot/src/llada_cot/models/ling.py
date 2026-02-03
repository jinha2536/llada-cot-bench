"""Ling autoregressive language model wrapper using vLLM.

Ling-mini-2.0 requires vLLM with the bailing_moe_v2 patch applied.
This wrapper handles the vLLM-based inference.
"""
import torch
import numpy as np

from ..config import ModelConfig, GenerationConfig


class LingModel:
    """Ling autoregressive MoE model from InclusionAI via vLLM.
    
    This serves as an autoregressive baseline for comparison with LLaDA.
    Ling-mini-2.0: 16B total params, 1.4B active (same scale as LLaDA2.0-mini)
    
    Requires vLLM with bailing_moe_v2 patch. See setup instructions in README.
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model_id = config.ling_model_id
        self.max_new_tokens = config.ling_max_new_tokens
        self.temperature = config.ling_temperature
        self.do_sample = config.ling_do_sample
        self.llm = None
        self.tokenizer = None
        self._device = None
    
    @property
    def name(self) -> str:
        return f"Ling ({self.model_id.split('/')[-1]})"
    
    @property
    def device(self):
        return self._device
    
    def load(self) -> None:
        """Load model using vLLM."""
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError(
                "vLLM is required for Ling model. Install with patched vLLM:\n"
                "  git clone -b v0.10.0 https://github.com/vllm-project/vllm.git\n"
                "  cd vllm\n"
                "  wget https://raw.githubusercontent.com/inclusionAI/Ling-V2/refs/heads/main/inference/vllm/bailing_moe_v2.patch\n"
                "  git apply bailing_moe_v2.patch\n"
                "  pip install -e ."
            )
        
        from transformers import AutoTokenizer
        
        print(f"Loading {self.model_id} via vLLM...")
        
        # Load tokenizer for chat template
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        
        # Load model via vLLM
        self.llm = LLM(
            model=self.model_id,
            dtype='bfloat16',
            trust_remote_code=True,
            gpu_memory_utilization=0.90,
        )
        
        self._device = torch.device("cuda:0")
        print(f"Loaded {self.name} via vLLM on {self._device}")
    
    def _format_prompt(self, prompt: str) -> str:
        """Format prompt using chat template."""
        messages = [
            {"role": "system", "content": "You are Ling, an assistant created by inclusionAI"},
            {"role": "user", "content": prompt}
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            return prompt
    
    def generate(
        self,
        prompt: str,
        gen_config: GenerationConfig,
    ) -> tuple[str, list[int], float]:
        """Generate using vLLM."""
        import time
        from vllm import SamplingParams
        
        formatted_prompt = self._format_prompt(prompt)
        
        sampling_params = SamplingParams(
            temperature=self.temperature if self.do_sample else 0.0,
            max_tokens=self.max_new_tokens,
        )
        
        t0 = time.time()
        outputs = self.llm.generate([formatted_prompt], sampling_params)
        latency = time.time() - t0
        
        gen_text = outputs[0].outputs[0].text
        gen_ids = outputs[0].outputs[0].token_ids
        
        return gen_text, list(gen_ids), latency
    
    def generate_with_trace(
        self,
        prompt: str,
        gen_config: GenerationConfig,
        trace_config,
    ):
        """Generate with pseudo-trace for AR model."""
        import time
        from ..trace import GenerationTrace, TraceMeta, TraceStep
        
        formatted_prompt = self._format_prompt(prompt)
        prompt_tokens = self.tokenizer.encode(formatted_prompt)
        prompt_len = len(prompt_tokens)
        
        gen_text, gen_ids, latency = self.generate(prompt, gen_config)
        gen_length = len(gen_ids)
        
        # Create pseudo-trace for AR model
        eos_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else -1
        
        trace = GenerationTrace(
            meta=TraceMeta(
                prompt_len=int(prompt_len),
                gen_length=int(gen_length),
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
