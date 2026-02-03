"""Qwen3 autoregressive language model wrapper.

Simple transformers-based AR baseline for comparison with LLaDA.
"""
import torch

from ..config import ModelConfig, GenerationConfig


class Qwen3Model:
    """Qwen3 autoregressive model as AR baseline.
    
    Qwen3-4B: 4B dense params (comparable to LLaDA2.0-mini's 1.4B active)
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model_id = config.qwen3_model_id
        self.max_new_tokens = config.qwen3_max_new_tokens
        self.temperature = config.qwen3_temperature
        self.do_sample = config.qwen3_do_sample
        self.model = None
        self.tokenizer = None
        self._device = None
    
    @property
    def name(self) -> str:
        return f"Qwen3 ({self.model_id.split('/')[-1]})"
    
    @property
    def device(self):
        return self._device
    
    def load(self) -> None:
        """Load model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"Loading {self.model_id}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=self.config.trust_remote_code,
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map=self.config.device_map,
            trust_remote_code=self.config.trust_remote_code,
        ).eval()
        
        try:
            self._device = next(self.model.parameters()).device
        except StopIteration:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loaded {self.name} on {self._device}")
    
    def _format_prompt(self, prompt: str) -> torch.Tensor:
        """Format prompt using chat template."""
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # Qwen3 specific: disable thinking mode
        )
        return self.tokenizer(text, return_tensors="pt")["input_ids"]
    
    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        gen_config: GenerationConfig,
    ) -> tuple[str, list[int], float]:
        """Standard autoregressive generation."""
        import time
        
        input_ids = self._format_prompt(prompt).to(self.device)
        prompt_len = input_ids.shape[1]
        
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        
        if self.do_sample and self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature
        
        t0 = time.time()
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
        
        AR models generate left-to-right, so trace shows sequential fixing.
        """
        from ..trace import GenerationTrace, TraceMeta, TraceStep
        
        input_ids = self._format_prompt(prompt).to(self.device)
        prompt_len = input_ids.shape[1]
        
        gen_text, gen_ids, latency = self.generate(prompt, gen_config)
        gen_length = len(gen_ids)
        
        eos_id = self.tokenizer.eos_token_id or -1
        
        # Create pseudo-trace: AR = sequential left-to-right
        trace = GenerationTrace(
            meta=TraceMeta(
                prompt_len=int(prompt_len),
                gen_length=int(gen_length),
                block_length=1,  # AR generates 1 token at a time
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
