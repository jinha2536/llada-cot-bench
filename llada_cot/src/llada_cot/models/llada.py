"""LLaDA diffusion language model wrapper."""
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
        """Load model and tokenizer."""
        print(f"Loading {self.model_id}...")
        
        # Determine dtype
        if self.config.torch_dtype == "auto":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = getattr(torch, self.config.torch_dtype)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=self.config.trust_remote_code,
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=self.config.trust_remote_code,
            device_map=self.config.device_map,
            torch_dtype=dtype,
        ).eval()
        
        self._device = self.model.device
        print(f"Loaded {self.name} on {self._device}, dtype={dtype}")
    
    def apply_chat_template(self, prompt: str) -> torch.Tensor:
        """Apply chat template and tokenize."""
        try:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
            )
        except Exception:
            return self.tokenizer(prompt, return_tensors="pt")["input_ids"]
    
    @torch.inference_mode()
    def generate(
        self, 
        prompt: str, 
        gen_config: GenerationConfig,
    ) -> tuple[str, list[int], float]:
        """Fast generation without trace.
        
        Returns:
            Tuple of (generated_text, token_ids, latency_seconds)
        """
        import time
        
        input_ids = self.apply_chat_template(prompt).to(self.device)
        prompt_len = input_ids.shape[1]
        
        t0 = time.time()
        
        # Use model's generate method (LLaDA 2.0 API)
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
        
        Uses the trace module for detailed step-by-step tracking.
        
        Returns:
            Tuple of (generated_text, token_ids, latency, trace)
        """
        import time
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
