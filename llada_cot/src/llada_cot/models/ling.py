"""Ling autoregressive language model wrapper."""
import torch
import numpy as np
import os
import sys
import re

from ..config import ModelConfig, GenerationConfig


def _patch_transformers_imports():
    """Patch missing imports in transformers for Ling model compatibility.
    
    Ling's custom modeling code uses deprecated/removed transformers imports.
    This adds them back for compatibility.
    """
    import transformers.utils.import_utils as import_utils
    
    # is_torch_fx_available was removed in newer transformers versions
    if not hasattr(import_utils, 'is_torch_fx_available'):
        def is_torch_fx_available():
            try:
                import torch.fx
                return True
            except ImportError:
                return False
        import_utils.is_torch_fx_available = is_torch_fx_available


def _find_and_patch_modeling_file(model_id: str) -> bool:
    """Find and patch the Ling modeling file in HF cache to add 'default' to ROPE_INIT_FUNCTIONS.
    
    Returns True if patching was successful or not needed.
    """
    from transformers import AutoConfig
    
    # First, load the config to trigger download of custom code to transformers_modules cache
    try:
        _ = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        print(f"Warning: Could not load config: {e}")
        # Continue anyway, maybe the files are already cached
    
    # Now find the transformers_modules cache location
    # Format: ~/.cache/huggingface/modules/transformers_modules/{org}/{repo}/...
    hf_cache_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    
    # Colab sometimes uses /root/.cache
    if not os.path.exists(hf_cache_home):
        hf_cache_home = os.path.expanduser("/root/.cache/huggingface")
    
    modules_cache = os.path.join(hf_cache_home, "modules", "transformers_modules")
    
    if not os.path.exists(modules_cache):
        print(f"Warning: transformers_modules cache not found at {modules_cache}")
        return False
    
    # The model_id gets transformed: inclusionAI/Ling-mini-2.0 -> inclusionAI/Ling_hyphen_mini_hyphen_2_dot_0
    org, repo = model_id.split("/")
    # Transform repo name to match HF's escaping
    repo_escaped = repo.replace("-", "_hyphen_").replace(".", "_dot_")
    
    model_modules_dir = os.path.join(modules_cache, org, repo_escaped)
    
    if not os.path.exists(model_modules_dir):
        # Try without escaping
        model_modules_dir = os.path.join(modules_cache, org, repo)
        if not os.path.exists(model_modules_dir):
            # Try to find it by listing the directory
            org_dir = os.path.join(modules_cache, org)
            if os.path.exists(org_dir):
                for dirname in os.listdir(org_dir):
                    if repo.replace("-", "_hyphen_") in dirname or repo in dirname:
                        model_modules_dir = os.path.join(org_dir, dirname)
                        break
            
            if not os.path.exists(model_modules_dir):
                print(f"Warning: Could not find modules cache for {model_id}")
                return False
    
    # Find the modeling file (could be in a versioned subdirectory)
    modeling_file = None
    for root, dirs, files in os.walk(model_modules_dir):
        if "modeling_bailing_moe_v2.py" in files:
            modeling_file = os.path.join(root, "modeling_bailing_moe_v2.py")
            break
    
    if not modeling_file or not os.path.exists(modeling_file):
        print(f"Warning: modeling_bailing_moe_v2.py not found in {model_modules_dir}")
        return False
    
    # Read and patch the file
    try:
        with open(modeling_file, 'r') as f:
            content = f.read()
        
        # Check if already patched or doesn't need patching
        if re.search(r'ROPE_INIT_FUNCTIONS\s*=\s*\{[^}]*["\']default["\']', content, re.DOTALL):
            print("ROPE_INIT_FUNCTIONS already has 'default'")
            return True  # Already has 'default'
        
        # Find ROPE_INIT_FUNCTIONS and add 'default'
        # Pattern: ROPE_INIT_FUNCTIONS = { ... }
        rope_pattern = r'(ROPE_INIT_FUNCTIONS\s*=\s*\{)'
        
        if not re.search(rope_pattern, content):
            print("Warning: ROPE_INIT_FUNCTIONS not found in modeling file")
            return False
        
        # Find which init function to use as default
        # Look for existing functions like _compute_llama3_parameters
        func_match = re.search(r'["\'](llama3|linear|dynamic)["\']:\s*(_compute_\w+_parameters)', content)
        if func_match:
            default_func = func_match.group(2)
        else:
            # Find any compute_*_parameters function
            func_match = re.search(r'["\'](\w+)["\']:\s*(_compute_\w+_parameters)', content)
            if func_match:
                default_func = func_match.group(2)
            else:
                print("Warning: Could not find a suitable rope init function")
                return False
        
        # Add 'default' entry
        new_content = re.sub(
            rope_pattern,
            f'\\1\n    "default": {default_func},',
            content
        )
        
        # Write back
        with open(modeling_file, 'w') as f:
            f.write(new_content)
        
        print(f"Patched {modeling_file} to add 'default' to ROPE_INIT_FUNCTIONS")
        
        # Clear any cached module imports
        modules_to_remove = [k for k in list(sys.modules.keys()) if 'bailing_moe' in k.lower()]
        for mod in modules_to_remove:
            del sys.modules[mod]
        
        return True
        
    except Exception as e:
        print(f"Warning: Could not patch modeling file: {e}")
        import traceback
        traceback.print_exc()
        return False


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
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Apply compatibility patches BEFORE loading
        _patch_transformers_imports()
        _find_and_patch_modeling_file(self.model_id)
        
        print(f"Loading {self.model_id}...")
        
        # Load tokenizer first
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=self.config.trust_remote_code,
        )
        
        # Load model - official docs say to use dtype="auto"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype="auto",  # Ling uses 'dtype' not 'torch_dtype'
            device_map=self.config.device_map,
            trust_remote_code=self.config.trust_remote_code,
        ).eval()
        
        # Get device
        try:
            self._device = next(self.model.parameters()).device
        except StopIteration:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loaded {self.name} on {self._device}")
    
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
            # Official docs use return_token_type_ids=False
            return self.tokenizer(text, return_tensors="pt", return_token_type_ids=False)["input_ids"]
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
