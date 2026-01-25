"""Benchmark runner for LLaDA CoT evaluation."""

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ExperimentConfig, GenerationConfig
from .evaluation import compute_metrics, extract_hash_answer, is_correct
from .prompts import build_prompt
from .trace import (
    GenerationTrace,
    analyze_answer_stability,
    generate_with_trace,
    save_trace_heatmaps,
)

logger = logging.getLogger(__name__)


class LLaDABenchmark:
    """
    Benchmark runner for LLaDA on BigGSM with various CoT methods.
    
    Example:
        >>> config = ExperimentConfig()
        >>> bench = LLaDABenchmark(config)
        >>> results = bench.run()
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._wandb = None
        
        # Ensure output directories exist
        config.output_dir.mkdir(parents=True, exist_ok=True)
        config.figures_dir.mkdir(parents=True, exist_ok=True)
    
    def setup(self) -> None:
        """Load model, tokenizer, and initialize logging."""
        logger.info(f"Loading model: {self.config.model.model_id}")
        
        # Determine dtype
        if self.config.model.torch_dtype == "auto":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = getattr(torch, self.config.model.torch_dtype)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.model_id,
            trust_remote_code=self.config.model.trust_remote_code,
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model.model_id,
            trust_remote_code=self.config.model.trust_remote_code,
            device_map=self.config.model.device_map,
            torch_dtype=dtype,
        ).eval()
        
        logger.info(f"Model loaded on {self.model.device}, dtype={dtype}")
        
        # Setup wandb if enabled
        if self.config.use_wandb:
            import wandb
            self._wandb = wandb
            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                config=asdict(self.config),
            )
    
    def load_dataset(self) -> Dataset:
        """Load and prepare the evaluation dataset."""
        ds_config = self.config.dataset
        
        logger.info(f"Loading dataset: {ds_config.dataset_id}")
        ds = load_dataset(ds_config.dataset_id, split=ds_config.split)
        ds = ds.shuffle(seed=ds_config.seed).select(range(ds_config.n_eval))
        
        logger.info(f"Loaded {len(ds)} examples")
        return ds
    
    def _apply_chat_template(self, prompt: str) -> torch.Tensor:
        """Apply chat template to prompt."""
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
        gen_config: GenerationConfig | None = None,
    ) -> tuple[str, list[int], float]:
        """
        Generate a response for the given prompt.
        
        Args:
            prompt: Input prompt.
            gen_config: Generation config (uses default if None).
            
        Returns:
            Tuple of (generated_text, token_ids, latency_seconds).
        """
        if gen_config is None:
            gen_config = self.config.generation
        
        input_ids = self._apply_chat_template(prompt).to(self.model.device)
        
        t0 = time.time()
        try:
            # Try LLaDA-specific generate
            output = self.model.generate(
                inputs=input_ids,
                eos_early_stop=gen_config.eos_early_stop,
                gen_length=gen_config.gen_length,
                block_length=gen_config.block_length,
                steps=gen_config.steps,
                temperature=gen_config.temperature,
            )
        except TypeError:
            # Fallback to standard HF generate
            output = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=gen_config.gen_length,
                do_sample=gen_config.temperature > 0,
                temperature=max(gen_config.temperature, 1e-6),
            )
        latency = time.time() - t0
        
        gen_ids = output[0, input_ids.shape[-1]:].detach().cpu().tolist()
        gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        
        return gen_text, gen_ids, latency
    
    def evaluate_method(
        self,
        method: str,
        dataset: Dataset,
    ) -> Iterator[dict[str, Any]]:
        """
        Evaluate a single method on the dataset.
        
        Yields result dicts for each example.
        """
        for i, example in enumerate(tqdm(dataset, desc=method)):
            question = example["question"]
            gold = extract_hash_answer(example["answer"])
            prompt = build_prompt(method, question)
            
            gen_text, gen_ids, latency = self.generate(prompt)
            pred = extract_hash_answer(gen_text)
            correct = is_correct(pred, gold)
            
            yield {
                "method": method,
                "example_idx": i,
                "dataset_idx": example.get("index", i),
                "gold": gold,
                "pred": pred,
                "correct": correct,
                "latency_sec": latency,
                "gen_text": gen_text,
            }
    
    def run_traces(self, dataset: Dataset) -> list[dict]:
        """Run trace analysis on selected examples."""
        trace_config = self.config.trace
        if not trace_config.enabled or trace_config.n_examples <= 0:
            return []
        
        logger.info(
            f"Running trace analysis: {trace_config.n_examples} examples, "
            f"method={trace_config.method}"
        )
        
        trace_results = []
        
        for i in range(min(trace_config.n_examples, len(dataset))):
            example = dataset[i]
            question = example["question"]
            gold = extract_hash_answer(example["answer"])
            prompt = build_prompt(trace_config.method, question)
            input_ids = self._apply_chat_template(prompt)
            
            try:
                final_seq, trace = generate_with_trace(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    input_ids=input_ids,
                    gen_length=self.config.generation.gen_length,
                    block_length=self.config.generation.block_length,
                    steps=self.config.generation.steps,
                    temperature=self.config.generation.temperature,
                    threshold=trace_config.threshold,
                    eos_early_stop=self.config.generation.eos_early_stop,
                )
            except Exception as e:
                logger.warning(f"Trace failed for example {i}: {e}")
                continue
            
            # Decode final output
            prompt_len = input_ids.shape[-1]
            gen_ids = final_seq[0, prompt_len:].detach().cpu().tolist()
            final_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            final_pred = extract_hash_answer(final_text)
            
            stability = analyze_answer_stability(trace)
            
            # Save trace JSON
            trace_path = (
                self.config.output_dir
                / f"trace_{trace_config.method}_ex{i}.json"
            )
            with open(trace_path, "w") as f:
                json.dump(
                    {
                        "method": trace_config.method,
                        "example_idx": i,
                        "gold": gold,
                        "final_pred": final_pred,
                        "stability": {
                            "final_answer": stability.final_answer,
                            "first_seen_step": stability.first_seen_step,
                            "first_stable_step": stability.first_stable_step,
                        },
                        "trace": trace.to_dict(),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            
            # Save heatmaps
            if trace_config.save_heatmaps:
                title = (
                    f"{trace_config.method} ex{i} | "
                    f"gold={gold} pred={final_pred} | "
                    f"first_seen={stability.first_seen_step} "
                    f"stable={stability.first_stable_step}"
                )
                fixed_path, transfer_path = save_trace_heatmaps(
                    trace,
                    self.config.figures_dir,
                    prefix=f"{trace_config.method}_ex{i}_",
                    title_suffix=title,
                )
                
                # Log to wandb
                if self._wandb:
                    self._wandb.log({
                        f"trace/{trace_config.method}/fixed_ex{i}": 
                            self._wandb.Image(str(fixed_path)),
                        f"trace/{trace_config.method}/transfer_ex{i}": 
                            self._wandb.Image(str(transfer_path)),
                    })
            
            logger.info(
                f"[trace ex{i}] gold={gold} pred={final_pred} "
                f"first_seen={stability.first_seen_step} "
                f"stable={stability.first_stable_step}"
            )
            
            trace_results.append({
                "example_idx": i,
                "gold": gold,
                "pred": final_pred,
                "correct": is_correct(final_pred, gold),
                "first_seen_step": stability.first_seen_step,
                "first_stable_step": stability.first_stable_step,
            })
        
        return trace_results
    
    def run(self) -> pd.DataFrame:
        """
        Run the full benchmark.
        
        Returns:
            Summary DataFrame with results per method.
        """
        self.setup()
        dataset = self.load_dataset()
        
        all_results = []
        
        # Evaluate each method
        for method in self.config.methods:
            results = list(self.evaluate_method(method, dataset))
            all_results.extend(results)
            
            metrics = compute_metrics(results)
            logger.info(
                f"[{method}] acc={metrics['accuracy']:.3f} "
                f"parse={metrics['parse_rate']:.3f}"
            )
            
            if self._wandb:
                self._wandb.log({
                    f"{method}/accuracy": metrics["accuracy"],
                    f"{method}/parse_rate": metrics["parse_rate"],
                })
        
        # Save detailed predictions
        pred_path = (
            self.config.output_dir
            / f"predictions_n{self.config.dataset.n_eval}.jsonl"
        )
        with open(pred_path, "w") as f:
            for r in all_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        
        # Compute summary
        df = pd.DataFrame(all_results)
        summary = (
            df.groupby("method")
            .agg(
                accuracy=("correct", "mean"),
                parse_rate=("pred", lambda x: x.notna().mean()),
                avg_latency_sec=("latency_sec", "mean"),
            )
            .reset_index()
        )
        
        # Save summary
        summary_path = (
            self.config.output_dir
            / f"summary_n{self.config.dataset.n_eval}.csv"
        )
        summary.to_csv(summary_path, index=False)
        
        logger.info(f"Saved predictions: {pred_path}")
        logger.info(f"Saved summary: {summary_path}")
        
        # Run trace analysis
        self.run_traces(dataset)
        
        # Log final results to wandb
        if self._wandb:
            self._wandb.log({"summary_table": self._wandb.Table(dataframe=summary)})
            self._wandb.finish()
        
        return summary
