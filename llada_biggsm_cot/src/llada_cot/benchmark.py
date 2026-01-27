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
from .trace import generate_with_trace

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
            
            # Clean special tokens from generated text
            gen_text_clean = self._clean_special_tokens(gen_text)
            pred = extract_hash_answer(gen_text_clean)
            correct = is_correct(pred, gold)
            
            yield {
                "method": method,
                "example_idx": i,
                "dataset_idx": example.get("index", i),
                "question": question,  # Original question text
                "gold": gold,
                "pred": pred,
                "correct": correct,
                "latency_sec": latency,
                "gen_text": gen_text_clean,
            }
    
    def _clean_special_tokens(self, text: str) -> str:
        """Remove special tokens from generated text."""
        import re
        # Common special tokens pattern
        patterns = [
            r"<\|.*?\|>",  # <|endoftext|>, <|eot_id|>, etc.
            r"</?s>",      # <s>, </s>
            r"\[/?INST\]", # [INST], [/INST]
        ]
        for pattern in patterns:
            text = re.sub(pattern, "", text)
        return text.strip()
    
    def run_traces(self, dataset: Dataset) -> dict[str, list]:
        """Run trace analysis on selected examples for all methods."""
        from .trace import (
            analyze_answer_stability,
            generate_with_trace,
            plot_average_fix_order_heatmap,
            plot_answer_stability_stats,
            compute_stability_summary,
        )
        
        trace_config = self.config.trace
        if not trace_config.enabled or trace_config.n_examples <= 0:
            return {}
        
        logger.info(
            f"Running trace analysis: {trace_config.n_examples} examples per method"
        )
        
        traces_by_method: dict[str, list] = {m: [] for m in self.config.methods}
        stability_by_method: dict[str, list] = {m: [] for m in self.config.methods}
        
        for method in self.config.methods:
            logger.info(f"Tracing method: {method}")
            
            for i in range(min(trace_config.n_examples, len(dataset))):
                example = dataset[i]
                question = example["question"]
                gold = extract_hash_answer(example["answer"])
                prompt = build_prompt(method, question)
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
                    logger.warning(f"Trace failed for {method} example {i}: {e}")
                    continue
                
                traces_by_method[method].append(trace)
                
                # Analyze stability
                stability = analyze_answer_stability(trace)
                stability_by_method[method].append(stability)
                
                # Decode final output
                prompt_len = input_ids.shape[-1]
                gen_ids = final_seq[0, prompt_len:].detach().cpu().tolist()
                final_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                final_pred = extract_hash_answer(self._clean_special_tokens(final_text))
                
                logger.info(
                    f"  [{method} ex{i}] gold={gold} pred={final_pred} "
                    f"first_seen={stability.first_seen_step} "
                    f"stable={stability.first_stable_step}"
                )
        
        # Generate aggregate visualizations
        if any(traces_by_method.values()):
            # 1. Average fix order heatmap
            heatmap_path = plot_average_fix_order_heatmap(
                traces_by_method,
                self.config.figures_dir / "avg_fix_order_heatmap.png",
            )
            logger.info(f"Saved: {heatmap_path}")
            
            # 2. Answer stability boxplot
            stability_path = plot_answer_stability_stats(
                stability_by_method,
                self.config.figures_dir / "answer_stability_stats.png",
            )
            logger.info(f"Saved: {stability_path}")
            
            # 3. Compute and save stability summary
            stability_summary = compute_stability_summary(stability_by_method)
            summary_path = self.config.output_dir / "stability_summary.json"
            with open(summary_path, "w") as f:
                json.dump(stability_summary, f, indent=2)
            logger.info(f"Saved: {summary_path}")
            
            # Log to wandb
            if self._wandb:
                self._wandb.log({
                    "trace/avg_fix_order_heatmap": self._wandb.Image(str(heatmap_path)),
                    "trace/answer_stability_stats": self._wandb.Image(str(stability_path)),
                })
                
                # Log stability summary as table
                stab_rows = []
                for method, stats in stability_summary.items():
                    stab_rows.append({
                        "method": method,
                        "mean_first_seen": stats.get("mean_first_seen"),
                        "mean_first_stable": stats.get("mean_first_stable"),
                        "stability_rate": stats.get("stability_rate"),
                    })
                self._wandb.log({
                    "trace/stability_summary": self._wandb.Table(
                        dataframe=pd.DataFrame(stab_rows)
                    )
                })
        
        return {
            "traces": traces_by_method,
            "stability": stability_by_method,
        }
    
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
        
        # Log to wandb
        if self._wandb:
            # Summary table
            self._wandb.log({"summary_table": self._wandb.Table(dataframe=summary)})
            
            # Detailed predictions table (truncate gen_text for readability)
            predictions_df = df[["method", "example_idx", "question", "gold", "pred", "correct", "gen_text"]].copy()
            predictions_df["gen_text"] = predictions_df["gen_text"].apply(
                lambda x: x[:500] + "..." if len(x) > 500 else x
            )
            self._wandb.log({
                "predictions_table": self._wandb.Table(dataframe=predictions_df)
            })
            
            # Accuracy bar chart
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']
            bars = ax.bar(summary["method"], summary["accuracy"], color=colors[:len(summary)])
            ax.set_ylabel("Accuracy")
            ax.set_title("Accuracy by CoT Method")
            ax.set_ylim(0, 1)
            for bar, acc in zip(bars, summary["accuracy"]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                       f'{acc:.2f}', ha='center', va='bottom', fontsize=12)
            plt.tight_layout()
            
            acc_chart_path = self.config.figures_dir / "accuracy_by_method.png"
            plt.savefig(acc_chart_path, dpi=150)
            plt.close()
            
            self._wandb.log({"accuracy_chart": self._wandb.Image(str(acc_chart_path))})
        
        # Run trace analysis
        self.run_traces(dataset)
        
        # Finish wandb
        if self._wandb:
            self._wandb.finish()
        
        return summary
