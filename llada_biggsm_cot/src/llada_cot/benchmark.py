"""Benchmark runner for LLaDA CoT evaluation with integrated trace analysis."""

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

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
    plot_average_fix_order_heatmap,
    plot_answer_stability_stats,
    compute_stability_summary,
)
from .analysis import (
    analyze_reasoning,
    analyze_digit_fix_order,
    aggregate_reasoning_stats,
    aggregate_digit_fix_stats,
    ReasoningAnalysis,
)

logger = logging.getLogger(__name__)


class LLaDABenchmark:
    """
    Benchmark runner for LLaDA on BigGSM with various CoT methods.
    
    All benchmark samples are traced for unified analysis.
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
    
    def _clean_special_tokens(self, text: str) -> str:
        """Remove special tokens from generated text."""
        import re
        patterns = [
            r"<\|.*?\|>",
            r"</?s>",
            r"\[/?INST\]",
        ]
        for pattern in patterns:
            text = re.sub(pattern, "", text)
        return text.strip()
    
    @torch.inference_mode()
    def evaluate_single(
        self,
        method: str,
        question: str,
        gold: str,
        example_idx: int,
        with_trace: bool = True,
    ) -> dict[str, Any]:
        """
        Evaluate a single example with optional trace.
        
        Returns result dict including trace if enabled.
        """
        prompt = build_prompt(method, question)
        input_ids = self._apply_chat_template(prompt)
        
        trace = None
        gen_ids = []
        t0 = time.time()
        
        if with_trace and self.config.trace.enabled:
            try:
                final_seq, trace = generate_with_trace(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    input_ids=input_ids,
                    gen_length=self.config.generation.gen_length,
                    block_length=self.config.generation.block_length,
                    steps=self.config.generation.steps,
                    temperature=self.config.generation.temperature,
                    threshold=self.config.trace.threshold,
                    eos_early_stop=self.config.generation.eos_early_stop,
                )
                latency = time.time() - t0
                
                prompt_len = input_ids.shape[-1]
                gen_ids = final_seq[0, prompt_len:].detach().cpu().tolist()
                gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                
            except Exception as e:
                logger.warning(f"Trace failed, falling back to fast generate: {e}")
                trace = None
                gen_text, gen_ids, latency = self._fast_generate(prompt)
        else:
            gen_text, gen_ids, latency = self._fast_generate(prompt)
        
        # Clean and extract answer
        gen_text_clean = self._clean_special_tokens(gen_text)
        pred = extract_hash_answer(gen_text_clean)
        correct = is_correct(pred, gold)
        
        # Analyze reasoning
        reasoning = analyze_reasoning(gen_text_clean, gen_ids)
        
        # Analyze trace if available
        stability = None
        digit_fix = None
        if trace is not None:
            stability = analyze_answer_stability(trace)
            digit_fix = analyze_digit_fix_order(
                trace, self.tokenizer, gen_ids, pred
            )
        
        return {
            "method": method,
            "example_idx": example_idx,
            "question": question,
            "prompt": prompt,
            "gold": gold,
            "pred": pred,
            "correct": correct,
            "latency_sec": latency,
            "gen_text": gen_text_clean,
            # Reasoning
            "char_count": reasoning.char_count,
            "word_count": reasoning.word_count,
            "token_count": reasoning.token_count,
            "step_count": reasoning.step_count,
            "equation_count": reasoning.equation_count,
            "answer_position_ratio": reasoning.answer_position_ratio,
            "has_step_markers": reasoning.has_step_markers,
            "has_therefore": reasoning.has_therefore,
            "operation_counts": reasoning.operation_counts,
            # Trace (can be None)
            "_trace": trace,
            "_stability": stability,
            "_digit_fix": digit_fix,
        }
    
    def _fast_generate(self, prompt: str) -> tuple[str, list[int], float]:
        """Fast generation without trace."""
        input_ids = self._apply_chat_template(prompt).to(self.model.device)
        
        t0 = time.time()
        try:
            out = self.model.generate(
                inputs=input_ids,
                eos_early_stop=True,
                gen_length=self.config.generation.gen_length,
                block_length=self.config.generation.block_length,
                steps=self.config.generation.steps,
                temperature=self.config.generation.temperature,
            )
        except TypeError:
            out = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=self.config.generation.gen_length,
                do_sample=self.config.generation.temperature > 0,
                temperature=max(self.config.generation.temperature, 1e-6),
            )
        latency = time.time() - t0
        
        gen_ids = out[0, input_ids.shape[-1]:].detach().cpu().tolist()
        gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        
        return gen_text, gen_ids, latency
    
    def run(self) -> pd.DataFrame:
        """
        Run the full benchmark with integrated trace analysis.
        
        Returns:
            Summary DataFrame with results per method.
        """
        self.setup()
        dataset = self.load_dataset()
        
        all_results = []
        traces_by_method: dict[str, list[GenerationTrace]] = {m: [] for m in self.config.methods}
        stability_by_method: dict[str, list] = {m: [] for m in self.config.methods}
        digit_fix_by_method: dict[str, list] = {m: [] for m in self.config.methods}
        reasoning_by_method: dict[str, list[ReasoningAnalysis]] = {m: [] for m in self.config.methods}
        
        # Evaluate each method
        for method in self.config.methods:
            logger.info(f"Evaluating method: {method}")
            
            for i, example in enumerate(tqdm(dataset, desc=method)):
                question = example["question"]
                gold = extract_hash_answer(example["answer"])
                
                result = self.evaluate_single(
                    method=method,
                    question=question,
                    gold=gold,
                    example_idx=i,
                    with_trace=self.config.trace.enabled,
                )
                
                # Extract trace objects before storing
                trace = result.pop("_trace")
                stability = result.pop("_stability")
                digit_fix = result.pop("_digit_fix")
                
                all_results.append(result)
                
                # Collect trace data
                if trace is not None:
                    traces_by_method[method].append(trace)
                if stability is not None:
                    stability_by_method[method].append(stability)
                if digit_fix is not None:
                    digit_fix_by_method[method].append(digit_fix)
                    if not digit_fix.analysis_success:
                        logger.debug(
                            f"  [{method} ex{i}] digit analysis failed: {digit_fix.failure_reason}"
                        )
                
                # Collect reasoning
                reasoning_by_method[method].append(ReasoningAnalysis(
                    char_count=result["char_count"],
                    word_count=result["word_count"],
                    token_count=result["token_count"],
                    step_count=result["step_count"],
                    equation_count=result["equation_count"],
                    answer_position_ratio=result["answer_position_ratio"],
                    has_step_markers=result["has_step_markers"],
                    has_therefore=result["has_therefore"],
                    operation_counts=result["operation_counts"],
                ))
                
                # Log progress for trace
                if trace is not None and (i + 1) % 10 == 0:
                    logger.info(
                        f"  [{method}] {i+1}/{len(dataset)} traced, "
                        f"acc so far: {sum(r['correct'] for r in all_results if r['method']==method)/(i+1):.2f}"
                    )
            
            # Log method summary
            method_results = [r for r in all_results if r["method"] == method]
            metrics = compute_metrics(method_results)
            logger.info(
                f"[{method}] acc={metrics['accuracy']:.3f} "
                f"parse={metrics['parse_rate']:.3f} "
                f"traces={len(traces_by_method[method])}"
            )
            
            if self._wandb:
                self._wandb.log({
                    f"{method}/accuracy": metrics["accuracy"],
                    f"{method}/parse_rate": metrics["parse_rate"],
                    f"{method}/n_traces": len(traces_by_method[method]),
                })
        
        # Create DataFrame
        df = pd.DataFrame(all_results)
        
        # Save detailed predictions
        pred_path = self.config.output_dir / f"predictions_n{self.config.dataset.n_eval}.jsonl"
        with open(pred_path, "w") as f:
            for r in all_results:
                # Remove non-serializable fields
                r_save = {k: v for k, v in r.items() if not k.startswith("_")}
                f.write(json.dumps(r_save, ensure_ascii=False, default=str) + "\n")
        
        # Compute summary
        summary = (
            df.groupby("method")
            .agg(
                accuracy=("correct", "mean"),
                parse_rate=("pred", lambda x: x.notna().mean()),
                avg_latency_sec=("latency_sec", "mean"),
                avg_word_count=("word_count", "mean"),
                avg_step_count=("step_count", "mean"),
                avg_equation_count=("equation_count", "mean"),
            )
            .reset_index()
        )
        
        summary_path = self.config.output_dir / f"summary_n{self.config.dataset.n_eval}.csv"
        summary.to_csv(summary_path, index=False)
        
        # Save reasoning stats
        reasoning_stats = {
            method: aggregate_reasoning_stats(analyses)
            for method, analyses in reasoning_by_method.items()
        }
        reasoning_path = self.config.output_dir / "reasoning_stats.json"
        with open(reasoning_path, "w") as f:
            json.dump(reasoning_stats, f, indent=2, default=float)
        
        logger.info(f"Saved predictions: {pred_path}")
        logger.info(f"Saved summary: {summary_path}")
        
        # Generate trace visualizations
        if self.config.trace.enabled and any(traces_by_method.values()):
            self._generate_trace_visualizations(
                traces_by_method,
                stability_by_method,
                digit_fix_by_method,
            )
        
        # Log to wandb
        if self._wandb:
            self._log_to_wandb(df, summary)
        
        # Finish wandb
        if self._wandb:
            self._wandb.finish()
        
        return summary
    
    def _generate_trace_visualizations(
        self,
        traces_by_method: dict[str, list],
        stability_by_method: dict[str, list],
        digit_fix_by_method: dict[str, list],
    ) -> None:
        """Generate all trace-related visualizations and statistics."""
        
        # 1. Average fix order heatmap
        try:
            heatmap_path = plot_average_fix_order_heatmap(
                traces_by_method,
                self.config.figures_dir / "avg_fix_order_heatmap.png",
            )
            logger.info(f"Saved: {heatmap_path}")
        except Exception as e:
            logger.warning(f"Failed to generate fix order heatmap: {e}")
        
        # 2. Answer stability boxplot
        try:
            stability_path = plot_answer_stability_stats(
                stability_by_method,
                self.config.figures_dir / "answer_stability_stats.png",
            )
            logger.info(f"Saved: {stability_path}")
        except Exception as e:
            logger.warning(f"Failed to generate stability stats: {e}")
        
        # 3. Stability summary JSON
        stability_summary = compute_stability_summary(stability_by_method)
        summary_path = self.config.output_dir / "stability_summary.json"
        with open(summary_path, "w") as f:
            json.dump(stability_summary, f, indent=2, default=float)
        logger.info(f"Saved: {summary_path}")
        
        # 4. Digit fix order summary
        digit_fix_summary = {
            method: aggregate_digit_fix_stats(fixes)
            for method, fixes in digit_fix_by_method.items()
        }
        digit_path = self.config.output_dir / "digit_fix_summary.json"
        with open(digit_path, "w") as f:
            json.dump(digit_fix_summary, f, indent=2, default=str)
        logger.info(f"Saved: {digit_path}")
        
        # Log figures to wandb
        if self._wandb:
            heatmap_file = self.config.figures_dir / "avg_fix_order_heatmap.png"
            stability_file = self.config.figures_dir / "answer_stability_stats.png"
            
            if heatmap_file.exists():
                self._wandb.log({
                    "trace/avg_fix_order_heatmap": self._wandb.Image(str(heatmap_file))
                })
            if stability_file.exists():
                self._wandb.log({
                    "trace/answer_stability_stats": self._wandb.Image(str(stability_file))
                })
            
            # Log stability summary table
            stab_rows = []
            for method, stats in stability_summary.items():
                stab_rows.append({
                    "method": method,
                    "mean_first_seen": stats.get("mean_first_seen"),
                    "mean_first_stable": stats.get("mean_first_stable"),
                    "stability_rate": stats.get("stability_rate"),
                    "n_samples": stats.get("n_samples"),
                })
            if stab_rows:
                self._wandb.log({
                    "trace/stability_summary": self._wandb.Table(
                        dataframe=pd.DataFrame(stab_rows)
                    )
                })
            
            # Log digit fix summary table
            digit_rows = []
            for method, stats in digit_fix_summary.items():
                if stats:
                    digit_rows.append({
                        "method": method,
                        "avg_first_digit_step": stats.get("avg_first_digit_step"),
                        "avg_last_digit_step": stats.get("avg_last_digit_step"),
                        "fix_order_dist": str(stats.get("fix_order_distribution", {})),
                        "n_successful": stats.get("n_successful", 0),
                        "n_failed": stats.get("n_failed", 0),
                        "n_total": stats.get("n_total", 0),
                    })
            if digit_rows:
                self._wandb.log({
                    "trace/digit_fix_summary": self._wandb.Table(
                        dataframe=pd.DataFrame(digit_rows)
                    )
                })
    
    def _log_to_wandb(self, df: pd.DataFrame, summary: pd.DataFrame) -> None:
        """Log results to Weights & Biases."""
        import matplotlib.pyplot as plt
        
        # Summary table
        self._wandb.log({"summary_table": self._wandb.Table(dataframe=summary)})
        
        # Detailed predictions table
        predictions_df = df[[
            "method", "example_idx", "question", "prompt",
            "gold", "pred", "correct",
            "word_count", "step_count", "equation_count",
            "gen_text"
        ]].copy()
        
        # Truncate long text for display
        predictions_df["gen_text"] = predictions_df["gen_text"].apply(
            lambda x: x[:500] + "..." if isinstance(x, str) and len(x) > 500 else x
        )
        predictions_df["prompt"] = predictions_df["prompt"].apply(
            lambda x: x[:500] + "..." if isinstance(x, str) and len(x) > 500 else x
        )
        predictions_df["question"] = predictions_df["question"].apply(
            lambda x: x[:300] + "..." if isinstance(x, str) and len(x) > 300 else x
        )
        
        self._wandb.log({
            "predictions_table": self._wandb.Table(dataframe=predictions_df)
        })
        
        # Accuracy bar chart
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
        
        # Reasoning comparison chart
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        ax1 = axes[0]
        ax1.bar(summary["method"], summary["avg_word_count"], color='steelblue')
        ax1.set_ylabel("Avg Word Count")
        ax1.set_title("Response Length")
        ax1.tick_params(axis='x', rotation=15)
        
        ax2 = axes[1]
        ax2.bar(summary["method"], summary["avg_step_count"], color='coral')
        ax2.set_ylabel("Avg Step Count")
        ax2.set_title("Reasoning Steps Detected")
        ax2.tick_params(axis='x', rotation=15)
        
        ax3 = axes[2]
        ax3.bar(summary["method"], summary["avg_equation_count"], color='seagreen')
        ax3.set_ylabel("Avg Equation Count")
        ax3.set_title("Calculations in Response")
        ax3.tick_params(axis='x', rotation=15)
        
        plt.tight_layout()
        reasoning_chart_path = self.config.figures_dir / "reasoning_comparison.png"
        plt.savefig(reasoning_chart_path, dpi=150)
        plt.close()
        self._wandb.log({"reasoning_chart": self._wandb.Image(str(reasoning_chart_path))})
