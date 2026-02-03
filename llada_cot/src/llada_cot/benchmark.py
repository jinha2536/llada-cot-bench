"""Benchmark runner for LLaDA CoT evaluation with integrated trace analysis."""

import json
import logging
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd
import numpy as np
from tqdm import tqdm

from .config import ExperimentConfig, DatasetType, ModelType
from .datasets import create_dataset, DatasetExample
from .models import create_model
from .prompts import build_prompt, PromptMethod
from .evaluation import extract_and_evaluate, compute_metrics
from .trace import (
    GenerationTrace,
    analyze_answer_stability,
    plot_average_fix_order_heatmap,
    plot_answer_stability_stats,
    compute_stability_summary,
)
from .analysis import (
    analyze_reasoning,
    analyze_digit_fix_order,
    aggregate_reasoning_stats,
    aggregate_digit_fix_stats,
)

logger = logging.getLogger(__name__)


class Benchmark:
    """
    Multi-dataset, multi-model benchmark runner with comprehensive analysis.
    
    Supports:
    - Datasets: BigGSM, MATH, Countdown
    - Models: LLaDA (diffusion), Qwen3 (autoregressive)
    - Analysis: Reasoning patterns, trace visualization, W&B logging
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.model = None
        self.examples: List[DatasetExample] = []
        self.dataset_name: str = ""
        self._wandb = None
        
        # Ensure output directories exist
        config.output_dir.mkdir(parents=True, exist_ok=True)
        config.figures_dir.mkdir(parents=True, exist_ok=True)
    
    def setup(self) -> None:
        """Load dataset and model."""
        logger.info("Setting up benchmark...")
        
        # Load dataset
        self.examples, self.dataset_name = create_dataset(self.config.dataset)
        
        # Load model
        self.model = create_model(self.config.model)
        self.model.load()
        
        # Setup W&B
        if self.config.use_wandb:
            import wandb
            self._wandb = wandb
            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                config=asdict(self.config),
                name=f"{self.config.model.type.value}_{self.config.dataset.type.value}",
            )
        
        logger.info(f"Dataset: {self.dataset_name} ({len(self.examples)} examples)")
        logger.info(f"Model: {self.model.name}")
        logger.info(f"Methods: {self.config.methods}")
    
    def _clean_special_tokens(self, text: str) -> str:
        """Remove special tokens from generated text."""
        patterns = [
            r"<\|.*?\|>",
            r"</?s>",
            r"\[/?INST\]",
        ]
        for pattern in patterns:
            text = re.sub(pattern, "", text)
        return text.strip()
    
    def evaluate_single(
        self,
        method: str,
        example: DatasetExample,
        with_trace: bool = True,
    ) -> dict[str, Any]:
        """Evaluate a single example with the given method."""
        
        # Build prompt based on dataset type
        if self.config.dataset.type == DatasetType.COUNTDOWN:
            prompt = build_prompt(
                method,
                example.question,
                dataset_type=self.config.dataset.type,
                numbers=example.numbers,
                target=example.target,
            )
        else:
            prompt = build_prompt(
                method,
                example.question,
                dataset_type=self.config.dataset.type,
            )
        
        # Generate
        trace = None
        if with_trace and self.config.trace.enabled:
            try:
                gen_text, gen_ids, latency, trace = self.model.generate_with_trace(
                    prompt,
                    self.config.generation,
                    self.config.trace,
                )
            except Exception as e:
                logger.warning(f"Trace failed, using fast generate: {e}")
                gen_text, gen_ids, latency = self.model.generate(
                    prompt,
                    self.config.generation,
                )
        else:
            gen_text, gen_ids, latency = self.model.generate(
                prompt,
                self.config.generation,
            )
        
        # Clean and extract answer
        gen_text_clean = self._clean_special_tokens(gen_text)
        
        pred, correct = extract_and_evaluate(
            gen_text_clean,
            example.gold_answer,
            self.config.dataset.type,
            numbers=example.numbers,
            target=example.target,
        )
        
        # Analyze reasoning
        reasoning = analyze_reasoning(gen_text_clean, gen_ids)
        
        # Analyze trace if available
        stability = None
        digit_fix = None
        if trace is not None:
            stability = analyze_answer_stability(trace)
            digit_fix = analyze_digit_fix_order(
                trace, self.model.tokenizer, gen_ids, pred
            )
        
        return {
            "method": method,
            "example_idx": example.idx,
            "question": example.question,
            "prompt": prompt,
            "gold": str(example.gold_answer),
            "pred": pred,
            "correct": correct,
            "latency_sec": latency,
            "gen_text": gen_text_clean,
            # Reasoning metrics
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
    
    def run(self) -> pd.DataFrame:
        """Run the full benchmark."""
        if self.model is None:
            self.setup()
        
        all_results = []
        traces_by_method = {m: [] for m in self.config.methods}
        stability_by_method = {m: [] for m in self.config.methods}
        digit_fix_by_method = {m: [] for m in self.config.methods}
        
        for method in self.config.methods:
            print(f"\n>>> Running {method}...")
            
            for example in tqdm(self.examples, desc=method):
                result = self.evaluate_single(method, example)
                all_results.append(result)
                
                # Collect traces
                if result["_trace"] is not None:
                    traces_by_method[method].append(result["_trace"])
                if result["_stability"] is not None:
                    stability_by_method[method].append(result["_stability"])
                if result["_digit_fix"] is not None:
                    digit_fix_by_method[method].append(result["_digit_fix"])
                
                # Log to W&B
                if self._wandb:
                    self._wandb.log({
                        "method": method,
                        "correct": result["correct"],
                        "latency": result["latency_sec"],
                    })
        
        # Create DataFrame (without trace objects)
        df_results = []
        for r in all_results:
            row = {k: v for k, v in r.items() if not k.startswith("_")}
            # Add digit fix info if available
            if r["_digit_fix"] is not None:
                row["first_digit_step"] = r["_digit_fix"].digit_fix_steps[0] if r["_digit_fix"].digit_fix_steps else None
                row["last_digit_step"] = r["_digit_fix"].digit_fix_steps[-1] if r["_digit_fix"].digit_fix_steps else None
                row["fix_order"] = r["_digit_fix"].fix_order
            df_results.append(row)
        
        df = pd.DataFrame(df_results)
        
        # Generate summary
        summary = self._create_summary(df)
        
        # Save results
        self._save_results(df, summary)
        
        # Generate trace visualizations
        if self.config.trace.enabled:
            self._generate_trace_visualizations(
                traces_by_method,
                stability_by_method,
                digit_fix_by_method,
                all_results,
            )
        
        # Log to W&B
        if self._wandb:
            self._log_to_wandb(df, summary)
            self._wandb.finish()
        
        return summary
    
    def _create_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create summary statistics by method."""
        summary = df.groupby("method").agg({
            "correct": ["mean", "sum", "count"],
            "latency_sec": "mean",
            "word_count": "mean",
            "step_count": "mean",
            "equation_count": "mean",
        })
        
        summary.columns = [
            "accuracy", "n_correct", "n_total",
            "avg_latency_sec", "avg_word_count",
            "avg_step_count", "avg_equation_count"
        ]
        summary = summary.round(3).reset_index()
        
        # Add digit fix stats if available
        if "first_digit_step" in df.columns:
            digit_stats = df.groupby("method").agg({
                "first_digit_step": "mean",
                "last_digit_step": "mean",
            }).round(2)
            digit_stats.columns = ["avg_first_digit_step", "avg_last_digit_step"]
            summary = summary.merge(digit_stats, on="method", how="left")
        
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(f"Dataset: {self.dataset_name}")
        print(f"Model: {self.model.name}")
        print("-" * 60)
        print(summary.to_string(index=False))
        print("=" * 60)
        
        return summary
    
    def _save_results(self, df: pd.DataFrame, summary: pd.DataFrame) -> None:
        """Save results to files."""
        dataset_name = self.config.dataset.type.value
        model_name = self.config.model.type.value
        
        # Save detailed results
        results_path = self.config.output_dir / f"results_{dataset_name}_{model_name}.csv"
        df.to_csv(results_path, index=False)
        logger.info(f"Saved: {results_path}")
        
        # Save summary
        summary_path = self.config.output_dir / f"summary_{dataset_name}_{model_name}.csv"
        summary.to_csv(summary_path, index=False)
        logger.info(f"Saved: {summary_path}")
        
        # Save config
        config_path = self.config.output_dir / f"config_{dataset_name}_{model_name}.json"
        with open(config_path, "w") as f:
            json.dump(asdict(self.config), f, indent=2, default=str)
    
    def _generate_trace_visualizations(
        self,
        traces_by_method: dict,
        stability_by_method: dict,
        digit_fix_by_method: dict,
        all_results: list = None,
    ) -> None:
        """Generate trace-related visualizations."""
        
        # 1. Average fix order heatmap
        try:
            valid_traces = {m: t for m, t in traces_by_method.items() if t}
            if valid_traces:
                heatmap_path = plot_average_fix_order_heatmap(
                    valid_traces,
                    self.config.figures_dir / "avg_fix_order_heatmap.png",
                )
                logger.info(f"Saved: {heatmap_path}")
        except Exception as e:
            logger.warning(f"Failed to generate fix order heatmap: {e}")
        
        # 2. Answer stability boxplot
        try:
            valid_stability = {m: s for m, s in stability_by_method.items() if s}
            if valid_stability:
                stability_path = plot_answer_stability_stats(
                    valid_stability,
                    self.config.figures_dir / "answer_stability_stats.png",
                )
                logger.info(f"Saved: {stability_path}")
        except Exception as e:
            logger.warning(f"Failed to generate stability stats: {e}")
        
        # 3. Stability summary JSON
        try:
            stability_summary = compute_stability_summary(stability_by_method)
            summary_path = self.config.output_dir / "stability_summary.json"
            with open(summary_path, "w") as f:
                json.dump(stability_summary, f, indent=2, default=float)
            logger.info(f"Saved: {summary_path}")
        except Exception as e:
            logger.warning(f"Failed to save stability summary: {e}")
        
        # 4. Digit fix order summary (with correct/incorrect split)
        try:
            digit_fix_summary = {}
            for method, fixes in digit_fix_by_method.items():
                # Get correctness for this method's samples
                if all_results:
                    method_results = [r for r in all_results if r["method"] == method]
                    correctness = [r["correct"] for r in method_results]
                    
                    # Ensure alignment
                    if len(correctness) == len(fixes):
                        digit_fix_summary[method] = aggregate_digit_fix_stats(fixes, correctness)
                    else:
                        digit_fix_summary[method] = aggregate_digit_fix_stats(fixes)
                else:
                    digit_fix_summary[method] = aggregate_digit_fix_stats(fixes)
            
            digit_path = self.config.output_dir / "digit_fix_summary.json"
            with open(digit_path, "w") as f:
                json.dump(digit_fix_summary, f, indent=2, default=str)
            logger.info(f"Saved: {digit_path}")
        except Exception as e:
            logger.warning(f"Failed to save digit fix summary: {e}")
    
    def _log_to_wandb(self, df: pd.DataFrame, summary: pd.DataFrame) -> None:
        """Log results to Weights & Biases."""
        import matplotlib.pyplot as plt
        
        # Summary table
        self._wandb.log({"summary_table": self._wandb.Table(dataframe=summary)})
        
        # Predictions table (cleaned for display)
        pred_cols = ["method", "example_idx", "question", "gold", "pred", "correct",
                     "word_count", "step_count", "equation_count", "gen_text"]
        pred_cols = [c for c in pred_cols if c in df.columns]
        predictions_df = df[pred_cols].copy()
        
        def clean_for_wandb(text, max_len=500):
            if not isinstance(text, str):
                return str(text) if text is not None else ""
            text = text.replace("\n", " ").replace("\t", " ")
            while "  " in text:
                text = text.replace("  ", " ")
            return text[:max_len] + "..." if len(text) > max_len else text
        
        for col in ["gen_text", "question"]:
            if col in predictions_df.columns:
                predictions_df[col] = predictions_df[col].apply(
                    lambda x: clean_for_wandb(x, 400)
                )
        
        try:
            self._wandb.log({
                "predictions_table": self._wandb.Table(dataframe=predictions_df)
            })
        except Exception as e:
            logger.warning(f"Failed to log predictions table: {e}")
        
        # Accuracy bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ['#808080', '#4CAF50', '#2196F3', '#FF9800', '#9C27B0']
        bars = ax.bar(summary["method"], summary["accuracy"], 
                     color=colors[:len(summary)])
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Accuracy by CoT Method ({self.dataset_name})")
        ax.set_ylim(0, 1)
        for bar, acc in zip(bars, summary["accuracy"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{acc:.2f}', ha='center', va='bottom', fontsize=12)
        plt.tight_layout()
        
        acc_path = self.config.figures_dir / "accuracy_by_method.png"
        plt.savefig(acc_path, dpi=150)
        plt.close()
        self._wandb.log({"accuracy_chart": self._wandb.Image(str(acc_path))})
        
        # Log trace visualizations if they exist
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


def run_benchmark(
    dataset: str = "biggsm",
    model: str = "llada",
    n_eval: int = 80,
    methods: List[str] = None,
    output_dir: str = "outputs",
    use_wandb: bool = False,
    **kwargs
) -> pd.DataFrame:
    """Convenience function to run benchmark.
    
    Args:
        dataset: Dataset name ("biggsm", "math", "countdown")
        model: Model name ("llada", "qwen3")
        n_eval: Number of examples to evaluate
        methods: List of methods to test
        output_dir: Output directory
        use_wandb: Whether to use W&B logging
        **kwargs: Additional config options
            - math_levels: List of MATH difficulty levels
            - countdown_num_count: Number of Countdown starting numbers
        
    Returns:
        Summary DataFrame
    """
    from .config import DatasetConfig, ModelConfig, ExperimentConfig
    
    # Build config
    dataset_config = DatasetConfig(
        type=DatasetType(dataset.lower()),
        n_eval=n_eval,
        math_levels=kwargs.get("math_levels", [3, 4, 5]),
        countdown_num_count=kwargs.get("countdown_num_count", 4),
    )
    
    model_config = ModelConfig(type=ModelType(model.lower()))
    
    config = ExperimentConfig(
        dataset=dataset_config,
        model=model_config,
        methods=methods or ["Direct", "Zero-CoT", "Complex-CoT", "MARP", "Diff-MARP"],
        output_dir=Path(output_dir),
        figures_dir=Path(output_dir) / "figures",
        use_wandb=use_wandb,
    )
    
    benchmark = Benchmark(config)
    return benchmark.run()
