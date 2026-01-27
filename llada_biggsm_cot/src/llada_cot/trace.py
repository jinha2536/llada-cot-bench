"""
LLaDA denoising trace analysis.

This module provides tools to trace and visualize the step-by-step
denoising process in LLaDA, capturing how tokens transition from
masked to fixed states across generation steps.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .evaluation import extract_hash_answer


@dataclass
class TraceMeta:
    """Metadata about a generation trace."""
    
    prompt_len: int
    gen_length: int
    block_length: int
    steps: int
    threshold: float
    mask_id: int
    eos_id: int


@dataclass
class TraceStep:
    """Single denoising step information."""
    
    global_step: int
    block: int
    step_in_block: int
    fixed_map: list[int]  # 0=masked, 1=fixed
    transfer_map: list[int]  # 1=newly fixed this step
    parsed_answer: str | None


@dataclass
class GenerationTrace:
    """Complete trace of a generation run."""
    
    meta: TraceMeta
    steps: list[TraceStep] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "meta": self.meta.__dict__,
            "per_step": [
                {
                    "global_step": s.global_step,
                    "block": s.block,
                    "step_in_block": s.step_in_block,
                    "fixed_map": s.fixed_map,
                    "transfer_map": s.transfer_map,
                    "parsed_answer": s.parsed_answer,
                }
                for s in self.steps
            ],
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "GenerationTrace":
        """Create from dictionary."""
        meta = TraceMeta(**data["meta"])
        steps = [
            TraceStep(
                global_step=s["global_step"],
                block=s["block"],
                step_in_block=s["step_in_block"],
                fixed_map=s["fixed_map"],
                transfer_map=s["transfer_map"],
                parsed_answer=s["parsed_answer"],
            )
            for s in data["per_step"]
        ]
        return cls(meta=meta, steps=steps)


@dataclass
class AnswerStability:
    """Answer stability analysis results."""
    
    final_answer: str | None
    first_seen_step: int | None  # When answer first appears
    first_stable_step: int | None  # When answer becomes stable


def infer_special_token_ids(tokenizer) -> tuple[int, int]:
    """
    Infer EOS and MASK token IDs from tokenizer.
    
    Args:
        tokenizer: HuggingFace tokenizer.
        
    Returns:
        Tuple of (eos_id, mask_id).
        
    Raises:
        RuntimeError: If token IDs cannot be inferred.
    """
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise RuntimeError("tokenizer.eos_token_id is None")
    
    mask_id = getattr(tokenizer, "mask_token_id", None)
    
    if mask_id is None:
        # Try common mask token formats
        candidates = ["<|mask|>", "<mask>", "[MASK]", "<|MASK|>"]
        for tok in candidates:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid >= 0 and tid != tokenizer.unk_token_id:
                mask_id = tid
                break
    
    if mask_id is None:
        raise RuntimeError("Could not infer mask_id from tokenizer")
    
    return eos_id, mask_id


def decode_with_masks(
    tokenizer,
    ids: list[int],
    mask_id: int,
    mask_repr: str = "▯",
) -> str:
    """
    Decode token IDs while preserving mask tokens visually.
    
    Args:
        tokenizer: HuggingFace tokenizer.
        ids: List of token IDs.
        mask_id: The mask token ID.
        mask_repr: String representation for mask tokens.
        
    Returns:
        Decoded string with mask representations.
    """
    parts: list[str] = []
    buffer: list[int] = []
    
    for tid in ids:
        if tid == mask_id:
            if buffer:
                parts.append(tokenizer.decode(buffer, skip_special_tokens=True))
                buffer = []
            parts.append(mask_repr)
        else:
            buffer.append(int(tid))
    
    if buffer:
        parts.append(tokenizer.decode(buffer, skip_special_tokens=True))
    
    return "".join(parts)


@torch.no_grad()
def generate_with_trace(
    model,
    tokenizer,
    input_ids: Tensor,
    gen_length: int = 512,
    block_length: int = 32,
    steps: int = 32,
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: int | None = None,
    threshold: float = 0.95,
    minimal_topk: int = 1,
    eos_early_stop: bool = True,
) -> tuple[Tensor, GenerationTrace]:
    """
    Generate with LLaDA while tracing the denoising process.
    
    This function replicates LLaDA's generation loop but records
    the state at each denoising step for later analysis.
    
    Args:
        model: LLaDA model with internal APIs.
        tokenizer: Associated tokenizer.
        input_ids: Input prompt tensor [1, seq_len].
        gen_length: Maximum generation length.
        block_length: Block size for semi-autoregressive generation.
        steps: Denoising steps per block.
        temperature: Sampling temperature (0 for greedy).
        top_p: Nucleus sampling threshold.
        top_k: Top-k sampling.
        threshold: Confidence threshold for early fixing.
        minimal_topk: Minimum tokens to transfer per step.
        eos_early_stop: Stop generation on EOS.
        
    Returns:
        Tuple of (final_sequence, generation_trace).
        
    Raises:
        RuntimeError: If model lacks required internal APIs.
    """
    # Validate model has required internal methods
    required_methods = ["_get_num_transfer_tokens", "_sample_with_temperature_topk_topp"]
    for method in required_methods:
        if not hasattr(model, method):
            raise RuntimeError(f"Model lacks required API: {method}")
    
    device = model.device
    dtype = getattr(model, "dtype", torch.bfloat16)
    eos_id, mask_id = infer_special_token_ids(tokenizer)
    
    input_ids = input_ids.to(device)
    prompt_len = input_ids.shape[1]
    
    # Compute schedule
    steps = min(steps, gen_length // minimal_topk)
    schedule = model._get_num_transfer_tokens(block_length, steps)
    
    # Setup attention
    num_blocks = (prompt_len + gen_length + block_length - 1) // block_length
    total_len = num_blocks * block_length
    
    neg_inf = torch.finfo(dtype).min
    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=device))
    attn = (
        block_mask.repeat_interleave(block_length, 0)
        .repeat_interleave(block_length, 1)
        .unsqueeze(0)
        .unsqueeze(0)
    ).bool()
    attn = torch.where(
        attn,
        torch.zeros((), device=device, dtype=dtype),
        torch.tensor(neg_inf, device=device, dtype=dtype),
    )
    pos = torch.arange(total_len, device=device).unsqueeze(0)
    
    # Initialize sequence with masks
    x = torch.full((1, total_len), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = input_ids.clone()
    
    # Track generated tokens on CPU
    gen_state = torch.full((gen_length,), mask_id, dtype=torch.long)
    
    # Create trace
    trace = GenerationTrace(
        meta=TraceMeta(
            prompt_len=int(prompt_len),
            gen_length=int(gen_length),
            block_length=int(block_length),
            steps=int(steps),
            threshold=float(threshold),
            mask_id=int(mask_id),
            eos_id=int(eos_id),
        )
    )
    
    prefill_blocks = prompt_len // block_length
    global_step = 0
    finished = False
    final_cut = prompt_len + gen_length
    
    for b in range(prefill_blocks, num_blocks):
        cur_end = (b + 1) * block_length
        cur_x = x[:, :cur_end]
        cur_attn = attn[:, :, :cur_end, :cur_end]
        cur_pos = pos[:, :cur_end]
        
        for s in range(steps):
            active = cur_x[:, -block_length:] == mask_id
            if active.sum().item() == 0:
                break
            
            logits = model(cur_x, attention_mask=cur_attn, position_ids=cur_pos).logits
            active_logits = logits[:, -block_length:, :]
            
            x0, x0_p = model._sample_with_temperature_topk_topp(
                active_logits, temperature=temperature, top_k=top_k, top_p=top_p
            )
            
            num_to_transfer = int(schedule[s].item())
            conf = torch.where(
                active, x0_p, torch.tensor(float("-inf"), device=device)
            )
            
            # Determine which tokens to fix
            transfer = torch.zeros_like(x0, dtype=torch.bool)
            high_conf = conf[0] > threshold
            
            if int(high_conf.sum().item()) >= num_to_transfer:
                transfer[0] = high_conf
            else:
                k = min(num_to_transfer, int(active.sum().item()))
                _, idx = torch.topk(conf[0], k=k)
                transfer[0, idx] = True
            
            if transfer.any():
                cur_x[:, -block_length:][transfer] = x0[transfer]
            
            # Record step
            newly_fixed: list[int] = []
            for j in range(block_length):
                gpos = b * block_length + j - prompt_len
                if 0 <= gpos < gen_length:
                    gen_state[gpos] = cur_x[0, -block_length + j].detach().cpu()
                    if transfer[0, j].item():
                        newly_fixed.append(gpos)
            
            fixed_map = (gen_state.numpy() != mask_id).astype(np.int8).tolist()
            transfer_map = [0] * gen_length
            for p in newly_fixed:
                transfer_map[p] = 1
            
            partial_text = decode_with_masks(
                tokenizer, gen_state.tolist(), mask_id=mask_id, mask_repr="▯"
            )
            parsed = extract_hash_answer(partial_text)
            
            trace.steps.append(
                TraceStep(
                    global_step=global_step,
                    block=int(b),
                    step_in_block=int(s),
                    fixed_map=fixed_map,
                    transfer_map=transfer_map,
                    parsed_answer=parsed,
                )
            )
            global_step += 1
            
            # Check for EOS
            if eos_early_stop and transfer.any() and (x0[transfer] == eos_id).any():
                eos_pos = (cur_x[0] == eos_id).nonzero(as_tuple=True)[0]
                if len(eos_pos) > 0:
                    eos_pos_val = int(eos_pos[0].item())
                    if (cur_x[0, prompt_len:eos_pos_val] != mask_id).all():
                        final_cut = eos_pos_val + 1
                        finished = True
                        break
        
        x[:, :cur_end] = cur_x
        if finished:
            break
    
    final_seq = x[:, : max(final_cut, prompt_len)]
    return final_seq, trace


def analyze_answer_stability(trace: GenerationTrace) -> AnswerStability:
    """
    Analyze when the answer first appears and stabilizes.
    
    Args:
        trace: Generation trace to analyze.
        
    Returns:
        AnswerStability with timing information.
    """
    answers = [s.parsed_answer for s in trace.steps]
    
    # Find final answer (last non-None)
    final = None
    for ans in reversed(answers):
        if ans is not None:
            final = ans
            break
    
    # Find first appearance
    first_seen = next((i for i, a in enumerate(answers) if a is not None), None)
    
    # Find stability point
    first_stable = None
    if final is not None:
        for i in range(len(answers)):
            if answers[i] == final:
                # Check if all subsequent are final or None
                if all(a == final or a is None for a in answers[i:]):
                    first_stable = i
                    break
    
    return AnswerStability(
        final_answer=final,
        first_seen_step=first_seen,
        first_stable_step=first_stable,
    )


def save_trace_heatmaps(
    trace: GenerationTrace,
    output_dir: Path,
    prefix: str = "",
    title_suffix: str = "",
) -> tuple[Path, Path]:
    """
    Save fixed-map and transfer-map heatmaps.
    
    Args:
        trace: Generation trace.
        output_dir: Directory for output files.
        prefix: Filename prefix.
        title_suffix: Additional title text.
        
    Returns:
        Tuple of (fixed_map_path, transfer_map_path).
    """
    import matplotlib.pyplot as plt
    
    if not trace.steps:
        raise RuntimeError("Empty trace - no steps to visualize")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fixed = np.array([s.fixed_map for s in trace.steps], dtype=np.int8)
    transfer = np.array([s.transfer_map for s in trace.steps], dtype=np.int8)
    
    # Fixed map
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(fixed, aspect="auto", cmap="Blues")
    ax.set_xlabel("Generated token position")
    ax.set_ylabel("Global denoising step")
    ax.set_title(f"Fixed Map (1=fixed, 0=masked) {title_suffix}")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    
    fixed_path = output_dir / f"{prefix}fixed_map.png"
    plt.savefig(fixed_path, dpi=150)
    plt.close()
    
    # Transfer map
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(transfer, aspect="auto", cmap="Oranges")
    ax.set_xlabel("Generated token position")
    ax.set_ylabel("Global denoising step")
    ax.set_title(f"Transfer Map (1=newly fixed) {title_suffix}")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    
    transfer_path = output_dir / f"{prefix}transfer_map.png"
    plt.savefig(transfer_path, dpi=150)
    plt.close()
    
    return fixed_path, transfer_path


def compute_fix_order(trace: GenerationTrace) -> np.ndarray:
    """
    Compute the step at which each position was fixed.
    
    Returns:
        Array of shape (gen_length,) with step index when each position was fixed.
        -1 if never fixed.
    """
    gen_length = trace.meta.gen_length
    fix_step = np.full(gen_length, -1, dtype=np.int32)
    
    for step in trace.steps:
        for pos, transferred in enumerate(step.transfer_map):
            if transferred == 1 and fix_step[pos] == -1:
                fix_step[pos] = step.global_step
    
    return fix_step


def plot_average_fix_order_heatmap(
    traces_by_method: dict[str, list[GenerationTrace]],
    output_path: Path,
    max_positions: int = 100,
) -> Path:
    """
    Plot heatmap showing average fix order by method and position.
    
    Args:
        traces_by_method: Dict mapping method name to list of traces.
        output_path: Path to save the figure.
        max_positions: Maximum positions to show (truncate for readability).
        
    Returns:
        Path to saved figure.
    """
    import matplotlib.pyplot as plt
    
    methods = list(traces_by_method.keys())
    
    # Compute average fix order for each method
    avg_fix_orders = []
    for method in methods:
        traces = traces_by_method[method]
        if not traces:
            continue
        
        # Stack fix orders, handle different lengths
        fix_orders = [compute_fix_order(t) for t in traces]
        max_len = min(max_positions, max(len(fo) for fo in fix_orders))
        
        # Pad/truncate to same length
        padded = []
        for fo in fix_orders:
            if len(fo) >= max_len:
                padded.append(fo[:max_len])
            else:
                padded.append(np.pad(fo, (0, max_len - len(fo)), constant_values=-1))
        
        stacked = np.array(padded, dtype=np.float32)
        stacked[stacked == -1] = np.nan  # Treat unfixed as NaN
        avg = np.nanmean(stacked, axis=0)
        avg_fix_orders.append(avg)
    
    data = np.array(avg_fix_orders)
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, max(3, len(methods) * 0.8)))
    im = ax.imshow(data, aspect="auto", cmap="viridis_r")  # Reverse: early=yellow, late=purple
    
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_xlabel("Token Position")
    ax.set_ylabel("Method")
    ax.set_title("Average Fix Step by Position (lower = fixed earlier)")
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Avg. Fix Step")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    return output_path


def plot_answer_stability_stats(
    stability_by_method: dict[str, list[AnswerStability]],
    output_path: Path,
) -> Path:
    """
    Plot answer stability statistics by method.
    
    Args:
        stability_by_method: Dict mapping method to list of AnswerStability.
        output_path: Path to save the figure.
        
    Returns:
        Path to saved figure.
    """
    import matplotlib.pyplot as plt
    
    methods = list(stability_by_method.keys())
    
    # Prepare data
    first_seen_data = []
    first_stable_data = []
    
    for method in methods:
        stabs = stability_by_method[method]
        seen = [s.first_seen_step for s in stabs if s.first_seen_step is not None]
        stable = [s.first_stable_step for s in stabs if s.first_stable_step is not None]
        first_seen_data.append(seen)
        first_stable_data.append(stable)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # First seen boxplot
    ax1 = axes[0]
    bp1 = ax1.boxplot(first_seen_data, labels=methods, patch_artist=True)
    for patch in bp1['boxes']:
        patch.set_facecolor('lightblue')
    ax1.set_ylabel("Step")
    ax1.set_title("Answer First Seen (lower = appears earlier)")
    ax1.tick_params(axis='x', rotation=15)
    
    # First stable boxplot
    ax2 = axes[1]
    bp2 = ax2.boxplot(first_stable_data, labels=methods, patch_artist=True)
    for patch in bp2['boxes']:
        patch.set_facecolor('lightgreen')
    ax2.set_ylabel("Step")
    ax2.set_title("Answer First Stable (lower = stabilizes earlier)")
    ax2.tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    return output_path


def compute_stability_summary(
    stability_by_method: dict[str, list[AnswerStability]]
) -> dict[str, dict[str, float]]:
    """
    Compute summary statistics for answer stability.
    
    Returns:
        Dict with method -> {mean_first_seen, mean_first_stable, stability_rate}
    """
    summary = {}
    
    for method, stabs in stability_by_method.items():
        seen = [s.first_seen_step for s in stabs if s.first_seen_step is not None]
        stable = [s.first_stable_step for s in stabs if s.first_stable_step is not None]
        
        summary[method] = {
            "mean_first_seen": np.mean(seen) if seen else None,
            "std_first_seen": np.std(seen) if seen else None,
            "mean_first_stable": np.mean(stable) if stable else None,
            "std_first_stable": np.std(stable) if stable else None,
            "stability_rate": len(stable) / len(stabs) if stabs else 0,
            "n_samples": len(stabs),
        }
    
    return summary
