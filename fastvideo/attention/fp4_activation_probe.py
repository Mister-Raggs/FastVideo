# SPDX-License-Identifier: Apache-2.0
"""Exact-activation diagnostics for the sm12x FP4 attention kernel.

This module is intentionally opt-in. The Cosmos experiment driver installs
forward-pre-hooks in a worker through ``Executor.collective_rpc``; normal model
construction and attention dispatch never import or invoke the probe.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from fastvideo.attention.backends.attn_qat_infer import (
    _get_attn_qat_infer,
    attn_qat_infer_receipt,
)
from fastvideo.forward_context import get_forward_context


def resolve_block_indices(num_blocks: int, requested: list[int] | None) -> list[int]:
    if num_blocks <= 0:
        raise ValueError("the transformer has no blocks")
    indices = requested if requested is not None else [0, num_blocks // 2, num_blocks - 1]
    unique = sorted(set(indices))
    invalid = [index for index in unique if index < 0 or index >= num_blocks]
    if invalid:
        raise ValueError(f"block indices {invalid} are outside [0, {num_blocks - 1}]")
    return unique


def _sample_token_indices(length: int, count: int, device: torch.device) -> torch.Tensor:
    if length <= count:
        return torch.arange(length, device=device)
    return torch.linspace(0, length - 1, steps=count, device=device).round().long().unique()


def tensor_distribution(tensor_bhld: torch.Tensor, token_samples: int = 256) -> dict[str, Any]:
    """Summarize a tensor using an evenly spaced token sample per head."""
    indices = _sample_token_indices(tensor_bhld.shape[2], token_samples, tensor_bhld.device)
    sample = tensor_bhld[:, :, indices, :].float()
    flat = sample.flatten()
    absolute = flat.abs()
    quantiles = torch.quantile(absolute, torch.tensor([0.5, 0.95, 0.99, 0.999], device=flat.device))
    head_rms = sample.square().mean(dim=(0, 2, 3)).sqrt()
    head_absmax = sample.abs().amax(dim=(0, 2, 3))
    return {
        "sampled_elements": flat.numel(),
        "finite_fraction": float(torch.isfinite(flat).float().mean().item()),
        "mean": float(flat.mean().item()),
        "std": float(flat.std().item()),
        "rms": float(flat.square().mean().sqrt().item()),
        "abs_mean": float(absolute.mean().item()),
        "abs_max": float(absolute.max().item()),
        "abs_p50": float(quantiles[0].item()),
        "abs_p95": float(quantiles[1].item()),
        "abs_p99": float(quantiles[2].item()),
        "abs_p999": float(quantiles[3].item()),
        "per_head_rms": [float(value) for value in head_rms.tolist()],
        "per_head_abs_max": [float(value) for value in head_absmax.tolist()],
    }


def sampled_logit_distribution(
    query_bhld: torch.Tensor,
    key_bhld: torch.Tensor,
    softmax_scale: float,
    query_samples: int = 64,
    key_samples: int = 256,
) -> dict[str, Any]:
    """Measure representative scaled QK logits without materializing QK^T."""
    query_indices = _sample_token_indices(query_bhld.shape[2], query_samples, query_bhld.device)
    key_indices = _sample_token_indices(key_bhld.shape[2], key_samples, key_bhld.device)
    sampled_query = query_bhld[:, :, query_indices, :].float()
    sampled_key = key_bhld[:, :, key_indices, :].float()
    logits = torch.matmul(sampled_query, sampled_key.transpose(-2, -1)) * softmax_scale
    flat = logits.flatten()
    absolute = flat.abs()
    quantiles = torch.quantile(absolute, torch.tensor([0.5, 0.95, 0.99, 0.999], device=flat.device))
    head_absmax = logits.abs().amax(dim=(0, 2, 3))
    return {
        "sampled_query_tokens": query_indices.numel(),
        "sampled_key_tokens": key_indices.numel(),
        "sampled_logits": flat.numel(),
        "mean": float(flat.mean().item()),
        "std": float(flat.std().item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "abs_p50": float(quantiles[0].item()),
        "abs_p95": float(quantiles[1].item()),
        "abs_p99": float(quantiles[2].item()),
        "abs_p999": float(quantiles[3].item()),
        "per_head_abs_max": [float(value) for value in head_absmax.tolist()],
    }


def output_error(fp4_output: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    fp4_float = fp4_output.float()
    reference_float = reference.float()
    difference = fp4_float - reference_float
    reduce_dims = (0, 2, 3)
    dot = (fp4_float * reference_float).sum(dim=reduce_dims)
    fp4_norm = fp4_float.square().sum(dim=reduce_dims).sqrt()
    reference_norm = reference_float.square().sum(dim=reduce_dims).sqrt()
    per_head_cosine = dot / (fp4_norm * reference_norm).clamp_min(torch.finfo(torch.float32).tiny)
    per_head_relative_l2 = difference.square().sum(dim=reduce_dims).sqrt() / reference_norm.clamp_min(
        torch.finfo(torch.float32).tiny)

    fp4_flat = fp4_float.flatten()
    reference_flat = reference_float.flatten()
    denominator = torch.linalg.vector_norm(reference_flat).clamp_min(torch.finfo(torch.float32).tiny)
    cosine = F.cosine_similarity(fp4_flat, reference_flat, dim=0)
    return {
        "finite": bool(torch.isfinite(fp4_float).all().item()),
        "cosine_similarity": float(cosine.item()),
        "relative_l2": float((torch.linalg.vector_norm(difference.flatten()) / denominator).item()),
        "mean_absolute_error": float(difference.abs().mean().item()),
        "max_absolute_error": float(difference.abs().max().item()),
        "per_head_cosine": [float(value) for value in per_head_cosine.tolist()],
        "per_head_relative_l2": [float(value) for value in per_head_relative_l2.tolist()],
    }


def _timing_summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "median_ms": statistics.median(ordered),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def _time_pair(sdpa, fp4, warmup: int, iterations: int) -> dict[str, Any]:
    for _ in range(warmup):
        sdpa()
        fp4()
    torch.cuda.synchronize()

    values: dict[str, list[float]] = {"sdpa": [], "fp4": []}
    functions = {"sdpa": sdpa, "fp4": fp4}
    for iteration in range(iterations):
        order = ("sdpa", "fp4") if iteration % 2 == 0 else ("fp4", "sdpa")
        for name in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = functions[name]()
            end.record()
            end.synchronize()
            values[name].append(start.elapsed_time(end))
            del output

    sdpa_summary = _timing_summary(values["sdpa"])
    fp4_summary = _timing_summary(values["fp4"])
    speedup = sdpa_summary["median_ms"] / fp4_summary["median_ms"]
    return {
        "sdpa": sdpa_summary,
        "fp4": fp4_summary,
        "speedup_x": speedup,
        "latency_reduction_percent": (1.0 - fp4_summary["median_ms"] / sdpa_summary["median_ms"]) * 100.0,
    }


def _current_timestep() -> Any:
    try:
        timestep = get_forward_context().current_timestep
    except AssertionError:
        return None
    if isinstance(timestep, torch.Tensor):
        values = timestep.detach().float()
        if values.numel() == 1:
            return float(values.item())
        return {
            "min": float(values.min().item()),
            "max": float(values.max().item()),
        }
    if isinstance(timestep, int | float):
        return timestep
    return str(timestep)


@dataclass
class _ProbeState:
    output_path: Path
    selected_blocks: list[int]
    capture_calls: set[int]
    warmup: int
    iterations: int
    metadata: dict[str, Any]
    counts: dict[str, int] = field(default_factory=dict)
    records: list[dict[str, Any]] = field(default_factory=list)
    handles: list[Any] = field(default_factory=list)
    active_capture: dict[str, Any] | None = None

    def receipt(self, *, complete: bool = False) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "selected_blocks": self.selected_blocks,
            "capture_calls": sorted(self.capture_calls),
            "observed_calls": dict(sorted(self.counts.items())),
            "active_capture": self.active_capture,
            "complete": complete,
            "records": self.records,
        }

    def checkpoint(self, *, complete: bool = False) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(self.receipt(complete=complete), indent=2) + "\n")


_ACTIVE_PROBE: _ProbeState | None = None


@torch.inference_mode()
def _capture_exact_activation(
    state: _ProbeState,
    block_index: int,
    role: str,
    module: torch.nn.Module,
    args: tuple[Any, ...],
) -> None:
    if len(args) < 3:
        raise RuntimeError(f"{role} attention hook expected q, k, v positional arguments")
    query_blhd, key_blhd, value_blhd = args[:3]
    key = f"block-{block_index}-{role}"
    call_index = state.counts.get(key, 0)
    state.counts[key] = call_index + 1
    if call_index not in state.capture_calls:
        return

    capture_id = f"block-{block_index}-{role}-call-{call_index}"
    state.active_capture = {
        "capture_id": capture_id,
        "block_index": block_index,
        "role": role,
        "call_index": call_index,
        "timestep": _current_timestep(),
    }
    state.checkpoint()
    print(f"[fp4-probe] capturing {capture_id} timestep={state.active_capture['timestep']}", flush=True)

    query = query_blhd.transpose(1, 2).contiguous()
    key_tensor = key_blhd.transpose(1, 2).contiguous()
    value = value_blhd.transpose(1, 2).contiguous()
    softmax_scale = float(module.softmax_scale)

    def sdpa() -> torch.Tensor:
        return F.scaled_dot_product_attention(query, key_tensor, value, is_causal=False, scale=softmax_scale)

    kernel = _get_attn_qat_infer()
    if kernel is None:
        raise RuntimeError("the sm12x FP4 attention extension is unavailable")

    def fp4() -> torch.Tensor:
        return kernel(query, key_tensor, value, is_causal=False, sm_scale=softmax_scale)

    record = dict(state.active_capture)
    record.update({
        "shape": {
            "batch": query.shape[0],
            "heads": query.shape[1],
            "q_len": query.shape[2],
            "kv_len": key_tensor.shape[2],
            "head_dim": query.shape[3],
            "q_aligned_128": query.shape[2] % 128 == 0,
            "kv_aligned_128": key_tensor.shape[2] % 128 == 0,
        },
        "dtype": str(query.dtype),
        "softmax_scale": softmax_scale,
    })

    try:
        reference = sdpa()
        fp4_result = fp4()
        torch.cuda.synchronize()
        record.update({
            "status": "ok",
            "activations": {
                "query": tensor_distribution(query),
                "key": tensor_distribution(key_tensor),
                "value": tensor_distribution(value),
                "sampled_scaled_qk_logits": sampled_logit_distribution(query, key_tensor, softmax_scale),
            },
            "output_error": output_error(fp4_result, reference),
            "timing": _time_pair(sdpa, fp4, state.warmup, state.iterations),
        })
        del reference, fp4_result
    except Exception as exc:
        record.update({
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        torch.cuda.empty_cache()

    state.records.append(record)
    state.active_capture = None
    state.checkpoint()
    if record["status"] == "ok":
        error = record["output_error"]
        timing = record["timing"]
        print(
            f"[fp4-probe] {capture_id} cos={error['cosine_similarity']:.6f} "
            f"rel_l2={error['relative_l2']:.4f} speedup={timing['speedup_x']:.3f}x",
            flush=True)
    else:
        print(f"[fp4-probe] {capture_id} ERROR {record['error_type']}: {record['error']}", flush=True)


def _find_transformer(worker) -> torch.nn.Module:
    pipeline = getattr(worker, "pipeline", None)
    modules = getattr(pipeline, "modules", None)
    if not isinstance(modules, dict) or "transformer" not in modules:
        raise RuntimeError("worker pipeline has no eagerly loaded transformer module")
    transformer = modules["transformer"]
    while not hasattr(transformer, "transformer_blocks") and hasattr(transformer, "module"):
        transformer = transformer.module
    if not hasattr(transformer, "transformer_blocks"):
        raise TypeError(f"expected a Cosmos transformer, got {type(transformer).__name__}")
    return transformer


def install_fp4_activation_probe(
    worker,
    *,
    output_path: str,
    capture_calls: list[int],
    block_indices: list[int] | None = None,
    warmup: int = 1,
    iterations: int = 3,
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Install exact-activation hooks inside a FastVideo GPU worker."""
    global _ACTIVE_PROBE
    if _ACTIVE_PROBE is not None:
        raise RuntimeError("an FP4 activation probe is already installed")
    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    if not capture_calls or any(index < 0 for index in capture_calls):
        raise ValueError("capture_calls must contain non-negative call indices")
    if _get_attn_qat_infer() is None:
        raise RuntimeError(f"FP4 kernel is unavailable ({attn_qat_infer_receipt()})")

    transformer = _find_transformer(worker)
    blocks = transformer.transformer_blocks
    selected_blocks = resolve_block_indices(len(blocks), block_indices)
    capability = torch.cuda.get_device_capability()
    state = _ProbeState(
        output_path=Path(output_path),
        selected_blocks=selected_blocks,
        capture_calls=set(capture_calls),
        warmup=warmup,
        iterations=iterations,
        metadata={
            "created_at": datetime.now(timezone.utc).isoformat(),
            "device_name": torch.cuda.get_device_name(),
            "device_capability": list(capability),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "fp4_kernel": attn_qat_infer_receipt(),
            "generation_attention_backend": "TORCH_SDPA",
            "comparison": "exact normalized/RoPE-applied QKV: FP4 vs BF16 SDPA",
            "warmup": warmup,
            "iterations": iterations,
            "run": run_metadata or {},
        },
    )

    for block_index in selected_blocks:
        block = blocks[block_index]
        targets = (("self", block.attn1.attn), ("cross", block.attn2.attn))
        for role, module in targets:
            hook = partial(_capture_exact_activation, state, block_index, role)
            state.handles.append(module.register_forward_pre_hook(hook))

    _ACTIVE_PROBE = state
    state.checkpoint()
    return {
        "status": "installed",
        "selected_blocks": selected_blocks,
        "capture_calls": sorted(state.capture_calls),
        "output_path": str(state.output_path),
    }


def finish_fp4_activation_probe(worker) -> dict[str, Any]:
    """Remove hooks and return the worker's JSON-safe receipt."""
    del worker
    global _ACTIVE_PROBE
    if _ACTIVE_PROBE is None:
        raise RuntimeError("no FP4 activation probe is installed")
    state = _ACTIVE_PROBE
    for handle in state.handles:
        handle.remove()
    state.handles.clear()
    state.active_capture = None
    state.checkpoint(complete=True)
    receipt = state.receipt(complete=True)
    _ACTIVE_PROBE = None
    return receipt
