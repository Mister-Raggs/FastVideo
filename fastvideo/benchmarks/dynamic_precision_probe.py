# SPDX-License-Identifier: Apache-2.0
"""Phase-A probe for timestep-aware precision routing in Wan blocks.

This is an experiment harness, not a production router.  The denoising
trajectory remains dense BF16.  At selected block/step pairs, the harness
replays the block once per target projection with only that projection using
FastVideo's existing on-the-fly NVFP4 path, then records the block-output
error.  It also records the inexpensive block-change signal at every step so
the analyzer can evaluate the actual one-step-lag predictor.

The design is intentionally process-safe for ``VideoGenerator``: the quant
config and Wan forward wrapper live in an importable FastVideo module, and the
worker writes JSONL directly to the requested path.
"""

from __future__ import annotations

import json
import math
import re
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from collections.abc import Callable, Iterable, Sequence

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from fastvideo.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from fastvideo.models.utils import set_weight_attrs

WAN_TARGET_SUFFIXES = (
    ".to_q",
    ".to_k",
    ".to_v",
    ".to_out",
    ".attn2.to_q",
    ".attn2.to_k",
    ".attn2.to_v",
    ".attn2.to_out",
    ".ffn.fc_in",
    ".ffn.fc_out",
)
_BLOCK_RE = re.compile(r"(?:^|\.)blocks\.(\d+)(?:\.|$)")


@dataclass(frozen=True)
class ProbeSpec:
    """Serializable experiment selection shared with the model worker."""

    output_path: str
    block_indices: tuple[int, ...] = (0, 5, 10, 15, 20, 25, 29)
    error_step_indices: tuple[int, ...] = (2, 8, 16, 24, 32, 40, 47)
    target_suffixes: tuple[str, ...] = WAN_TARGET_SUFFIXES
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if not self.output_path:
            raise ValueError("output_path must not be empty")
        if not self.block_indices:
            raise ValueError("at least one block index is required")
        if not self.error_step_indices:
            raise ValueError("at least one error step index is required")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")


@dataclass(frozen=True)
class ProbeRecord:
    kind: str
    step_index: int
    block_index: int
    gamma: float
    projection: str | None = None
    relative_l2_error: float | None = None


class ProbeState:
    """Mutable worker-local state; always restored after a perturbation."""

    def __init__(self, spec: ProbeSpec):
        self.spec = spec
        self.active_prefix: str | None = None
        self.in_block_replay = False
        self._write_lock = threading.Lock()

    def __getstate__(self) -> dict[str, Any]:
        # Pipeline configs cross the VideoGenerator worker boundary. Locks are
        # process-local and cannot be pickled, so recreate this one on load.
        state = self.__dict__.copy()
        state.pop("_write_lock", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._write_lock = threading.Lock()

    def write(self, record: ProbeRecord) -> None:
        path = Path(self.spec.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(asdict(record), sort_keys=True)
        with self._write_lock, path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
            stream.flush()


def relative_l1_block_change(x: torch.Tensor, y: torch.Tensor, epsilon: float = 1e-12) -> float:
    """Return ||y - x||_1 / max(||x||_1, epsilon), accumulated in fp32."""

    numerator = torch.linalg.vector_norm((y.float() - x.float()).reshape(-1), ord=1)
    denominator = torch.linalg.vector_norm(x.float().reshape(-1), ord=1).clamp_min(epsilon)
    return float((numerator / denominator).item())


def relative_l2_error(reference: torch.Tensor, candidate: torch.Tensor, epsilon: float = 1e-12) -> float:
    """Return ||reference - candidate||_2 / max(||reference||_2, epsilon)."""

    numerator = torch.linalg.vector_norm((reference.float() - candidate.float()).reshape(-1), ord=2)
    denominator = torch.linalg.vector_norm(reference.float().reshape(-1), ord=2).clamp_min(epsilon)
    return float((numerator / denominator).item())


class DynamicPrecisionProbeLinearMethod(QuantizeMethodBase):
    """Dense master weight with worker-local, one-projection NVFP4 routing."""

    def __init__(self, prefix: str, state: ProbeState):
        self.prefix = prefix
        self.state = state

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        del input_size, output_size
        weight = Parameter(
            torch.empty(sum(output_partition_sizes), input_size_per_partition, dtype=params_dtype),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.state.active_prefix != self.prefix:
            return F.linear(x, layer.weight, bias)

        # This import is deliberately lazy: importing/analyzing the scaffold on
        # a CPU-only laptop must not initialize CUDA or require FP4 libraries.
        from fastvideo.layers.fp4linear import _LinearFWD4BWD16Fn

        return _LinearFWD4BWD16Fn.apply(x, layer.weight, bias, "cutlass", 16, True)


class DynamicPrecisionProbeConfig(QuantizationConfig):
    """Quant config that tags Wan projections without changing the main path."""

    def __init__(self, spec: ProbeSpec):
        super().__init__()
        self.spec = spec
        self.state = ProbeState(spec)

    def get_name(self) -> str:
        return "dynamic_precision_probe"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 100

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DynamicPrecisionProbeConfig:
        return cls(ProbeSpec(**config))

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> QuantizeMethodBase | None:
        from fastvideo.layers.linear import LinearBase

        if not isinstance(layer, LinearBase):
            return None
        block_index = block_index_from_prefix(prefix)
        if block_index not in self.spec.block_indices:
            return None
        if not any(prefix.endswith(suffix) for suffix in self.spec.target_suffixes):
            return None
        install_wan_block_probe()
        return DynamicPrecisionProbeLinearMethod(prefix, self.state)


def block_index_from_prefix(prefix: str) -> int | None:
    match = _BLOCK_RE.search(prefix)
    return int(match.group(1)) if match else None


def _probe_methods(block: torch.nn.Module) -> list[DynamicPrecisionProbeLinearMethod]:
    methods: list[DynamicPrecisionProbeLinearMethod] = []
    for module in block.modules():
        method = getattr(module, "quant_method", None)
        if isinstance(method, DynamicPrecisionProbeLinearMethod):
            methods.append(method)
    return sorted(methods, key=lambda method: method.prefix)


def run_block_probe(
    original_forward: Callable[..., torch.Tensor],
    block: torch.nn.Module,
    args: Sequence[Any],
    kwargs: dict[str, Any],
    step_index: int,
) -> torch.Tensor:
    """Run the BF16 block result and optional isolated projection replays."""

    methods = _probe_methods(block)
    if not methods:
        return original_forward(block, *args, **kwargs)
    state = methods[0].state
    if state.in_block_replay:
        return original_forward(block, *args, **kwargs)
    if any(method.state is not state for method in methods):
        raise RuntimeError("a probed block contains methods from multiple probe states")

    hidden_states = kwargs.get("hidden_states", args[0] if args else None)
    if not isinstance(hidden_states, torch.Tensor):
        raise TypeError("Wan block probe could not locate hidden_states")
    block_index = block_index_from_prefix(methods[0].prefix)
    if block_index is None:
        raise RuntimeError(f"could not parse block index from {methods[0].prefix!r}")

    baseline = original_forward(block, *args, **kwargs)
    metric_input = hidden_states
    if metric_input.ndim == baseline.ndim + 1 and metric_input.shape[1] == 1:
        metric_input = metric_input.squeeze(1)
    if metric_input.shape != baseline.shape:
        raise RuntimeError(
            f"Wan block input/output shape mismatch for gamma: {tuple(metric_input.shape)} vs {tuple(baseline.shape)}")
    gamma = relative_l1_block_change(metric_input, baseline, state.spec.epsilon)
    state.write(ProbeRecord(kind="block", step_index=step_index, block_index=block_index, gamma=gamma))

    if step_index not in state.spec.error_step_indices:
        return baseline

    previous_prefix = state.active_prefix
    try:
        state.in_block_replay = True
        for method in methods:
            state.active_prefix = method.prefix
            candidate = original_forward(block, *args, **kwargs)
            error = relative_l2_error(baseline, candidate, state.spec.epsilon)
            state.write(
                ProbeRecord(
                    kind="projection",
                    step_index=step_index,
                    block_index=block_index,
                    projection=method.prefix,
                    gamma=gamma,
                    relative_l2_error=error,
                ))
    finally:
        state.active_prefix = previous_prefix
        state.in_block_replay = False
    return baseline


_WAN_WRAPPER_INSTALLED = False


def install_wan_block_probe() -> None:
    """Install the guarded Wan block wrapper once in the current process."""

    global _WAN_WRAPPER_INSTALLED
    if _WAN_WRAPPER_INSTALLED:
        return
    from fastvideo.forward_context import get_forward_context
    from fastvideo.models.dits.wanvideo import WanTransformerBlock

    original_forward = WanTransformerBlock.forward

    def probed_forward(block: torch.nn.Module, *args: Any, **kwargs: Any) -> torch.Tensor:
        methods = _probe_methods(block)
        if not methods:
            return original_forward(block, *args, **kwargs)
        context = get_forward_context()
        if context.forward_batch is not None and getattr(context.forward_batch, "is_cfg_negative", False):
            raise RuntimeError("dynamic precision probe requires guidance_scale=1.0 (CFG disabled)")
        step_index = int(context.current_timestep)
        return run_block_probe(original_forward, block, args, kwargs, step_index)

    WanTransformerBlock.forward = probed_forward
    _WAN_WRAPPER_INSTALLED = True


def load_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at line {line_number}: {error}") from error
    return records


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        average = (position + end - 1) / 2.0
        for ordered_index in order[position:end]:
            ranks[ordered_index] = average
        position = end
    return ranks


def pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return math.nan
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    x_scale = math.sqrt(sum((x - x_mean)**2 for x in xs))
    y_scale = math.sqrt(sum((y - y_mean)**2 for y in ys))
    return numerator / (x_scale * y_scale) if x_scale and y_scale else math.nan


def spearman_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    return pearson_correlation(_average_ranks(xs), _average_ranks(ys))


def linear_fit(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return math.nan, math.nan
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denominator = sum((x - x_mean)**2 for x in xs)
    if denominator == 0:
        return math.nan, math.nan
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True)) / denominator
    return slope, y_mean - slope * x_mean


def best_threshold_route(
    predictors: Sequence[float],
    errors: Sequence[float],
    high_error_fraction: float = 0.1,
    minimum_recall: float = 0.9,
) -> dict[str, float] | None:
    """Find the largest FP4 share while BF16 catches high-error samples."""

    if len(predictors) != len(errors) or not predictors:
        return None
    high_count = max(1, math.ceil(len(errors) * high_error_fraction))
    high_error_indices = set(sorted(range(len(errors)), key=errors.__getitem__, reverse=True)[:high_count])
    best: dict[str, float] | None = None
    for threshold in sorted(set(predictors)):
        bf16 = {index for index, predictor in enumerate(predictors) if predictor >= threshold}
        recall = len(bf16 & high_error_indices) / high_count
        fp4_fraction = 1.0 - len(bf16) / len(predictors)
        if recall >= minimum_recall and (best is None or fp4_fraction > best["fp4_fraction"]):
            best = {"threshold": threshold, "high_error_recall": recall, "fp4_fraction": fp4_fraction}
    return best


def analyze_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Analyze exact previous-step gamma against current projection error."""

    materialized = list(records)
    gamma_by_block_step = {
        (int(record["block_index"]), int(record["step_index"])): float(record["gamma"])
        for record in materialized if record.get("kind") == "block"
    }
    groups: dict[tuple[int, str], list[tuple[float, float, float]]] = {}
    skipped_without_previous_step = 0
    for record in materialized:
        if record.get("kind") != "projection":
            continue
        block = int(record["block_index"])
        step = int(record["step_index"])
        previous_gamma = gamma_by_block_step.get((block, step - 1))
        if previous_gamma is None:
            skipped_without_previous_step += 1
            continue
        projection = str(record["projection"])
        groups.setdefault((block, projection), []).append(
            (previous_gamma, float(record["gamma"]), float(record["relative_l2_error"])))

    summaries = []
    all_lagged: list[float] = []
    all_same_step: list[float] = []
    all_errors: list[float] = []
    for (block, projection), values in sorted(groups.items()):
        lagged = [value[0] for value in values]
        same_step = [value[1] for value in values]
        errors = [value[2] for value in values]
        slope, intercept = linear_fit(lagged, errors)
        summaries.append({
            "block_index": block,
            "projection": projection,
            "samples": len(values),
            "lagged_spearman": spearman_correlation(lagged, errors),
            "same_step_spearman": spearman_correlation(same_step, errors),
            "linear_slope": slope,
            "linear_intercept": intercept,
            "route": best_threshold_route(lagged, errors),
        })
        all_lagged.extend(lagged)
        all_same_step.extend(same_step)
        all_errors.extend(errors)

    slope, intercept = linear_fit(all_lagged, all_errors)
    return {
        "paired_samples": len(all_errors),
        "skipped_without_previous_step": skipped_without_previous_step,
        "overall": {
            "lagged_spearman": spearman_correlation(all_lagged, all_errors),
            "same_step_spearman": spearman_correlation(all_same_step, all_errors),
            "linear_slope": slope,
            "linear_intercept": intercept,
            "route": best_threshold_route(all_lagged, all_errors),
        },
        "by_block_projection": summaries,
    }
