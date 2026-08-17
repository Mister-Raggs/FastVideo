# SPDX-License-Identifier: Apache-2.0
"""
D1 S2: per-layer KV budget allocation for causal/AR DiTs.

Today every layer gets the same KV window (`local_attn_size`), a hand-picked
constant. The Phase-A/S1 profiles show that is the wrong shape: attention mass
by token age varies enormously across layers (newest-bucket spread 0.633 on
Matrix-Game, 0.660 on SFWan), and the SAME layer indices are the recency-light
ones in both models (L13/15/18/22/23) despite different checkpoints and window
regimes.

Calibration also showed uniform *shrinking* is not viable: cutting the window
6->4 halves the motion in the output. So the claim this module exists to test is
reallocation at constant cost --

    protect the recency-light layers, starve the recency-bound ones,
    same total bytes, and keep the motion a uniform cut destroys.

Config (all no-ops unless FASTVIDEO_KV_BUDGET_LAYERS is set):

    FASTVIDEO_KV_BUDGET_LAYERS="13,15,18,22,23"   layers to protect
    FASTVIDEO_KV_BUDGET_LONG=12                   their window (latent frames)
    FASTVIDEO_KV_BUDGET_SHORT=4                   everyone else

The plan is applied to each attention module's `local_attn_size` and recorded on
the transformer as `_kv_budget_plan`, which the denoising stage reads to size
each layer's cache buffer to match. Both are required: the buffer bounds the
rolling cache, `local_attn_size` bounds the attention window.

THIS CHANGES OUTPUT. It is an experiment, not instrumentation.
"""

from __future__ import annotations

import os

from torch import nn

from fastvideo.hooks.kv_probe import _is_causal_self_attn
from fastvideo.logger import init_logger

logger = init_logger(__name__)


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _parse_layers() -> set[int] | None:
    raw = os.getenv("FASTVIDEO_KV_BUDGET_LAYERS", "").strip()
    if not raw:
        return None
    out = set()
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.add(int(tok))
        except ValueError:
            logger.warning("FASTVIDEO_KV_BUDGET_LAYERS: ignoring non-integer %r", tok)
    return out or None


def apply_kv_budget_plan(model: nn.Module | None) -> list[int] | None:
    """Assign a per-layer `local_attn_size` and record the plan on *model*.

    Returns the plan (one window per causal attention layer, in block order), or
    None when unset. Logs the total against the uniform baseline so an
    equal-budget claim can be checked rather than assumed.
    """
    protect = _parse_layers()
    if protect is None or model is None:
        return None

    long_w = _int_env("FASTVIDEO_KV_BUDGET_LONG", 12)
    short_w = _int_env("FASTVIDEO_KV_BUDGET_SHORT", 4)
    if long_w < 1 or short_w < 1:
        logger.warning("KV budget windows must be >= 1; got long=%d short=%d -- ignoring plan", long_w, short_w)
        return None

    plan: list[int] = []
    baseline: list[int] = []
    for _, module in model.named_modules():
        if not _is_causal_self_attn(module) or not hasattr(module, "local_attn_size"):
            continue
        idx = len(plan)
        baseline.append(int(module.local_attn_size))
        window = long_w if idx in protect else short_w
        module.local_attn_size = window
        plan.append(window)

    if not plan:
        logger.warning("FASTVIDEO_KV_BUDGET_LAYERS set but no causal attention layers found")
        return None

    model._kv_budget_plan = plan

    total = sum(plan)
    # the baseline may be -1 (unwindowed); only compare when it is a real window
    base_total = sum(baseline) if all(b > 0 for b in baseline) else None
    unknown = sorted(i for i in protect if i >= len(plan))
    if unknown:
        logger.warning("KV budget: layers %s are out of range (model has %d layers)", unknown, len(plan))

    logger.warning(
        "KV BUDGET PLAN: %d layers protected @%d, %d @%d -- total %d frame-units%s. "
        "THIS CHANGES OUTPUT.",
        sum(1 for w in plan if w == long_w),
        long_w,
        sum(1 for w in plan if w != long_w),
        short_w,
        total,
        f" vs uniform baseline {base_total} ({100.0 * total / base_total:.1f}%)" if base_total else "",
    )
    return plan
