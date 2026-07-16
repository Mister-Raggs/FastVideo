# SPDX-License-Identifier: Apache-2.0
"""Shared-input quantization helpers for attention QKV projections.

When q/k/v (self-attention) — or k/v (cross-attention) — project the *same*
source tensor, a quantized linear method that re-quantizes its input inside
``apply()`` ends up quantizing that identical activation two or three times.
The activation quant is a memory-bound pass over the full [tokens, dim]
tensor, so at video-DiT sequence lengths (~32k tokens) the redundant copies
are a real cost.

These helpers let a caller quantize the shared source once and thread the
pre-quantized tuple through each projection. They are deliberately generic
(they key off the linear's ``quant_method`` duck-type), so they no-op cleanly
for bf16 / unquantized linears and only take effect on the fp8 / fp4 / nvfp4
paths. Originally lived in ``models/dits/ltx2.py``; hoisted here so Wan,
Cosmos, Kandinsky and LTX-2 can share one implementation.
"""
from typing import Any

import torch


def supports_prequantized_input(linear: torch.nn.Module) -> bool:
    """Whether ``linear``'s quant method accepts a pre-quantized input tuple.

    True only when the method exposes ``quantize_input`` and reports
    ``wants_prequantized_input()`` — i.e. a quantized linear that can skip its
    in-``apply`` quantize step. Unquantized / bf16 linears return False.
    """
    quant_method = getattr(linear, "quant_method", None)
    if not callable(getattr(quant_method, "quantize_input", None)):
        return False
    wants_prequant = getattr(quant_method, "wants_prequantized_input", None)
    if callable(wants_prequant):
        try:
            return bool(wants_prequant())
        except Exception:  # noqa: BLE001 - a broken probe means "don't risk it"
            return False
    return True


def project_with_optional_prequant(
    linear: torch.nn.Module,
    x: torch.Tensor,
    pre_quantized: tuple[torch.Tensor, torch.Tensor, Any] | None,
) -> torch.Tensor:
    """Project ``x`` through ``linear``, reusing ``pre_quantized`` if given.

    Falls back to the standard ``linear(x)`` call (dropping the bias
    pass-through tuple element) when there is no pre-quantized input or the
    method does not support one, so callers can use it unconditionally.
    """
    if pre_quantized is None or not supports_prequantized_input(linear):
        return linear(x)[0]
    bias = linear.bias if not linear.skip_bias_add else None
    return linear.quant_method.apply(  # type: ignore[union-attr]
        linear,
        x,
        bias=bias,
        pre_quantized=pre_quantized,
    )
