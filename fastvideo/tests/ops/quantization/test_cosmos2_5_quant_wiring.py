# SPDX-License-Identifier: Apache-2.0
"""Cosmos-2.5 linear-quantization wiring contract tests.

Locks in the quant wiring between :class:`FP8Config` /
:class:`NVFP4QATConfig` and the Cosmos-2.5 DiT (``quant_config`` +
``prefix`` plumbing through ``Cosmos25SelfAttention`` /
``Cosmos25CrossAttention`` / ``Cosmos25TransformerBlock``) so it doesn't
silently regress:

- attention ``to_q/to_k/to_v/to_out`` and FFN ``mlp.fc_in/fc_out`` get the
  quant method; nothing else in the block does (AdaLN modulation, norms
  stay plain),
- prefixes match the real module paths (the suffix matchers key off them),
- ``quant_config=None`` keeps every linear on ``UnquantizedLinearMethod``
  (the bf16 path is untouched by the wiring).

The FP8 tests are CPU-only. The NVFP4-QAT method allocates a CUDA tensor
in its constructor, so those attachment tests require a GPU (they run on
Blackwell boxes / GPU CI and skip elsewhere); the pure suffix-list
contract is still asserted on CPU.
"""
from __future__ import annotations

import pytest
import torch

from fastvideo.layers.linear import ReplicatedLinear, UnquantizedLinearMethod
from fastvideo.layers.quantization.fp8_config import _FP8_SUFFIXES, FP8Config, FP8QuantizeMethod
from fastvideo.layers.quantization.nvfp4_qat_config import DEFAULT_FP4_LAYERS
from fastvideo.models.dits.cosmos2_5 import Cosmos25TransformerBlock
from fastvideo.platforms import AttentionBackendEnum

# Linears the quant configs must tag, as (module path, ) relative to a block.
QUANT_TARGET_PATHS = tuple(f"{attn}.{proj}" for attn in ("attn1", "attn2")
                           for proj in ("to_q", "to_k", "to_v", "to_out")) + ("mlp.fc_in", "mlp.fc_out")


def _block(quant_config=None) -> Cosmos25TransformerBlock:
    return Cosmos25TransformerBlock(
        num_attention_heads=2,
        attention_head_dim=32,
        cross_attention_dim=64,
        supported_attention_backends=(AttentionBackendEnum.TORCH_SDPA, ),
        quant_config=quant_config,
        prefix="transformer_blocks.0",
    )


def _quantized_linears(block: Cosmos25TransformerBlock, method_cls) -> list[str]:
    return sorted(name for name, mod in block.named_modules() if isinstance(getattr(mod, "quant_method", None), method_cls))


def test_fp8_tags_attention_and_ffn_linears_only() -> None:
    block = _block(FP8Config())
    assert _quantized_linears(block, FP8QuantizeMethod) == sorted(QUANT_TARGET_PATHS)


def test_fp8_layer_prefixes_match_module_paths() -> None:
    block = _block(FP8Config())
    for path in QUANT_TARGET_PATHS:
        linear = block.get_submodule(path)
        assert isinstance(linear, ReplicatedLinear), (f"{path} must be ReplicatedLinear, got {type(linear).__name__}")
        assert linear.prefix == f"transformer_blocks.0.{path}"


def test_no_quant_config_keeps_every_linear_unquantized() -> None:
    """bf16 path: the wiring must be a no-op when quant_config is None."""
    block = _block(quant_config=None)
    for name, mod in block.named_modules():
        if isinstance(mod, ReplicatedLinear):
            assert isinstance(mod.quant_method, UnquantizedLinearMethod), (
                f"{name} unexpectedly quantized: {type(mod.quant_method).__name__}")


def test_cosmos_mlp_names_are_in_default_suffix_lists() -> None:
    """Cosmos names its FFN attribute ``mlp`` (Wan uses ``ffn``); both
    default target lists must match it or the FFN GEMMs stay bf16."""
    for suffix in ("mlp.fc_in", "mlp.fc_out"):
        assert suffix in _FP8_SUFFIXES
        assert suffix in DEFAULT_FP4_LAYERS


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVFP4QATQuantizeMethod allocates a CUDA tensor at construction")
def test_nvfp4_qat_tags_attention_and_ffn_linears_only() -> None:
    from fastvideo.layers.quantization.nvfp4_qat_config import NVFP4QATConfig, NVFP4QATQuantizeMethod

    block = _block(NVFP4QATConfig())
    assert _quantized_linears(block, NVFP4QATQuantizeMethod) == sorted(QUANT_TARGET_PATHS)
