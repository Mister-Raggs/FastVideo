"""Correctness gate for the experimental native 128-token Triton forward."""

from __future__ import annotations

import pytest
import torch

from fastvideo_kernel.block_sparse_attn_256 import (
    _expand_mask_and_sizes_128_to_64,
    block_sparse_attn_128,
)
from fastvideo_kernel.triton_kernels.block_sparse_attn_triton import triton_block_sparse_attn_forward
from fastvideo_kernel.triton_kernels.index import map_to_index


@pytest.mark.cuda
def test_native_vsa128_triton_matches_64_route_a(monkeypatch) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    torch.manual_seed(61)
    batch, heads, dim = 1, 2, 128
    q_blocks, kv_blocks, topk = 3, 4, 2
    q = torch.randn(batch, heads, q_blocks * 128, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, heads, kv_blocks * 128, dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    kv_sizes_128 = torch.tensor([128, 91, 37, 128], device="cuda", dtype=torch.int32)

    scores = torch.randn(batch, heads, q_blocks, kv_blocks, device="cuda")
    selected = scores.topk(topk, dim=-1).indices
    mask_128 = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, selected, True)

    monkeypatch.setenv("FASTVIDEO_VSA_TRITON", "1")
    monkeypatch.setenv("FASTVIDEO_VSA_TRITON_NATIVE_128", "1")
    with torch.inference_mode():
        actual, _ = block_sparse_attn_128(q, k, v, mask_128, kv_sizes_128)

    mask_64, kv_sizes_64 = _expand_mask_and_sizes_128_to_64(mask_128, kv_sizes_128)
    idx_64, num_64 = map_to_index(mask_64)
    expected, _ = triton_block_sparse_attn_forward(q, k, v, idx_64, num_64, kv_sizes_64)

    difference = (actual.float() - expected.float()).abs()
    print(
        "[vsa128-native-triton] "
        f"mean_abs={difference.mean().item():.6e}, max_abs={difference.max().item():.6e}"
    )
    assert torch.isfinite(actual).all().item()
    assert difference.mean().item() < 2e-3
    assert difference.max().item() < 0.1
