# SPDX-License-Identifier: Apache-2.0
"""VSA-256 Triton routes must select exactly the same tokens.

The 256-block Triton path expands the logical [Qb256, KVb256] map down to a
physical block size before calling the kernel:

  route A (default): 4x along Q and 4x along KV  -> 64x64  tiles
  route B:           2x along Q and 2x along KV  -> 128x128 tiles

Both are only re-tilings of one selection, so the *token-level* mask they
imply must be identical -- same (q_token, k_token) pairs valid, same
variable-block truncation. If that holds, any output difference between the
routes is float reassociation, not a change in what was attended to.

The expansion math is pure torch, so this runs on CPU and needs no kernel
build. The GPU parity check is a separate, skipped-by-default companion.
"""

import pytest
import torch

from fastvideo_kernel.block_sparse_attn_256 import (
    _KV_BLOCK_PHYS,
    _KV_BLOCK_TRITON,
    _expand_mask_and_sizes_256_to_64,
    _expand_mask_and_sizes_256_to_128,
)

LOGICAL_BLOCK = 256


def _token_mask_from_expansion(mask, sizes, phys_block, q_repeat):
    """Materialize the [Qtok, KVtok] boolean mask an expansion implies.

    `mask` is [B, H, Qb_phys, KVb_phys] after the KV split; `q_repeat` is how
    many physical q-tiles each logical q-block became.
    """
    b, h = 0, 0
    m = mask[b, h].bool()
    n_qb, n_kvb = m.shape
    # k token t is valid iff its block is selected and t's offset < that
    # block's valid count
    offs = torch.arange(phys_block)
    valid_in_block = offs[None, :] < sizes.reshape(-1)[:n_kvb, None]  # [KVb, phys]
    kv_tok = (m[:, :, None] & valid_in_block[None, :, :]).reshape(n_qb, n_kvb * phys_block)
    return kv_tok.repeat_interleave(phys_block, dim=0), q_repeat


def _make_case(n_qb, n_kvb, seed, partial_tail=True):
    g = torch.Generator().manual_seed(seed)
    mask = torch.rand(1, 2, n_qb, n_kvb, generator=g) < 0.4
    mask[:, :, :, 0] = True  # guarantee at least one selected block per row
    sizes = torch.full((n_kvb, ), LOGICAL_BLOCK, dtype=torch.int32)
    if partial_tail:
        # boundary tiles are partially filled -- the case the size clamping exists for
        sizes[-1] = 137
        if n_kvb > 2:
            sizes[n_kvb // 2] = 64
            sizes[1] = 200
    return mask, sizes


@pytest.mark.parametrize("n_qb,n_kvb", [(1, 1), (2, 3), (4, 4), (3, 7)])
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("partial_tail", [False, True])
def test_route_a_and_b_imply_the_same_token_mask(n_qb, n_kvb, seed, partial_tail):
    mask, sizes = _make_case(n_qb, n_kvb, seed, partial_tail)

    mask_64, sizes_64 = _expand_mask_and_sizes_256_to_64(mask, sizes)
    mask_128, sizes_128 = _expand_mask_and_sizes_256_to_128(mask, sizes)
    # route B splits Q in the caller (the helper only splits KV)
    mask_128 = mask_128.repeat_interleave(2, dim=2)

    tok_a, _ = _token_mask_from_expansion(mask_64, sizes_64, _KV_BLOCK_TRITON, 4)
    tok_b, _ = _token_mask_from_expansion(mask_128, sizes_128, _KV_BLOCK_PHYS, 2)

    assert tok_a.shape == tok_b.shape, (tok_a.shape, tok_b.shape)
    assert torch.equal(tok_a, tok_b), (
        f"routes disagree on {(tok_a ^ tok_b).sum().item()} of {tok_a.numel()} token pairs")


@pytest.mark.parametrize("n_kvb", [1, 3, 8])
def test_expansions_preserve_total_valid_tokens(n_kvb):
    """Both expansions must chop the logical valid count without losing tokens."""
    _, sizes = _make_case(2, n_kvb, seed=0, partial_tail=True)
    _, sizes_64 = _expand_mask_and_sizes_256_to_64(torch.ones(1, 1, 2, n_kvb, dtype=torch.bool), sizes)
    _, sizes_128 = _expand_mask_and_sizes_256_to_128(torch.ones(1, 1, 2, n_kvb, dtype=torch.bool), sizes)

    assert sizes_64.reshape(n_kvb, 4).sum(1).tolist() == sizes.tolist()
    assert sizes_128.reshape(n_kvb, 2).sum(1).tolist() == sizes.tolist()
    assert sizes_64.max() <= _KV_BLOCK_TRITON
    assert sizes_128.max() <= _KV_BLOCK_PHYS


def test_route_b_q_tiles_share_the_selection_route_a_replicates():
    """The premise of route B: within one logical q-block every physical
    q-tile carries an identical top-k list, so the 4x split is pure overhead."""
    mask, sizes = _make_case(3, 5, seed=7)
    mask_64, _ = _expand_mask_and_sizes_256_to_64(mask, sizes)
    for logical_q in range(3):
        rows = mask_64[0, 0, logical_q * 4:(logical_q + 1) * 4]
        assert torch.equal(rows, rows[:1].expand_as(rows)), \
            "q-tiles within a logical block should carry identical selections"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA + built fastvideo-kernel")
@pytest.mark.parametrize("seq_blocks", [4, 8])
def test_routes_agree_numerically_on_gpu(seq_blocks):
    """End-to-end: route A and route B outputs must match to bf16 tolerance."""
    import os

    from fastvideo_kernel.block_sparse_attn_256 import block_sparse_attn_256

    torch.manual_seed(0)
    b, h, d = 1, 4, 128
    seq = seq_blocks * LOGICAL_BLOCK
    q, k, v = (torch.randn(b, h, seq, d, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    mask = torch.rand(b, h, seq_blocks, seq_blocks, device="cuda") < 0.5
    mask[..., 0] = True
    sizes = torch.full((seq_blocks, ), LOGICAL_BLOCK, dtype=torch.int32, device="cuda")
    sizes[-1] = 137

    prev = os.environ.get("FASTVIDEO_VSA_TRITON_ROUTE")
    try:
        os.environ["FASTVIDEO_VSA_TRITON_ROUTE"] = "a"
        out_a = block_sparse_attn_256(q, k, v, mask, sizes)[0]
        os.environ["FASTVIDEO_VSA_TRITON_ROUTE"] = "b"
        out_b = block_sparse_attn_256(q, k, v, mask, sizes)[0]
    finally:
        if prev is None:
            os.environ.pop("FASTVIDEO_VSA_TRITON_ROUTE", None)
        else:
            os.environ["FASTVIDEO_VSA_TRITON_ROUTE"] = prev

    torch.testing.assert_close(out_a.float(), out_b.float(), rtol=2e-2, atol=2e-2)
