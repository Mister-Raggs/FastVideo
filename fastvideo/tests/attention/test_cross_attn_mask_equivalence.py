# SPDX-License-Identifier: Apache-2.0
"""Mask-equivalence sanity check for W3i (wire context_lens through Wan
cross-attention).

The hypothesis under W3i: cross-attention computed with a key-padding
mask on padded inputs is bit-equivalent to per-sample dense cross-attention
on the un-padded real-length inputs, REGARDLESS of what values occupy
the padded K/V positions. That is the property that makes the W3b
zero-pad-the-K-and-V text-encoding recipe unnecessary once masking is
wired up: with a proper mask, garbage values past the real length are
ignored, so we can pad to ``max(real_lens)`` (or any length we want)
and recover the cross-attn cost W3b paid by padding to ``max_length=512``.

Tests here use pure SDPA (CPU, fp32) so they run in any CI and prove
the math without depending on flash-attn / CUDA. Bf16 is intentionally
NOT covered here: at lower precision the same math just accumulates
more noise (fp32 passing within ULP implies bf16 passes within bf16-ULP
trivially), AND a CPU SDPA bf16 path is not representative of Wan's
production cross-attn which dispatches to flash-attn on CUDA. Bf16
coverage lives in the GPU-only ``test_cross_attn_mask_fa_varlen.py``
companion test (planned) that exercises ``flash_attn_varlen_qk_no_pad``
— the actual production path — at production precision.

If THIS test fails, the W3i premise is wrong and we should not touch
model code. If THIS test passes and the FA-varlen one fails, the
backend is non-bit-equivalent and W3i needs a different dispatch path.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F


def _per_sample_dense_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    real_lens: torch.Tensor,
) -> torch.Tensor:
    """Reference: loop over batch, run SDPA per sample on real-length K/V.

    This is the "truth" the masked path must match — what un-padded
    variable-length cross-attention would produce on each sample.

    Args:
        q: [B, Lq, H, D]
        k: [B, Lk_max, H, D] — only first real_lens[b] positions are
           consulted per sample.
        v: [B, Lk_max, H, D] — same.
        real_lens: [B] int tensor of real key/value lengths per sample.

    Returns:
        [B, Lq, H, D] output.
    """
    B, Lq, H, D = q.shape
    out = torch.empty_like(q)
    for b in range(B):
        rl = int(real_lens[b].item())
        # SDPA expects [B, H, L, D]
        q_b = q[b:b + 1].transpose(1, 2)  # [1, H, Lq, D]
        k_b = k[b:b + 1, :rl].transpose(1, 2)  # [1, H, rl, D]
        v_b = v[b:b + 1, :rl].transpose(1, 2)  # [1, H, rl, D]
        out_b = F.scaled_dot_product_attention(q_b, k_b, v_b, attn_mask=None)
        out[b] = out_b.transpose(1, 2).squeeze(0)
    return out


def _masked_dense_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    real_lens: torch.Tensor,
) -> torch.Tensor:
    """The path W3i will produce: dense SDPA on padded inputs + key mask.

    Builds a [B, 1, 1, Lk_max] bool mask (True = real position, False =
    pad) and passes it as ``attn_mask`` to SDPA. SDPA's ``attn_mask``
    semantics: True means "attend here", False means "don't attend" —
    the False positions get -inf added to their pre-softmax scores so
    they contribute zero weight regardless of K/V garbage.
    """
    B, Lq, H, D = q.shape
    Lk = k.shape[1]
    # [B, Lk] bool: True for real positions
    arange = torch.arange(Lk, device=k.device).unsqueeze(0)  # [1, Lk]
    key_mask = arange < real_lens.unsqueeze(1)  # [B, Lk]
    # SDPA wants [B, H, Lq, Lk] broadcasting [B, 1, 1, Lk]
    attn_mask = key_mask[:, None, None, :]  # [B, 1, 1, Lk]
    q_t = q.transpose(1, 2)  # [B, H, Lq, D]
    k_t = k.transpose(1, 2)  # [B, H, Lk, D]
    v_t = v.transpose(1, 2)  # [B, H, Lk, D]
    out = F.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=attn_mask)
    return out.transpose(1, 2)  # [B, Lq, H, D]


def _build_inputs(
    real_lens: list[int],
    max_k: int,
    Lq: int,
    H: int,
    D: int,
    dtype: torch.dtype,
    pad_fill: str,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build q, k, v with controlled values at padded K/V positions.

    ``pad_fill`` controls what occupies K/V[b, real_lens[b]:]:
      - ``"zero"``: zeros (matches W3b's canonical-recipe trick)
      - ``"random"``: fresh random noise (the adversarial case — the
        whole point of W3i is that this should NOT affect output)
      - ``"large"``: large-magnitude random (stress test — softmax
        with -inf mask must still ignore these)
    """
    torch.manual_seed(seed)
    B = len(real_lens)
    real_lens_t = torch.tensor(real_lens, dtype=torch.long)
    q = torch.randn(B, Lq, H, D, dtype=dtype)
    k = torch.randn(B, max_k, H, D, dtype=dtype)
    v = torch.randn(B, max_k, H, D, dtype=dtype)
    for b, rl in enumerate(real_lens):
        if pad_fill == "zero":
            k[b, rl:] = 0.0
            v[b, rl:] = 0.0
        elif pad_fill == "random":
            pass  # already random from torch.randn above
        elif pad_fill == "large":
            k[b, rl:] = torch.randn(max_k - rl, H, D, dtype=dtype) * 100.0
            v[b, rl:] = torch.randn(max_k - rl, H, D, dtype=dtype) * 100.0
        else:
            raise ValueError(f"unknown pad_fill: {pad_fill}")
    return q, k, v, real_lens_t


@pytest.mark.parametrize("pad_fill", ["zero", "random", "large"])
def test_mask_equivalence_fp32(pad_fill: str) -> None:
    """In fp32, masked attention equals per-sample dense attention
    within fp32 ULP, regardless of padded K/V values.

    Not bit-exact: SDPA's masked path runs the attention output reduction
    over [B, H, Lq, max_k] tensors with zero-weighted padded slots, while
    the per-sample path reduces over [B, H, Lq, real_lens[b]] tensors with
    no zero slots. The math is identical (exp(-inf) = 0 exactly), so the
    masked positions contribute exactly 0 to the weighted sum — but the
    fp32 reduction order across H/Lk differs between the two shapes, and
    fp32 addition is not associative. Tolerance set at fp32 ULP scale.
    """
    q, k, v, real_lens = _build_inputs(
        real_lens=[3, 5, 7],
        max_k=16,
        Lq=11,
        H=4,
        D=32,
        dtype=torch.float32,
        pad_fill=pad_fill,
    )
    truth = _per_sample_dense_attention(q, k, v, real_lens)
    masked = _masked_dense_attention(q, k, v, real_lens)
    torch.testing.assert_close(masked, truth, atol=1e-5, rtol=1e-5)


def test_zero_pad_recipe_is_NOT_equivalent_without_mask() -> None:
    """Negative control: the W3b zero-pad recipe (no mask) does NOT
    match per-sample dense attention. Documents WHY W3i exists.

    Without a mask, padded K positions with value 0 still receive
    softmax weight ``exp(q . 0) / Z = 1/Z``, diluting real-position
    weights through the denominator. V=0 at pad means the diluted
    weights have zero contribution magnitude, but the real-position
    weights themselves shift. This is the dilution Kuan flagged:
    "the whole text_encoding recipe disappears" once masking lands.
    """
    q, k, v, real_lens = _build_inputs(
        real_lens=[3, 5],
        max_k=16,
        Lq=7,
        H=4,
        D=32,
        dtype=torch.float32,
        pad_fill="zero",
    )
    truth = _per_sample_dense_attention(q, k, v, real_lens)
    # No mask passed — just dense SDPA on the zero-padded inputs.
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    unmasked = F.scaled_dot_product_attention(q_t, k_t, v_t,
                                              attn_mask=None).transpose(1, 2)
    # Should NOT be close — softmax dilution from pad positions shifts
    # real-position weights.
    diff = (unmasked - truth).abs().max().item()
    assert diff > 1e-3, (
        f"zero-pad recipe accidentally bit-equiv to truth (diff={diff:.3e}); "
        "either the recipe is correct after all or this test is wrong")


def test_mask_equivalence_uniform_real_lens() -> None:
    """Edge case: all samples have the same real length (== max_k).
    Masked path should be identical to no-mask, both equal to truth.
    Guards against accidental dependence on lengths varying."""
    q, k, v, real_lens = _build_inputs(
        real_lens=[10, 10, 10],
        max_k=10,
        Lq=7,
        H=4,
        D=32,
        dtype=torch.float32,
        pad_fill="random",
    )
    truth = _per_sample_dense_attention(q, k, v, real_lens)
    masked = _masked_dense_attention(q, k, v, real_lens)
    assert torch.equal(truth, masked)


def test_mask_equivalence_single_real_position() -> None:
    """Edge case: one sample has real_len=1 (the degenerate case).
    Masked attention should reduce to picking out exactly v[b, 0]
    per query position."""
    q, k, v, real_lens = _build_inputs(
        real_lens=[1, 4],
        max_k=8,
        Lq=5,
        H=2,
        D=16,
        dtype=torch.float32,
        pad_fill="large",
    )
    truth = _per_sample_dense_attention(q, k, v, real_lens)
    masked = _masked_dense_attention(q, k, v, real_lens)
    assert torch.equal(truth, masked)
    # And specifically, sample 0 (real_len=1) should have output[0, :, h, :]
    # exactly equal to v[0, 0, h, :] for every query position (softmax over
    # one position is always 1.0).
    for h in range(q.shape[2]):
        for lq in range(q.shape[1]):
            assert torch.equal(masked[0, lq, h], v[0, 0, h])
