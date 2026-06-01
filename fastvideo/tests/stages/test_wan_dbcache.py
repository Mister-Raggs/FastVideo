# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Wan DiT DBCache step-caching loop.

These exercise ``WanTransformer3DModel._forward_blocks_dbcache`` in isolation on
CPU (sp=1, no process group, no real weights). A lightweight fake ``self`` holds
counting stub blocks so we can assert exactly which blocks ran on each step.

Covers: warmup never skips; an unchanged Fn residual triggers a middle-block
skip; cond/uncond passes use separate cache slots; ``reset_dbcache_state``
clears state; and the skip path reproduces the no-skip output when the cached
residual is exact.

The sequence-parallel world size is patched to 1 so the all-reduce branch is not
taken (no distributed init needed). The all-reduce path itself is exercised only
on the pod with a real SP group.
"""
import pytest
import torch
import torch.nn as nn

import fastvideo.models.dits.wanvideo as wanmod
from fastvideo.forward_context import set_forward_context
from fastvideo.models.dits.wanvideo import WanTransformer3DModel


@pytest.fixture(autouse=True)
def _sp_world_size_one(monkeypatch):
    # _forward_blocks_dbcache calls get_sp_world_size() unconditionally; with no
    # process group that would assert. Force the sp=1 (no all-reduce) path.
    monkeypatch.setattr(wanmod, "get_sp_world_size", lambda: 1)


class _CountingBlock(nn.Module):
    """Stub DiT block: adds a fixed per-block delta to hidden_states and records
    how many times it was called."""

    def __init__(self, delta: float) -> None:
        super().__init__()
        self.delta = delta
        self.calls = 0

    def forward(self, hidden_states, encoder_hidden_states, timestep_proj,
                freqs_cis, original_seq_len):
        self.calls += 1
        return hidden_states + self.delta


class _FakeWan:
    """Minimal stand-in carrying just what ``_forward_blocks_dbcache`` touches."""

    _forward_blocks_dbcache = WanTransformer3DModel._forward_blocks_dbcache
    reset_dbcache_state = WanTransformer3DModel.reset_dbcache_state

    def __init__(self, n_blocks=12, fn=4, bn=2, threshold=0.05, warmup=2):
        self.blocks = nn.ModuleList(
            [_CountingBlock(delta=0.1 * (i + 1)) for i in range(n_blocks)])
        self.dbcache_fn_compute_blocks = fn
        self.dbcache_bn_compute_blocks = bn
        self.dbcache_residual_threshold = threshold
        self.dbcache_max_warmup_steps = warmup
        self.reset_dbcache_state()

    def _calls(self):
        return [b.calls for b in self.blocks]


def _run(model, x, encoder, step_idx):
    with set_forward_context(current_timestep=step_idx, attn_metadata=None):
        return model._forward_blocks_dbcache(x, encoder, None, None, 8)


def test_warmup_runs_all_blocks():
    m = _FakeWan(n_blocks=12, fn=4, bn=2, warmup=3)
    enc = torch.zeros(1, 4, 8)
    # Same input every step -> residual diff is 0, but warmup must still run all.
    for step in range(3):
        _run(m, torch.ones(1, 16, 8), enc, step)
    assert m._calls() == [3] * 12


def test_skip_after_warmup_on_unchanged_residual():
    m = _FakeWan(n_blocks=12, fn=4, bn=2, threshold=0.05, warmup=2)
    enc = torch.zeros(1, 4, 8)
    x = torch.ones(1, 16, 8)
    # Identical input every step => Fn residual identical => diff 0 < threshold.
    for step in range(5):
        _run(m, x, enc, step)
    calls = m._calls()
    # Fn blocks [0:4] and Bn blocks [10:12] always run: 5 calls each.
    assert calls[0:4] == [5, 5, 5, 5]
    assert calls[10:12] == [5, 5]
    # Middle blocks [4:10]: the cache is populated during the 2 warmup steps, so
    # steps 2/3/4 all skip -> 2 calls each.
    assert calls[4:10] == [2, 2, 2, 2, 2, 2]


def test_skip_output_matches_full_when_residual_exact():
    # With deterministic stub blocks the cached middle residual is exact, so a
    # skipped step must produce the same output as running every block.
    m = _FakeWan(n_blocks=12, fn=4, bn=2, threshold=0.05, warmup=1)
    enc = torch.zeros(1, 4, 8)
    x = torch.ones(1, 16, 8)
    out_warm = _run(m, x, enc, 0)  # warmup, full compute (populates cache)
    out_skip1 = _run(m, x, enc, 1)  # middle skipped
    out_skip2 = _run(m, x, enc, 2)  # middle skipped
    assert torch.allclose(out_warm, out_skip1)
    assert torch.allclose(out_warm, out_skip2)


def test_cond_uncond_use_separate_slots():
    # Two forwards per step (same step_idx) must not share a cache slot.
    m = _FakeWan(n_blocks=12, fn=4, bn=2, threshold=0.05, warmup=1)
    enc = torch.zeros(1, 4, 8)
    cond = torch.ones(1, 16, 8)
    uncond = torch.full((1, 16, 8), 2.0)
    for step in range(4):
        _run(m, cond, enc, step)    # slot 0
        _run(m, uncond, enc, step)  # slot 1
    # Per slot: warmup step 0 computes + caches the middle, steps 1-3 skip.
    # Middle runs once per slot -> 2 calls total across the two slots.
    assert m._calls()[4:10] == [2, 2, 2, 2, 2, 2]


def test_reset_clears_state():
    m = _FakeWan(n_blocks=12, fn=4, bn=2, warmup=0)
    enc = torch.zeros(1, 4, 8)
    x = torch.ones(1, 16, 8)
    _run(m, x, enc, 0)  # no cache yet -> full compute, populate
    _run(m, x, enc, 1)  # skip
    m.reset_dbcache_state()
    assert m._dbcache_state == {}
    assert m._dbcache_last_step_idx is None
    pre = m._calls()
    _run(m, x, enc, 0)  # fresh: no cache -> full compute again
    post = m._calls()
    assert [p - q for p, q in zip(post, pre)] == [1] * 12
