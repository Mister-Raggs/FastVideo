# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the model-agnostic EasyCache step cache (CPU, no model)."""
import pytest
import torch

from fastvideo.pipelines.easycache import EasyCache


def test_warmup_and_tail_are_forced_to_compute():
    ec = EasyCache(thresh=0.05, warmup_steps=2, tail_steps=1)
    ec.start(num_steps=6)  # cutoff_step = 6 - 1 = 5
    x = torch.zeros(4)
    assert ec.should_compute(x, 0)  # warmup
    assert ec.should_compute(x, 1)  # warmup
    assert ec.should_compute(x, 5)  # tail (step >= cutoff_step)


def test_reuse_reconstructs_input_plus_residual():
    ec = EasyCache()
    x = torch.ones(4)
    cond = torch.full((4, ), 3.0)
    uncond = torch.full((4, ), 2.0)
    ec.update(x, cond, uncond)
    x2 = torch.full((4, ), 1.5)
    # residual = output - input, reconstruction = new_input + residual
    assert torch.allclose(ec.reuse(x2, cond=True), x2 + (cond - x))  # 1.5 + (3-1) = 3.5
    assert torch.allclose(ec.reuse(x2, cond=False), x2 + (uncond - x))  # 1.5 + (2-1) = 2.5


def test_k_is_learned_between_consecutive_computes():
    ec = EasyCache(warmup_steps=0, tail_steps=0)
    ec.start(num_steps=10)
    x0 = torch.zeros(4)
    ec.update(x0, torch.full((4, ), 2.0))  # first compute: no k yet
    x1 = torch.full((4, ), 1.0)
    ec.update(x1, torch.full((4, ), 4.0))  # output_change=|4-2|=2, input_change=|1-0|=1
    assert ec.k == pytest.approx(2.0)


def test_skips_when_input_stable_and_computes_when_input_jumps():
    ec = EasyCache(thresh=0.05, warmup_steps=2, tail_steps=0)
    ec.start(num_steps=20)

    # Two warmup computes to anchor k (= |Δout| / |Δin| = 0.1 / 0.1 = 1.0).
    x0 = torch.zeros(8)
    assert ec.should_compute(x0, 0)
    ec.update(x0, x0 + 1.0)
    x1 = torch.full((8, ), 0.1)
    assert ec.should_compute(x1, 1)
    ec.update(x1, x1 + 1.0)
    assert ec.k == pytest.approx(1.0)

    # A nearly-identical input -> tiny predicted change -> skip.
    x2 = torch.full((8, ), 0.1001)
    assert not ec.should_compute(x2, 2)

    # A large input jump -> accumulated error crosses thresh -> compute.
    x3 = torch.full((8, ), 5.0)
    assert ec.should_compute(x3, 3)


def test_counters_and_summary_track_skips():
    ec = EasyCache(thresh=1e9, warmup_steps=1, tail_steps=0)  # huge thresh => always skip post-warmup
    ec.start(num_steps=5)
    x = torch.zeros(4)
    # step 0: warmup compute
    assert ec.should_compute(x, 0)
    ec.update(x, x + 1.0)
    ec.update(x + 0.1, x + 1.1)  # second compute to anchor k so the adaptive branch is reachable
    # remaining steps skip (thresh is enormous)
    for s in range(2, 5):
        assert not ec.should_compute(x + 0.1 * s, s)
    assert ec.skipped == 3
    assert "skipped=3" in ec.summary()


def test_reset_clears_state():
    ec = EasyCache()
    ec.start(num_steps=4)
    ec.update(torch.zeros(4), torch.ones(4))
    assert ec._cache_cond is not None
    ec.start(num_steps=4)  # start() doubles as reset
    assert ec._cache_cond is None
    assert ec.k is None
    assert ec.accumulated_error == 0.0
