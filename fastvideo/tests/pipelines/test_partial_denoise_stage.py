# SPDX-License-Identifier: Apache-2.0
"""Unit tests for PartialDenoiseStage.

Weight-free and CPU-only: the stage only touches the timestep schedule and the
latent tensor, so a stub scheduler exercising the real add_noise contract is
enough.
"""

import pytest
import torch

from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.partial_denoise import PartialDenoiseStage


class _StubScheduler:
    """Mirrors FlowMatchEulerDiscreteScheduler.add_noise's contract.

    Notably it reshapes sigma to (-1, 1, 1, 1), which is why callers must hand
    in a 4D view -- the test fails loudly if the stage forgets to flatten.
    """

    def __init__(self, num_train_timesteps: int = 1000, num_steps: int = 50) -> None:
        self.timesteps = torch.linspace(num_train_timesteps, 0, num_steps)
        self.sigmas = self.timesteps / num_train_timesteps

    def add_noise(self, clean_latent, noise, timestep):
        if clean_latent.ndim != 4:
            raise ValueError(f"add_noise expects a 4D view, got {clean_latent.ndim}D")
        if timestep.ndim == 1 and timestep.shape[0] == 1:
            timestep = timestep.expand(clean_latent.shape[0])
        timestep_id = torch.argmin((self.timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma = self.sigmas[timestep_id].reshape(-1, 1, 1, 1)
        return (1 - sigma) * clean_latent + sigma * noise


def _batch(num_steps: int = 50, shape=(1, 16, 3, 8, 8)):
    scheduler = _StubScheduler(num_steps=num_steps)
    batch = ForwardBatch(data_type="video")
    batch.timesteps = scheduler.timesteps.clone()
    batch.latents = torch.randn(shape)
    return scheduler, batch


def test_reads_from_batch_extra_escape_hatch():
    """Callers drive the stage via the _BATCH_EXTRA_PASSTHROUGH_KEYS route."""
    scheduler, batch = _batch(num_steps=50)
    init = torch.randn_like(batch.latents)
    batch.extra["denoise_strength"] = 0.4
    batch.extra["init_latents"] = init

    out = PartialDenoiseStage(scheduler).forward(batch, None)

    assert len(out.timesteps) == 20
    assert not torch.equal(out.latents, init)


def test_dataclass_fields_win_over_extra():
    scheduler, batch = _batch(num_steps=50)
    batch.init_latents = torch.randn_like(batch.latents)
    batch.denoise_strength = 0.4
    batch.extra["denoise_strength"] = 0.9

    out = PartialDenoiseStage(scheduler).forward(batch, None)

    assert len(out.timesteps) == 20


def test_noop_without_strength_or_init_latents():
    """The default t2v path must be untouched -- same objects, not just equal."""
    scheduler, batch = _batch()
    latents_before, timesteps_before = batch.latents, batch.timesteps

    out = PartialDenoiseStage(scheduler).forward(batch, None)

    assert out.latents is latents_before
    assert out.timesteps is timesteps_before


def test_noop_when_only_one_of_the_pair_is_set():
    scheduler, batch = _batch()
    latents_before = batch.latents

    batch.denoise_strength = 0.5  # init_latents still None
    out = PartialDenoiseStage(scheduler).forward(batch, None)
    assert out.latents is latents_before

    batch.denoise_strength = None
    batch.init_latents = torch.randn_like(latents_before)
    out = PartialDenoiseStage(scheduler).forward(batch, None)
    assert out.latents is latents_before


@pytest.mark.parametrize("strength,expected", [(0.5, 25), (0.3, 15), (0.02, 1), (0.001, 1)])
def test_schedule_is_cropped_to_the_tail(strength, expected):
    scheduler, batch = _batch(num_steps=50)
    full = batch.timesteps.clone()
    batch.denoise_strength = strength
    batch.init_latents = torch.randn_like(batch.latents)

    out = PartialDenoiseStage(scheduler).forward(batch, None)

    assert len(out.timesteps) == expected
    # The tail is kept: schedules run high noise -> low noise.
    torch.testing.assert_close(out.timesteps, full[len(full) - expected:])


def test_strength_one_keeps_full_schedule_and_takes_init_latents():
    scheduler, batch = _batch(num_steps=50)
    batch.denoise_strength = 1.0
    init = torch.randn_like(batch.latents)
    batch.init_latents = init

    out = PartialDenoiseStage(scheduler).forward(batch, None)

    assert len(out.timesteps) == 50
    torch.testing.assert_close(out.latents, init)


def test_latent_is_renoised_toward_noise_as_strength_rises():
    """Higher strength starts earlier => noisier start => further from init."""
    distances = []
    for strength in (0.1, 0.5, 0.9):
        scheduler, batch = _batch(num_steps=50)
        init = torch.randn_like(batch.latents)
        batch.denoise_strength = strength
        batch.init_latents = init
        out = PartialDenoiseStage(scheduler).forward(batch, None)
        assert out.latents.shape == init.shape
        distances.append((out.latents - init).abs().mean().item())

    assert distances[0] < distances[1] < distances[2], distances


def test_rejects_mismatched_init_latents_shape():
    scheduler, batch = _batch()
    batch.denoise_strength = 0.5
    batch.init_latents = torch.randn(1, 16, 3, 4, 4)

    with pytest.raises(ValueError, match="does not match"):
        PartialDenoiseStage(scheduler).forward(batch, None)


@pytest.mark.parametrize("strength", [0.0, -0.1, 1.5])
def test_rejects_out_of_range_strength(strength):
    scheduler, batch = _batch()
    batch.denoise_strength = strength
    batch.init_latents = torch.randn_like(batch.latents)

    with pytest.raises(ValueError, match="denoise_strength"):
        PartialDenoiseStage(scheduler).forward(batch, None)
