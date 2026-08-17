# SPDX-License-Identifier: Apache-2.0
"""
Partial-denoise entry: start the denoising loop from an intermediate timestep on
top of a supplied latent, instead of from pure noise at t=T.

This is the ``strength`` / img2img primitive, generalised. It lets a caller hand
in an already-estimated clean latent ``x0_hat`` and spend only the tail of the
schedule refining it, which is the shared mechanism behind image-to-image,
video-to-video, restart sampling, iterative refinement, and using a distilled
few-step model as an initialiser for the full model.

LongCat already does this for its own refinement path
(``LongCatRefineTimestepStage``), but that implementation is bound to LongCat's
``refine_from`` / ``stage1_video`` inputs and its own sigma construction. This
stage is the model-agnostic version: it consumes ``batch.init_latents`` +
``batch.denoise_strength`` and works on any pipeline whose denoising loop reads
``batch.timesteps``.

Placed AFTER ``LatentPreparationStage`` so the seeded pure-noise tensor is
already available to re-noise with -- reusing it keeps the generator/seed path
untouched, so a given seed produces the same noise whether or not this stage
runs.

No-op unless BOTH ``denoise_strength`` (< 1.0) and ``init_latents`` are set, so
the default text-to-video path is bit-identical.
"""

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult

logger = init_logger(__name__)


class PartialDenoiseStage(PipelineStage):
    """Crop the timestep schedule and re-noise a supplied latent to its head.

    ``denoise_strength`` follows the diffusers convention: the fraction of the
    schedule to actually run. 1.0 runs everything (equivalent to starting from
    noise), 0.3 runs the last 30% of the steps.
    """

    def __init__(self, scheduler) -> None:
        super().__init__()
        self.scheduler = scheduler

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        # `batch.extra` is the framework's designed escape hatch (populated via
        # generate_video(_extra_overrides=...)), so callers can drive this stage
        # today without a SamplingParam change. A real img2img surface would
        # promote `denoise_strength` to SamplingParam; the fields win when set.
        strength = batch.denoise_strength
        if strength is None:
            strength = batch.extra.get("denoise_strength")
        init_latents = batch.init_latents
        if init_latents is None:
            init_latents = batch.extra.get("init_latents")

        if strength is None or init_latents is None:
            return batch
        if not 0.0 < strength <= 1.0:
            raise ValueError(f"denoise_strength must be in (0.0, 1.0], got {strength}")
        if strength == 1.0:
            # Nothing to crop. Still honour the supplied latent so callers can
            # use init_latents alone as a plain latent override.
            batch.latents = init_latents.to(batch.latents.device, batch.latents.dtype)
            return batch

        timesteps = batch.timesteps
        assert timesteps is not None, "timesteps must be prepared before partial denoise"
        num_steps = len(timesteps)
        # Keep the TAIL of the schedule: timesteps run high noise -> low noise,
        # and a partial denoise starts partway down.
        num_kept = max(1, int(round(num_steps * strength)))
        cropped = timesteps[num_steps - num_kept:]
        start_timestep = cropped[:1]

        noise = batch.latents
        assert noise is not None, "latents must be prepared before partial denoise"
        init_latents = init_latents.to(device=noise.device, dtype=noise.dtype)
        if init_latents.shape != noise.shape:
            raise ValueError(f"init_latents shape {tuple(init_latents.shape)} does not match the prepared "
                             f"latent shape {tuple(noise.shape)}")

        # add_noise reshapes sigma to (-1, 1, 1, 1), so it needs a 4D view.
        # Flattening the leading two dims is the convention the DMD loop already
        # uses (denoising.py); sigma is uniform across the batch here, so which
        # two dims get merged does not change the result.
        leading = init_latents.shape[:2]
        noised = self.scheduler.add_noise(
            init_latents.flatten(0, 1),
            noise.flatten(0, 1),
            start_timestep.to(noise.device),
        ).unflatten(0, leading)

        batch.latents = noised
        batch.timesteps = cropped

        logger.info(
            "Partial denoise: strength=%.3f, running %d/%d steps from timestep %s",
            strength,
            num_kept,
            num_steps,
            int(start_timestep.item()),
        )
        return batch

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify partial denoise stage inputs."""
        result = VerificationResult()
        result.add_check("latents", batch.latents, [V.is_tensor, V.min_dims(2)])
        result.add_check("timesteps", batch.timesteps, [V.is_tensor, V.min_dims(1)])
        result.add_check("init_latents", batch.init_latents, V.none_or_tensor)
        return result

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify partial denoise stage outputs."""
        result = VerificationResult()
        result.add_check("latents", batch.latents, [V.is_tensor, V.min_dims(2)])
        result.add_check("timesteps", batch.timesteps, [V.is_tensor, V.min_dims(1)])
        return result
