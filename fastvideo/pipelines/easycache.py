# SPDX-License-Identifier: Apache-2.0
"""Model-agnostic, training-free runtime-adaptive denoise-step caching (EasyCache).

A clean-room port of EasyCache (Zhou et al., "Less is Enough: Training-Free Video
Diffusion Acceleration via Runtime-Adaptive Caching", arXiv:2507.02860), adapted
to FastVideo's shared denoising stage.

Why this is *one module for every model* (unlike block-level caches such as
cache-dit's DBCache/FBCache, which need a per-model ``BlockAdapter``): EasyCache
operates purely on the DiT's **whole-forward residual** ``output - input``. It
never inspects internal blocks, so it needs nothing model-specific — just the
latent going into the transformer and the prediction coming out, which the
denoise stage already exposes uniformly for every model.

Mechanism (faithful to the reference ``easycache_forward``):
  * Each step, estimate how much the output *would* change from how much the
    input changed since the previous step, scaled by a runtime-learned
    sensitivity ``k``:  ``pred_change = k * (input_change / output_norm)``.
  * Accumulate ``pred_change``; while it stays under ``thresh`` (the SSIM knob),
    **skip the DiT forward** and reuse the cached residual: ``output = input + cache``.
  * On a computed step, re-learn ``k`` from the observed output/input change
    between consecutive computes, refresh the residual cache, and reset the
    accumulator.
  * The first ``warmup_steps`` and the last ``tail_steps`` are always computed
    (anchor ``k``/cache; preserve final refinement).

Cond and uncond (CFG) predictions are cached separately, mirroring the reference
even/odd split; the skip decision is made once per step (on the cond input) and
applied to both, so an entire step's compute is skipped together.
"""

from __future__ import annotations

import torch

from fastvideo.logger import init_logger

logger = init_logger(__name__)


class EasyCache:
    """Runtime-adaptive denoise-step cache. One instance per generation.

    Usage in the denoise loop, wrapping the (possibly CFG) DiT compute::

        cache.start(num_inference_steps)          # once, before the loop
        ...
        if cache.should_compute(latent_model_input, step_idx):
            noise_pred_cond = dit(latent_model_input, cond, ...)
            noise_pred_uncond = dit(latent_model_input, uncond, ...) if cfg else None
            cache.update(latent_model_input, noise_pred_cond, noise_pred_uncond)
            # ... combine into noise_pred as usual ...
        else:
            noise_pred_cond = cache.reuse(latent_model_input, cond=True)
            noise_pred_uncond = cache.reuse(latent_model_input, cond=False) if cfg else None
            # ... combine into noise_pred as usual ...
    """

    def __init__(self, thresh: float = 0.05, warmup_steps: int = 1, tail_steps: int = 1) -> None:
        self.thresh = float(thresh)
        self.warmup_steps = int(warmup_steps)
        self.tail_steps = int(tail_steps)
        self.start(0)

    def start(self, num_steps: int) -> None:
        """Reset all per-generation state for a run of ``num_steps`` steps."""
        self.num_steps = int(num_steps)
        self.cutoff_step = max(self.num_steps - self.tail_steps, 0)
        self.k: float | None = None
        self.accumulated_error: float = 0.0
        self._prev_step_input: torch.Tensor | None = None  # input of the previous step
        self._last_compute_input: torch.Tensor | None = None  # input at the last computed step
        self._last_compute_output: torch.Tensor | None = None  # cond output at the last computed step
        self._cache_cond: torch.Tensor | None = None  # residual = cond_output - input
        self._cache_uncond: torch.Tensor | None = None  # residual = uncond_output - input
        self.computed = 0
        self.skipped = 0

    @staticmethod
    def _mean_abs(t: torch.Tensor) -> float:
        return t.detach().float().abs().mean().item()

    def should_compute(self, x: torch.Tensor, step_idx: int) -> bool:
        """Decide whether this step's DiT forward must be computed (vs reuse cache)."""
        if step_idx < self.warmup_steps or step_idx >= self.cutoff_step:
            calc = True  # forced: warmup + final refinement
            self.accumulated_error = 0.0
        elif (self._prev_step_input is not None and self._last_compute_output is not None and self.k is not None):
            input_change = self._mean_abs(x - self._prev_step_input)
            output_norm = max(self._mean_abs(self._last_compute_output), 1e-8)
            self.accumulated_error += self.k * input_change / output_norm
            calc = self.accumulated_error >= self.thresh
            if calc:
                self.accumulated_error = 0.0
        else:
            calc = True  # no history / no k yet
        self._prev_step_input = x.detach()  # updated every step (per-step delta)
        if not calc:
            self.skipped += 1
        return calc

    def reuse(self, x: torch.Tensor, cond: bool = True) -> torch.Tensor:
        """Reconstruct a skipped step's prediction: ``output = input + residual``."""
        cache = self._cache_cond if cond else self._cache_uncond
        assert cache is not None, "reuse() before any computed step"
        return x + cache

    def update(self, x: torch.Tensor, output_cond: torch.Tensor, output_uncond: torch.Tensor | None = None) -> None:
        """Record a freshly computed step: re-learn ``k``, refresh caches."""
        if self._last_compute_output is not None and self._last_compute_input is not None:
            output_change = self._mean_abs(output_cond - self._last_compute_output)
            input_change = self._mean_abs(x - self._last_compute_input)
            if input_change > 1e-8:
                self.k = output_change / input_change
        self._last_compute_input = x.detach()
        self._last_compute_output = output_cond.detach()
        self._cache_cond = (output_cond - x).detach()
        if output_uncond is not None:
            self._cache_uncond = (output_uncond - x).detach()
        self.computed += 1

    def summary(self) -> str:
        total = self.computed + self.skipped
        frac = (self.skipped / total) if total else 0.0
        return (f"EasyCache: computed={self.computed} skipped={self.skipped} "
                f"({frac:.0%} steps skipped) thresh={self.thresh}")
