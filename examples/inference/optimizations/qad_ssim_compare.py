"""SSIM / PSNR between two QAD arms' videos — corroborating metric for the
FP4-attention quality A/B on sm_121.

Both arms use the SAME checkpoint and SAME seed; only the attention (and/or
linear) numerics differ. So SSIM(A, B) should be HIGH — a high value confirms
the eyeball read. A low value is NOT disqualifying: a few-step distilled sampler
can land on a different-but-equally-valid trajectory under a tiny perturbation
(the eye stays the primary quality gate; see #1594). This is the safe-upside
number, not the verdict.

Self-contained (torch + imageio + numpy only — no scikit-image), runs on CPU.

Usage:
    python qad_ssim_compare.py <arm_a> <arm_b> [--base qad_fp4_samples]

An arm is either a directory under --base (its newest *.mp4 is used) or a direct
mp4 path. Examples:
    # FP4 attention vs bf16 reference (the key A/B):
    python qad_ssim_compare.py lin-bf16_attn-default lin-bf16_attn-attn_qat_infer
    # full 4-bit vs bf16 reference:
    python qad_ssim_compare.py lin-bf16_attn-default lin-fp4_attn-attn_qat_infer
"""
from __future__ import annotations

import argparse
import glob
import math
import os

import imageio
import numpy as np
import torch
import torch.nn.functional as F


def _gaussian_window(size: int = 11, sigma: float = 1.5, channels: int = 3) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window2d = g[:, None] @ g[None, :]
    return window2d.expand(channels, 1, size, size).contiguous()


def _ssim(a: torch.Tensor, b: torch.Tensor, window: torch.Tensor) -> float:
    """SSIM for two [1, C, H, W] tensors in [0, 1] (mean over channels)."""
    channels = a.shape[1]
    pad = window.shape[-1] // 2
    mu_a = F.conv2d(a, window, padding=pad, groups=channels)
    mu_b = F.conv2d(b, window, padding=pad, groups=channels)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sig_a = F.conv2d(a * a, window, padding=pad, groups=channels) - mu_a2
    sig_b = F.conv2d(b * b, window, padding=pad, groups=channels) - mu_b2
    sig_ab = F.conv2d(a * b, window, padding=pad, groups=channels) - mu_ab
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_ab + c1) * (2 * sig_ab + c2)) / ((mu_a2 + mu_b2 + c1) * (sig_a + sig_b + c2))
    return float(s.mean())


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = float(F.mse_loss(a, b))
    return 99.0 if mse == 0 else 10 * math.log10(1.0 / mse)


def _read_frames(path: str) -> list[np.ndarray]:
    reader = imageio.get_reader(path)
    try:
        return [np.asarray(im) for im in reader]
    finally:
        reader.close()


def _resolve(arg: str, base: str) -> str:
    if os.path.isfile(arg):
        return arg
    # arg may be a full dir path (possibly already including base) OR a bare
    # arm name under base — try both.
    tried = []
    for cand in (arg, os.path.join(base, arg)):
        tried.append(cand)
        if os.path.isdir(cand):
            mp4s = sorted(glob.glob(os.path.join(cand, "*.mp4")), key=os.path.getmtime)
            if mp4s:
                return mp4s[-1]
    raise SystemExit(f"no mp4 found for {arg!r} (looked at {tried})")


def _to_tensor(frame: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(frame[..., :3].astype(np.float32) / 255.0)
    return t.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]


def main() -> None:
    parser = argparse.ArgumentParser(description="SSIM/PSNR between two QAD arms")
    parser.add_argument("arm_a")
    parser.add_argument("arm_b")
    parser.add_argument("--base", default="qad_fp4_samples")
    args = parser.parse_args()

    path_a = _resolve(args.arm_a, args.base)
    path_b = _resolve(args.arm_b, args.base)
    print(f"A: {path_a}")
    print(f"B: {path_b}")

    frames_a = _read_frames(path_a)
    frames_b = _read_frames(path_b)
    n = min(len(frames_a), len(frames_b))
    if len(frames_a) != len(frames_b):
        print(f"note: frame counts differ ({len(frames_a)} vs {len(frames_b)}); "
              f"comparing first {n}")
    if n == 0:
        raise SystemExit("no frames decoded")
    if frames_a[0].shape[:2] != frames_b[0].shape[:2]:
        raise SystemExit(f"frame size mismatch {frames_a[0].shape} vs {frames_b[0].shape}")

    window = _gaussian_window()
    ssims, psnrs = [], []
    with torch.no_grad():
        for i in range(n):
            ta, tb = _to_tensor(frames_a[i]), _to_tensor(frames_b[i])
            ssims.append(_ssim(ta, tb, window))
            psnrs.append(_psnr(ta, tb))

    ssims_t, psnrs_t = torch.tensor(ssims), torch.tensor(psnrs)
    print(f"\nframes compared: {n}")
    print(f"SSIM  mean {ssims_t.mean():.4f}  min {ssims_t.min():.4f}  "
          f"max {ssims_t.max():.4f}")
    print(f"PSNR  mean {psnrs_t.mean():.2f} dB  min {psnrs_t.min():.2f} dB")


if __name__ == "__main__":
    main()
