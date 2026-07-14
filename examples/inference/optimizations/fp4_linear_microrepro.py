"""Isolate the FP4-linear (flashinfer ``mm_fp4``) crash on sm_121 (DGX Spark GB10).

The Wan QAD full-4-bit path segfaults on sm_121 in the FP4 *linear* layers. This
reproduces the two flashinfer kernels the linear ``apply()`` uses, in isolation
and with no pipeline around them, so a single Spark session yields a real crash
signature instead of "it segfaulted again":

  1. ``nvfp4_quantize``  (activation + weight quantize)
  2. ``mm_fp4``          (the FP4 GEMM) — for ONE backend, chosen by MM_BACKEND

Why one backend per process: a hard segfault in flashinfer's cutlass FP4 GEMM
kills the whole process, so a single-process backend loop would stop at the first
crash. The runbook loops this script in the shell instead:

    for b in cutlass auto trtllm cudnn; do MM_BACKEND=$b python fp4_linear_microrepro.py; done

and for the exact illegal-address + kernel name:

    MM_BACKEND=cutlass compute-sanitizer --tool memcheck python fp4_linear_microrepro.py

``nvfp4_qat_config.apply`` hardcodes backend="cutlass"; if another backend runs
clean on sm_121, the fix is a one-line arch-gated backend selection there.
"""
from __future__ import annotations

import faulthandler
import os
import sys

import torch

faulthandler.enable()  # segfault -> C traceback on stderr


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")

    cap = torch.cuda.get_device_capability()
    backend = os.environ.get("MM_BACKEND", "cutlass")
    M = int(os.environ.get("MM_M", "256"))   # activation rows
    K = int(os.environ.get("MM_K", "1536"))  # Wan-1.3B hidden
    N = int(os.environ.get("MM_N", "1536"))  # out features

    print(f"[repro] GPU {torch.cuda.get_device_name()} (cc {cap[0]}.{cap[1]})")
    print(f"[repro] torch {torch.__version__}, CUDA {torch.version.cuda}")
    try:
        import flashinfer  # noqa: F401
        print(f"[repro] flashinfer {getattr(flashinfer, '__version__', '?')}")
    except Exception as exc:  # pragma: no cover - diagnostic only
        raise SystemExit(f"[repro] flashinfer import failed: {exc!r}")

    from fastvideo.layers.quantization.nvfp4_config import (
        _mm_fp4,
        _nvfp4_quantize,
        _require_flashinfer,
    )
    SfLayout, _, _ = _require_flashinfer()
    layout = SfLayout.layout_128x4

    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)

    # ---- Step 1: quantize (isolates the quantize kernel) --------------------
    x_gsf = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    w_gsf = (448 * 6) / w.abs().amax().float()
    try:
        x_fp4, x_scale = _nvfp4_quantize(x, x_gsf, sfLayout=layout, do_shuffle=False)
        w_fp4, w_scale = _nvfp4_quantize(w, w_gsf, sfLayout=layout, do_shuffle=False)
        torch.cuda.synchronize()
        print(f"[repro] nvfp4_quantize OK  x_fp4={tuple(x_fp4.shape)}/{x_fp4.dtype} "
              f"x_scale={tuple(x_scale.shape)}")
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"[repro] nvfp4_quantize FAIL: {exc!r}")
        sys.exit(2)

    # ---- Step 2: mm_fp4 for the one selected backend ------------------------
    print(f"[repro] mm_fp4 backend={backend!r}  ({M}x{K} @ {K}x{N}) ...")
    try:
        out = _mm_fp4(
            x_fp4,
            w_fp4.T,
            x_scale,
            w_scale.T,
            1.0 / (x_gsf * w_gsf),
            torch.bfloat16,
            None,
            backend=backend,
        )
        torch.cuda.synchronize()
        print(f"[repro] mm_fp4[{backend}] OK  out={tuple(out.shape)}/{out.dtype} "
              f"|mean|={out.float().abs().mean().item():.4f}")
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"[repro] mm_fp4[{backend}] FAIL: {exc!r}")
        sys.exit(3)


if __name__ == "__main__":
    main()
