"""torch.compile A/B benchmark — model-agnostic.

Step 4 of the graph-break workflow: measure the e2e speedup of
`enable_torch_compile=True` vs baseline, on the same pod/GPU/config
(hold the environment fixed — the only valid way to attribute a delta).

Defaults to Wan2.1-T2V-1.3B (repo CI perf benchmark; comparable to
Kuan-Hao's runs and the perf dashboard). Override MODEL for a
secondary cross-model check.

CRITICAL: the first compiled generation pays a one-time graph-build
cost (can be minutes). We do ONE un-measured warmup, then a SECOND
measured run with IDENTICAL shapes so the compiled graph is reused.
Measuring the warmup is the #1 way people wrongly conclude
"compile is slower."

Env:
  MODEL    HF id / path     (default Wan-AI/Wan2.1-T2V-1.3B-Diffusers)
  COMPILE  0 (default) | 1  -> enable_torch_compile
  AB_STEPS inference steps  (default: model's SamplingParam default)
  AB_FRAMES num_frames      (default: model's SamplingParam default)

Usage on pod (run BOTH on the SAME pod, compare measured lines):
    cd /FastVideo
    MODEL=Wan-AI/Wan2.1-T2V-1.3B-Diffusers COMPILE=0 \
      python profiling/compile_ab.py 2>&1 | tee /tmp/ab_base.log
    MODEL=Wan-AI/Wan2.1-T2V-1.3B-Diffusers COMPILE=1 \
      python profiling/compile_ab.py 2>&1 | tee /tmp/ab_comp.log
    # confirm graph-break count dropped after the fix:
    python profiling/analyze_graphbreaks.py /tmp/ab_comp.log
"""

import os
import time

import torch

from fastvideo import VideoGenerator
from fastvideo.api.sampling_param import SamplingParam

MODEL = os.environ.get("MODEL", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
COMPILE = os.environ.get("COMPILE") == "1"

PROMPT = (
    "A high-definition video of a robotic arm welding a metal structure, "
    "bright sparks and smoke, industrial setting."
)


def _sampling_param() -> SamplingParam:
    sp = SamplingParam.from_pretrained(MODEL)  # per-model defaults
    if os.environ.get("AB_STEPS"):
        sp.num_inference_steps = int(os.environ["AB_STEPS"])
    if os.environ.get("AB_FRAMES"):
        sp.num_frames = int(os.environ["AB_FRAMES"])
    return sp


def _generate(generator: VideoGenerator, tag: str, save: bool) -> float:
    sp = _sampling_param()  # identical shapes each call -> graph reused
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    generator.generate_video(
        PROMPT, sampling_param=sp,
        output_path=f"outputs_video/ab_{tag}.mp4", save_video=save,
    )
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def main() -> None:
    mode = "COMPILE" if COMPILE else "BASELINE"
    print(f"[ab] MODEL={MODEL}  mode={mode}  "
          f"enable_torch_compile={COMPILE}", flush=True)

    generator = VideoGenerator.from_pretrained(
        MODEL,
        num_gpus=1,             # single GPU; SP/multi-GPU not needed for a delta
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
        enable_torch_compile=COMPILE,
    )

    # Warmup — un-measured. When COMPILE, this pays the one-time build.
    print("[ab] warmup generation (un-measured)...", flush=True)
    w = _generate(generator, "warmup", save=False)
    print(f"[ab] warmup wall clock: {w:.2f}s "
          f"({'incl. graph build' if COMPILE else 'cold-start only'})",
          flush=True)

    # Measured — steady state, compiled graph reused.
    print("[ab] measured generation...", flush=True)
    m = _generate(generator, mode.lower(), save=True)
    print(f"\n[ab] === {mode} measured e2e wall clock: {m:.2f}s ===",
          flush=True)
    generator.shutdown()


if __name__ == "__main__":
    main()
