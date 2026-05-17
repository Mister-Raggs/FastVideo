"""Enumerate torch.compile graph breaks in a FastVideo DiT.

Model-agnostic. Defaults to Wan2.1-T2V-1.3B because that is the repo's
canonical CI perf benchmark (.buildkite/performance-benchmarks/tests/
wan-t2v-1.3b.json, PR #1248) and was used for the #1362 cross-model
A/B — measuring on the same yardstick the project tracks makes the
number comparable and reviewable. Override MODEL for a secondary
check (e.g. Cosmos 2.5).

WHY a runner + log scrape and not `torch._dynamo.explain()`: the DiT
is compiled inside the MultiprocExecutor *worker* process (same reason
parent-side torch.profiler saw nothing in the postdecode work).
`explain()` is in-process and can't reach the worker. The portable
mechanism is the `TORCH_LOGS` env var: process-global, inherited by
the spawned worker, one record per break/recompile from whichever
process compiles. Set BEFORE importing torch so the worker inherits it.

  graph_breaks  -> one record per break (reason + user frame)
  recompiles    -> guard failures forcing a re-trace (dynamic-shape
                   thrash is its own perf problem)
  TORCHDYNAMO_VERBOSE=1 -> full reason + source location

Env:
  MODEL      HF id / path     (default Wan-AI/Wan2.1-T2V-1.3B-Diffusers)
  GB_STEPS   inference steps  (default 4 — warmup-only; compiles every block)
  GB_FRAMES  num_frames       (default 29)

Usage on pod (capture EVERYTHING; records come from the worker):
    cd /FastVideo
    python profiling/compile_graphbreaks.py 2>&1 | tee /tmp/gb.log
    python profiling/analyze_graphbreaks.py /tmp/gb.log
    # secondary check on a different model:
    MODEL=KyleShao/Cosmos-Predict2.5-2B-Diffusers \
      python profiling/compile_graphbreaks.py 2>&1 | tee /tmp/gb_cosmos.log
"""

import os

# MUST be set before torch is imported anywhere (incl. by the worker).
os.environ.setdefault("TORCH_LOGS", "graph_breaks,recompiles")
os.environ.setdefault("TORCHDYNAMO_VERBOSE", "1")

import time  # noqa: E402

from fastvideo import VideoGenerator  # noqa: E402
from fastvideo.api.sampling_param import SamplingParam  # noqa: E402

MODEL = os.environ.get("MODEL", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
GB_STEPS = int(os.environ.get("GB_STEPS", "4"))
GB_FRAMES = int(os.environ.get("GB_FRAMES", "29"))

PROMPT = (
    "A high-definition video of a robotic arm welding a metal structure, "
    "bright sparks and smoke, industrial setting."
)


def main() -> None:
    print("=== GRAPHBREAK-RUN-START ===", flush=True)
    print(f"MODEL={MODEL}", flush=True)
    print(f"TORCH_LOGS={os.environ['TORCH_LOGS']} "
          f"TORCHDYNAMO_VERBOSE={os.environ['TORCHDYNAMO_VERBOSE']} "
          f"steps={GB_STEPS} frames={GB_FRAMES}", flush=True)

    generator = VideoGenerator.from_pretrained(
        MODEL,
        num_gpus=1,            # single GPU is fine for break discovery
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
        enable_torch_compile=True,  # the whole point
    )

    sp = SamplingParam.from_pretrained(MODEL)  # per-model defaults
    sp.num_inference_steps = GB_STEPS
    sp.num_frames = GB_FRAMES

    # One generation: the first denoise step compiles every block and
    # emits all first-trace graph breaks; later steps surface recompiles
    # (guard failures) if shapes/conditions drift across steps.
    print("=== COMPILE-AND-RUN (graph breaks emitted by worker below) ===",
          flush=True)
    t0 = time.perf_counter()
    generator.generate_video(
        PROMPT, sampling_param=sp,
        output_path="outputs_video/gb_probe.mp4", save_video=False,
    )
    print(f"=== GRAPHBREAK-RUN-END  ({time.perf_counter() - t0:.1f}s) ===",
          flush=True)
    generator.shutdown()


if __name__ == "__main__":
    main()
