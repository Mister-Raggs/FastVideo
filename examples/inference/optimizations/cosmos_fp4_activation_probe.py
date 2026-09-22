"""Probe FP4 attention with exact Cosmos 2.5 activations on DGX Spark.

The generation itself stays on BF16 Torch SDPA. Worker-local hooks inspect the
normalized, RoPE-applied Q/K/V tensors at early, middle, and late transformer
blocks and side-compute FP4 versus BF16 outputs. No full activation is saved.

Default workload: distilled Cosmos 2.5 2B, 704x1280x77, four denoising steps,
guidance 1.0, with blocks 0/14/27 captured at calls 0/2/3. Output remains
latent so VAE decode cannot dominate this diagnostic.
"""
from __future__ import annotations

import argparse
import faulthandler
import os
import sys
from datetime import datetime
from pathlib import Path

import torch


faulthandler.enable()

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL = "~/models/Cosmos-Predict2.5-2B-Distilled-Diffusers"
DEFAULT_PROMPT = (
    "A high-definition video captures the precision of robotic welding in an industrial setting. "
    "The first frame showcases a robotic arm, equipped with a welding torch, positioned over a large metal structure. "
    "The welding process is in full swing, with bright sparks and intense light illuminating the scene, "
    "creating a vivid display of blue and white hues. "
    "A significant amount of smoke billows around the welding area, partially obscuring the view but emphasizing the heat and activity. "
    "The background reveals parts of the workshop environment, including a ventilation system and various pieces of machinery, "
    "indicating a busy and functional industrial workspace. "
    "As the video progresses, the robotic arm maintains its steady position, continuing the welding process and moving to its left. "
    "The welding torch consistently emits sparks and light, and the smoke continues to rise, diffusing slightly as it moves upward. "
    "The metal surface beneath the torch shows ongoing signs of heating and melting. "
    "The scene retains its industrial ambiance, with the welding sparks and smoke dominating the visual field, "
    "underscoring the ongoing nature of the welding operation."
)


def _parse_indices(value: str, *, steps: int) -> list[int] | None:
    if value == "auto":
        return None
    if value == "first,middle,last":
        return sorted({0, steps // 2, steps - 1})
    try:
        indices = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid comma-separated indices: {value!r}") from exc
    if not indices or any(index < 0 for index in indices):
        raise argparse.ArgumentTypeError("indices must be a non-empty list of non-negative integers")
    return indices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--frames", type=int, default=77)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--blocks", default="auto", help="auto or comma-separated zero-based block indices")
    parser.add_argument("--capture-calls", default="first,middle,last", help="named default or comma-separated calls")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("--warmup must be non-negative and --iterations must be positive")
    try:
        block_indices = _parse_indices(args.blocks, steps=args.steps)
        capture_calls = _parse_indices(args.capture_calls, steps=args.steps)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    assert capture_calls is not None
    if max(capture_calls) >= args.steps:
        parser.error("capture call indices must be smaller than --steps for the default guidance=1 probe")

    model = str(Path(args.model).expanduser()) if args.model.startswith("~") else args.model
    output = args.output
    if output is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output = Path(f"cosmos-fp4-exact-activation-{stamp}.json")
    output = output.expanduser().resolve()

    print(f"[fp4-probe] model={model}")
    print(f"[fp4-probe] shape={args.height}x{args.width}x{args.frames} steps={args.steps} "
          f"guidance={args.guidance} seed={args.seed}")
    print(f"[fp4-probe] blocks={block_indices or 'auto(first/middle/last)'} calls={capture_calls}")
    print(f"[fp4-probe] receipt={output}")
    if args.dry_run:
        return 0

    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    capability = tuple(torch.cuda.get_device_capability())
    if capability not in {(12, 0), (12, 1)}:
        parser.error(f"the CUTLASS FP4 comparison requires sm_120 or sm_121, got {capability}")

    # Resolve the model's normal backend before workers are spawned. The probe
    # invokes FP4 directly on copies and never changes generation outputs.
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "TORCH_SDPA"

    from fastvideo import VideoGenerator
    from fastvideo.attention.fp4_activation_probe import (
        finish_fp4_activation_probe,
        install_fp4_activation_probe,
    )

    generator = VideoGenerator.from_pretrained(
        model,
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
        enable_torch_compile=False,
        output_type="latent",
    )
    installed = False
    receipt = None
    try:
        install_results = generator.executor.collective_rpc(
            install_fp4_activation_probe,
            kwargs={
                "output_path": str(output),
                "capture_calls": capture_calls,
                "block_indices": block_indices,
                "warmup": args.warmup,
                "iterations": args.iterations,
                "run_metadata": {
                    "model": model,
                    "prompt": args.prompt,
                    "steps": args.steps,
                    "guidance": args.guidance,
                    "height": args.height,
                    "width": args.width,
                    "frames": args.frames,
                    "fps": args.fps,
                    "seed": args.seed,
                    "output_type": "latent",
                },
            },
        )
        installed = True
        print(f"[fp4-probe] worker install: {install_results[0]}")
        generator.generate(request={
            "prompt": args.prompt,
            "sampling": {
                "seed": args.seed,
                "num_inference_steps": args.steps,
                "guidance_scale": args.guidance,
                "height": args.height,
                "width": args.width,
                "num_frames": args.frames,
                "fps": args.fps,
            },
            "output": {
                "save_video": False,
                "return_frames": False,
            },
        })
    finally:
        if installed:
            receipt = generator.executor.collective_rpc(finish_fp4_activation_probe)[0]
        generator.shutdown()

    assert receipt is not None
    expected = len(receipt["selected_blocks"]) * 2 * len(receipt["capture_calls"])
    records = receipt["records"]
    errors = [record for record in records if record["status"] != "ok"]
    print(f"[fp4-probe] captured={len(records)}/{expected} errors={len(errors)}")
    for record in records:
        if record["status"] != "ok":
            print(f"[fp4-probe] {record['capture_id']} ERROR {record['error_type']}: {record['error']}")
            continue
        error = record["output_error"]
        timing = record["timing"]
        logits = record["activations"]["sampled_scaled_qk_logits"]
        print(f"[fp4-probe] {record['capture_id']} t={record['timestep']} "
              f"Q/K={record['shape']['q_len']}/{record['shape']['kv_len']} "
              f"logit_p99={logits['abs_p99']:.3f} logit_max={max(abs(logits['min']), abs(logits['max'])):.3f} "
              f"cos={error['cosine_similarity']:.6f} rel_l2={error['relative_l2']:.4f} "
              f"speedup={timing['speedup_x']:.3f}x")
    print(f"[fp4-probe] receipt: {output}")
    return 1 if errors or len(records) != expected else 0


if __name__ == "__main__":
    raise SystemExit(main())
