"""Benchmark Cosmos Predict2.5 DFD on DGX Spark across four runtime arms.

Each arm runs in its own process so attention-backend selection and
``torch.compile`` caches cannot leak between measurements:

* ``sdpa``: BF16 Torch SDPA baseline.
* ``compile``: BF16 Torch SDPA plus DiT block compilation.
* ``fp4``: FP4 self-attention with BF16 SDPA cross-attention.
* ``fp4_compile``: FP4 self-attention plus DiT block compilation.

The first generation is a discarded warmup by default. This is required for
compiled arms and keeps the comparison symmetric. Measured runs include image
conditioning, four DFD denoising steps, VAE decode, and frame transfer. One
review MP4 is encoded from the final frames after the timed window. The script
writes one JSON result and one boundary image per arm, plus ``matrix.json``
when ``--arm all`` is selected.

Example:

    python examples/inference/optimizations/cosmos2_5_dfd_matrix.py \
        --model /home/raghav/models/Cosmos-Predict2.5-2B-DFD-FastVideo \
        --image /home/raghav/dfd-eval/eval_inputs/hybrid_t2w_last.png \
        --arm all --warmups 1 --runs 2
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_PROMPT = (
    "The same red Dodge Challenger approaches a sweeping bend and begins turning smoothly to the right, "
    "maintaining its appearance, speed, and the same low rear three-quarter tracking camera."
)


@dataclass(frozen=True)
class Arm:
    name: str
    attention_backend: str
    compile: bool


ARMS = {
    "sdpa": Arm("sdpa", "TORCH_SDPA", False),
    "compile": Arm("compile", "TORCH_SDPA", True),
    "fp4": Arm("fp4", "ATTN_QAT_INFER", False),
    "fp4_compile": Arm("fp4_compile", "ATTN_QAT_INFER", True),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Converted FastVideo DFD model directory")
    parser.add_argument("--image", required=True, help="One-frame continuation image")
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt", default=DEFAULT_PROMPT)
    prompt_group.add_argument("--prompt-file", help="UTF-8 text file containing one prompt")
    parser.add_argument("--arm", choices=("all", *ARMS), default="all")
    parser.add_argument("--output-dir", default="outputs/cosmos25_dfd_matrix")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--dry-run", action="store_true", help="Print resolved arms without loading FastVideo")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.warmups < 0:
        raise SystemExit("--warmups must be non-negative")
    if args.runs < 1:
        raise SystemExit("--runs must be positive")
    for name in ("height", "width", "frames", "fps"):
        if getattr(args, name) < 1:
            raise SystemExit(f"--{name} must be positive")
    if not args.dry_run:
        if not Path(args.model).is_dir():
            raise SystemExit(f"--model directory not found: {args.model}")
        if not Path(args.image).is_file():
            raise SystemExit(f"--image file not found: {args.image}")
        if args.prompt_file and not Path(args.prompt_file).is_file():
            raise SystemExit(f"--prompt-file not found: {args.prompt_file}")


def _prompt(args: argparse.Namespace) -> str:
    prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip() if args.prompt_file else args.prompt.strip()
    if not prompt:
        raise SystemExit("The benchmark prompt must not be empty")
    return prompt


def _arm_command(args: argparse.Namespace, arm_name: str) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model",
        args.model,
        "--image",
        args.image,
        "--arm",
        arm_name,
        "--output-dir",
        args.output_dir,
        "--warmups",
        str(args.warmups),
        "--runs",
        str(args.runs),
        "--seed",
        str(args.seed),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--frames",
        str(args.frames),
        "--fps",
        str(args.fps),
    ]
    if args.prompt_file:
        command.extend(("--prompt-file", args.prompt_file))
    else:
        command.extend(("--prompt", args.prompt))
    if args.dry_run:
        command.append("--dry-run")
    return command


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _frame_array(frame: Any):
    import numpy as np

    pixels = np.asarray(frame)
    if pixels.ndim != 3 or pixels.shape[-1] != 3:
        raise RuntimeError(f"Expected an HWC RGB frame, got {pixels.shape}")
    return pixels.astype(np.float32)


def _quality_receipt(image_path: str, frames: list[Any], output_path: Path) -> dict[str, float]:
    import numpy as np
    from PIL import Image

    first = _frame_array(frames[0])
    second = _frame_array(frames[1])
    last = _frame_array(frames[-1])
    with Image.open(image_path) as source_image:
        source = np.asarray(source_image.convert("RGB").resize((first.shape[1], first.shape[0]))).astype(np.float32)

    boundary = Image.new("RGB", (first.shape[1] * 2, first.shape[0]))
    boundary.paste(Image.fromarray(source.astype(np.uint8)), (0, 0))
    boundary.paste(Image.fromarray(first.astype(np.uint8)), (first.shape[1], 0))
    boundary.save(output_path)

    return {
        "input_to_frame0_mae": float(np.mean(np.abs(source - first))),
        "frame0_to_frame1_mae": float(np.mean(np.abs(first - second))),
        "input_to_last_control_mae": float(np.mean(np.abs(source - last))),
    }


def _save_review_video(frames: list[Any], output_path: Path, fps: int) -> float:
    import imageio

    started = time.perf_counter()
    imageio.mimsave(output_path, frames, fps=fps, format="mp4")
    return time.perf_counter() - started


def _run_arm(args: argparse.Namespace, arm: Arm) -> dict[str, Any]:
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = arm.attention_backend
    os.environ.setdefault("FASTVIDEO_STAGE_LOGGING", "1")

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the Cosmos DFD performance matrix")

    if arm.attention_backend == "ATTN_QAT_INFER":
        from fastvideo.attention.backends.attn_qat_infer import (
            attn_qat_infer_receipt,
            is_attn_qat_infer_available,
        )

        receipt = attn_qat_infer_receipt()
        print(f"[{arm.name}] {receipt}", flush=True)
        if not is_attn_qat_infer_available():
            raise SystemExit(f"ATTN_QAT_INFER requested but unavailable: {receipt}")

    from fastvideo import VideoGenerator
    from fastvideo.api.sampling_param import SamplingParam

    arm_dir = Path(args.output_dir).resolve() / arm.name
    arm_dir.mkdir(parents=True, exist_ok=True)
    prompt = _prompt(args)
    print(
        f"[{arm.name}] backend={arm.attention_backend} compile={arm.compile} "
        f"shape={args.width}x{args.height}x{args.frames} warmups={args.warmups} runs={args.runs}",
        flush=True,
    )

    generator = VideoGenerator.from_pretrained(
        args.model,
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
        attention_backend=arm.attention_backend,
        enable_torch_compile=arm.compile,
    )
    sampling = SamplingParam(
        num_inference_steps=4,
        num_frames=args.frames,
        height=args.height,
        width=args.width,
        fps=args.fps,
        seed=args.seed,
        guidance_scale=1.0,
        num_cond_frames=1,
        return_frames=True,
    )

    def generate(*, save_video: bool, run_index: int):
        output_path = arm_dir / f"{arm.name}_run{run_index}.mp4"
        return generator.generate_video(
            prompt,
            sampling_param=sampling,
            image_path=args.image,
            output_path=str(output_path),
            save_video=save_video,
            return_frames=True,
        )

    try:
        for warmup_index in range(args.warmups):
            started = time.perf_counter()
            generate(save_video=False, run_index=warmup_index)
            print(f"[{arm.name}] warmup {warmup_index + 1}/{args.warmups}: {time.perf_counter() - started:.2f}s",
                  flush=True)

        runs: list[dict[str, Any]] = []
        last_result: dict[str, Any] | None = None
        for run_index in range(args.runs):
            started = time.perf_counter()
            result = generate(save_video=False, run_index=run_index + 1)
            wall_time = time.perf_counter() - started
            if not isinstance(result, dict):
                raise RuntimeError(f"Expected one result dictionary, got {type(result)!r}")
            logging_info = result.get("logging_info")
            stages = getattr(logging_info, "stages", {}) if logging_info is not None else {}
            run = {
                "wall_time_s": wall_time,
                "generation_time_s": result.get("generation_time"),
                "e2e_latency_s": result.get("e2e_latency"),
                "peak_memory_mb": result.get("peak_memory_mb"),
                "stages": _json_ready(stages),
            }
            runs.append(run)
            last_result = result
            print(
                f"[{arm.name}] measured {run_index + 1}/{args.runs}: wall={wall_time:.2f}s "
                f"generation={float(result.get('generation_time') or 0):.2f}s "
                f"e2e={float(result.get('e2e_latency') or 0):.2f}s",
                flush=True,
            )

        assert last_result is not None
        frames = last_result.get("frames")
        if not isinstance(frames, list) or len(frames) != args.frames:
            actual = len(frames) if isinstance(frames, list) else None
            raise RuntimeError(f"DFD frame contract failed: expected {args.frames}, got {actual}")

        review_video_path = arm_dir / f"{arm.name}_review.mp4"
        video_encode_time = _save_review_video(frames, review_video_path, args.fps)
        quality = _quality_receipt(args.image, frames, arm_dir / f"{arm.name}_boundary.png")
        e2e_values = [float(run["e2e_latency_s"] or run["wall_time_s"]) for run in runs]
        summary = {
            "arm": asdict(arm),
            "model": str(Path(args.model).resolve()),
            "image": str(Path(args.image).resolve()),
            "prompt": prompt,
            "sampling": {
                "seed": args.seed,
                "height": args.height,
                "width": args.width,
                "frames": args.frames,
                "fps": args.fps,
                "steps": 4,
                "guidance_scale": 1.0,
            },
            "warmups": args.warmups,
            "runs": runs,
            "median_e2e_latency_s": statistics.median(e2e_values),
            "review_video_path": str(review_video_path),
            "review_video_encode_time_s": video_encode_time,
            "quality_receipt": quality,
        }
        result_path = arm_dir / "result.json"
        result_path.write_text(json.dumps(_json_ready(summary), indent=2) + "\n", encoding="utf-8")
        print(f"[{arm.name}] result: {result_path}", flush=True)
        return summary
    finally:
        generator.shutdown()


def _run_matrix(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for arm_name in ARMS:
        print(f"[matrix] starting {arm_name}", flush=True)
        subprocess.run(_arm_command(args, arm_name), check=True)

    summaries = {
        arm_name: json.loads((output_dir / arm_name / "result.json").read_text(encoding="utf-8"))
        for arm_name in ARMS
    }
    baseline = float(summaries["sdpa"]["median_e2e_latency_s"])
    for summary in summaries.values():
        latency = float(summary["median_e2e_latency_s"])
        summary["speedup_vs_sdpa"] = baseline / latency
        summary["latency_reduction_vs_sdpa_percent"] = (baseline - latency) / baseline * 100.0
    matrix_path = output_dir / "matrix.json"
    matrix_path.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print("\narm           median_e2e_s  speedup  reduction", flush=True)
    for arm_name, summary in summaries.items():
        print(
            f"{arm_name:<14} {summary['median_e2e_latency_s']:>12.2f}  "
            f"{summary['speedup_vs_sdpa']:>6.2f}x  "
            f"{summary['latency_reduction_vs_sdpa_percent']:>8.1f}%",
            flush=True,
        )
    print(f"[matrix] result: {matrix_path}", flush=True)


def main() -> None:
    args = _parser().parse_args()
    _validate_args(args)
    selected = list(ARMS) if args.arm == "all" else [args.arm]
    if args.dry_run:
        for arm_name in selected:
            print(asdict(ARMS[arm_name]))
        return
    if args.arm == "all":
        _run_matrix(args)
    else:
        _run_arm(args, ARMS[args.arm])


if __name__ == "__main__":
    main()
