"""Benchmark quality-safe Cosmos2.5 DFD latency profiles through DreamVerse.

Every arm keeps BF16 weights, the four-step DFD schedule, seed, resolution,
frame count, and conditioning image fixed. Arms vary only component residency,
the dense attention implementation, and precision-preserving regional compile.
Each arm runs in an isolated process so backend selection, allocator state, and
Inductor caches cannot leak across measurements.

The ``core`` matrix answers the first production question cheaply:

* ``lazy_sdpa`` reproduces GB10's prior automatic per-request module reload.
* ``resident_sdpa`` keeps both Cosmos 2B stacks resident.
* ``resident_sdpa_regional`` adds per-transformer-block regional compile.
* ``hybrid_sdpa_regional`` leaves the one-use T2W stack lazy while keeping the
  repeated DFD stack resident and compiled.

The optional FlashAttention arms retain BF16 and are subject to decoded-frame
parity plus visual review before promotion.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_PROMPT = ("The same red Dodge Challenger approaches a sweeping bend and begins turning smoothly to the right, "
                  "maintaining its appearance, speed, and the same low rear three-quarter tracking camera.")


@dataclass(frozen=True)
class Arm:
    name: str
    attention_backend: str
    bootstrap_lazy_module_load: bool
    continuation_lazy_module_load: bool
    bootstrap_inference_torch_compile: bool
    continuation_inference_torch_compile: bool
    bootstrap_compile_vae: bool = False
    continuation_compile_vae: bool = False


ARMS = {
    "lazy_sdpa": Arm("lazy_sdpa", "TORCH_SDPA", True, True, False, False),
    "resident_sdpa": Arm("resident_sdpa", "TORCH_SDPA", False, False, False, False),
    "resident_sdpa_regional": Arm("resident_sdpa_regional", "TORCH_SDPA", False, False, True, True),
    "hybrid_sdpa_regional": Arm("hybrid_sdpa_regional", "TORCH_SDPA", True, False, False, True),
    "hybrid_sdpa_regional_vae": Arm(
        "hybrid_sdpa_regional_vae",
        "TORCH_SDPA",
        True,
        False,
        False,
        True,
        False,
        True,
    ),
    "resident_flash": Arm("resident_flash", "FLASH_ATTN", False, False, False, False),
    "resident_flash_regional": Arm("resident_flash_regional", "FLASH_ATTN", False, False, True, True),
}
CORE_ARMS = ("lazy_sdpa", "resident_sdpa_regional", "hybrid_sdpa_regional")
DECODE_ARMS = ("hybrid_sdpa_regional", "hybrid_sdpa_regional_vae")
PRODUCTION_ARMS = ("lazy_sdpa", "hybrid_sdpa_regional_vae")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-model", required=True)
    parser.add_argument("--continuation-model", required=True)
    parser.add_argument("--image", help="Conditioning image; required for --role continuation")
    parser.add_argument("--role", choices=("continuation", "bootstrap"), default="continuation")
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt", default=DEFAULT_PROMPT)
    prompt_group.add_argument("--prompt-file")
    parser.add_argument("--arm", choices=("core", "decode", "production", "all", *ARMS), default="core")
    parser.add_argument("--output-dir", default="outputs/cosmos25_dfd_dreamverse_matrix")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--bootstrap-frames", type=int, default=77)
    parser.add_argument("--continuation-frames", type=int, default=81)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--quality-scale", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _selected_arms(selection: str) -> tuple[str, ...]:
    if selection == "core":
        return CORE_ARMS
    if selection == "decode":
        return DECODE_ARMS
    if selection == "production":
        return PRODUCTION_ARMS
    if selection == "all":
        return tuple(ARMS)
    return (selection, )


def _validate_args(args: argparse.Namespace) -> None:
    if args.warmups < 0:
        raise SystemExit("--warmups must be non-negative")
    if args.runs < 1:
        raise SystemExit("--runs must be positive")
    if args.quality_scale < 1:
        raise SystemExit("--quality-scale must be positive")
    for name in ("height", "width", "bootstrap_frames", "continuation_frames", "fps"):
        if getattr(args, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.dry_run:
        return
    if not Path(args.bootstrap_model).is_dir():
        raise SystemExit(f"Bootstrap model directory not found: {args.bootstrap_model}")
    if not Path(args.continuation_model).is_dir():
        raise SystemExit(f"Continuation model directory not found: {args.continuation_model}")
    if args.role == "continuation" and (not args.image or not Path(args.image).is_file()):
        raise SystemExit("--role continuation requires an existing --image")
    if args.prompt_file and not Path(args.prompt_file).is_file():
        raise SystemExit(f"Prompt file not found: {args.prompt_file}")


def _prompt(args: argparse.Namespace) -> str:
    prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip() if args.prompt_file else args.prompt.strip()
    if not prompt:
        raise SystemExit("Prompt must not be empty")
    return prompt


def _arm_command(args: argparse.Namespace, arm_name: str) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--bootstrap-model",
        args.bootstrap_model,
        "--continuation-model",
        args.continuation_model,
        "--role",
        args.role,
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
        "--bootstrap-frames",
        str(args.bootstrap_frames),
        "--continuation-frames",
        str(args.continuation_frames),
        "--fps",
        str(args.fps),
        "--quality-scale",
        str(args.quality_scale),
    ]
    if args.image:
        command.extend(("--image", args.image))
    if args.prompt_file:
        command.extend(("--prompt-file", args.prompt_file))
    else:
        command.extend(("--prompt", args.prompt))
    return command


def _model_config(args: argparse.Namespace, arm: Arm) -> dict[str, Any]:
    return {
        "name": "Cosmos Predict2.5 DFD benchmark",
        "generation_backend": "cosmos25_dfd",
        "default_sp_size": 1,
        "model_path": str(Path(args.bootstrap_model).resolve()),
        "continuation_model_path": str(Path(args.continuation_model).resolve()),
        "attention_backend": arm.attention_backend,
        "height": args.height,
        "width": args.width,
        "bootstrap_num_frames": args.bootstrap_frames,
        "continuation_num_frames": args.continuation_frames,
        "fps": args.fps,
        "num_inference_steps": 4,
        "seed": args.seed,
        "bootstrap_lazy_module_load": arm.bootstrap_lazy_module_load,
        "continuation_lazy_module_load": arm.continuation_lazy_module_load,
        "bootstrap_inference_torch_compile": arm.bootstrap_inference_torch_compile,
        "continuation_inference_torch_compile": arm.continuation_inference_torch_compile,
        "bootstrap_compile_vae": arm.bootstrap_compile_vae,
        "continuation_compile_vae": arm.continuation_compile_vae,
        # The harness owns warmup so its cost is measured separately.
        "startup_warmup": False,
    }


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
        raise RuntimeError(f"Expected HWC RGB frame, got {pixels.shape}")
    if pixels.dtype != np.uint8:
        pixels = np.clip(pixels, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(pixels)


def _save_receipts(
    frames: list[Any],
    *,
    source_image: str | None,
    arm_dir: Path,
    fps: int,
    quality_scale: int,
) -> dict[str, Any]:
    import imageio
    import numpy as np
    from PIL import Image

    arrays = [_frame_array(frame) for frame in frames]
    review_path = arm_dir / "review.mp4"
    imageio.mimsave(review_path, arrays, fps=fps, format="mp4")

    first = arrays[0]
    boundary_path = arm_dir / "boundary.png"
    if source_image:
        with Image.open(source_image) as image:
            left = np.asarray(image.convert("RGB").resize((first.shape[1], first.shape[0])))
    else:
        left = first
    boundary = Image.new("RGB", (first.shape[1] * 2, first.shape[0]))
    boundary.paste(Image.fromarray(left), (0, 0))
    boundary.paste(Image.fromarray(first), (first.shape[1], 0))
    boundary.save(boundary_path)

    sample_width = max(1, first.shape[1] // quality_scale)
    sample_height = max(1, first.shape[0] // quality_scale)
    sample = np.stack([
        np.asarray(Image.fromarray(frame).resize((sample_width, sample_height), Image.Resampling.BOX))
        for frame in arrays
    ])
    sample_path = arm_dir / "quality_sample.npz"
    np.savez_compressed(sample_path, frames=sample)
    return {
        "review_video": str(review_path),
        "boundary_image": str(boundary_path),
        "quality_sample": str(sample_path),
        "quality_sample_shape": list(sample.shape),
    }


def _median_timings(runs: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted(set().union(*(run.keys() for run in runs)))
    return {key: statistics.median(float(run[key]) for run in runs if key in run) for key in keys}


def _run_arm(args: argparse.Namespace, arm: Arm) -> dict[str, Any]:
    os.environ["FASTVIDEO_STAGE_LOGGING"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from dreamverse.cosmos25_dfd_generation import Cosmos25DFDGenerationBackend

    arm_dir = Path(args.output_dir).resolve() / args.role / arm.name
    arm_dir.mkdir(parents=True, exist_ok=True)
    prompt = _prompt(args)
    backend = Cosmos25DFDGenerationBackend(gpu_id=0)

    print(
        f"[{arm.name}] role={args.role} attention={arm.attention_backend} "
        f"bootstrap(lazy={arm.bootstrap_lazy_module_load}, "
        f"regional_compile={arm.bootstrap_inference_torch_compile}) "
        f"continuation(lazy={arm.continuation_lazy_module_load}, "
        f"regional_compile={arm.continuation_inference_torch_compile}) "
        f"vae_compile=(bootstrap={arm.bootstrap_compile_vae}, continuation={arm.continuation_compile_vae})",
        flush=True)
    initialize_started = time.perf_counter()
    backend.initialize(_model_config(args, arm))
    initialize_ms = (time.perf_counter() - initialize_started) * 1000.0
    print(f"[{arm.name}] initialize={initialize_ms:.0f}ms", flush=True)

    image_path = args.image if args.role == "continuation" else None
    expected_frames = args.continuation_frames if image_path else args.bootstrap_frames
    warmup_times_ms: list[float] = []
    runs: list[dict[str, float]] = []
    last_frames: list[Any] | None = None
    try:
        for index in range(args.warmups):
            started = time.perf_counter()
            backend.generate_step(prompt, 1, image_path, True)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            warmup_times_ms.append(elapsed_ms)
            print(f"[{arm.name}] warmup {index + 1}/{args.warmups}: {elapsed_ms:.0f}ms", flush=True)

        for index in range(args.runs):
            result = backend.generate_step(prompt, 1, image_path, True)
            if len(result.frames) != expected_frames:
                raise RuntimeError(f"Expected {expected_frames} frames, got {len(result.frames)}")
            timing = {key: float(value) for key, value in result.timings.items() if isinstance(value, int | float)}
            runs.append(timing)
            last_frames = result.frames
            print(
                f"[{arm.name}] measured {index + 1}/{args.runs}: "
                f"e2e={timing.get('e2e_latency_ms', 0.0):.0f}ms "
                f"prompt={timing.get('stage_prompt_encoding_stage_ms', 0.0):.0f}ms "
                f"denoise={timing.get('stage_denoising_stage_ms', 0.0):.0f}ms "
                f"decode={timing.get('stage_decoding_stage_ms', 0.0):.0f}ms",
                flush=True)

        assert last_frames is not None
        receipts = _save_receipts(
            last_frames,
            source_image=image_path,
            arm_dir=arm_dir,
            fps=args.fps,
            quality_scale=args.quality_scale,
        )
        summary = {
            "arm": asdict(arm),
            "role": args.role,
            "initialize_ms": initialize_ms,
            "warmup_times_ms": warmup_times_ms,
            "runs": runs,
            "median_timings": _median_timings(runs),
            "prompt": prompt,
            "sampling": {
                "height": args.height,
                "width": args.width,
                "frames": expected_frames,
                "fps": args.fps,
                "steps": 4,
                "seed": args.seed,
            },
            **receipts,
        }
        result_path = arm_dir / "result.json"
        result_path.write_text(json.dumps(_json_ready(summary), indent=2) + "\n", encoding="utf-8")
        print(f"[{arm.name}] result={result_path}", flush=True)
        return summary
    finally:
        backend.shutdown()


def _quality_comparison(reference_path: str, candidate_path: str) -> dict[str, Any]:
    import numpy as np

    with np.load(reference_path) as reference_data, np.load(candidate_path) as candidate_data:
        reference = reference_data["frames"].astype(np.float32)
        candidate = candidate_data["frames"].astype(np.float32)
    if reference.shape != candidate.shape:
        return {
            "shape_match": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }
    difference = np.abs(reference - candidate)
    mse = float(np.square(reference - candidate).mean())
    psnr = math.inf if mse == 0 else 20.0 * math.log10(255.0 / math.sqrt(mse))
    return {
        "shape_match": True,
        "identical": bool(np.array_equal(reference, candidate)),
        "mean_abs": float(difference.mean()),
        "max_abs": float(difference.max()),
        "psnr_db": psnr,
    }


def _run_matrix(args: argparse.Namespace, arm_names: tuple[str, ...]) -> None:
    output_dir = Path(args.output_dir).resolve() / args.role
    output_dir.mkdir(parents=True, exist_ok=True)
    for arm_name in arm_names:
        print(f"[matrix] starting {arm_name}", flush=True)
        subprocess.run(_arm_command(args, arm_name), check=True)

    summaries = {
        arm_name: json.loads((output_dir / arm_name / "result.json").read_text(encoding="utf-8"))
        for arm_name in arm_names
    }
    baseline_name = "lazy_sdpa" if "lazy_sdpa" in summaries else arm_names[0]
    baseline = summaries[baseline_name]
    baseline_latency = float(baseline["median_timings"]["e2e_latency_ms"])
    for arm_name, summary in summaries.items():
        latency = float(summary["median_timings"]["e2e_latency_ms"])
        summary["speedup_vs_baseline"] = baseline_latency / latency
        summary["latency_reduction_vs_baseline_percent"] = (baseline_latency - latency) / baseline_latency * 100.0
        summary["quality_vs_baseline"] = _quality_comparison(
            baseline["quality_sample"],
            summary["quality_sample"],
        )

    matrix_path = output_dir / "matrix.json"
    matrix_path.write_text(json.dumps(_json_ready(summaries), indent=2) + "\n", encoding="utf-8")
    print("\narm                          e2e_s  speedup  reduction  quality_psnr", flush=True)
    for arm_name, summary in summaries.items():
        latency_s = float(summary["median_timings"]["e2e_latency_ms"]) / 1000.0
        quality = summary["quality_vs_baseline"]
        psnr = quality.get("psnr_db")
        psnr_label = "inf" if psnr == math.inf else (f"{psnr:.2f}" if isinstance(psnr, int | float) else "n/a")
        print(
            f"{arm_name:<28} {latency_s:>6.2f}  {summary['speedup_vs_baseline']:>6.2f}x  "
            f"{summary['latency_reduction_vs_baseline_percent']:>8.1f}%  {psnr_label:>12}",
            flush=True)
    print(f"[matrix] result={matrix_path}", flush=True)


def main() -> None:
    args = _parser().parse_args()
    _validate_args(args)
    arm_names = _selected_arms(args.arm)
    if args.dry_run:
        for arm_name in arm_names:
            print(asdict(ARMS[arm_name]))
        return
    if len(arm_names) == 1:
        _run_arm(args, ARMS[arm_names[0]])
    else:
        _run_matrix(args, arm_names)


if __name__ == "__main__":
    main()
