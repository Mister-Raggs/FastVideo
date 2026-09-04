# SPDX-License-Identifier: Apache-2.0
"""Paired Wan VSA-128 route-A versus native-Triton benchmark.

The model is loaded once. Both arms use the same 128-token ``(8, 4, 4)``
geometry, sparsity, prompt, seed, and output shape; only the physical sparse
kernel route changes. Route choice travels through ``ForwardBatch.extra`` so
the worker process sees every per-request arm explicitly.

The default workload uses the small three-step FastWan 1.3B checkpoint and the
DGX Spark-safe 448x832x81 shape. One excluded warmup per arm is followed by an
ABBA-balanced six-request measurement (three requests per arm).
"""
from __future__ import annotations

import argparse
import math
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch


MODEL = "FastVideo/FastWan2.1-T2V-1.3B-Diffusers"
PROMPT = (
    "A curious raccoon peers through a vibrant field of yellow sunflowers, "
    "soft natural light, warm cheerful tones, mid-shot, cinematic."
)


def _label(native: bool) -> str:
    return "native128" if native else "route64"


def _balanced_schedule(runs_per_arm: int) -> list[bool]:
    schedule: list[bool] = []
    counts = {False: 0, True: 0}
    for native in (False, True, True, False):
        if counts[native] < runs_per_arm:
            schedule.append(native)
            counts[native] += 1
    while counts[False] < runs_per_arm or counts[True] < runs_per_arm:
        for native in (False, True):
            if counts[native] < runs_per_arm:
                schedule.append(native)
                counts[native] += 1
    return schedule


def _frame_array(result: object) -> np.ndarray | None:
    if not isinstance(result, dict):
        return None
    frames = result.get("frames")
    if frames is None:
        return None
    if isinstance(frames, np.ndarray):
        return frames.astype(np.float32)
    try:
        return np.stack([np.asarray(frame) for frame in frames]).astype(np.float32)
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--output", default="outputs/wan_vsa128_native_ab")
    parser.add_argument("--runs-per-arm", type=int, default=3)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--height", type=int, default=448)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.runs_per_arm < 1:
        parser.error("--runs-per-arm must be at least 1")

    # Boot-time backend selection remains immutable. Tile geometry and the
    # route-A/native choice are explicit request data handled inside the worker.
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "VIDEO_SPARSE_ATTN"
    os.environ["FASTVIDEO_VSA_TRITON"] = "1"
    os.environ["FASTVIDEO_VSA_CUTEDSL"] = "0"
    os.environ["FASTVIDEO_VSA_TRITON_NATIVE_128"] = "0"
    os.environ.setdefault("FASTVIDEO_STAGE_LOGGING", "1")

    from fastvideo import VideoGenerator
    from fastvideo.api.sampling_param import SamplingParam

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    print(f"device={torch.cuda.get_device_name(0)} model={args.model}", flush=True)
    print("logical geometry=(8,4,4), tokens=128; only physical kernel route changes", flush=True)

    load_start = time.perf_counter()
    generator = VideoGenerator.from_pretrained(
        args.model,
        num_gpus=1,
        use_fsdp_inference=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        VSA_sparsity=0.8,
    )
    print(f"model_load_s={time.perf_counter() - load_start:.2f}", flush=True)

    sampling = SamplingParam.from_pretrained(args.model)
    sampling.num_frames = args.frames
    sampling.height = args.height
    sampling.width = args.width
    sampling.num_inference_steps = args.steps
    sampling.seed = args.seed

    def generate(native: bool, stem: str):
        path = output / f"{stem}_{_label(native)}.mp4"
        start = time.perf_counter()
        result = generator.generate_video(
            args.prompt,
            sampling_param=sampling,
            output_path=str(path),
            save_video=True,
            vsa_tile_size=128,
            vsa_native_128=native,
        )
        wall = time.perf_counter() - start
        reported = result.get("generation_time") if isinstance(result, dict) else None
        return result, float(reported if reported is not None else wall), wall, path

    try:
        if args.warmup:
            for native in (False, True):
                print(f"warmup arm={_label(native)}", flush=True)
                generate(native, "warmup")

        timings: dict[bool, list[float]] = {False: [], True: []}
        first_frames: dict[bool, np.ndarray] = {}
        for index, native in enumerate(_balanced_schedule(args.runs_per_arm), start=1):
            result, reported, wall, path = generate(native, f"measured_{index:02d}")
            timings[native].append(reported)
            frames = _frame_array(result)
            if frames is not None and native not in first_frames:
                first_frames[native] = frames
            print(
                f"run={index} arm={_label(native)} generation_s={reported:.3f} "
                f"wall_s={wall:.3f} output={path}",
                flush=True,
            )

        route_median = statistics.median(timings[False])
        native_median = statistics.median(timings[True])
        print(f"route64_median_s={route_median:.3f}", flush=True)
        print(f"native128_median_s={native_median:.3f}", flush=True)
        print(f"generation_speedup={route_median / native_median:.4f}x", flush=True)

        if set(first_frames) == {False, True} and first_frames[False].shape == first_frames[True].shape:
            delta = np.abs(first_frames[False] - first_frames[True])
            mse = float(np.square(delta).mean())
            psnr = math.inf if mse == 0 else 20 * math.log10(255.0 / math.sqrt(mse))
            print(
                f"decoded_frame_parity shape={first_frames[False].shape} "
                f"mean_abs={float(delta.mean()):.6f} max_abs={float(delta.max()):.1f} psnr_db={psnr:.3f}",
                flush=True,
            )
        else:
            print("decoded_frame_parity=unavailable; compare the first video from each arm visually", flush=True)
    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()
