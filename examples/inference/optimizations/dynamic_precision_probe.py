# SPDX-License-Identifier: Apache-2.0
"""Profile and analyze Wan block sensitivity for adaptive precision.

The ``profile`` command is intentionally a slow research run: it keeps the
real denoising trajectory in BF16 and replays selected blocks with one NVFP4
projection at a time.  The ``analyze`` command is CPU-only and safe to run on a
laptop.

Examples::

    python examples/inference/optimizations/dynamic_precision_probe.py analyze \
        --input artifacts/dynamic_precision/probe.jsonl

    python examples/inference/optimizations/dynamic_precision_probe.py profile \
        --output artifacts/dynamic_precision/probe.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

DEFAULT_PROMPT = (
    "A curious raccoon peers through a vibrant field of yellow sunflowers, "
    "soft natural light, detailed fur, gentle camera movement."
)


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="analyze an existing JSONL probe")
    analyze.add_argument("--input", required=True, help="probe JSONL path")
    analyze.add_argument("--output", help="optional summary JSON path")

    profile = subparsers.add_parser("profile", help="run the GPU profiling experiment")
    profile.add_argument("--model", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    profile.add_argument("--output", required=True, help="new JSONL output path")
    profile.add_argument("--overwrite", action="store_true")
    profile.add_argument("--prompt", default=DEFAULT_PROMPT)
    profile.add_argument("--seed", type=int, default=42)
    profile.add_argument("--steps", type=int, default=50)
    profile.add_argument("--height", type=int, default=480)
    profile.add_argument("--width", type=int, default=832)
    profile.add_argument("--frames", type=int, default=81)
    profile.add_argument("--blocks", type=_csv_ints, default=_csv_ints("0,5,10,15,20,25,29"))
    profile.add_argument("--error-steps", type=_csv_ints, default=_csv_ints("2,8,16,24,32,40,47"))
    return parser


def _analyze(args: argparse.Namespace) -> None:
    from fastvideo.benchmarks.dynamic_precision_probe import analyze_records, load_records

    summary = analyze_records(load_records(args.input))
    rendered = json.dumps(summary, indent=2, sort_keys=True, allow_nan=True)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


def _profile(args: argparse.Namespace) -> None:
    # Pin the comparison to dense SDPA before importing FastVideo's platform
    # selection.  Attention is deliberately outside this experiment's axis.
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "TORCH_SDPA"

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("profile requires a CUDA GPU; use 'analyze' locally")
    if max(args.error_steps) >= args.steps:
        raise SystemExit("every --error-steps value must be smaller than --steps")

    output = Path(args.output).resolve()
    if output.exists() and not args.overwrite:
        raise SystemExit(f"refusing to append to existing {output}; pass --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    from fastvideo import VideoGenerator
    from fastvideo.benchmarks.dynamic_precision_probe import (
        DynamicPrecisionProbeConfig,
        ProbeSpec,
    )
    from fastvideo.configs.pipelines.base import PipelineConfig

    spec = ProbeSpec(
        output_path=str(output),
        block_indices=args.blocks,
        error_step_indices=args.error_steps,
    )
    pipeline_config = PipelineConfig.from_pretrained(args.model)
    pipeline_config.dit_precision = "bf16"
    pipeline_config.vae_precision = "bf16"
    pipeline_config.text_encoder_precisions = ("bf16", )
    pipeline_config.dit_config.quant_config = DynamicPrecisionProbeConfig(spec)

    print("[probe] BF16 main path; isolated NVFP4 projection replays")
    print(f"[probe] blocks={args.blocks}, error_steps={args.error_steps}")
    print(f"[probe] JSONL={output}")
    generator = VideoGenerator.from_pretrained(
        args.model,
        pipeline_config=pipeline_config,
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,
        pin_cpu_memory=False,
        enable_torch_compile=False,
        output_type="latent",
    )
    try:
        generator.generate(request={
            "prompt": args.prompt,
            "sampling": {
                "seed": args.seed,
                "num_inference_steps": args.steps,
                "guidance_scale": 1.0,
                "height": args.height,
                "width": args.width,
                "num_frames": args.frames,
            },
            "output": {
                "save_video": False,
                "return_frames": False,
            },
        })
    finally:
        generator.shutdown()
    print(f"[probe] complete: {output}")


def main() -> None:
    args = _parser().parse_args()
    if args.command == "analyze":
        _analyze(args)
    else:
        _profile(args)


if __name__ == "__main__":
    main()
