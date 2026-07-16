"""
Inner-script half of the Cosmos-2.5 quant A/B harness.

Runs one pass (bf16 baseline or quantized) over the harness's prompt + seed
set, records per-prompt walls, and writes a JSON results blob. Invoked by
``cosmos25_quant_ab.py`` inside the image's venv (Modal's main process
Python lacks torch).

Quantization is LOSSY and Cosmos-2.5 is a full-step CFG model, so the SSIM
column is a quality measurement (target >= ~0.95), not a 1.0 gate.
"""
import argparse
import json
import os
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("run_pass", "compute_ssim"), default="run_pass")
    parser.add_argument("--config-json")
    parser.add_argument("--results-json")
    parser.add_argument("--baseline-results-json")
    parser.add_argument("--patched-results-json")
    parser.add_argument("--ssim-output-json")
    args = parser.parse_args()

    if args.mode == "compute_ssim":
        return _compute_ssim_main(args)

    if not args.config_json or not args.results_json:
        parser.error("--config-json and --results-json are required for mode=run_pass")
    with open(args.config_json) as f:
        cfg = json.load(f)

    model_id = cfg["model_id"]
    num_gpus = cfg["num_gpus"]
    quant_method = cfg["quant_method"]  # None (bf16) | "FP8" | "nvfp4_qat"
    granularity = cfg.get("granularity", "tensor")
    enable_compile = cfg.get("enable_compile", False)
    num_inference_steps = cfg.get("num_inference_steps", 0)
    output_dir = cfg["output_dir"]
    prompts = cfg["prompts"]
    seed_base = cfg["seed_base"]

    from fastvideo import VideoGenerator

    # transformer_quant needs a QuantizationConfig *instance* — the bare
    # string is not resolved on the from_pretrained kwarg path.
    extra = {}
    if quant_method == "FP8":
        from fastvideo.layers.quantization import get_quantization_config
        extra["transformer_quant"] = get_quantization_config("FP8")(granularity=granularity)
    elif quant_method == "nvfp4_qat":
        from fastvideo.layers.quantization import get_quantization_config
        extra["transformer_quant"] = get_quantization_config("nvfp4_qat")()
    elif quant_method is not None:
        raise ValueError(f"unknown quant_method: {quant_method}")

    generator = VideoGenerator.from_pretrained(
        model_id,
        num_gpus=num_gpus,
        enable_torch_compile=enable_compile,
        **extra,
    )

    # Verify the wiring actually engaged: the DiT config must carry the
    # quant_config instance (a silent no-op here would produce a flattering
    # SSIM of 1.0 and a wall delta of 0).
    if quant_method is not None:
        dit_config = getattr(generator.fastvideo_args.pipeline_config, "dit_config", None)
        resolved = getattr(dit_config, "quant_config", None)
        if resolved is None:
            raise RuntimeError("transformer_quant did not propagate to dit_config.quant_config — "
                               "quant pass would silently run bf16")
        print(f"[inner] quant engaged: {type(resolved).__name__}", flush=True)
    else:
        print("[inner] bf16 baseline", flush=True)

    os.makedirs(output_dir, exist_ok=True)
    records = []
    for i, prompt in enumerate(prompts):
        prompt_out = os.path.join(output_dir, f"prompt_{i:02d}")
        gen_kwargs = {}
        if num_inference_steps:
            gen_kwargs["num_inference_steps"] = num_inference_steps
        t0 = time.perf_counter()
        generator.generate_video(
            prompt,
            output_path=prompt_out,
            save_video=True,
            seed=seed_base + i,
            **gen_kwargs,
        )
        wall = time.perf_counter() - t0
        # generate_video appends _1, _2, ... rather than overwriting; pick the
        # mp4 we just wrote by mtime (newest).
        mp4s = sorted([os.path.join(prompt_out, f) for f in os.listdir(prompt_out) if f.endswith(".mp4")],
                      key=os.path.getmtime)
        if not mp4s:
            raise RuntimeError(f"no .mp4 produced for prompt {i} at {prompt_out}")
        records.append({"i": i, "wall_s": wall, "mp4": mp4s[-1]})
        print(f"[inner] [{quant_method or 'baseline'}] prompt {i}: {wall:.3f}s -> {mp4s[-1]}", flush=True)

    with open(args.results_json, "w") as f:
        json.dump(records, f)
    print(f"[inner] wrote {len(records)} records to {args.results_json}", flush=True)
    return 0


def _compute_ssim_main(args) -> int:
    """Pairwise SSIM between baseline and quant output mp4s. Avoids importing
    fastvideo (its top-level import pulls triton, which needs a CUDA driver);
    runs on CPU with pytorch_msssim + torchvision/av directly."""
    if not args.ssim_output_json or not (args.baseline_results_json and args.patched_results_json):
        raise SystemExit("compute_ssim needs --baseline-results-json, --patched-results-json, --ssim-output-json")

    import torch
    from pytorch_msssim import ssim as pm_ssim

    def _read_video_frames(path):
        try:
            from torchvision.io import read_video
            frames, _, _ = read_video(path, pts_unit="sec", output_format="TCHW")
            if frames.shape[0] > 0:
                return frames
        except Exception:
            # torchvision's backend (FFmpeg/PyAV) can raise more than
            # ImportError/AttributeError; fall back to the PyAV path below.
            pass
        import av
        container = av.open(path)
        frames = []
        for frame in container.decode(video=0):
            frames.append(torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1))
        container.close()
        if not frames:
            raise RuntimeError(f"No video frames decoded from {path}")
        return torch.stack(frames)

    def _ssim(p1, p2):
        f1, f2 = _read_video_frames(p1), _read_video_frames(p2)
        n = min(f1.shape[0], f2.shape[0])
        if n == 0:
            raise RuntimeError(f"no decodable frames to compare: {p1} ({f1.shape[0]}) vs {p2} ({f2.shape[0]})")
        f1 = (f1[:n].float() / 255.0).contiguous()
        f2 = (f2[:n].float() / 255.0).contiguous()
        return [pm_ssim(f1[i:i + 1], f2[i:i + 1], data_range=1.0).item() for i in range(n)]

    with open(args.baseline_results_json) as f:
        baseline = json.load(f)
    with open(args.patched_results_json) as f:
        patched = json.load(f)

    rows = []
    for b, p in zip(baseline, patched, strict=True):
        if b["i"] != p["i"]:
            raise ValueError(f"baseline/patched prompt index mismatch: {b['i']} != {p['i']}")
        vals = _ssim(b["mp4"], p["mp4"])
        rows.append({
            "i": b["i"],
            "baseline_wall_s": b["wall_s"],
            "patched_wall_s": p["wall_s"],
            "ssim_mean": float(sum(vals) / len(vals)),
            "ssim_worst": float(min(vals)),
        })
        print(f"[inner]   prompt {b['i']} SSIM mean={rows[-1]['ssim_mean']:.6f} worst={rows[-1]['ssim_worst']:.6f}",
              flush=True)
    with open(args.ssim_output_json, "w") as f:
        json.dump(rows, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
