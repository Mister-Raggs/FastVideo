"""FP8 weight quantization inference for Cosmos-Predict2.5 2B.

Runs Cosmos-2.5 2B t2w with FP8 e4m3 quantized DiT linear layers (attention
to_q/k/v/out projections and FFN mlp.fc_in/fc_out). Weights are quantized
in-place after loading; activations are quantized dynamically at runtime.

Unlike the few-step distilled Wan recipes, Cosmos-2.5 is a full-step CFG
model (the preset runs guidance 7.0), so quantization error accumulates
across every denoise step — compare outputs against a same-seed bf16 run
before trusting any speed number.

Requirements:
    - GPU: sm89+ (H100, L40S, RTX 4090, Ada Lovelace, or newer)
      Falls back to a bf16 dequant path on older GPUs.

Usage:
    python fp8_cosmos2_5_2b.py              # FP8 per-tensor (default)
    python fp8_cosmos2_5_2b.py --bf16       # BF16 baseline
    python fp8_cosmos2_5_2b.py --granularity channel  # per-channel (higher accuracy but slower)
"""

import argparse
import os
import time

OUTPUT_PATH = "video_samples"


def main():
    parser = argparse.ArgumentParser(description="FP8 Cosmos-2.5 2B generation benchmark")
    parser.add_argument("--bf16", action="store_true",
                        help="BF16 baseline (no FP8 quantization)")
    parser.add_argument("--granularity", choices=["tensor", "channel"], default="tensor",
                        help="FP8 weight scale granularity: tensor (faster) or channel (more accurate)")
    parser.add_argument("--model", default="KyleShao/Cosmos-Predict2.5-2B-Diffusers",
                        help="Model path or HuggingFace ID")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile for the DiT")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--infer_steps", type=int, default=None,
                        help="Denoise steps; default uses the cosmos25_predict2_2b preset")
    parser.add_argument("--seed", type=int, default=1024,
                        help="Generation seed; keep fixed across bf16/fp8 arms for SSIM comparison")
    args = parser.parse_args()

    from fastvideo import VideoGenerator
    from fastvideo.layers.quantization import get_quantization_config

    mode = "bf16" if args.bf16 else f"fp8_{args.granularity}"
    if args.compile:
        mode += "_compile"
    print(f"Mode: {mode.upper()}")

    # transformer_quant needs a QuantizationConfig *instance* — the bare string
    # is not resolved on the from_pretrained kwarg path.
    extra = {} if args.bf16 else {
        "transformer_quant": get_quantization_config("FP8")(granularity=args.granularity)
    }
    generator = VideoGenerator.from_pretrained(
        args.model,
        num_gpus=args.num_gpus,
        enable_torch_compile=args.compile,
        **extra,
    )

    prompt = (
        "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes "
        "wide with interest. The playful yet serene atmosphere is complemented by soft "
        "natural light filtering through the petals. Mid-shot, warm and cheerful tones."
    )

    sampling = {"seed": args.seed}
    if args.infer_steps is not None:
        sampling["num_inference_steps"] = args.infer_steps

    if args.compile:
        generator.generate(request={"prompt": prompt,
                                    "sampling": {**sampling, "num_inference_steps": 2},
                                    "output": {"save_video": False}})

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    start = time.time()
    generator.generate(request={
        "prompt": prompt,
        "sampling": sampling,
        "output": {"save_video": True, "output_path": os.path.join(OUTPUT_PATH, f"raccoon_{mode}.mp4")},
    })
    elapsed = time.time() - start
    print(f"[{mode.upper()}] generated in {elapsed:.2f}s (seed {args.seed})")

    generator.shutdown()


if __name__ == "__main__":
    main()
