"""D1 step 3 -- KV window sweep for quality-signal calibration.

Generates one Matrix-Game 2.0 clip with EVERYTHING pinned except the KV window,
so the only variable is how much history the model can attend to. Run it once
per window value; the point is to find a metric that responds monotonically as
the window shrinks, BEFORE building any allocation policy that would need such a
metric to be scoreable.

    FASTVIDEO_KV_WINDOW=6 python examples/inference/optimizations/kvpolicy_window_sweep.py
    FASTVIDEO_KV_WINDOW=4 python examples/inference/optimizations/kvpolicy_window_sweep.py
    FASTVIDEO_KV_WINDOW=3 python examples/inference/optimizations/kvpolicy_window_sweep.py

Run the default window TWICE to establish the noise floor first -- if two
identical-input runs differ, no metric can resolve a window effect smaller than
that difference.

Pinned: generation seed, and the ACTION SEQUENCE. The stock example calls
create_action_presets(...) without a seed, so its actions are re-randomised every
run and two runs are not comparable. That would silently masquerade as a window
effect.
"""

import argparse
import json
import os
import time

import torch

from fastvideo import VideoGenerator
from fastvideo.models.dits.matrixgame2.utils import create_action_presets

MODEL = "FastVideo/Matrix-Game-2.0-Base-Distilled-Diffusers"
IMAGE = ("https://raw.githubusercontent.com/SkyworkAI/Matrix-Game/main/"
         "Matrix-Game-2/demo_images/universal/0000.png")
KEYBOARD_DIM = 4


def _extract_peak_mb(result) -> float | None:
    """Peak memory as measured IN THE WORKER, from whatever shape it comes back in."""
    if isinstance(result, dict):
        return result.get("peak_memory_mb")
    for attr in ("peak_memory_mb", "extra"):
        val = getattr(result, attr, None)
        if isinstance(val, dict):
            return val.get("peak_memory_mb")
        if val is not None:
            return val
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-frames", type=int, default=597)
    ap.add_argument("--seed", type=int, default=1024)
    ap.add_argument("--action-seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="suffix for the output dir, e.g. 'run2'")
    ap.add_argument("--outdir", default="kvsweep_out")
    args = ap.parse_args()

    window = os.getenv("FASTVIDEO_KV_WINDOW", "default")
    label = f"w{window}" + (f"_{args.tag}" if args.tag else "")
    outdir = os.path.join(args.outdir, label)
    os.makedirs(outdir, exist_ok=True)

    # latent frames must be divisible by num_frames_per_block (=3)
    latent = (args.num_frames - 1) // 4 + 1
    if (args.num_frames - 1) % 4 or latent % 3:
        raise SystemExit(f"--num-frames {args.num_frames} -> {latent} latent frames; "
                         "need 4k+1 and latent divisible by 3 (597, 165, 153, ...)")

    generator = VideoGenerator.from_pretrained(
        MODEL,
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=True,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
    )

    # seeded so the action sequence is identical across every run in the sweep
    actions = create_action_presets(args.num_frames, keyboard_dim=KEYBOARD_DIM, seed=args.action_seed)
    grid_sizes = torch.tensor([latent, 44, 80])

    t0 = time.perf_counter()
    result = generator.generate_video(
        prompt="",
        image_path=IMAGE,
        mouse_cond=actions["mouse"].unsqueeze(0),
        keyboard_cond=actions["keyboard"].unsqueeze(0),
        grid_sizes=grid_sizes,
        num_frames=args.num_frames,
        height=352,
        width=640,
        num_inference_steps=50,
        seed=args.seed,
        output_path=outdir,
        save_video=True,
    )
    gen_s = time.perf_counter() - t0

    meta = {
        "kv_window": window,
        "tag": args.tag,
        "num_frames": args.num_frames,
        "latent_frames": latent,
        "seed": args.seed,
        "action_seed": args.action_seed,
        "gen_s": round(gen_s, 2),
        # MUST come from the returned result: generation runs in a spawned
        # worker, so torch.cuda.max_memory_allocated() in THIS process reads a
        # cold allocator and reports 0.0.
        "peak_memory_mb": _extract_peak_mb(result),
    }
    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("\n=== kv window sweep ===")
    for k, v in meta.items():
        print(f"  {k:16}: {v}")
    print(f"  video + meta.json -> {outdir}")


if __name__ == "__main__":
    main()
