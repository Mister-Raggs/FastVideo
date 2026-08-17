#!/usr/bin/env python3
"""
D1 Phase-A runner.

The measurement itself lives in `fastvideo/hooks/kv_probe.py` on the
`perf/causal-kv-policy` branch, because generation runs in a SPAWNED WORKER --
anything monkeypatched in this process never sees the model. This script only:
sets the env the worker reads, runs one generation, then collects the JSON the
worker wrote and prints the summary.

Usage (on the GB10, from the branch checkout):
    python kvpolicy_probe.py --frames 81 --outdir kvprobe_out

num_frames constraint: the causal DMD stage requires the LATENT frame count to
be divisible by num_frames_per_block (=3), and latent t = (frames-1)/4 + 1.
That works out to **num_frames = 9 (mod 12)**: 81, 93, 105, 117, 129, 141, 153,
165. 161 and 121 are both INVALID -- they raise
"num_frames must be divisible by num_frames_per_block".
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time


VALID_HINT = "num_frames must satisfy (frames-1)/4+1 divisible by 3, i.e. frames = 9 mod 12 (81, 93, 105, 117, 129, 141, 153, 165)"


def check_frames(n: int) -> None:
    if (n - 1) % 4 != 0:
        raise SystemExit(f"--frames {n}: not a valid Wan latent count. {VALID_HINT}")
    t = (n - 1) // 4 + 1
    if t % 3 != 0:
        raise SystemExit(f"--frames {n} -> {t} latent frames, {t} % 3 = {t % 3}. {VALID_HINT}")


def summarize(rep: dict, meta: dict) -> None:
    print("\n" + "=" * 74)
    print("D1 Phase-A probe summary")
    print("=" * 74)
    for k, v in meta.items():
        print(f"  {k:26}: {v}")
    print(f"  {'layers seen':26}: {rep['num_layers_seen']}")
    print(f"  {'KV cache est (all layers)':26}: {rep['kv_cache_total_gib_est']} GiB")
    print(f"  {'worker peak mem':26}: {rep.get('peak_mem_gib')} GiB")

    ls = [x for x in rep["layers"] if x.get("age_mass")]
    if not ls:
        print("\n  !! no attention statistics -- check 'errors' in the JSON")
        return

    las = {x.get("local_attn_size") for x in ls}
    sink = {x.get("sink_size") for x in ls}
    kvl = {x.get("kv_len_max") for x in ls}
    print(f"  {'local_attn_size':26}: {las}   <- -1 means the 21-latent-frame hard cap applies")
    print(f"  {'sink_size':26}: {sink}")
    print(f"  {'kv_len_max distinct':26}: {sorted(kvl)}   <- one value == uniform budget (today's policy)")

    nb = len(ls[0]["age_mass"])
    print(f"\n  attention mass by token age ({nb} buckets, bucket 0 = oldest):")
    for x in ls:
        bars = " ".join(f"{m:5.3f}" for m in x["age_mass"])
        hs, hr = x.get("head_sink_mass") or [0], x.get("head_recent_mass") or [0]
        print(f"   L{x['layer']:3d} | {bars} | sink {sum(hs)/len(hs):5.3f} | recent {sum(hr)/len(hr):5.3f}")

    # Bucket-based metrics. These do NOT depend on local_attn_size or sink_size.
    # (head_recent_mass/head_sink_mass are derived from local_attn_size and are
    # structurally 0 when it is -1 -- do not use them for the decision.)
    newest = [x["age_mass"][-1] for x in ls]
    oldest = [x["age_mass"][0] for x in ls]
    uniform = 1.0 / nb

    per_head_spreads = []
    for x in ls:
        ham = x.get("head_age_mass")
        if ham:
            tail = [h[-1] for h in ham]  # each head's mass in the newest bucket
            per_head_spreads.append((x["layer"], max(tail) - min(tail)))

    print("\n  >>> THE NUMBERS THAT DECIDE PHASE B <<<")
    print(f"   uniform-attention baseline per bucket : {uniform:.3f}")
    print(f"   newest-bucket mass  : min {min(newest):.3f} (L{ls[newest.index(min(newest))]['layer']})  "
          f"max {max(newest):.3f} (L{ls[newest.index(max(newest))]['layer']})  "
          f"spread {max(newest) - min(newest):.3f}  mean {sum(newest)/len(newest):.3f}")
    print(f"   oldest-bucket mass  : min {min(oldest):.3f}  max {max(oldest):.3f}  "
          f"spread {max(oldest) - min(oldest):.3f}  mean {sum(oldest)/len(oldest):.3f}")
    # Window-RELATIVE, not absolute. An absolute "oldest bucket > 1.5x uniform"
    # test silently returns [] on a short window (with 6 latent frames the
    # oldest bucket holds ~0.02 everywhere), which would quietly misallocate.
    # Normalising by the model's own mean gives the same layers on both a
    # 21-frame global model and a 6-frame windowed one.
    mean_newest = sum(newest) / len(newest)
    long_range = [x["layer"] for x in ls if x["age_mass"][-1] < 0.7 * mean_newest]
    ranked = sorted(ls, key=lambda x: x["age_mass"][-1])[:5]
    bottom5 = ", ".join("L{}={:.3f}".format(x["layer"], x["age_mass"][-1]) for x in ranked)
    print(f"   recency-light layers (newest < 0.7x mean {mean_newest:.3f}): {long_range}")
    print(f"   bottom-5 by newest-bucket mass: {bottom5}")
    if per_head_spreads:
        worst = max(per_head_spreads, key=lambda t: t[1])
        med = sorted(s for _, s in per_head_spreads)[len(per_head_spreads) // 2]
        print(f"   per-head spread in newest bucket : max {worst[1]:.3f} (L{worst[0]})  median {med:.3f}")
    print("\n   across-layer spread < ~0.05 -> a layer-adaptive budget has nothing to")
    print("   exploit. Large spread + a few long-range layers -> the premise holds.")
    print("=" * 74 + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers")
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--prompt", default="A lone hiker walks along a ridgeline at sunrise, camera slowly tracking "
                    "alongside, clouds drifting through the valley below, cinematic.")
    ap.add_argument("--outdir", default="kvprobe_out")
    ap.add_argument("--output-path", default="kvprobe_samples")
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--queries", type=int, default=32)
    ap.add_argument("--buckets", type=int, default=12)
    ap.add_argument("--max-calls", type=int, default=24)
    ap.add_argument("--save-video", action="store_true")
    ap.add_argument("--summarize", metavar="JSON",
                    help="skip generation; just print the summary for an existing probe JSON")
    args = ap.parse_args()

    if args.summarize:
        with open(args.summarize) as f:
            rep = json.load(f)
        summarize(rep, {"source": args.summarize, "worker pid": rep.get("pid")})
        return

    check_frames(args.frames)

    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    # the worker substitutes its own pid
    os.environ["FASTVIDEO_KV_PROBE"] = "1"
    os.environ["FASTVIDEO_KV_PROBE_OUTPUT"] = os.path.join(outdir, "kvprobe_<pid>.json")
    os.environ["FASTVIDEO_KV_PROBE_HEADS"] = str(args.heads)
    os.environ["FASTVIDEO_KV_PROBE_QUERIES"] = str(args.queries)
    os.environ["FASTVIDEO_KV_PROBE_BUCKETS"] = str(args.buckets)
    os.environ["FASTVIDEO_KV_PROBE_MAX_CALLS"] = str(args.max_calls)

    from fastvideo import VideoGenerator
    from fastvideo.api.sampling_param import SamplingParam

    started = time.time()
    t0 = time.perf_counter()
    generator = VideoGenerator.from_pretrained(
        args.model,
        num_gpus=1,
        use_fsdp_inference=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
    )
    load_s = time.perf_counter() - t0

    sampling_param = SamplingParam.from_pretrained(args.model)
    sampling_param.num_frames = args.frames

    t0 = time.perf_counter()
    generator.generate_video(args.prompt,
                             output_path=args.output_path,
                             save_video=args.save_video,
                             sampling_param=sampling_param)
    gen_s = time.perf_counter() - t0

    # The worker dumps its JSON on atexit, and atexit only fires when the worker
    # process exits -- i.e. at executor shutdown. Polling before that can never
    # succeed, so tear the executor down explicitly first.
    try:
        generator.executor.shutdown()
    except Exception as exc:
        print(f"[warn] explicit executor shutdown failed ({exc!r}); falling back to polling")

    files: list[str] = []
    for _ in range(30):
        files = [f for f in glob.glob(os.path.join(outdir, "kvprobe_*.json")) if os.path.getmtime(f) >= started]
        if files:
            break
        time.sleep(1.0)

    if not files:
        raise SystemExit(f"no probe JSON appeared in {outdir}. Is the branch checked out (fastvideo/hooks/kv_probe.py "
                         f"+ the attach call in composed_pipeline_base.post_init)? Did the worker log "
                         f"'KV probe attached to N causal attention layers'?")

    newest = max(files, key=os.path.getmtime)
    with open(newest) as f:
        rep = json.load(f)

    meta = {
        "model": args.model,
        "frames": args.frames,
        "latent_frames": (args.frames - 1) // 4 + 1,
        "load_s": round(load_s, 2),
        "gen_s": round(gen_s, 2),
        "worker json": os.path.basename(newest),
    }
    summarize(rep, meta)
    print(f"probe JSON: {newest}")
    if len(files) > 1:
        print(f"(note: {len(files)} worker files this run: {[os.path.basename(f) for f in files]})")


if __name__ == "__main__":
    main()
