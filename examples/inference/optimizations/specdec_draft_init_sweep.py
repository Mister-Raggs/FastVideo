"""D5b: is a distilled model's x0 a better starting point than noise?

Runs one phase per invocation so each model gets a clean process (loading two
DiTs in one process is unnecessary here -- we compare quality, and the per-step
costs are already known).

    # 1. draft: FastWan 3-step -> x0_hat, saved as a latent
    python specdec_draft_init_sweep.py draft --out ~/specdec

    # 2. reference: full Wan at 50 steps, the thing everything is scored against
    python specdec_draft_init_sweep.py reference --out ~/specdec

    # 3. the sweep: for each strength, BOTH arms
    python specdec_draft_init_sweep.py sweep --out ~/specdec

    # 4. score every arm against the reference
    python specdec_draft_init_sweep.py score --out ~/specdec

**The comparison that matters is not "faster than 50 steps".** Running plain Wan
with fewer steps is free and needs no code, so every draft-init arm is paired
with an equal-compute control: plain Wan from noise at the same total step
budget. Draft-init only means anything if it beats that control.

Compute accounting (GB10 Phase-0 numbers): full Wan denoise is 13.1 s/step,
FastWan is ~5.5 s/step, so the 3-step draft costs ~1.3 full-Wan-steps. An arm
running N full steps after the draft is charged N + 1.3 steps, and its control
is plain Wan at round(N + 1.3) steps.

Every arm shares one seed. Height/width/frames are pinned across models rather
than taken from each model's own SamplingParam, otherwise the draft latent will
not match the shape the verifier prepares.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch

from fastvideo import SamplingParam, VideoGenerator

DRAFT_MODEL = "FastVideo/FastWan2.1-T2V-1.3B-Diffusers"
VERIFIER_MODEL = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

# Pinned across every arm -- see module docstring.
SEED = 42
HEIGHT, WIDTH, NUM_FRAMES = 480, 832, 81
REFERENCE_STEPS = 50

# Cost of the 3-step draft, in units of full-Wan steps (13.1 vs ~5.5 s/step).
DRAFT_COST_IN_VERIFIER_STEPS = 1.3

# FastWan ships VSA-trained gate weights and is meant to run with VSA; the
# verifier (stock Wan) is dense. That asymmetry is not a confound -- the draft
# under test IS the distilled model as shipped -- but every verifier step in
# both arms is dense, so the arms stay comparable.
DRAFT_VSA_SPARSITY = 0.8

STRENGTHS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7]

# The strength the eyeball phases render. 0.3 is the interesting middle: the
# verifier runs 15 of 50 steps, enough compute that it should show if it is
# doing anything, while still a 3x nominal saving.
EYEBALL_STRENGTH = 0.3

PROMPT = ("A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes "
          "wide with interest. The playful yet serene atmosphere is complemented by soft "
          "natural light filtering through the petals. Mid-shot, warm and cheerful tones.")


def _sampling_param(model: str, steps: int | None = None) -> SamplingParam:
    param = SamplingParam.from_pretrained(model)
    param.seed = SEED
    param.height, param.width, param.num_frames = HEIGHT, WIDTH, NUM_FRAMES
    if steps is not None:
        param.num_inference_steps = steps
    return param


def _generator(model: str, output_type: str = "pil", vsa_sparsity: float | None = None) -> VideoGenerator:
    """Build a generator. Offload is left at FastVideo defaults.

    FastWan checkpoints ship VSA gate weights (``to_gate_compress``), and the
    attention backend must be selected BEFORE the transformer is built --
    otherwise the DiT is constructed gateless and loading dies with
    "Parameter blocks.0.to_gate_compress.bias not found in custom model state
    dict". This mirrors examples/inference/basic/basic_dmd.py.
    """
    kwargs = {}
    if vsa_sparsity is not None:
        os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "VIDEO_SPARSE_ATTN"
        kwargs["VSA_sparsity"] = vsa_sparsity
    else:
        # Each phase runs in its own process, but stay explicit: the dense
        # verifier must not inherit a VSA backend selection.
        os.environ.pop("FASTVIDEO_ATTENTION_BACKEND", None)
    return VideoGenerator.from_pretrained(model,
                                          num_gpus=1,
                                          use_fsdp_inference=False,
                                          output_type=output_type,
                                          **kwargs)


def _latent_from_result(result) -> torch.Tensor:
    """Mirrors fastvideo/tests/ssim/latent_similarity_utils._extract_latent_from_result."""
    if not isinstance(result, dict):
        raise RuntimeError(f"generate_video returned {type(result)!r}; expected dict with 'samples'")
    samples = result.get("samples")
    if samples is None:
        raise RuntimeError("No latent samples returned. output_type='latent' requires return_frames=True.")
    latent = samples.detach().to(torch.float32).cpu()
    if latent.dim() != 5:
        raise RuntimeError(f"Expected a 5-D (B,C,T,H,W) video latent; got {tuple(latent.shape)}")
    return latent


def _generate(gen: VideoGenerator, param: SamplingParam, **extra):
    # return_frames=True is REQUIRED alongside output_type='latent' -- without
    # it the pipeline skips the samples buffer and `samples` comes back None,
    # but only after paying the full generation cost.
    return gen.generate_video(PROMPT, sampling_param=param, save_video=False, return_frames=True, **extra)


def cmd_draft(out: Path) -> None:
    """FastWan 3 steps -> x0_hat. Its own config supplies the DMD schedule."""
    gen = _generator(DRAFT_MODEL, output_type="latent", vsa_sparsity=DRAFT_VSA_SPARSITY)
    param = _sampling_param(DRAFT_MODEL)
    start = time.perf_counter()
    result = _generate(gen, param)
    wall = time.perf_counter() - start

    latent = _latent_from_result(result)
    torch.save({"latent": latent, "wall_s": wall, "model": DRAFT_MODEL}, out / "draft_latent.pt")
    print(f"[draft] {tuple(latent.shape)} {latent.dtype} in {wall:.1f}s -> {out / 'draft_latent.pt'}")


def cmd_reference(out: Path) -> None:
    gen = _generator(VERIFIER_MODEL, output_type="latent")
    param = _sampling_param(VERIFIER_MODEL, steps=REFERENCE_STEPS)
    start = time.perf_counter()
    result = _generate(gen, param)
    wall = time.perf_counter() - start

    latent = _latent_from_result(result)
    torch.save({"latent": latent, "wall_s": wall, "steps": REFERENCE_STEPS}, out / "reference_latent.pt")
    print(f"[reference] {REFERENCE_STEPS} steps, {tuple(latent.shape)} in {wall:.1f}s")


def cmd_sweep(out: Path) -> None:
    """Both arms per strength, in one process so the model loads once."""
    draft_path = out / "draft_latent.pt"
    if not draft_path.exists():
        raise SystemExit(f"missing {draft_path}; run the `draft` phase first")
    draft_latent = torch.load(draft_path)["latent"]

    gen = _generator(VERIFIER_MODEL, output_type="latent")
    records = []

    for strength in STRENGTHS:
        n_full = max(1, round(REFERENCE_STEPS * strength))

        # Arm A: start from the draft's x0_hat, run only the schedule tail.
        param = _sampling_param(VERIFIER_MODEL, steps=REFERENCE_STEPS)
        start = time.perf_counter()
        result = _generate(gen, param, denoise_strength=strength, init_latents=draft_latent)
        wall_a = time.perf_counter() - start
        torch.save(_latent_from_result(result), out / f"arm_draftinit_s{strength:.2f}.pt")

        # Arm B: equal-compute control -- plain Wan from noise, charged for the
        # draft it did not run.
        control_steps = max(1, round(n_full + DRAFT_COST_IN_VERIFIER_STEPS))
        param_b = _sampling_param(VERIFIER_MODEL, steps=control_steps)
        start = time.perf_counter()
        result_b = _generate(gen, param_b)
        wall_b = time.perf_counter() - start
        torch.save(_latent_from_result(result_b), out / f"arm_control_s{strength:.2f}.pt")

        records.append({
            "strength": strength,
            "draftinit_full_steps": n_full,
            "draftinit_charged_steps": round(n_full + DRAFT_COST_IN_VERIFIER_STEPS, 2),
            "control_steps": control_steps,
            "draftinit_wall_s": round(wall_a, 2),
            "control_wall_s": round(wall_b, 2),
        })
        print(f"[sweep] strength={strength:.2f}  draft-init {n_full} steps ({wall_a:.1f}s)  "
              f"vs control {control_steps} steps ({wall_b:.1f}s)")

    (out / "sweep_arms.json").write_text(json.dumps(records, indent=2))


def cmd_eyeball_draft(out: Path) -> None:
    """Render FastWan's raw 3-step output. Separate process from the Wan arms:
    the VSA backend is selected by an env var read at model-build time."""
    videos = out / "videos"
    videos.mkdir(exist_ok=True)
    gen = _generator(DRAFT_MODEL, vsa_sparsity=DRAFT_VSA_SPARSITY)
    param = _sampling_param(DRAFT_MODEL)
    gen.generate_video(PROMPT, sampling_param=param, output_path=str(videos / "a_draft_fastwan_3step"),
                       save_video=True)
    print(f"[eyeball] draft -> {videos / 'a_draft_fastwan_3step'}")


def cmd_eyeball_wan(out: Path) -> None:
    """Render draft-init and its equal-compute control at EYEBALL_STRENGTH."""
    videos = out / "videos"
    videos.mkdir(exist_ok=True)
    draft_latent = torch.load(out / "draft_latent.pt")["latent"]
    n_full = max(1, round(REFERENCE_STEPS * EYEBALL_STRENGTH))
    control_steps = max(1, round(n_full + DRAFT_COST_IN_VERIFIER_STEPS))

    gen = _generator(VERIFIER_MODEL)

    param = _sampling_param(VERIFIER_MODEL, steps=REFERENCE_STEPS)
    gen.generate_video(PROMPT, sampling_param=param,
                       output_path=str(videos / f"b_draftinit_s{EYEBALL_STRENGTH:.2f}"), save_video=True,
                       denoise_strength=EYEBALL_STRENGTH, init_latents=draft_latent)
    print(f"[eyeball] draft-init ({n_full} verifier steps) -> b_draftinit_s{EYEBALL_STRENGTH:.2f}")

    param_b = _sampling_param(VERIFIER_MODEL, steps=control_steps)
    gen.generate_video(PROMPT, sampling_param=param_b,
                       output_path=str(videos / f"c_control_{control_steps}step"), save_video=True)
    print(f"[eyeball] control ({control_steps} steps) -> c_control_{control_steps}step")
    print("\nCompare a_draft vs b_draftinit: if they are indistinguishable, the "
          "verifier steps bought nothing and D5b closes.\nThen b_draftinit vs "
          "c_control: that is the actual thesis, at equal compute.")


def _cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    a32, b32 = a.flatten().float(), b.flatten().float()
    return float(1.0 - torch.nn.functional.cosine_similarity(a32.unsqueeze(0), b32.unsqueeze(0), dim=1).item())


def cmd_score(out: Path) -> None:
    """Score every arm against the 50-step reference latent.

    Cosine distance on latents is the repo's own convention for this comparison
    (fastvideo/tests/ssim/latent_similarity_utils.py). It is the SCREENING
    signal: if it does not move monotonically with strength, the direction is
    unscoreable and stops here, before anyone builds on it.
    """
    reference = torch.load(out / "reference_latent.pt")["latent"]
    arms = json.loads((out / "sweep_arms.json").read_text())

    rows = []
    for rec in arms:
        s = rec["strength"]
        row = dict(rec)
        for arm in ("draftinit", "control"):
            path = out / f"arm_{arm}_s{s:.2f}.pt"
            row[f"{arm}_cosine_dist"] = round(_cosine_distance(torch.load(path), reference), 6)
        row["draftinit_wins"] = row["draftinit_cosine_dist"] < row["control_cosine_dist"]
        rows.append(row)

    (out / "scores.json").write_text(json.dumps(rows, indent=2))

    print(f"\n{'strength':>9} {'steps(A/B)':>12} {'draft-init':>12} {'control':>12}  winner")
    for r in rows:
        winner = "DRAFT-INIT" if r["draftinit_wins"] else "control"
        print(f"{r['strength']:>9.2f} {str(r['draftinit_charged_steps']) + '/' + str(r['control_steps']):>12} "
              f"{r['draftinit_cosine_dist']:>12.6f} {r['control_cosine_dist']:>12.6f}  {winner}")

    dists = [r["draftinit_cosine_dist"] for r in rows]
    monotonic = all(x >= y for x, y in zip(dists, dists[1:], strict=False))
    wins = sum(r["draftinit_wins"] for r in rows)
    print(f"\nmonotonic in strength: {monotonic}  (if False, the signal is untrustworthy -- stop)")
    print(f"draft-init beats equal-compute control in {wins}/{len(rows)} arms")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",
                        choices=["draft", "reference", "sweep", "score", "eyeball-draft", "eyeball-wan"])
    parser.add_argument("--out", default="specdec_out")
    args = parser.parse_args()

    out = Path(os.path.expanduser(args.out))
    out.mkdir(parents=True, exist_ok=True)

    {
        "draft": cmd_draft,
        "reference": cmd_reference,
        "sweep": cmd_sweep,
        "score": cmd_score,
        "eyeball-draft": cmd_eyeball_draft,
        "eyeball-wan": cmd_eyeball_wan,
    }[args.phase](out)


if __name__ == "__main__":
    main()
