"""
Unconflated head-to-head: EasyCache (forward-level, model-agnostic) vs cache-dit
(block-level, per-model adapter), in ONE container against ONE shared baseline.

Why this harness exists (and not two separate runs of easycache_ab.py /
cachedit_ab.py): comparing a ``-28%`` from one run against a ``-X%`` from another
silently assumes the two runs' baselines were identical. They never quite are
(different builds, different baseline passes, eyeballed quality matching). Here a
SINGLE build (the ``compare/easycache-vs-cachedit`` branch wires both caching
paths, mutually exclusive at runtime) runs:

    baseline (both caches OFF)  ->  the one shared denominator
    easycache @ --ec-thresh
    cachedit  @ each --cd-thresholds (+ TaylorSeer)

Every method's SSIM is computed against the SAME baseline output, and every wall
is a fraction of the SAME baseline wall. No cross-run drift, nothing to "trust".

Both caches are LOSSY: SSIM is a quality cost (target >= ~0.95), not a 1.0 gate.
The "win" is the wall delta. The honest comparison is speedup at MATCHED SSIM —
sweep ``--cd-thresholds`` so one cache-dit point lands near the EasyCache point.

Usage (from the repo root of the compare worktree):

    # Wan2.1 on H100: EasyCache @0.015 vs cache-dit @{0.08,0.15}+TaylorSeer
    modal run --detach fastvideo/tests/modal/easycache_vs_cachedit_ab.py \
        --gpu H100 --num-prompts 2

Requires a Modal Secret named ``huggingface-token`` with key ``HF_TOKEN``.
"""
import os

import modal

app = modal.App("easycache-vs-cachedit-ab")

model_vol = modal.Volume.from_name("hf-model-weights")
hf_secret = modal.Secret.from_name("huggingface-token")
image_tag = f"ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:{os.getenv('IMAGE_VERSION', 'py3.12-latest')}"

# Prebuilt FA3 wheel matching the fastvideo-dev image ABI (Hopper leg only).
FA3_WHEEL_URL = (
    "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/"
    "flash_attn_3-3.0.0%2Bcu128torch2.11gite2743ab-cp39-abi3-linux_x86_64.whl")

image = (modal.Image.from_registry(image_tag, add_python="3.12").apt_install(
    "cmake", "pkg-config", "build-essential", "curl", "libssl-dev", "ffmpeg", "libgl1", "libglib2.0-0",
).run_commands(
    "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable",
    "echo 'source ~/.cargo/env' >> ~/.bashrc",
).env({"PATH": "/root/.cargo/bin:$PATH", "HF_HUB_ENABLE_HF_TRANSFER": "1"}))

# Same five prompts + seed as every prior Wan A/B so deltas compare directly.
AB_PROMPTS = [
    "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes wide with interest. "
    "The playful yet serene atmosphere is complemented by soft natural light filtering through the petals. "
    "Mid-shot, warm and cheerful tones.",
    "A majestic lion strides across the golden savanna, its powerful frame glistening under the warm afternoon "
    "sun. The tall grass ripples gently in the breeze, enhancing the lion's commanding presence. Low angle, "
    "steady tracking shot, cinematic.",
    "A sailing ship cuts through dark stormy waves under a sky of rolling thunderclouds. Sea spray catches the "
    "lantern light along the hull. Dramatic, painterly tones; medium-wide tracking shot.",
    "A street vendor in Tokyo flips okonomiyaki on a sizzling iron griddle while neon shop signs reflect in "
    "rain-slicked pavement. Steam rises around her hands; warm color grade; handheld close-up.",
    "An astronaut floats slowly past the cupola of a space station, the curve of Earth glowing blue beyond "
    "the glass. Calm, contemplative pacing; smooth dolly; cinematic.",
]
AB_SEED = 42

MODEL_PRESETS = {
    "wan2_1-t2v-1.3b": {
        "model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "height": 720, "width": 1280, "num_frames": 77, "num_inference_steps": 30, "default_num_gpus": 1,
    },
    # Footnote only (Will scoped the comparison to Wan2.1). The A14B MoE swaps
    # experts at the boundary; EasyCache is a weaker fit there. 480p keeps the
    # two-resident-expert footprint inside H100:2 with offload off (720p OOMs),
    # and matches the resolution the prior 0.68 Wan2.2 number was measured at.
    "wan2_2-t2v-14b": {
        "model_id": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "height": 480, "width": 832, "num_frames": 49, "num_inference_steps": 30, "default_num_gpus": 2,
    },
}

# Inner runners (already in the tree). Both take --config-json/--results-json and
# emit [{i, wall_s, mp4}]; both expose --mode compute_ssim. We always score SSIM
# with the EasyCache inner so the metric is identical across methods.
_EC_INNER = "/FastVideo/fastvideo/tests/modal/_easycache_ab_inner.py"
_CD_INNER = "/FastVideo/fastvideo/tests/modal/_cachedit_ab_inner.py"


def _build_workspace_command(git_repo: str, git_ref: str, install_fa3: bool) -> str:
    import shlex
    fa3_step = f'uv pip install --no-cache-dir "{FA3_WHEEL_URL}"' if install_fa3 else 'echo "[fa3] skipped"'
    return f"""
set -euxo pipefail
source $HOME/.local/bin/env
source $HOME/.cargo/env
source /opt/venv/bin/activate
if [ -d /FastVideo/.git ]; then
  cd /FastVideo && git remote set-url origin {shlex.quote(git_repo)} && git fetch --prune origin
else
  git clone {shlex.quote(git_repo)} /FastVideo && cd /FastVideo
fi
git checkout {shlex.quote(git_ref)}
git submodule update --init --recursive
{fa3_step}
uv pip install -e ".[test,cache]"
uv pip install hf_transfer
cd fastvideo-kernel && ./build.sh && cd ..
export HF_HOME=/root/data/.cache
hf auth login --token "$HF_TOKEN"
"""


def _run_inner(inner: str, config: dict) -> list[dict]:
    import json
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        config_path = f.name
    results_path = config_path.replace(".json", ".results.json")
    cmd = (f"source /opt/venv/bin/activate && exec python {inner} "
           f"--config-json {config_path} --results-json {results_path}")
    try:
        subprocess.run(["/bin/bash", "-lc", cmd], check=True)
        with open(results_path) as f:
            return json.load(f)
    finally:
        for p in (config_path, results_path):
            try:
                os.unlink(p)
            except OSError:
                pass


def _common_cfg(preset: dict, model_id: str, num_gpus: int, enable_compile: bool, selected: list[str]) -> dict:
    return {
        "model_id": model_id, "num_gpus": num_gpus, "enable_compile": enable_compile,
        "height": preset["height"], "width": preset["width"], "num_frames": preset["num_frames"],
        "num_inference_steps": preset["num_inference_steps"], "prompts": selected, "seed_base": AB_SEED,
    }


def _run_baseline(common: dict, output_dir: str) -> list[dict]:
    cfg = {**common, "enable_easycache": False, "output_dir": output_dir}
    return _run_inner(_EC_INNER, cfg)


def _run_easycache(common: dict, output_dir: str, thresh: float, warmup: int, tail: int) -> list[dict]:
    cfg = {**common, "enable_easycache": True, "easycache_thresh": thresh, "easycache_warmup": warmup,
           "easycache_tail": tail, "output_dir": output_dir}
    return _run_inner(_EC_INNER, cfg)


def _run_cachedit(common: dict, output_dir: str, threshold: float, fn: int, bn: int, warmup: int,
                  taylorseer: bool, taylorseer_order: int) -> list[dict]:
    cfg = {**common, "use_cachedit": True, "cachedit_fn_compute_blocks": fn, "cachedit_bn_compute_blocks": bn,
           "cachedit_residual_threshold": threshold, "cachedit_max_warmup_steps": warmup,
           "cachedit_taylorseer": taylorseer, "cachedit_taylorseer_order": taylorseer_order,
           "output_dir": output_dir}
    return _run_inner(_CD_INNER, cfg)


def _ssim_vs_baseline(baseline: list[dict], patched: list[dict]) -> list[dict]:
    """Pairwise SSIM of each patched output against the SHARED baseline output."""
    import json
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".baseline.json", delete=False) as f:
        json.dump(baseline, f)
        bpath = f.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".patched.json", delete=False) as f:
        json.dump(patched, f)
        ppath = f.name
    spath = bpath.replace(".baseline.json", ".ssim.json")
    cmd = (f"source /opt/venv/bin/activate && exec python {_EC_INNER} --mode compute_ssim "
           f"--baseline-results-json {bpath} --patched-results-json {ppath} --ssim-output-json {spath}")
    try:
        subprocess.run(["/bin/bash", "-lc", cmd], check=True)
        with open(spath) as f:
            return json.load(f)
    finally:
        for p in (bpath, ppath, spath):
            try:
                os.unlink(p)
            except OSError:
                pass


def _summarize(rows: list[dict]) -> dict:
    base = sum(r["baseline_wall_s"] for r in rows)
    patched = sum(r["patched_wall_s"] for r in rows)
    means = [r["ssim_mean"] for r in rows]
    worsts = [r["ssim_worst"] for r in rows]
    return {
        "baseline_total_s": base, "patched_total_s": patched,
        "wall_delta_pct": (patched - base) / base * 100.0 if base else 0.0,
        "ssim_mean": sum(means) / len(means) if means else 0.0,
        "ssim_worst": min(worsts) if worsts else 0.0,
    }


def _print_report(model_preset: str, method_summaries: list[tuple[str, dict]]) -> None:
    print(f"\n================ EasyCache vs cache-dit — {model_preset} ================")
    print("ONE shared baseline; every row scored against it. Both caches lossy (SSIM = quality cost).")
    print("Negative wall delta = faster. Honest comparison = speedup at MATCHED SSIM.\n")
    print(f"{'method':<28} {'wall_total_s':>13} {'delta_pct':>10} {'ssim_mean':>10} {'ssim_worst':>11}")
    for label, s in method_summaries:
        print(f"{label:<28} {s['patched_total_s']:>13.3f} {s['wall_delta_pct']:>9.2f}% "
              f"{s['ssim_mean']:>10.6f} {s['ssim_worst']:>11.6f}")
    if method_summaries:
        print(f"\n(shared baseline total: {method_summaries[0][1]['baseline_total_s']:.3f}s)")
    print("\nGenerality (independent of the table): cache-dit caches at the BLOCK level and "
          "needs a per-model BlockAdapter; EasyCache caches the whole-forward residual and "
          "needs NOTHING model-specific — one module for every model.")


@app.function(image=image, timeout=14400, volumes={"/root/data": model_vol}, secrets=[hf_secret], gpu="H100:1")
def run_ab(*, git_repo: str, git_ref: str, model_preset: str, num_gpus: int, install_fa3: bool,
           enable_compile: bool, num_prompts: int, ec_thresholds: list[float], ec_warmup: int, ec_tail: int,
           cd_thresholds: list[float], cd_fn: int, cd_bn: int, cd_warmup: int, taylorseer: bool,
           taylorseer_order: int) -> dict:
    import subprocess

    if model_preset not in MODEL_PRESETS:
        raise ValueError(f"unknown model_preset: {model_preset}")
    if "HF_TOKEN" not in os.environ:
        raise RuntimeError("HF_TOKEN not set — Modal Secret 'huggingface-token' missing the HF_TOKEN key.")
    preset = MODEL_PRESETS[model_preset]

    result = subprocess.run(["/bin/bash", "-lc", _build_workspace_command(git_repo, git_ref, install_fa3)],
                            env=os.environ.copy(), capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"workspace setup failed (rc={result.returncode})")
    os.chdir("/FastVideo")

    selected = list(AB_PROMPTS) if num_prompts <= 0 else list(AB_PROMPTS)[:num_prompts]
    common = _common_cfg(preset, preset["model_id"], num_gpus, enable_compile, selected)
    out_root = f"/root/data/ec_vs_cd_out/{model_preset}"

    # The one shared baseline — every method below is scored against THIS.
    print(f"\n--- baseline pass (both caches OFF) on {model_preset} ---")
    baseline = _run_baseline(common, f"{out_root}/baseline")

    method_summaries: list[tuple[str, dict]] = []

    for thr in ec_thresholds:
        print(f"\n--- EasyCache pass (thresh={thr} warmup={ec_warmup} tail={ec_tail}) ---")
        try:
            ec = _run_easycache(common, f"{out_root}/easycache_t{thr}", thr, ec_warmup, ec_tail)
            method_summaries.append((f"easycache t={thr}", _summarize(_ssim_vs_baseline(baseline, ec))))
        except Exception as e:  # noqa: BLE001 — one bad EasyCache point must not lose the rest
            print(f"[WARN] EasyCache pass at threshold={thr} failed, skipping: {e}")

    for thr in cd_thresholds:
        ts = "+ts" if taylorseer else ""
        print(f"\n--- cache-dit pass (threshold={thr} fn={cd_fn} bn={cd_bn} taylorseer={taylorseer}) ---")
        try:
            cd = _run_cachedit(common, f"{out_root}/cachedit_t{thr}", thr, cd_fn, cd_bn, cd_warmup, taylorseer,
                               taylorseer_order)
            method_summaries.append((f"cachedit t={thr}{ts}", _summarize(_ssim_vs_baseline(baseline, cd))))
        except Exception as e:  # noqa: BLE001 — one bad cache-dit point must not lose the EasyCache rows
            print(f"[WARN] cache-dit pass at threshold={thr} failed, skipping: {e}")

    _print_report(model_preset, method_summaries)
    return {"model_preset": model_preset, "methods": method_summaries}


@app.local_entrypoint()
def main(gpu: str = "H100", model: str = "wan2_1-t2v-1.3b", num_gpus: int = 0, git_repo: str = "",
         git_ref: str = "compare/easycache-vs-cachedit", enable_compile: bool = False, num_prompts: int = 0,
         ec_thresholds: str = "0.015", ec_warmup: int = 1, ec_tail: int = 1, cd_thresholds: str = "0.08,0.15",
         cd_fn: int = 8, cd_bn: int = 0, cd_warmup: int = 8, taylorseer: bool = True, taylorseer_order: int = 1):
    """Drive the unconflated EasyCache-vs-cache-dit A/B. ``--ec-thresholds`` and
    ``--cd-thresholds`` are comma lists (sweep to land matched-SSIM points so the
    comparison is speedup-at-matched-quality). Pass ``--cd-thresholds ""`` to skip
    cache-dit (e.g. extend the EasyCache frontier against a baseline you've already
    paired cache-dit to). ``git_repo`` defaults to ``fork``."""
    import subprocess

    if model not in MODEL_PRESETS:
        raise ValueError(f"unknown model: {model}; choose from {list(MODEL_PRESETS)}")
    if num_gpus == 0:
        num_gpus = MODEL_PRESETS[model]["default_num_gpus"]
    if not git_repo:
        for remote in ("fork", "origin"):
            try:
                git_repo = subprocess.check_output(["git", "config", "--get", f"remote.{remote}.url"],
                                                   text=True, stderr=subprocess.DEVNULL).strip()
                break
            except subprocess.CalledProcessError:
                continue
        if not git_repo:
            raise RuntimeError("Could not resolve git_repo. Pass --git-repo or configure a 'fork'/'origin' remote.")

    ec_list = [float(x) for x in ec_thresholds.split(",") if x.strip()]
    cd_list = [float(x) for x in cd_thresholds.split(",") if x.strip()]
    install_fa3 = gpu.upper().startswith("H100") or gpu.upper().startswith("H200")
    print(f"GPU: {gpu}:{num_gpus}  model: {model}  ref: {git_ref}  install_fa3: {install_fa3}  "
          f"prompts: {num_prompts or 'all'}  easycache: t={ec_list}  "
          f"cachedit: t={cd_list} taylorseer={taylorseer}(o{taylorseer_order})  repo: {git_repo}")

    run_ab.with_options(gpu=f"{gpu}:{num_gpus}").remote(
        git_repo=git_repo, git_ref=git_ref, model_preset=model, num_gpus=num_gpus, install_fa3=install_fa3,
        enable_compile=enable_compile, num_prompts=num_prompts, ec_thresholds=ec_list, ec_warmup=ec_warmup,
        ec_tail=ec_tail, cd_thresholds=cd_list, cd_fn=cd_fn, cd_bn=cd_bn, cd_warmup=cd_warmup,
        taylorseer=taylorseer, taylorseer_order=taylorseer_order)
