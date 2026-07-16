"""
A/B harness for Cosmos-2.5 2B linear quantization (fp8 / nvfp4_qat).

Single-container pattern: clone the branch, build once, then run the same N
seed-pinned prompts twice — once bf16 (baseline) and once with
``transformer_quant`` set — and print per-prompt wall + pairwise SSIM.

Quantization is LOSSY, and Cosmos-2.5 is a full-step CFG model (the preset
runs guidance 7.0 / 35 steps), so quant error accumulates across every
denoise step: the SSIM column is the quality cost (target >= ~0.95), NOT a
1.0 correctness gate. The "win" is the wall delta.

Usage (from FastVideo repo root):

    # fp8 per-tensor vs bf16 on L40S (the default arm)
    modal run fastvideo/tests/modal/cosmos25_quant_ab.py --gpu L40S

    # per-channel weight scales (higher accuracy, slower)
    modal run fastvideo/tests/modal/cosmos25_quant_ab.py --granularity channel

    # nvfp4 needs Blackwell — run that arm on the DGX Spark instead
    # (see the Spark runbook); this Modal harness is the fp8/sm89 leg.

Requires a Modal Secret named ``huggingface-token`` with key ``HF_TOKEN``.
"""
import os

import modal

app = modal.App("cosmos25-quant-ab")

model_vol = modal.Volume.from_name("hf-model-weights")
hf_secret = modal.Secret.from_name("huggingface-token")
image_tag = f"ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:{os.getenv('IMAGE_VERSION', 'py3.12-latest')}"

image = (modal.Image.from_registry(image_tag, add_python="3.12").apt_install(
    "cmake", "pkg-config", "build-essential", "curl", "libssl-dev", "ffmpeg", "libgl1", "libglib2.0-0",
).run_commands(
    "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable",
    "echo 'source ~/.cargo/env' >> ~/.bashrc",
).env({"PATH": "/root/.cargo/bin:$PATH"}))

# Same prompts as the prior Wan A/Bs so deltas compare directly.
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

MODEL_ID = "KyleShao/Cosmos-Predict2.5-2B-Diffusers"


# TEMP until hao-ai-lab/FastVideo#1607 merges: reason1's compute_text_embeddings
# rejects the BatchEncoding that new transformers 5.x returns from
# apply_chat_template, so every Cosmos-2.5 run on current main crashes in the
# text-encoding stage. Cherry-pick the fix commits on top of the A/B ref.
# Drop this (and the fetch fallback) once the branch rebases past the merge.
REASON1_FIX_COMMITS = ("bcfa81e4291caa6ab06734e4b098566d21223ad6", "0a6f4ffd93520e4672c5a55f95da939d7a983fab")
REASON1_FIX_BRANCH = "fix/reason1-batchencoding"


def _build_workspace_command(git_repo: str, git_ref: str) -> str:
    import shlex
    fix_shas = " ".join(REASON1_FIX_COMMITS)
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
if ! git merge-base --is-ancestor {REASON1_FIX_COMMITS[0]} HEAD; then
  git fetch origin {REASON1_FIX_BRANCH} || git fetch origin pull/1607/head
  git -c user.name=ab -c user.email=ab@local cherry-pick {fix_shas}
fi
git submodule update --init --recursive
uv pip install -e ".[test]"
cd fastvideo-kernel && ./build.sh && cd ..
export HF_HOME=/root/data/.cache
hf auth login --token "$HF_TOKEN"
"""


def _run_pass(*, quant_method: str | None, granularity: str, num_gpus: int, output_dir: str,
              enable_compile: bool, num_prompts, num_inference_steps: int) -> list[dict]:
    import json
    import subprocess
    import tempfile

    selected = list(AB_PROMPTS) if num_prompts is None else list(AB_PROMPTS)[:num_prompts]
    config = {
        "model_id": MODEL_ID, "num_gpus": num_gpus, "quant_method": quant_method, "granularity": granularity,
        "enable_compile": enable_compile, "num_inference_steps": num_inference_steps,
        "output_dir": output_dir, "prompts": selected, "seed_base": AB_SEED,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        config_path = f.name
    results_path = config_path.replace(".json", ".results.json")
    inner = "/FastVideo/fastvideo/tests/modal/_cosmos25_quant_ab_inner.py"
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


def _pairwise_ssim(baseline: list[dict], patched: list[dict]) -> list[dict]:
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
    inner = "/FastVideo/fastvideo/tests/modal/_cosmos25_quant_ab_inner.py"
    cmd = (f"source /opt/venv/bin/activate && exec python {inner} --mode compute_ssim "
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


def _print_table(rows: list[dict], label: str) -> None:
    print(f"\n=== Cosmos-2.5 quant A/B results ({label}) ===")
    print("NOTE: quantization is lossy — SSIM is a quality cost (target >= ~0.95), not a 1.0 gate.")
    print(f"{'prompt':>6} {'baseline_wall_s':>16} {'quant_wall_s':>14} {'delta_pct':>10} {'ssim_mean':>10} "
          f"{'ssim_worst':>11}")
    base_sum = patched_sum = 0.0
    means, worsts = [], []
    for r in rows:
        delta = (r["patched_wall_s"] - r["baseline_wall_s"]) / r["baseline_wall_s"] * 100.0
        print(f"{r['i']:>6} {r['baseline_wall_s']:>16.3f} {r['patched_wall_s']:>14.3f} {delta:>9.2f}% "
              f"{r['ssim_mean']:>10.6f} {r['ssim_worst']:>11.6f}")
        base_sum += r["baseline_wall_s"]
        patched_sum += r["patched_wall_s"]
        means.append(r["ssim_mean"])
        worsts.append(r["ssim_worst"])
    print(f"\nbaseline total: {base_sum:.3f}s")
    print(f"quant total:    {patched_sum:.3f}s")
    print(f"wall delta:     {(patched_sum - base_sum) / base_sum * 100.0:+.2f}%   (negative = faster)")
    print(f"SSIM mean-of-means:  {sum(means) / len(means):.6f}")
    print(f"SSIM worst-of-worst: {min(worsts):.6f}")


@app.function(image=image, timeout=14400, volumes={"/root/data": model_vol}, secrets=[hf_secret], gpu="L40S:1")
def run_ab(*, git_repo: str, git_ref: str, quant_method: str, granularity: str, num_gpus: int,
           enable_compile: bool = False, num_prompts: int = 0, num_inference_steps: int = 0):
    import subprocess

    if "HF_TOKEN" not in os.environ:
        raise RuntimeError("HF_TOKEN not set — Modal Secret 'huggingface-token' missing the HF_TOKEN key.")

    result = subprocess.run(["/bin/bash", "-lc", _build_workspace_command(git_repo, git_ref)],
                            env=os.environ.copy(), capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"workspace setup failed (rc={result.returncode})")
    os.chdir("/FastVideo")

    common = dict(granularity=granularity, num_gpus=num_gpus, enable_compile=enable_compile,
                  num_prompts=(num_prompts if num_prompts > 0 else None),
                  num_inference_steps=num_inference_steps)

    cfg_tag = quant_method.lower() + (f"_{granularity}" if quant_method == "FP8" else "")
    baseline_dir = "/root/data/cosmos25_quant_out/baseline"
    patched_dir = f"/root/data/cosmos25_quant_out/{cfg_tag}"
    label = f"cosmos2.5-2b | {cfg_tag} vs bf16 | compile={enable_compile}"

    print("\n--- baseline pass (bf16) ---")
    baseline = _run_pass(quant_method=None, output_dir=baseline_dir, **common)
    print(f"\n--- quant pass ({label}) ---")
    patched = _run_pass(quant_method=quant_method, output_dir=patched_dir, **common)

    rows = _pairwise_ssim(baseline, patched)
    _print_table(rows, label)
    return rows


@app.local_entrypoint()
def main(gpu: str = "L40S", quant_method: str = "FP8", granularity: str = "tensor", num_gpus: int = 1,
         git_repo: str = "", git_ref: str = "perf/cosmos-linear-quant", enable_compile: bool = False,
         num_prompts: int = 3, num_inference_steps: int = 0):
    """Drive the Cosmos-2.5 quant A/B from your laptop. ``--quant-method FP8``
    (default; sm89+). nvfp4_qat needs Blackwell — use the Spark runbook for
    that leg. ``--num-inference-steps 0`` keeps the model preset's step count
    (guidance 7.0 / 35 steps). ``git_repo`` defaults to the ``fork`` remote."""
    import subprocess

    if quant_method not in ("FP8", "nvfp4_qat"):
        raise ValueError(f"unknown quant_method: {quant_method}")
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

    print(f"GPU: {gpu}:{num_gpus}  quant: {quant_method}({granularity})  ref: {git_ref}  "
          f"compile: {enable_compile}  prompts: {num_prompts or 'all'}  repo: {git_repo}")

    run_ab.with_options(gpu=f"{gpu}:{num_gpus}").remote(
        git_repo=git_repo, git_ref=git_ref, quant_method=quant_method, granularity=granularity,
        num_gpus=num_gpus, enable_compile=enable_compile, num_prompts=num_prompts,
        num_inference_steps=num_inference_steps)
