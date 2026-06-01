"""
A/B harness for ``use_dbcache`` (W4 — CacheDiT-style DBCache step caching)
on the Wan DiT.

Single-container pattern: clone the patched branch, build once, then run
the same N seed-pinned prompts twice — once with ``use_dbcache=False``
(baseline, every block runs every step) and once with ``use_dbcache=True``
(first ``Fn`` + last ``Bn`` blocks always run; middle blocks skipped when
the Fn-residual barely changed). Records per-prompt wall + pairwise SSIM,
prints a delta table.

Modeled on the W3b overlay harness (``batched_cfg_ab.py``) — same
container, same model load, isolates the one-flag change from
container/topology variance.

Unlike the W3 harnesses, DBCache is LOSSY: the SSIM column measures the
quality cost of caching (target >= 0.95), it is NOT a 1.0 correctness
gate. The "win" is the wall delta; the "cost" is the SSIM drop. Sweep
``--dbcache-threshold`` / ``--dbcache-fn`` / ``--dbcache-bn`` to find the
knee.

Usage (from FastVideo repo root):

    # F8B0, threshold 0.08 (cache-dit default profile), L40S Wan 1.3B
    modal run fastvideo/tests/modal/dbcache_ab.py --gpu L40S \\
        --model wan2_1-t2v-1.3b --num-gpus 1 \\
        --dbcache-fn 8 --dbcache-bn 0 --dbcache-threshold 0.08

    # Sweep the threshold by re-invoking; outputs are tagged per config.
    modal run fastvideo/tests/modal/dbcache_ab.py --gpu L40S \\
        --dbcache-threshold 0.12

Requires a Modal Secret named ``huggingface-token`` with key ``HF_TOKEN``.
"""
import os

import modal

app = modal.App("dbcache-ab")

model_vol = modal.Volume.from_name("hf-model-weights")
hf_secret = modal.Secret.from_name("huggingface-token")
# Pin to py3.12-latest (built from docker/Dockerfile.python3.12); its baked
# FA2 wheel (cu128torch2.11-cp312) matches the torch the [test] install
# pulls — no ABI mismatch on the FastVideo import.
image_tag = f"ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:{os.getenv('IMAGE_VERSION', 'py3.12-latest')}"

# Prebuilt FA3 wheel for the same ABI the fastvideo-dev image ships
# (cu128torch2.11, cp39-abi3, works on cp312). Hopper leg only.
FA3_WHEEL_URL = (
    "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/"
    "flash_attn_3-3.0.0%2Bcu128torch2.11gite2743ab-cp39-abi3-linux_x86_64.whl")

image = (modal.Image.from_registry(image_tag, add_python="3.12").apt_install(
    "cmake",
    "pkg-config",
    "build-essential",
    "curl",
    "libssl-dev",
    "ffmpeg",
    "libgl1",
    "libglib2.0-0",
).run_commands(
    "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable",
    "echo 'source ~/.cargo/env' >> ~/.bashrc",
).env({"PATH": "/root/.cargo/bin:$PATH"}))

# Same five prompts as the W3 A/Bs (#1395 / batched_cfg) so DBCache wall +
# SSIM are directly comparable to the SSIM=1.0 rearrangements they stack on.
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
        "height": 720,
        "width": 1280,
        "num_frames": 77,
        "num_inference_steps": 30,
        "default_num_gpus": 1,
    },
    "wan2_2-t2v-14b": {
        "model_id": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "height": 720,
        "width": 1280,
        "num_frames": 49,
        "num_inference_steps": 30,
        "default_num_gpus": 2,
    },
}


def _build_workspace_command(git_repo: str, git_ref: str, install_fa3: bool, full_install: bool = True) -> str:
    """Container-side bootstrap. Reads ``HF_TOKEN`` from the Modal Secret.
    When ``install_fa3`` is set, installs the prebuilt FA3 wheel before the
    FastVideo install. With ``full_install=False`` (used by
    ``recover_ssim``), skip the heavy install — the SSIM recompute path
    only needs the inner script on disk + pytorch_msssim/av.
    """
    import shlex
    fa3_step = f'uv pip install --no-cache-dir "{FA3_WHEEL_URL}"' if install_fa3 else 'echo "[fa3] skipped"'
    install_block = f"""
{fa3_step}
uv pip install -e ".[test]"
cd fastvideo-kernel && ./build.sh && cd ..
export HF_HOME=/root/data/.cache
hf auth login --token "$HF_TOKEN"
python -c "from fastvideo.attention.backends.flash_attn import fa_version; print(f'[fa] resolved fa_version={{fa_version}}')"
""" if full_install else """
# Minimal recovery bootstrap: clone only, no fastvideo install. The inner
# script's compute_ssim mode uses pytorch_msssim + torchvision/av directly.
uv pip install --no-cache-dir pytorch_msssim av || echo "[recover] pytorch_msssim/av already present"
echo "[recover] skipped fastvideo install + kernel build"
"""
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
{install_block}
"""


def _run_pass(*, use_dbcache: bool, model_id: str, num_gpus: int, height: int, width: int, num_frames: int,
              num_inference_steps: int, output_dir: str, enable_compile: bool, dit_precision: str = "bf16",
              num_prompts: int | None = None, dbcache_fn_compute_blocks: int = 8,
              dbcache_bn_compute_blocks: int = 0, dbcache_residual_threshold: float = 0.08,
              dbcache_max_warmup_steps: int = 8) -> list[dict]:
    """Invoke the inner pass script via /opt/venv/bin/python so FastVideo
    runs inside the image's venv (Modal's main process Python lacks torch).
    """
    import json
    import subprocess
    import tempfile

    selected_prompts = list(AB_PROMPTS) if num_prompts is None else list(AB_PROMPTS)[:num_prompts]
    config = {
        "model_id": model_id,
        "num_gpus": num_gpus,
        "use_dbcache": use_dbcache,
        "dbcache_fn_compute_blocks": dbcache_fn_compute_blocks,
        "dbcache_bn_compute_blocks": dbcache_bn_compute_blocks,
        "dbcache_residual_threshold": dbcache_residual_threshold,
        "dbcache_max_warmup_steps": dbcache_max_warmup_steps,
        "enable_compile": enable_compile,
        "dit_precision": dit_precision,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "num_inference_steps": num_inference_steps,
        "output_dir": output_dir,
        "prompts": selected_prompts,
        "seed_base": AB_SEED,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        config_path = f.name
    results_path = config_path.replace(".json", ".results.json")

    inner_script = "/FastVideo/fastvideo/tests/modal/_dbcache_ab_inner.py"
    backend_env = ("export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA && "
                   if dit_precision == "fp32" else "")
    cmd = (f"source /opt/venv/bin/activate && {backend_env}"
           f"exec python {inner_script} "
           f"--config-json {config_path} --results-json {results_path}")
    subprocess.run(["/bin/bash", "-lc", cmd], check=True)

    with open(results_path) as f:
        return json.load(f)


def _pairwise_ssim(baseline_records: list[dict], patched_records: list[dict]) -> list[dict]:
    """Pairwise SSIM between baseline and patched outputs, via the inner
    script (subprocess into the venv — fastvideo.tests.utils imports torch)."""
    import json
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".baseline.json", delete=False) as f:
        json.dump(baseline_records, f)
        baseline_path = f.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".patched.json", delete=False) as f:
        json.dump(patched_records, f)
        patched_path = f.name
    ssim_path = baseline_path.replace(".baseline.json", ".ssim.json")

    inner_script = "/FastVideo/fastvideo/tests/modal/_dbcache_ab_inner.py"
    cmd = (f"source /opt/venv/bin/activate && exec python {inner_script} --mode compute_ssim "
           f"--baseline-results-json {baseline_path} --patched-results-json {patched_path} "
           f"--ssim-output-json {ssim_path}")
    subprocess.run(["/bin/bash", "-lc", cmd], check=True)

    with open(ssim_path) as f:
        return json.load(f)


def _print_table(rows: list[dict], cfg_label: str) -> None:
    print(f"\n=== DBCache A/B results ({cfg_label}) ===")
    print("NOTE: DBCache is lossy — SSIM is a quality cost (target >= 0.95), not a 1.0 gate.")
    print(f"{'prompt':>6} {'baseline_wall_s':>16} {'patched_wall_s':>16} {'delta_pct':>10} {'ssim_mean':>10} "
          f"{'ssim_worst':>11}")
    base_sum = patched_sum = 0.0
    ssim_means = []
    ssim_worsts = []
    for r in rows:
        delta = (r["patched_wall_s"] - r["baseline_wall_s"]) / r["baseline_wall_s"] * 100.0
        print(f"{r['i']:>6} {r['baseline_wall_s']:>16.3f} {r['patched_wall_s']:>16.3f} {delta:>9.2f}% "
              f"{r['ssim_mean']:>10.6f} {r['ssim_worst']:>11.6f}")
        base_sum += r["baseline_wall_s"]
        patched_sum += r["patched_wall_s"]
        ssim_means.append(r["ssim_mean"])
        ssim_worsts.append(r["ssim_worst"])
    delta_total = (patched_sum - base_sum) / base_sum * 100.0
    print(f"\nbaseline total: {base_sum:.3f}s")
    print(f"dbcache  total: {patched_sum:.3f}s")
    print(f"wall delta:     {delta_total:+.2f}%   (negative = faster)")
    print(f"SSIM mean-of-means:  {sum(ssim_means) / len(ssim_means):.6f}")
    print(f"SSIM worst-of-worst: {min(ssim_worsts):.6f}")


@app.function(
    image=image,
    timeout=10800,
    volumes={"/root/data": model_vol},
    secrets=[hf_secret],
    gpu="L40S:1",
)
def run_ab(
    *,
    git_repo: str,
    git_ref: str,
    model_preset: str,
    num_gpus: int,
    install_fa3: bool = False,
    enable_compile: bool = False,
    dit_precision: str = "bf16",
    num_prompts: int = 0,
    dbcache_fn: int = 8,
    dbcache_bn: int = 0,
    dbcache_threshold: float = 0.08,
    dbcache_warmup: int = 8,
):
    """A/B: baseline (use_dbcache=False) vs patched (use_dbcache=True with
    the given Fn/Bn/threshold/warmup). Both passes share container, model
    load, seeds, and resolution — so the only variable is the cache."""
    import subprocess

    if model_preset not in MODEL_PRESETS:
        raise ValueError(f"unknown model_preset: {model_preset}")
    if "HF_TOKEN" not in os.environ:
        raise RuntimeError("HF_TOKEN not set — Modal Secret 'huggingface-token' missing the HF_TOKEN key.")
    preset = MODEL_PRESETS[model_preset]

    result = subprocess.run(
        ["/bin/bash", "-lc", _build_workspace_command(git_repo, git_ref, install_fa3=install_fa3)],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"workspace setup failed (rc={result.returncode})")
    os.chdir("/FastVideo")

    common = dict(
        model_id=preset["model_id"],
        num_gpus=num_gpus,
        height=preset["height"],
        width=preset["width"],
        num_frames=preset["num_frames"],
        num_inference_steps=preset["num_inference_steps"],
        enable_compile=enable_compile,
        dit_precision=dit_precision,
        num_prompts=(num_prompts if num_prompts > 0 else None),
    )

    # Tag outputs by mode + cache config so sweeps don't clobber each other.
    mode_tag = "compile" if enable_compile else "eager"
    if dit_precision != "bf16":
        mode_tag = f"{mode_tag}-{dit_precision}"
    cfg_tag = f"f{dbcache_fn}b{dbcache_bn}t{str(dbcache_threshold).replace('.', 'p')}w{dbcache_warmup}"
    baseline_dir = f"/root/data/dbcache_out/{model_preset}/{mode_tag}/baseline"
    patched_dir = f"/root/data/dbcache_out/{model_preset}/{mode_tag}/{cfg_tag}"

    cfg_label = (f"{model_preset} {mode_tag} | Fn={dbcache_fn} Bn={dbcache_bn} "
                 f"threshold={dbcache_threshold} warmup={dbcache_warmup}")
    print(f"\n--- baseline pass (use_dbcache=False, {mode_tag}) on {model_preset} ---")
    baseline = _run_pass(use_dbcache=False, output_dir=baseline_dir, **common)
    print(f"\n--- dbcache pass ({cfg_label}) ---")
    patched = _run_pass(use_dbcache=True, output_dir=patched_dir,
                        dbcache_fn_compute_blocks=dbcache_fn, dbcache_bn_compute_blocks=dbcache_bn,
                        dbcache_residual_threshold=dbcache_threshold, dbcache_max_warmup_steps=dbcache_warmup,
                        **common)

    rows = _pairwise_ssim(baseline, patched)
    _print_table(rows, cfg_label)
    return rows


@app.local_entrypoint()
def main(
    gpu: str = "L40S",
    model: str = "wan2_1-t2v-1.3b",
    num_gpus: int = 0,
    git_repo: str = "",
    git_ref: str = "perf/wan-dbcache-spike",
    enable_compile: bool = False,
    dit_precision: str = "bf16",
    num_prompts: int = 0,
    dbcache_fn: int = 8,
    dbcache_bn: int = 0,
    dbcache_threshold: float = 0.08,
    dbcache_warmup: int = 8,
):
    """Drive the DBCache A/B from your laptop. ``gpu`` is the Modal GPU
    class (``L40S``, ``H100``, ...). ``num_gpus=0`` uses the model preset
    default. ``git_repo`` defaults to the ``fork`` remote (where the spike
    branch lives) and falls back to ``origin``.

    Sweep the cache by re-invoking with different --dbcache-threshold /
    --dbcache-fn / --dbcache-bn; each config tags its own output dir."""
    import subprocess

    if model not in MODEL_PRESETS:
        raise ValueError(f"unknown model: {model}; choose from {list(MODEL_PRESETS)}")
    if num_gpus == 0:
        num_gpus = MODEL_PRESETS[model]["default_num_gpus"]
    if not git_repo:
        for remote in ("fork", "origin"):
            try:
                git_repo = subprocess.check_output(
                    ["git", "config", "--get", f"remote.{remote}.url"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
                break
            except subprocess.CalledProcessError:
                continue
        if not git_repo:
            raise RuntimeError("Could not resolve git_repo. Pass --git-repo or configure a 'fork'/'origin' remote.")

    # FA3 only on Hopper (capability 9.0); L40S is sm_89 -> baked FA2.
    install_fa3 = gpu.upper().startswith("H100") or gpu.upper().startswith("H200")
    print(f"GPU: {gpu}:{num_gpus}  model: {model}  ref: {git_ref}  install_fa3: {install_fa3}  "
          f"compile: {enable_compile}  precision: {dit_precision}  prompts: {num_prompts or 'all'}  "
          f"DBCache: Fn={dbcache_fn} Bn={dbcache_bn} threshold={dbcache_threshold} warmup={dbcache_warmup}  "
          f"repo: {git_repo}")

    run_ab.with_options(gpu=f"{gpu}:{num_gpus}").remote(
        git_repo=git_repo,
        git_ref=git_ref,
        model_preset=model,
        num_gpus=num_gpus,
        install_fa3=install_fa3,
        enable_compile=enable_compile,
        dit_precision=dit_precision,
        num_prompts=num_prompts,
        dbcache_fn=dbcache_fn,
        dbcache_bn=dbcache_bn,
        dbcache_threshold=dbcache_threshold,
        dbcache_warmup=dbcache_warmup,
    )
