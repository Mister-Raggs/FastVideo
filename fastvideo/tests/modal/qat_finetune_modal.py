"""Stage-1 QAT finetune (full-step, pre-DMD) on Modal — builds a full-step QAT
Wan2.1-T2V-1.3B checkpoint that tolerates FP4 attention.

Runs the QAD recipe's quantization-aware finetune (the same flags as
``examples/training/finetune/wan_t2v_1.3B/mixkit/finetune_qat.sh``, driven through
``run_qat_finetune.sh``) on 4x L40S. Output = a *full-step* checkpoint (the
distill-schedule threshold stays above ``--max-steps`` so the few-step DMD phase
never triggers) — the thing the public 3-step QAD model can't give us at full
sharpness.

Pilot vs full is just ``--max-steps`` (300 vs 2000). Run the PILOT first to shake
out ATTN_QAT_TRAIN backend selection / NCCL / data download / first-validation
before committing to the full run.

Usage (from the FastVideo repo root):

    # PILOT — ~300 steps, watch it live (~30-45 min, a few $)
    modal run fastvideo/tests/modal/qat_finetune_modal.py \
        --max-steps 300 --validation-steps 100 --ckpt-steps 300

    # FULL — 2000 steps, detached (survives your client exiting; ~4-6 h)
    modal run --detach fastvideo/tests/modal/qat_finetune_modal.py

Checkpoints land on the ``hf-model-weights`` volume under
``/root/data/checkpoints/wan_t2v_qat_finetune`` (every ``--ckpt-steps``).
Requires the Modal Secret ``huggingface-token`` (key ``HF_TOKEN``).
"""
import os

import modal

app = modal.App("qat-finetune")

model_vol = modal.Volume.from_name("hf-model-weights")
hf_secret = modal.Secret.from_name("huggingface-token")
image_tag = f"ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:{os.getenv('IMAGE_VERSION', 'py3.12-latest')}"

image = (modal.Image.from_registry(image_tag, add_python="3.12").apt_install(
    "cmake", "pkg-config", "build-essential", "curl", "libssl-dev", "ffmpeg", "libgl1", "libglib2.0-0",
).run_commands(
    "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable",
    "echo 'source ~/.cargo/env' >> ~/.bashrc",
).env({"PATH": "/root/.cargo/bin:$PATH"}))


def _workspace_and_train_command(git_repo: str, git_ref: str, num_gpus: int, max_steps: int,
                                 validation_steps: int, ckpt_steps: int) -> str:
    import shlex
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
uv pip install -e ".[test]"
cd fastvideo-kernel && ./build.sh && cd ..
export HF_HOME=/root/data/.cache
hf auth login --token "$HF_TOKEN"
export NUM_GPUS={num_gpus} MAX_STEPS={max_steps} VALIDATION_STEPS={validation_steps} CKPT_STEPS={ckpt_steps}
export WANDB_MODE=offline
bash examples/training/finetune/wan_t2v_1.3B/mixkit/run_qat_finetune.sh
"""


@app.function(image=image, timeout=86400, volumes={"/root/data": model_vol}, secrets=[hf_secret], gpu="L40S:4")
def run_qat(*, git_repo: str, git_ref: str, num_gpus: int, max_steps: int, validation_steps: int,
            ckpt_steps: int) -> dict:
    import subprocess

    if "HF_TOKEN" not in os.environ:
        raise RuntimeError("HF_TOKEN not set — Modal Secret 'huggingface-token' missing the HF_TOKEN key.")
    cmd = _workspace_and_train_command(git_repo, git_ref, num_gpus, max_steps, validation_steps, ckpt_steps)
    try:
        subprocess.run(["/bin/bash", "-lc", cmd], env=os.environ.copy(), check=True)
    finally:
        # Persist whatever checkpoints landed (final commit; see module docstring).
        model_vol.commit()
    return {"max_steps": max_steps, "output_dir": "/root/data/checkpoints/wan_t2v_qat_finetune"}


@app.local_entrypoint()
def main(gpu: str = "L40S", num_gpus: int = 4, git_repo: str = "", git_ref: str = "spark/qad-fp4-quality",
         max_steps: int = 2000, validation_steps: int = 200, ckpt_steps: int = 500):
    """Drive the QAT finetune from your laptop. Pilot: ``--max-steps 300
    --validation-steps 100 --ckpt-steps 300``. ``git_repo`` defaults to the
    ``fork`` remote."""
    import subprocess

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

    print(f"GPU: {gpu}:{num_gpus}  ref: {git_ref}  max_steps: {max_steps}  "
          f"validation_steps: {validation_steps}  ckpt_steps: {ckpt_steps}  repo: {git_repo}")
    run_qat.with_options(gpu=f"{gpu}:{num_gpus}").remote(
        git_repo=git_repo, git_ref=git_ref, num_gpus=num_gpus, max_steps=max_steps,
        validation_steps=validation_steps, ckpt_steps=ckpt_steps)
