#!/bin/bash
# CHEAP PILOT of the stage-1 QAT finetune on Modal (4x L40S, ~300 steps).
#
# Purpose: shake out install / ATTN_QAT_TRAIN-backend-selection / NCCL /
# multi-GPU / data download / first-validation BEFORE spending on the full run.
# Watch for: "Selected backend: ATTN_QAT_TRAIN" (not a Flash fallback), healthy
# loss/grad, and the step-100 validation rendering coherent video.
#
# Prereqs (local): `modal` configured, and HF_TOKEN (or HUGGINGFACE_HUB_TOKEN)
# exported so the container can pull the Wan base weights. WANDB optional.
#
# Runs synchronously so you see the logs live. ~30-45 min, roughly a few $.
set -euo pipefail
REPO=${GIT_REPO:-https://github.com/Mister-Raggs/FastVideo.git}
BRANCH=${GIT_COMMIT:-spark/qad-fp4-quality}

modal run fastvideo/tests/modal/launch_l40s_job.py \
  --gpu-type L40S --num-gpus 4 \
  --git-repo "${REPO}" --git-commit "${BRANCH}" \
  --install-extra dev \
  --commit-volume \
  --env-vars "MAX_STEPS=300,VALIDATION_STEPS=100,CKPT_STEPS=300,WANDB_MODE=offline" \
  --command "bash examples/training/finetune/wan_t2v_1.3B/mixkit/run_qat_finetune.sh"
