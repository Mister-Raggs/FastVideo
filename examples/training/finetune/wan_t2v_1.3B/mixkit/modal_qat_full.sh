#!/bin/bash
# FULL stage-1 QAT finetune on Modal (4x L40S, 2000 steps). ONLY run this after
# the pilot is clean (and ideally after Will confirms the recipe/step count).
#
# Detached (--detach + --no-wait) so it survives your local client exiting;
# it prints a FunctionCall id — poll with:
#   python -c "import modal; print(modal.FunctionCall.from_id('<id>').get())"
#
# Checkpoints land every 500 steps under /root/data/checkpoints/
# wan_t2v_qat_finetune on the Modal volume (hf-model-weights); the volume is
# committed when the run finishes. NOTE: mid-run checkpoints only persist on that
# final commit — if you need crash-safety, run in resumable stages instead.
#
# Prereqs (local): `modal` configured + HF_TOKEN exported. ~4-6 h, ~$30-45.
set -euo pipefail
REPO=${GIT_REPO:-https://github.com/Mister-Raggs/FastVideo.git}
BRANCH=${GIT_COMMIT:-spark/qad-fp4-quality}

modal run --detach fastvideo/tests/modal/launch_l40s_job.py \
  --gpu-type L40S --num-gpus 4 \
  --git-repo "${REPO}" --git-commit "${BRANCH}" \
  --install-extra dev \
  --commit-volume --no-wait \
  --env-vars "MAX_STEPS=2000,VALIDATION_STEPS=200,CKPT_STEPS=500,WANDB_MODE=offline" \
  --command "bash examples/training/finetune/wan_t2v_1.3B/mixkit/run_qat_finetune.sh"
