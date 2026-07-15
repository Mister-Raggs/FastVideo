#!/bin/bash
# Parameterized stage-1 QAT finetune (the QAD recipe, BEFORE the DMD step-distill)
# for Modal / any multi-GPU box. Mirrors finetune_qat.sh flag-for-flag; the only
# changes are env-overridable steps/paths + WANDB offline-by-default, so the same
# script runs a cheap pilot (MAX_STEPS=300) or the full run (MAX_STEPS=2000).
#
# FULL-STEP by construction: --multi_phased_distill_schedule "4000-1" with
# MAX_STEPS < 4000 means the few-step distill phase never triggers — the model
# trains at 50 euler timesteps throughout. Output = a full-step Wan2.1-1.3B that
# tolerates FP4 attention.
#
# Env knobs (all optional):
#   NUM_GPUS(4) MAX_STEPS(2000) VALIDATION_STEPS(200) CKPT_STEPS(500)
#   DATA_ROOT(/root/data/HD-Mixkit-Finetune-Wan) DATA_DIR(<root>/combined_parquet_dataset/)
#   OUTPUT_DIR(/root/data/checkpoints/wan_t2v_qat_finetune)
#   WANDB_MODE(offline)  — set online + export WANDB_API_KEY to track
set -euo pipefail

export FASTVIDEO_ATTENTION_BACKEND=ATTN_QAT_TRAIN   # fake-quant (Triton STE) attention
export WANDB_MODE=${WANDB_MODE:-offline}
export TOKENIZERS_PARALLELISM=false

MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
NUM_GPUS=${NUM_GPUS:-4}
DATA_ROOT=${DATA_ROOT:-/root/data/HD-Mixkit-Finetune-Wan}
DATA_DIR=${DATA_DIR:-${DATA_ROOT}/combined_parquet_dataset/}
OUTPUT_DIR=${OUTPUT_DIR:-/root/data/checkpoints/wan_t2v_qat_finetune}
MAX_STEPS=${MAX_STEPS:-2000}
VALIDATION_STEPS=${VALIDATION_STEPS:-200}
CKPT_STEPS=${CKPT_STEPS:-500}
VALIDATION_FILE="$(dirname "$0")/../crush_smol/validation.json"

echo "[qat] NUM_GPUS=${NUM_GPUS} MAX_STEPS=${MAX_STEPS} VALIDATION_STEPS=${VALIDATION_STEPS} CKPT_STEPS=${CKPT_STEPS}"
echo "[qat] DATA_DIR=${DATA_DIR}"
echo "[qat] OUTPUT_DIR=${OUTPUT_DIR}"

# 1. Data — preprocessed MixKit parquet (VAE latents + text embeds), once.
if [ ! -d "${DATA_DIR}" ]; then
  echo "[qat] downloading MixKit parquet to ${DATA_ROOT} ..."
  python scripts/huggingface/download_hf.py \
    --repo_id "weizhou03/HD-Mixkit-Finetune-Wan" \
    --local_dir "${DATA_ROOT}" --repo_type "dataset"
fi

# 1b. PREFLIGHT — the fake-quant train kernel must actually import, else the
#     selector silently filters ATTN_QAT_TRAIN and attention falls back to Flash
#     (a NON-QAT run) upstream of the cuda.py hard-fail guard. Surface the real
#     error and refuse to proceed rather than waste a run.
echo "[qat] preflight: importing fastvideo_kernel.triton_kernels.attn_qat_train ..."
python - <<'PY' || { echo "[qat] PREFLIGHT FAILED — attn_qat_train kernel not importable; aborting (would train NON-QAT)."; exit 3; }
import importlib, sys, traceback
sys.path.insert(0, "fastvideo-kernel/python")
sys.path.insert(0, "fastvideo-kernel")
try:
    m = importlib.import_module("fastvideo_kernel.triton_kernels.attn_qat_train")
    _ = m.attention
    print("[qat] preflight OK: attn_qat_train.attention importable")
except Exception:
    traceback.print_exc()
    sys.exit(1)
PY

# 2. Train — identical hyperparams to finetune_qat.sh (LR 5e-5, wd 1e-4,
#    grad-norm 1.0, bf16 + fp32 master, cfg-rate 0.1, 50 euler timesteps).
torchrun --nnodes 1 --nproc_per_node "${NUM_GPUS}" \
    fastvideo/training/wan_training_pipeline.py \
    --num_gpus "${NUM_GPUS}" --sp_size "${NUM_GPUS}" --tp_size 1 \
    --hsdp_replicate_dim 1 --hsdp_shard_dim "${NUM_GPUS}" \
    --model_path "${MODEL_PATH}" --pretrained_model_name_or_path "${MODEL_PATH}" \
    --data_path "${DATA_DIR}" --dataloader_num_workers 1 \
    --max_train_steps "${MAX_STEPS}" --train_batch_size 1 --train_sp_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --num_latent_t 20 --num_height 480 --num_width 832 --num_frames 77 \
    --enable_gradient_checkpointing_type full \
    --log_validation --validation_dataset_file "${VALIDATION_FILE}" \
    --validation_steps "${VALIDATION_STEPS}" --validation_sampling_steps 50 --validation_guidance_scale 3.0 \
    --learning_rate 5e-5 --mixed_precision bf16 --weight_decay 1e-4 --max_grad_norm 1.0 \
    --weight_only_checkpointing_steps "${CKPT_STEPS}" --training_state_checkpointing_steps "${CKPT_STEPS}" \
    --tracker_project_name wan_t2v_qat_finetune --output_dir "${OUTPUT_DIR}" \
    --inference_mode False --training_cfg_rate 0.1 --not_apply_cfg_solver \
    --dit_precision fp32 --num_euler_timesteps 50 --ema_start_step 0 \
    --multi_phased_distill_schedule "4000-1"

echo "[qat] done. checkpoints under ${OUTPUT_DIR} (every ${CKPT_STEPS} steps)."
