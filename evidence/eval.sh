#!/usr/bin/env bash
# Score a checkpoint on LIBERO-Object with 50-action chunks open loop.
# 50 episodes per task x 10 tasks = 500 tries, the paper's scoring protocol.
#
#   CKPT=repro-out/bf16/checkpoints/last/pretrained_model OUT=eval-bf16 ./eval.sh
set -euo pipefail
CKPT=${CKPT:?set CKPT to a pretrained_model directory}
OUT=${OUT:-eval}
mkdir -p "$OUT"

RENAME='{"observation.images.image":"observation.images.camera1","observation.images.wrist_image":"observation.images.camera2"}'
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

lerobot-eval \
  --policy.path="$CKPT" \
  --policy.device=cuda \
  --env.type=libero \
  --env.task=libero_object \
  --env.episode_length=300 \
  --eval.n_episodes=${N_EPISODES:-50} \
  --eval.batch_size=${EVAL_BATCH:-14} \
  --rename_map="$RENAME" \
  --output_dir="$OUT" \
  2>&1 | tee "$OUT/eval.log"
