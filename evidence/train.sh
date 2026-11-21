#!/usr/bin/env bash
# Train one arm of the SmolVLA dtype comparison.
#
#   ARM=bf16 ./train.sh          # LeRobot default: expert stored in bf16
#   ARM=fp32 ./train.sh          # same run, fp32 master copy (applies the one-line cast)
#
# Everything is overridable: STEPS BATCH LR SEED DATASET POLICY OUT EXTRA_ARGS
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ARM=${ARM:-bf16}
RUN=${RUN:-repro-out/$ARM}      # probe.jsonl + train.log + checkpoints/
OUT="$RUN/train"                # LeRobot creates this itself and refuses a pre-existing dir
mkdir -p "$RUN"

if [ "$ARM" = "fp32" ]; then
  python "$HERE/apply_fp32_cast.py" on
  trap 'python "$HERE/apply_fp32_cast.py" off' EXIT
fi

# Python only auto-imports a module literally named `sitecustomize`, so expose the probe
# under that name in a scratch dir instead of renaming the readable file.
PROBE_DIR="$RUN/.probe"
mkdir -p "$PROBE_DIR"
cp "$HERE/probe_sitecustomize.py" "$PROBE_DIR/sitecustomize.py"
export PYTHONPATH="$PROBE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PROBE_LOG="$RUN/probe.jsonl"
export PROBE_EVERY=${PROBE_EVERY:-100}

# LIBERO-Object has two cameras; smolvla_base's config declares three.
RENAME='{"observation.images.image":"observation.images.camera1","observation.images.wrist_image":"observation.images.camera2"}'

lerobot-train \
  --policy.path=${POLICY:-lerobot/smolvla_base} \
  --dataset.repo_id=${DATASET:-lerobot/libero_object_image} \
  --batch_size=${BATCH:-16} \
  --steps=${STEPS:-10000} \
  --policy.optimizer_lr=${LR:-1e-4} \
  --policy.device=cuda \
  --policy.empty_cameras=1 \
  --policy.push_to_hub=false \
  --rename_map="$RENAME" \
  --accelerator.mixed_precision=bf16 \
  --output_dir="$OUT" \
  --job_name="smolvla-$ARM" \
  --seed=${SEED:-1000} \
  --num_workers=8 \
  --log_freq=50 \
  --save_checkpoint=true \
  --save_freq=5000 \
  ${EXTRA_ARGS:-} \
  2>&1 | tee "$RUN/train.log"
