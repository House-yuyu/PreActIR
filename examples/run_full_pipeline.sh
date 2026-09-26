#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/preactir_small.yaml}
CLEAN_ROOT=${CLEAN_ROOT:-data/clean}
DATA_ROOT=${DATA_ROOT:-data/interveneir}
DEVICE=${DEVICE:-cuda}

python scripts/build_intervention_dataset.py \
  --config "$CONFIG" \
  --clean-root "$CLEAN_ROOT" \
  --output-root "$DATA_ROOT"

python scripts/train_belief.py --config "$CONFIG" --data-root "$DATA_ROOT" --device "$DEVICE"
python scripts/train_world_model.py --config "$CONFIG" --data-root "$DATA_ROOT" --device "$DEVICE"
python scripts/train_verifier.py --config "$CONFIG" --data-root "$DATA_ROOT" --device "$DEVICE"

python scripts/evaluate_belief.py \
  --config "$CONFIG" --data-root "$DATA_ROOT" --split test \
  --checkpoint outputs/belief/best.pt --device "$DEVICE"

python scripts/evaluate_world_model.py \
  --config "$CONFIG" --data-root "$DATA_ROOT" --split test \
  --checkpoint outputs/world/best.pt --device "$DEVICE"

python scripts/evaluate_verifier.py \
  --config "$CONFIG" --data-root "$DATA_ROOT" --split test \
  --checkpoint outputs/verifier/best.pt --device "$DEVICE"

python scripts/evaluate_agent.py \
  --config "$CONFIG" --data-root "$DATA_ROOT" --split test \
  --belief-checkpoint outputs/belief/best.pt \
  --world-checkpoint outputs/world/best.pt \
  --verifier-checkpoint outputs/verifier/best.pt \
  --output outputs/agent_eval --device "$DEVICE"
