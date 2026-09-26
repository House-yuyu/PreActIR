#!/usr/bin/env bash
set -euo pipefail

rm -rf outputs/debug data/toy_debug
python scripts/make_toy_clean_data.py --output data/toy_debug --train 4 --val 2 --test 2 --size 64
python scripts/build_intervention_dataset.py \
  --config configs/preactir_debug.yaml \
  --clean-root data/toy_debug \
  --output-root outputs/debug/interveneir

python scripts/train_belief.py \
  --config configs/preactir_debug.yaml --data-root outputs/debug/interveneir --device cpu --epochs 1
python scripts/train_world_model.py \
  --config configs/preactir_debug.yaml --data-root outputs/debug/interveneir --device cpu --epochs 1
python scripts/train_verifier.py \
  --config configs/preactir_debug.yaml --data-root outputs/debug/interveneir --device cpu --epochs 1

python scripts/evaluate_agent.py \
  --config configs/preactir_debug.yaml \
  --data-root outputs/debug/interveneir --split test \
  --belief-checkpoint outputs/debug/belief/best.pt \
  --world-checkpoint outputs/debug/world/best.pt \
  --verifier-checkpoint outputs/debug/verifier/best.pt \
  --output outputs/debug/agent_eval --device cpu
