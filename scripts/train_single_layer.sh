#!/bin/bash
# Train a single CLT layer
# Usage: bash scripts/train_single_layer.sh LAYER_NUM [CONFIG_PATH]

set -e

LAYER="${1:-0}"
CONFIG="${2:-./config.yaml}"

echo "========================================="
echo "Training CLT for Layer $LAYER"
echo "========================================="

cd "$(dirname "$0")/.."

python scripts/train_clt.py \
  --config "$CONFIG" \
  --layer "$LAYER" \
  --steps 5000 \
  --lr 3e-4 \
  --feature-dim 8192 \
  --batch-samples 16 \
  --cache-batches 800 \
  --val-interval 200 \
  --val-samples 20 \
  --compile

echo "========================================="
echo "Training complete for Layer $LAYER!"
echo "========================================="
