#!/bin/bash
# Train a single CLT layer using the working training script
# Usage: bash scripts/train_single_layer.sh LAYER_NUM

set -e

LAYER="${1:-0}"
CONFIG="${2:-./config.yaml}"

echo "========================================="
echo "Training CLT for Layer $LAYER"
echo "========================================="

source /home/fkalghan/.venv/bin/activate

cd "$(dirname "$0")/.."

python scripts/train_llava_clt.py \
  --config "$CONFIG" \
  --layer "$LAYER" \
  --clt-mode \
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

