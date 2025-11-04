#!/bin/bash
# CLT Training for All Layers - LLaVA 1.5 7B
# Uses EleutherAI's forward hook method with VLM support

set -e

echo "========================================"
echo "LLaVA CLT Training - All 32 Layers"
echo "========================================"

# Activate environment
source /home/fkalghan/.venv/bin/activate

# Change to repo directory
cd /scratch/fkalghan/vlm-clt-training

# Model and dataset configuration
MODEL="llava-hf/llava-1.5-7b-hf"
DATASET="/scratch/fkalghan/circuit_discovery_and_supression/data/llava_instruct_150k/train.jsonl"

# Training configuration
LAYERS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31"
N_TARGETS=16  # Predict 16 future layers (true CLT)
BATCH_SIZE=2
GRAD_ACC=16
LR=3e-4
LOSS_FN="fvu"
RUN_NAME="llava-clt-all-layers-$(date +%Y%m%d_%H%M%S)"

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Dataset: $DATASET"
echo "  Layers: All 32 layers (0-31)"
echo "  Targets per layer: $N_TARGETS"
echo "  Batch size: $BATCH_SIZE"
echo "  Gradient accumulation: $GRAD_ACC"
echo "  Learning rate: $LR"
echo "  Loss function: $LOSS_FN"
echo ""

# Run training
python -m sparsify \
  "$MODEL" \
  "$DATASET" \
  --sae.transcode \
  --sae.n_targets $N_TARGETS \
  --layers $LAYERS \
  --loss-fn $LOSS_FN \
  --batch-size $BATCH_SIZE \
  --grad-acc-steps $GRAD_ACC \
  --lr $LR \
  --run-name "$RUN_NAME" \
  --log-to-wandb \
  --ctx-len 512 \
  --sae.expansion-factor 2 \
  --sae.k 32 \
  --max-examples 100000

echo ""
echo "========================================"
echo "✅ Training Complete!"
echo "========================================"

