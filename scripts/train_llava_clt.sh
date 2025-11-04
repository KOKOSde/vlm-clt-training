#!/bin/bash
# Train Cross-Layer Transcoders (CLTs) for LLaVA
# Based on EleutherAI's CLT training methodology
# Paper: https://transformer-circuits.pub/2025/attribution-graphs/methods.html

set -e

MODEL_NAME="${1:-llava-hf/llava-1.5-7b-hf}"
RUN_NAME="${2:-llava-clt-training}"
ACTIVATIONS_DIR="${3:-./activations}"

echo "========================================="
echo "Training CLTs for VLM"
echo "========================================="
echo "Model: $MODEL_NAME"
echo "Run name: $RUN_NAME"
echo "Activations: $ACTIVATIONS_DIR"
echo "========================================="

# Training configuration
CROSS_LAYER=16  # Predict 16 layers ahead
K=32           # 32 active features (sparsity)
EXPANSION=2    # 2x expansion factor (8192 features for 4096 hidden dim)
BATCH_SIZE=8
LR=1e-4

# Run training
python -m sparsify.train_vlm \
  --model_name "$MODEL_NAME" \
  --activations_dir "$ACTIVATIONS_DIR" \
  --run_name "$RUN_NAME" \
  --cross_layer $CROSS_LAYER \
  --k $K \
  --expansion_factor $EXPANSION \
  --batch_size $BATCH_SIZE \
  --lr $LR \
  --optimizer adam \
  --b1 0.0 \
  --b2 0.999 \
  --lr_warmup_steps 50 \
  --train_post_encoder \
  --post_encoder_scale \
  --coalesce_topk concat \
  --skip_connection \
  --save_every 1000 \
  --log_to_wandb \
  "$@"

echo "========================================="
echo "Training complete!"
echo "Checkpoints saved to: checkpoints/$RUN_NAME"
echo "========================================="

