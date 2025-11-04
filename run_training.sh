#!/bin/bash
# Train CLTs for LLaVA using EleutherAI's framework with forward hooks
# This uses on-the-fly activation capture - no pre-captured activations needed!

set -e

MODEL="llava-hf/llava-1.5-7b-hf"
DATASET="your_dataset_path"  # Must have input_ids and pixel_values
LAYERS="0"  # Start with layer 0

source /home/fkalghan/.venv/bin/activate

cd /scratch/fkalghan/vlm-clt-training

python -m sparsify \
  "$MODEL" \
  "$DATASET" \
  --sae.transcode \
  --sae.n_targets 16 \
  --layers $LAYERS \
  --loss-fn fvu \
  --batch-size 4 \
  --grad-acc-steps 8 \
  --lr 3e-4 \
  --run-name "llava-clt-layer${LAYERS}"

echo "✅ Training complete!"

