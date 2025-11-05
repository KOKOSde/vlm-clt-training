#!/bin/bash
# Train full cross-layer CLT on LLaVA with images
# Uses --cross_layer to auto-calculate targets for each layer

cd /scratch/fkalghan/vlm-clt-training
source /home/fkalghan/.venv/bin/activate

echo "========================================="
echo "Training LLaVA CLT - All 32 Layers"
echo "Auto-calculating targets per layer"
echo "========================================="

python -m sparsify \
  llava-hf/llava-1.5-7b-hf \
  /scratch/fkalghan/circuit_discovery_and_supression/data/llava_instruct_150k/train.jsonl \
  --transcode \
  --cross_layer 32 \
  --layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 \
  --loss_fn fvu \
  --batch_size 2 \
  --grad_acc_steps 16 \
  --lr 3e-4 \
  --run_name llava-clt-full \
  --ctx_len 512 \
  --expansion_factor 2 \
  -k 32 \
  --max_examples 100000 \
  --text_column prompt \
  --save_every 1000

echo ""
echo "========================================="
echo "Training Complete!"
echo "Checkpoints: checkpoints/llava-clt-full/"
echo "========================================="

