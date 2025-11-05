#!/bin/bash
# Simple test with EleutherAI's own dataset format

cd /scratch/fkalghan/vlm-clt-training
source /home/fkalghan/.venv/bin/activate

echo "=== Testing CLT Training with EleutherAI dataset ==="

nohup python -m sparsify \
  llava-hf/llava-1.5-7b-hf \
  EleutherAI/pile \
  --transcode \
  --n_targets 16 \
  --layers 0 1 2 \
  --loss_fn fvu \
  --batch_size 1 \
  --grad_acc_steps 4 \
  --lr 3e-4 \
  --run_name llava-clt-test \
  --ctx_len 256 \
  --expansion_factor 2 \
  -k 32 \
  --max_examples 100 \
  --split train \
  > training_simple.log 2>&1 &

TRAIN_PID=$!
echo "Test training started with PID: $TRAIN_PID"
echo "Check logs: tail -f /scratch/fkalghan/vlm-clt-training/training_simple.log"
