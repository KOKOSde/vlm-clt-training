#!/bin/bash
# Test CLT training with WikiText to verify framework works

cd /scratch/fkalghan/vlm-clt-training
source /home/fkalghan/.venv/bin/activate

echo "=== Testing CLT Training with WikiText dataset ==="
echo "This tests the framework first, then we'll use VLM data"

nohup python -m sparsify \
  llava-hf/llava-1.5-7b-hf \
  wikitext \
  --transcode \
  --n_targets 16 \
  --layers 0 1 2 \
  --loss_fn fvu \
  --batch_size 2 \
  --grad_acc_steps 4 \
  --lr 3e-4 \
  --run_name llava-clt-test \
  --ctx_len 512 \
  --expansion_factor 2 \
  -k 32 \
  --max_examples 1000 \
  --split test \
  > training_test.log 2>&1 &

TRAIN_PID=$!
echo "Test training started with PID: $TRAIN_PID"
echo "Check logs: tail -f /scratch/fkalghan/vlm-clt-training/training_test.log"
echo ""
echo "Once this works, we'll preprocess the VLM dataset properly"


