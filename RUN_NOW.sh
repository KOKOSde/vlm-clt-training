#!/bin/bash
# Simple script to commit and start training
# Run this directly: bash RUN_NOW.sh

cd /scratch/fkalghan/vlm-clt-training

echo "=== Committing changes ==="
git add -A
git commit -m "Surgical VLM support: minimal updates for CLT training" || echo "Already committed or no changes"
git push origin main || echo "Push skipped or already pushed"

echo ""
echo "=== Starting training ==="
source /home/fkalghan/.venv/bin/activate

nohup python -m sparsify \
  llava-hf/llava-1.5-7b-hf \
  /scratch/fkalghan/circuit_discovery_and_supression/data/llava_instruct_150k/train.jsonl \
  --sae.transcode \
  --sae.n_targets 16 \
  --layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 \
  --loss-fn fvu \
  --batch-size 2 \
  --grad-acc-steps 16 \
  --lr 3e-4 \
  --run-name llava-clt-all-layers \
  --ctx-len 512 \
  --sae.expansion-factor 2 \
  --sae.k 32 \
  --max-examples 100000 \
  > training.log 2>&1 &

TRAIN_PID=$!
echo "Training started with PID: $TRAIN_PID"
echo "Check status: ps aux | grep $TRAIN_PID"
echo "View logs: tail -f /scratch/fkalghan/vlm-clt-training/training.log"
echo "Stop training: kill $TRAIN_PID"

