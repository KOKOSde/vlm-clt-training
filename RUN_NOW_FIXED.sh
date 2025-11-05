#!/bin/bash
# Fixed training command with correct argument format

cd /scratch/fkalghan/vlm-clt-training
source /home/fkalghan/.venv/bin/activate

echo "=== Starting CLT Training with Correct Arguments ==="

nohup python -m sparsify \
  llava-hf/llava-1.5-7b-hf \
  /scratch/fkalghan/circuit_discovery_and_supression/data/llava_instruct_150k/train.jsonl \
  --transcode \
  --n_targets 16 \
  --layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 \
  --loss_fn fvu \
  --batch_size 2 \
  --grad_acc_steps 16 \
  --lr 3e-4 \
  --run_name llava-clt-all-layers \
  --ctx_len 512 \
  --expansion_factor 2 \
  -k 32 \
  --max_examples 100000 \
  > training.log 2>&1 &

TRAIN_PID=$!
echo "Training started with PID: $TRAIN_PID"
echo "Check logs: tail -f /scratch/fkalghan/vlm-clt-training/training.log"
echo "Check status: ps aux | grep $TRAIN_PID"

