#!/bin/bash
# Direct execution - commit and train

set -e

cd /scratch/fkalghan/vlm-clt-training

echo "========================================"
echo "Step 1: Committing to GitHub"
echo "========================================"

git add -A
git commit -m "Surgical VLM support: minimal updates for CLT training" || echo "Already committed"
git push origin main || echo "Push failed or already pushed"

echo ""
echo "========================================"
echo "Step 2: Starting CLT Training"
echo "========================================"

source /home/fkalghan/.venv/bin/activate

python -m sparsify \
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
  --max-examples 100000

