#!/usr/bin/env python3
"""Commit changes and start training - Python 3.6 compatible."""

import os
import subprocess
import sys

os.chdir('/scratch/fkalghan/vlm-clt-training')

print("=" * 60)
print("Step 1: Committing changes")
print("=" * 60)

# Git commit
result = subprocess.call(['git', 'add', '-A'])
if result != 0:
    print("⚠️  git add failed")
    sys.exit(1)

result = subprocess.call([
    'git', 'commit', '-m',
    'Surgical VLM support: minimal updates for CLT training\n\n'
    '- Added VLM detection (AutoModelForVision2Seq)\n'
    '- Pass pixel_values in all forward calls\n'
    '- Support VLM model structure\n'
    '- Only 2 files modified, 6 changes total'
])

if result == 0:
    print("✅ Committed!")
    print("\nPushing to GitHub...")
    result = subprocess.call(['git', 'push', 'origin', 'main'])
    if result == 0:
        print("✅ Pushed to GitHub!")
    else:
        print("⚠️  Push failed, but commit succeeded")
else:
    print("⚠️  Nothing to commit (may already be committed)")

print("\n" + "=" * 60)
print("Step 2: Starting CLT training for all 32 layers")
print("=" * 60)

# Start training
cmd = """source /home/fkalghan/.venv/bin/activate && \
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
  --max-examples 100000"""

print("\n🚀 Starting training...")
print("Command:", cmd.replace(' && ', '\n   '))
print("\n" + "=" * 60)

# Run in background
import signal
process = subprocess.Popen(cmd, shell=True, executable='/bin/bash')
print(f"✅ Training started with PID: {process.pid}")
print(f"   Check status: ps aux | grep {process.pid}")
print(f"   Or check: ps aux | grep 'python -m sparsify'")

