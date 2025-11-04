#!/usr/bin/env python3
"""Commit changes and start CLT training for all layers."""

import os
import subprocess
import sys

def run_cmd(cmd, cwd=None):
    """Run a command and return output."""
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.returncode

# Change to repo directory
repo_dir = "/scratch/fkalghan/vlm-clt-training"
os.chdir(repo_dir)

print("=" * 60)
print("Step 1: Committing changes to GitHub")
print("=" * 60)

# Git operations
run_cmd("git add -A", cwd=repo_dir)
run_cmd('git commit -m "Surgical VLM support for CLT training"', cwd=repo_dir)
run_cmd("git push origin main", cwd=repo_dir)

print("\n✅ Changes committed!")

print("\n" + "=" * 60)
print("Step 2: Starting CLT training for all layers")
print("=" * 60)

# Check if we have a preprocessed dataset
dataset_path = "/scratch/fkalghan/circuit_discovery_and_supression/data/llava_instruct_150k_processed"
if not os.path.exists(dataset_path):
    print(f"\n⚠️  Preprocessed dataset not found at {dataset_path}")
    print("We need a dataset with 'input_ids' and 'pixel_values'")
    print("\nOptions:")
    print("1. Use EleutherAI's text-only approach (no images, LLM-style)")
    print("2. Preprocess the JSONL dataset with images first")
    
    # For now, let's use the JSONL dataset path
    dataset_path = "/scratch/fkalghan/circuit_discovery_and_supression/data/llava_instruct_150k/train.jsonl"
    print(f"\n→ Using: {dataset_path}")

# Training configuration
config = {
    "model": "llava-hf/llava-1.5-7b-hf",
    "dataset": dataset_path,
    "layers": list(range(32)),  # All 32 layers
    "loss_fn": "fvu",
    "sae_transcode": True,
    "sae_n_targets": 16,
    "batch_size": 2,
    "grad_acc_steps": 16,
    "lr": 3e-4,
    "run_name": "llava-clt-all-layers",
}

# Build command
cmd = f"""
source /home/fkalghan/.venv/bin/activate && \
python -m sparsify \
  {config['model']} \
  {config['dataset']} \
  --sae.transcode \
  --sae.n_targets {config['sae_n_targets']} \
  --layers {' '.join(map(str, config['layers']))} \
  --loss-fn {config['loss_fn']} \
  --batch-size {config['batch_size']} \
  --grad-acc-steps {config['grad_acc_steps']} \
  --lr {config['lr']} \
  --run-name {config['run_name']} \
  --log-to-wandb
"""

print("\n" + "=" * 60)
print("Training command:")
print("=" * 60)
print(cmd)
print("=" * 60)

print("\n🚀 Starting training...")
os.system(cmd)

