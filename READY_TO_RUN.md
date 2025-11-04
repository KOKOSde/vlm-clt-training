# ✅ VLM CLT Training - Ready to Run!

## What Was Done (Surgical Updates)

### Files Modified:
1. **`sparsify/__main__.py`** - Added VLM model loading and dataset handling
2. **`sparsify/trainer.py`** - Added pixel_values passthrough for VLM forward calls

### Changes Summary:
- ✅ VLM detection (LLaVA, Qwen-VL, etc.)
- ✅ Load VLMs with `AutoModelForVision2Seq`
- ✅ Pass `pixel_values` in all model forward calls
- ✅ Support VLM structure (`language_model.model.layers`)
- ✅ Backward compatible with LLMs

**Total: 2 files, 6 targeted changes** - pure surgical updates!

---

## How to Commit Changes

```bash
cd /scratch/fkalghan/vlm-clt-training
git add -A
git commit -m "Surgical VLM support for CLT training"
git push origin main
```

Or run:
```bash
python3 /scratch/fkalghan/vlm-clt-training/COMMIT_CHANGES.py
```

---

## How to Start Training

### Quick Start (All 32 Layers):
```bash
bash /scratch/fkalghan/vlm-clt-training/START_TRAINING.sh
```

### Manual Command:
```bash
source /home/fkalghan/.venv/bin/activate
cd /scratch/fkalghan/vlm-clt-training

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
  --log-to-wandb \
  --ctx-len 512 \
  --sae.expansion-factor 2 \
  --sae.k 32
```

---

## What This Does

### True CLT (Cross-Layer Transcoder):
- **Not** PLT (Per-Layer Transcoder) - that's what you had before
- Each layer's encoder produces features
- **16 decoder heads** per layer (predicts 16 future layers)
- Uses EleutherAI's forward hook method (captures activations on-the-fly)
- No need for pre-captured activations!

### Training Flow:
1. Load LLaVA model on GPU
2. Load dataset (text + images if available)
3. Register forward hooks on all 32 MLP layers
4. For each batch:
   - Forward pass through model
   - Hooks capture MLP inputs/outputs
   - Train CLT with multi-target loss
   - Update features to predict 16 future layers

### Output:
- CLT checkpoints saved to `checkpoints/llava-clt-all-layers/`
- Each layer gets a separate CLT with 16 decoders
- Can load and use for circuit discovery later

---

## Key Parameters Explained

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `--sae.transcode` | - | Enable transcoder mode (CLT) |
| `--sae.n_targets` | 16 | Each layer predicts 16 future layers |
| `--layers` | 0-31 | Train all 32 layers |
| `--loss-fn` | fvu | Fraction of Variance Unexplained |
| `--batch-size` | 2 | Sequences per batch |
| `--grad-acc-steps` | 16 | Effective batch = 2×16 = 32 |
| `--sae.k` | 32 | Top-K sparse features active |
| `--sae.expansion-factor` | 2 | Feature dim = 2× hidden dim |

---

## Monitoring Training

### Check if running:
```bash
ps aux | grep "python -m sparsify"
```

### Watch logs (if using script):
```bash
tail -f /scratch/fkalghan/vlm-clt-training/training.log
```

### Check GPU usage:
```bash
nvidia-smi
```

### View in W&B:
Training logs automatically sync to Weights & Biases if `--log-to-wandb` is set.

---

## Troubleshooting

### If dataset doesn't have pixel_values:
The framework will still work with text-only data (LLM-style). For full VLM training with images, preprocess the dataset to include `pixel_values`.

### If CUDA out of memory:
- Reduce `--batch-size` to 1
- Increase `--grad-acc-steps` to 32
- Or train fewer layers at a time

### If hooks don't find layers:
Check model structure:
```python
from transformers import AutoModelForVision2Seq
model = AutoModelForVision2Seq.from_pretrained("llava-hf/llava-1.5-7b-hf")
for name, _ in model.named_modules():
    if 'mlp' in name:
        print(name)
```

---

## Next Steps After Training

1. **Load trained CLTs** from checkpoints
2. **Replace MLPs** in LLaVA with CLTs
3. **Run inference** to capture feature activations
4. **Build attribution graphs** (Anthropic's method)
5. **Discover circuits** for specific behaviors

---

## Summary

✅ **Surgical updates complete** - VLM support added with minimal changes  
✅ **True CLT architecture** - multi-target decoders, not PLT  
✅ **EleutherAI method** - forward hooks, on-the-fly capture  
✅ **Ready to train** - all 32 layers, 16 targets each  

**Just run:** `bash START_TRAINING.sh`


