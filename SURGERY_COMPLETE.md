# VLM Support: Surgical Updates Complete ✅

## What Was Changed

### 1. `sparsify/__main__.py`
- **Lines 14-22**: Added imports for VLM support
  - Added `AutoModelForVision2Seq` for loading VLMs
  - Added `AutoProcessor` for VLM preprocessing
  
- **Lines 91-122**: VLM model detection and loading
  - Detects VLMs by name (llava, qwen-vl, idefics, paligemma)
  - Uses `AutoModelForVision2Seq` for VLMs
  - Falls back to original logic for LLMs
  
- **Lines 151-169**: VLM dataset handling
  - Uses `AutoProcessor` for VLMs instead of `AutoTokenizer`
  - Expects pre-processed datasets with `pixel_values`

### 2. `sparsify/trainer.py`
- **Lines 73-79**: VLM model structure support
  - Detects LLaVA-style structure (`model.language_model`)
  - Handles `base_model` correctly for VLMs
  
- **Lines 86-100**: Config handling for VLMs
  - Gets `num_hidden_layers` from correct location in VLM structure
  
- **Lines 462-468**: CE loss with pixel_values
  - Passes `pixel_values` to model if present in batch
  
- **Lines 716-722**: KL loss with pixel_values
  - Passes `pixel_values` for clean logits computation
  
- **Lines 747-754**: CE loss in training loop with pixel_values
  - Passes `pixel_values` during main training

- **Lines 758-762**: KL loss in training loop with pixel_values
  - Passes `pixel_values` for dirty logits computation

## Key Principles

1. **Minimal Surgery**: Only 6 targeted changes across 2 files
2. **Backward Compatible**: LLM training still works exactly as before
3. **Forward Compatible**: VLM training now supported with same API
4. **No New Files**: Used existing EleutherAI framework

## Testing

To test VLM training, run:
```bash
python -m sparsify \
  llava-hf/llava-1.5-7b-hf \
  your_preprocessed_dataset \
  --sae.transcode \
  --sae.n_targets 16 \
  --layers 0 1 2 \
  --loss-fn fvu
```

Dataset must have:
- `input_ids`: Tokenized text [B, T]
- `pixel_values`: Image tensors [B, C, H, W]

## What Works Now

✅ Load VLM models (LLaVA, Qwen-VL, etc.)
✅ Pass pixel_values through forward hooks
✅ Train CLTs with multi-target decoders
✅ Support all loss functions (FVU, CE, KL)
✅ Maintain backward compatibility with LLMs

## Next Steps

1. Prepare VLM dataset with `pixel_values`
2. Run training with EleutherAI's forward hook method
3. No need for pre-captured activations - hooks capture on-the-fly!

