# CLT Architecture Analysis

## Overview

This document explains the Cross-Layer Transcoder (CLT) architecture for Vision-Language Models, based on Anthropic's [Circuit-Tracer methodology](https://transformer-circuits.pub/2025/attribution-graphs/methods.html).

---

## 🔬 Anthropic's CLT Architecture

### Key Principle

> "Each feature reads from the residual stream at one layer and **contributes to the outputs of ALL subsequent MLP layers**"

### Architecture Components

```python
# Single transcoder at layer L
Encoder: [hidden_dim] → [feature_dim]  # Read from residual stream
TopK: Select K active features
Decoders: [
    W_dec[0]: [feature_dim] → [hidden_dim]  # Contribute to layer L+1
    W_dec[1]: [feature_dim] → [hidden_dim]  # Contribute to layer L+2
    ...
    W_dec[n-1]: [feature_dim] → [hidden_dim]  # Contribute to layer L+n
]
```

### Training

- **Input**: Residual stream at layer L
- **Targets**: MLP outputs at layers L+1, L+2, ..., L+n
- **Loss**: `Σ MSE(decoder[i](features), target[i])` across all targets
- **Result**: ~50% token-level match with original model

### Why CLTs Work

- Features "bridge" across multiple layers
- Errors don't compound linearly (each layer gets fresh contribution)
- Richer feature semantics (features know their multi-layer impact)
- Enable building attribution graphs for circuit discovery

---

## 🔍 EleutherAI Implementation Details

### Config Parameters

```python
# Training CLTs for a 12-layer model
python -m sparsify gpt2 \
  --transcode=True \           # Enable transcoder mode
  --cross_layer=12 \           # Predict 12 layers ahead
  --coalesce_topk=concat \     # Concatenate features across layers
  --topk_coalesced=False \     # Apply topk before coalescing
  --post_encoder_scale=True \  # Affine scaling after encoder
  --train_post_encoder=True \  # Trainable bias after encoder
  --skip_connection=True \     # Add skip connection
  --k=32 \                     # 32 active features
  --expansion_factor=128       # 128x hidden dim for feature dim
```

### How `cross_layer` Works

From `trainer.py`:

```python
if cfg.cross_layer > 0:
    n_targets = cfg.cross_layer
    n_targets = min(n_targets, len(hookpoints) - position)  # Don't exceed model depth
    n_sources = len(sources)  # How many earlier layers write to this one
```

Example for 31-layer LLaVA with `cross_layer=16`:

| Layer | n_targets | Predicts Layers | n_sources |
|-------|-----------|-----------------|-----------|
| 0 | 16 | 1-16 | 0 |
| 5 | 16 | 6-21 | 5 |
| 15 | 16 | 16-30 | 15 |
| 20 | 11 | 21-30 | 16 |
| 30 | 0 | - | 16 |

### Multi-Decoder Architecture

From `sparse_coder.py`:

```python
class SparseCoder(nn.Module):
    def __init__(self, d_in, cfg):
        self.multi_target = cfg.n_targets > 0 and cfg.transcode
        
        if self.multi_target:
            self.W_decs = nn.ParameterList()
            self.b_decs = nn.ParameterList()
            self.post_encs = nn.ParameterList()
            
            for i in range(cfg.n_targets):
                self.W_decs.append(nn.Parameter(torch.zeros(num_latents, d_in)))
                self.b_decs.append(nn.Parameter(torch.zeros(d_in)))
                self.post_encs.append(nn.Parameter(torch.zeros(num_latents)))
```

### Forward Pass

```python
def forward(self, x, targets):
    # x: input from residual stream at layer L
    # targets: [y_L+1, y_L+2, ..., y_L+n]
    
    # Encode once
    top_acts, top_indices = self.encode(x)  # [batch, k], [batch, k]
    
    # Decode to each target
    loss = 0
    for i, target in enumerate(targets):
        y_hat = self.decode(top_acts, top_indices, index=i)  # Use W_decs[i]
        loss += F.mse_loss(y_hat, target)
    
    return loss / len(targets)
```

---

## 🛠️ Adapting for VLMs

### Changes Needed

1. **Add VLM Support** to `trainer.py`:
   ```python
   # Support for LlavaForConditionalGeneration
   # Use language_model.model.layers.*.mlp hookpoints
   ```

2. **Update Data Loading**:
   ```python
   # Multi-modal datasets (images + text)
   # Reorganize: batch_*_x.pt contains layer L input
   #             batch_*_y.pt contains [y_L+1, y_L+2, ..., y_L+n]
   ```

3. **Modify Hookpoints**:
   ```python
   # LLaVA uses: "language_model.model.layers.0.mlp", ...
   # vs GPT-2: "h.0.mlp", "h.1.mlp", ...
   ```

### Implementation

See `sparsify/vlm_data.py` and `sparsify/vlm_hooks.py` for VLM-specific implementations.

---

## 📊 CLT Architecture Summary

| Component | Description |
|-----------|-------------|
| **Number of Transcoders** | One per layer (e.g., 31 for LLaVA-7B) |
| **Decoders per Transcoder** | n_targets (e.g., 16 for cross_layer=16) |
| **Training Targets** | Multiple future layers (L+1, L+2, ..., L+n) |
| **Decoder Size** | [feature_dim, hidden_dim] × n_targets |
| **Loss Function** | Σ MSE(y_hat[i], y_L+i) across all targets |
| **Generation Quality** | ~50% token-level match ✅ |
| **Attribution Graphs** | ✅ Yes |
| **Feature Semantics** | Cross-layer circuits |

---

## 📚 Key References

1. **Anthropic Circuit-Tracer Paper**: https://transformer-circuits.pub/2025/attribution-graphs/methods.html
2. **EleutherAI CLT Training**: https://github.com/EleutherAI/clt-training
3. **Original Crosscoders Paper**: https://transformer-circuits.pub/2024/crosscoders/index.html

---

## 🎯 Next Steps

1. **Capture Activations**: Use `scripts/capture_activations.py` with `--n_targets 16`
2. **Train CLTs**: Use `scripts/train_llava_clt.sh` or Python API
3. **Build Attribution Graphs**: Use trained CLTs with `VLMReplacementModel`
4. **Discover Circuits**: Analyze feature→feature interactions
