# CLT Architecture Analysis: EleutherAI vs Current Implementation

## Executive Summary

**Current Status**: You trained **PLT (Per-Layer Transcoders)**, not CLT (Cross-Layer Transcoders)  
**Impact**: Replacement model generates only newlines (~0% match vs expected ~50%)  
**Solution**: Need to retrain with true CLT architecture (multi-target decoders)

---

## 🔬 Anthropic's CLT Architecture

Based on [Circuit-Tracer Paper](https://transformer-circuits.pub/2025/attribution-graphs/methods.html):

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

### Why It Works
- Features "bridge" across multiple layers
- Errors don't compound linearly (each layer gets fresh contribution)
- Richer feature semantics (features know their multi-layer impact)

---

## ⚠️ Your Current Implementation (PLT)

### Architecture

```python
# 31 SEPARATE transcoders (one per layer)
Transcoder_L0: [hidden_dim] → [feature_dim] → [hidden_dim]  # Only layer 0 → 1
Transcoder_L1: [hidden_dim] → [feature_dim] → [hidden_dim]  # Only layer 1 → 2
...
Transcoder_L30: [hidden_dim] → [feature_dim] → [hidden_dim]  # Only layer 30 → 31
```

### Training
- **Input**: Residual stream at layer L
- **Target**: MLP output at layer L+1 ONLY
- **Loss**: `MSE(decoder(features), target)` for single layer
- **Result**: ~0% token-level match (catastrophic error compounding)

### Why It Fails
1. **Error Compounding**:
   - Layer 0: 94% accurate → 6% error
   - Layer 1: compounds to ~88% (0.94²)
   - Layer 30: **0.94³¹ ≈ 14%** signal remaining
   
2. **No Cross-Layer Information**:
   - Features at layer L don't know about layers L+2, L+3, etc.
   - Can't learn multi-hop circuits

3. **Degenerate Generation**:
   - Small errors accumulate
   - By layer 10-15, the model is lost
   - Defaults to repetitive patterns (newlines in your case)

---

## 📊 Comparison Table

| Feature | Your PLT | EleutherAI CLT | Anthropic CLT |
|---------|----------|----------------|---------------|
| **Number of Transcoders** | 31 (one per layer) | 31 (one per layer) | Unified across layers |
| **Decoders per Transcoder** | 1 | n_targets (e.g., 12) | n_targets |
| **Training Targets** | Next layer only | Multiple future layers | All subsequent layers |
| **Decoder Size** | [8192, 4096] | [8192, 4096] × 12 | [feature_dim, hidden_dim] × n |
| **Loss Function** | MSE(y_hat, y_L+1) | Σ MSE(y_hat[i], y_L+i) | Same |
| **Generation Quality** | 0% match ❌ | ~50% match ✅ | ~50% match ✅ |
| **Can Build Attribution Graphs** | ❌ No | ✅ Yes | ✅ Yes |
| **Feature Semantics** | Per-layer only | Cross-layer circuits | Cross-layer circuits |

---

## 🔍 EleutherAI Implementation Details

### Config Parameters (from their codebase)

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

## 🛠️ Minimal Changes for VLMs

### Option 1: Adapt EleutherAI's Code (Recommended)

**Pros**: Battle-tested, distributed training, full CLT support  
**Cons**: Requires understanding their codebase, may need VLM-specific hooks

**Changes Needed**:

1. **Add VLM Support** to `trainer.py`:
   ```python
   # Current: only supports HF text models
   # Need: support for LlavaForConditionalGeneration
   
   # Add vision encoder hooks (if needed)
   # Most likely: just use language_model.model.layers.*.mlp
   ```

2. **Update Data Loading**:
   ```python
   # Current: text-only datasets
   # Need: multimodal datasets (images + text)
   
   # Your existing activation capture can provide data!
   # Just need to reorganize: batch_*_x.pt contains layer L input
   #                           batch_*_y.pt contains [y_L+1, y_L+2, ..., y_L+n]
   ```

3. **Modify Hookpoints**:
   ```python
   # EleutherAI uses: "h.0.mlp", "h.1.mlp", ...
   # LLaVA uses: "language_model.model.layers.0.mlp", ...
   ```

### Option 2: Update Your Existing Code

**Pros**: You control everything, familiar codebase  
**Cons**: More work, need to implement multi-target training from scratch

**Changes Needed**:

1. **Update `Transcoder` class** in `train_llava_transcoder_layer.py`:
   ```python
   class Transcoder(nn.Module):
       def __init__(self, hidden_dim, feature_dim, n_targets=16):
           super().__init__()
           self.n_targets = n_targets
           self.enc = nn.Sequential(
               nn.LayerNorm(hidden_dim),
               nn.Linear(hidden_dim, feature_dim),
           )
           
           # Multiple decoder heads
           self.decoders = nn.ModuleList([
               nn.Linear(feature_dim, hidden_dim)
               for _ in range(n_targets)
           ])
       
       def forward(self, x, targets):
           # x: [B, T, H] - input from layer L
           # targets: list of [B, T, H] - outputs for layers L+1, ..., L+n
           
           z_pre = self.enc(x)
           z = torch.relu(z_pre)  # or topk
           
           losses = []
           y_hats = []
           for i, (decoder, target) in enumerate(zip(self.decoders, targets)):
               y_hat = decoder(z)
               loss = F.mse_loss(y_hat, target)
               losses.append(loss)
               y_hats.append(y_hat)
           
           return sum(losses) / len(losses), z, y_hats
   ```

2. **Update Data Capture**:
   ```python
   # Currently: capture_activations.py saves (X_L, Y_L) pairs
   # Need: save (X_L, [Y_L+1, Y_L+2, ..., Y_L+n]) tuples
   
   # Example structure:
   # activations/L0/batch_0.pt = {
   #     'x': residual_L0,
   #     'targets': [mlp_out_L1, mlp_out_L2, ..., mlp_out_L16]
   # }
   ```

3. **Update Training Loop**:
   ```python
   # Load multi-target data
   batch = torch.load(batch_file)
   x = batch['x']  # [B, T, H]
   targets = batch['targets']  # list of n_targets tensors
   
   # Forward
   loss, z, y_hats = transcoder(x, targets)
   
   # Backward
   loss.backward()
   optimizer.step()
   ```

4. **Update Replacement Model**:
   ```python
   # Currently: hooks replace MLP output with single decoder
   # Need: aggregate contributions from ALL source layers
   
   def forward_hook(module, input, output):
       hidden_states = input[0]
       layer_idx = ...
       
       # Collect contributions from all source layers
       total_contribution = torch.zeros_like(output)
       
       for source_layer in range(max(0, layer_idx - 16), layer_idx):
           transcoder = self.transcoders[source_layer]
           decoder_idx = layer_idx - source_layer - 1
           
           # Get features from source layer (cached during forward)
           source_features = self.cached_features[source_layer]
           
           # Decode to current layer
           contribution = transcoder.decoders[decoder_idx](source_features)
           total_contribution += contribution
       
       return total_contribution
   ```

---

## 💡 Recommendation

### Short-term (Next 1-2 weeks):
**Continue with PLT analysis** using baseline model responses
- You already have 1004 activation files ✅
- You already have baseline responses ✅
- Can still discover interesting per-layer features
- **Deliverable**: "Hallucinogenic Features in VLM MLP Layers"

### Long-term (Research Paper):
**Retrain with true CLT architecture** using EleutherAI's code
- Adapt their `sparsify` library for LLaVA
- Use your existing multimodal activation data
- Get ~50% generation match
- Build full attribution graphs
- **Deliverable**: "Cross-Layer Hallucination Circuits in VLMs"

---

## 📚 Key References

1. **Anthropic Circuit-Tracer Paper**: https://transformer-circuits.pub/2025/attribution-graphs/methods.html
2. **EleutherAI CLT Training**: https://github.com/EleutherAI/clt-training
3. **Original Crosscoders Paper**: https://transformer-circuits.pub/2024/crosscoders/index.html
4. **Your Current Implementation**: `/home/fkalghan/circuit_discovery_and_supression/train/train_llava_transcoder_layer.py`

---

## 🎯 Next Steps

1. **Decision Point**: Continue with PLT or retrain with CLT?
2. **If PLT**: Proceed to Phase 2-4 using baseline responses
3. **If CLT**: Start adapting EleutherAI's code for VLMs
4. **Either way**: You'll get valuable interpretability results!

The question is: Do you want **per-layer features** (PLT, faster) or **cross-layer circuits** (CLT, more powerful)?

