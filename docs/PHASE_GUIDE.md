# Feature Discovery Pipeline for Vision-Language Models
## Generalizable Phase Guide for Any Dataset/Benchmark

---

## Overview
This pipeline identifies interpretable features from Cross-Layer Transcoders (CLTs) that distinguish between good and bad model responses on vision-language benchmarks, then maps them back to MLP neurons for production deployment.

**Based on**: [Circuit-tracer methodology](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) (influence-based feature selection)  
**Applicable to**: AMBER, POPE, MM-Hal, Object-Hal, LLaVA-Bench, or any custom benchmark

### Complete Workflow

```
1. Train CLTs to replace MLPs ✅ DONE
   - Model: llava-hf/llava-1.5-7b-hf
   - CLTs: KokosDev/llava15-7b-clt (31 layers, 8192 features/layer)
   - Training: Multimodal (image+text) data
   - Script: /home/fkalghan/circuit_discovery_and_supression/train/train_llava_transcoder_layer.py
   - Local: /scratch/fkalghan/circuit_discovery_and_supression/transcoders_llava15_7b/
   - HuggingFace: https://huggingface.co/KokosDev/llava15-7b-clt
   - Training Quality: 0% dead features, 2-5% sparsity (early/mid), 7-29% (deep)
   - MLP→CLT Mappings: Included (co-activation based) ✅
   
2. Build Replacement Model ✅ DONE
   - LLaVAReplacementModel (mlp_layers → clt_transcoders)
   - Path: benchmarks_llava/llava_replacement_model.py
   
3. Capture Activations (PHASE 1)
   - Run inference with replacement model
   - Save responses + CLT feature activations
   
4. Evaluate with Official Metrics (PHASE 2)
   - AMBER: CHAIR, Cover, HalRate, Cog
   - POPE: Accuracy, Precision
   
5. Classify GOOD/BAD Samples (PHASE 3)
   - AMBER: 100% coverage = GOOD
   - POPE: Correct answer = GOOD
   
6. Analyze Feature Influence (PHASE 4)
   - Compute goodness_score per feature
   - Round-robin selection (layer diversity)
   - Result: Top 200 features
   
7. Map CLT → MLP Neurons (PHASE 6)
   - Extract decoder weights
   - Identify top MLP neurons per CLT feature
   - Ready for production deployment
   
8. (Optional) Steering & Validation (PHASE 7)
   - Amplify good features
   - Measure metric improvements
   - Validate feature importance
```

### Key Paths

| Component | Path |
|-----------|------|
| **Base Model** | `llava-hf/llava-1.5-7b-hf` (HuggingFace) |
| **CLT Training Script** | `/home/fkalghan/circuit_discovery_and_supression/train/train_llava_transcoder_layer.py` |
| **CLT Transcoders** | `/scratch/fkalghan/circuit_discovery_and_supression/transcoders_llava15_7b/` |
| **CLT HuggingFace** | [`KokosDev/llava15-7b-clt`](https://huggingface.co/KokosDev/llava15-7b-clt) |
| **MLP→CLT Mappings** | `/scratch/fkalghan/circuit_discovery_and_supression/transcoders_llava15_7b/mapping_L{0-30}.pt` |
| **Replacement Model** | `benchmarks_llava/llava_replacement_model.py` |
| **Analysis Scripts** | `benchmarks_llava/analyze_good_features.py` |
| **Results (AMBER)** | `benchmarks_llava/analysis/amber_good_features.json` |
| **Activations (AMBER)** | `benchmarks_llava/activations/amber/` |

---

## PHASE 0: Train Cross-Layer Transcoders (CLTs) ✅ COMPLETED

### Goal
Train sparse autoencoders (CLTs) on LLaVA's MLP layers to decompose dense activations into interpretable features.

### Status
**✅ COMPLETED** - All 31 layers trained and uploaded to HuggingFace

### Training Details

**Script**: `/home/fkalghan/circuit_discovery_and_supression/train/train_llava_transcoder_layer.py`

**Command** (example for single layer):
```bash
python train/train_llava_transcoder_layer.py \
  --config config/transcoder_llava15_7b.yaml \
  --layer 10 \
  --steps 5000 \
  --lr 3e-4 \
  --feature-dim 8192 \
  --batch-samples 16 \
  --cache-batches 800 \
  --val-interval 200 \
  --val-samples 20 \
  --compile
```

**Training All Layers**:
```bash
for LAYER in {0..30}; do
  python train/train_llava_transcoder_layer.py \
    --config config/transcoder_llava15_7b.yaml \
    --layer $LAYER \
    --steps 5000 --lr 3e-4 --feature-dim 8192 \
    --batch-samples 16 --cache-batches 800 \
    --val-interval 200 --val-samples 20 --compile
done
```

### Training Results

| Metric | Value |
|--------|-------|
| **Layers trained** | 31 (L0-L30) |
| **Steps per layer** | 5,000 |
| **Hidden dim** | 4,096 |
| **Feature dim** | 8,192 (2× expansion) |
| **Dead features** | 0% (all layers) |
| **Sparsity (early layers L0-L10)** | 2-4% L0 |
| **Sparsity (middle layers L11-L18)** | 3-7% L0 |
| **Sparsity (deep layers L19-L30)** | 7-29% L0 (expected) |
| **Reconstruction loss** | <0.1 for most layers |

### Output Files

**Per-layer files** (in `/scratch/fkalghan/circuit_discovery_and_supression/transcoders_llava15_7b/`):
1. `transcoder_L{layer}.pt` - Trained transcoder with state_dict + metadata
2. `mapping_L{layer}.pt` - MLP→CLT co-activation mapping + decoder weights

**HuggingFace Repository**: https://huggingface.co/KokosDev/llava15-7b-clt
- All 31 transcoders uploaded
- All 31 MLP→CLT mappings uploaded
- Full documentation in README

### Key Features
- ✅ **Multimodal training**: Trained on both image and text data (Flickr30K + instruction tasks)
- ✅ **MLP→CLT mapping**: Co-activation based correlation matrix saved during training
- ✅ **Decoder weights**: Explicit CLT→MLP reconstruction weights for mapping back
- ✅ **0% dead features**: All features are active and useful
- ✅ **Layer-appropriate sparsity**: Early layers denser (2-4%), deep layers sparser (7-29%)

### Next Steps
With CLTs trained, proceed to:
1. **Build Replacement Model** (LLaVAReplacementModel) ✅ DONE
2. **Capture activations** on AMBER/POPE using replacement model → **NEXT PHASE**

---

## PHASE 1: Capture Activations and Generate Responses

### Goal
Generate model responses using the **LLaVA Replacement Model** (MLPs replaced by CLT transcoders) and capture CLT feature activations during forward pass.

### Input
- **Dataset**: AMBER, POPE, or other vision-language benchmark
- **Base Model**: `llava-hf/llava-1.5-7b-hf` (HuggingFace)
- **CLT Transcoders**: 
  - Local: `/scratch/fkalghan/circuit_discovery_and_supression/transcoders_llava15_7b/`
  - HuggingFace: [`KokosDev/llava15-7b-clt`](https://huggingface.co/KokosDev/llava15-7b-clt)
  - 31 layers (L0-L30), 8192 features per layer
  - Trained on multimodal (image+text) data

### Process
1. **Load Replacement Model**: 
   ```python
   from llava_replacement_model import LLaVAReplacementModel
   
   replacement_model = LLaVAReplacementModel.from_pretrained(
       model_name="llava-hf/llava-1.5-7b-hf",
       transcoder_dir="/scratch/fkalghan/circuit_discovery_and_supression/transcoders_llava15_7b",
       capture_activations=True
   )
   ```
   - This replaces all MLP layers with CLT transcoders
   - Forward pass computes: `X (layer output) → CLT → Y_hat (MLP replacement)`

2. **For each sample**:
   - Load image and question
   - Run inference through replacement model
   - CLT features computed **during forward pass**, not post-hoc
   - Save response and feature activations

### Output
- `results/{benchmark}/answer/model_responses.jsonl` - Model responses (using CLT-replaced MLPs)
- `activations/{benchmark}/sample_XXXX.pt` - CLT activations per sample (dict {layer: tensor})

### Script
`capture_{benchmark}_activations.py` (updated to use `LLaVAReplacementModel`)

### Key Code
**Path**: `/scratch/fkalghan/circuit_discovery_and_supression/benchmarks_llava/llava_replacement_model.py`

**Architecture**:
```
Image + Text → Vision Encoder (CLIP) → Image features
                                             ↓
Text tokens → LLaVA Language Model:
              Layer 0: Self-Attn → [CLT replaces MLP] → Residual
              Layer 1: Self-Attn → [CLT replaces MLP] → Residual
              ...
              Layer 30: Self-Attn → [CLT replaces MLP] → Output
```

**Key Parameters**:
- `model_name`: `"llava-hf/llava-1.5-7b-hf"`
- `transcoder_dir`: `/scratch/fkalghan/circuit_discovery_and_supression/transcoders_llava15_7b`
- `capture_activations`: `True` (captures features during forward pass)
- `temperature`, `top_p`, `max_new_tokens`: Generation parameters

### Why Replacement Model?
Following [Transformer Circuits paper](https://transformer-circuits.pub/2025/attribution-graphs/methods.html):
- ✅ **True feature computation**: Features computed during forward pass, not post-hoc
- ✅ **Attribution graphs**: Can trace feature→feature interactions
- ✅ **Interventions**: Can modify features and observe output changes
- ✅ **~50% match**: CLTs approximate original model outputs on most prompts

---

## PHASE 2: Evaluate with Official Benchmark Metrics

### Goal
Compute official benchmark metrics for each model response to establish ground truth quality.

### Input
- Model responses from Phase 1
- Benchmark annotations/ground truth

### Process
1. Run official benchmark evaluator
2. Compute per-sample metrics (e.g., CHAIR, Cover, HalRate, Cog for AMBER; Accuracy, Precision for POPE)
3. Save metrics alongside responses

### Output
- Evaluation results with official metrics per sample

### Script
Use official benchmark evaluator (e.g., `AMBER_eval.py`, `eval_pope.py`)

### Benchmark-Specific Metrics
**AMBER (Generative)**:
- CHAIR: Hallucinated objects (lower is better)
- Cover: Coverage of ground truth (higher is better)
- HalRate: % sentences with hallucinations (lower is better)
- Cog: Cognitive hallucinations (lower is better)

**POPE (Yes/No)**:
- Accuracy: Overall correctness
- Precision: % correct when model says "yes"

---

## PHASE 3: Classify Samples as GOOD/BAD

### Goal
Classify each sample based on official metrics to create training labels for feature analysis.

### Input
- Model responses with metrics from Phase 2
- Activation files from Phase 1

### Process
1. Analyze metric distributions
2. Define classification threshold based on dataset characteristics
3. Classify samples:
   - **GOOD**: High-quality responses (e.g., 100% coverage for AMBER, correct for POPE)
   - **BAD**: Low-quality responses
4. Verify activation files exist for classified samples

### Output
- `analysis/{benchmark}_classification.json`:
```json
{
  "GOOD": [{"id": 1, "coverage": 1.0, ...}, ...],
  "BAD": [{"id": 2, "coverage": 0.4, ...}, ...]
}
```

### Script
`classify_{benchmark}_simple.py`

### Threshold Guidelines
- **Be strict for GOOD**: Ensures high-quality positive examples
- **Balance sample sizes**: Aim for at least 50+ GOOD samples if possible
- **Dataset-specific**: AMBER (coverage-based), POPE (correctness-based)

---

## PHASE 4: Analyze Feature Influence (Circuit-Tracer Method)

### Goal
Identify CLT features that distinguish GOOD from BAD responses using influence-based selection with layer diversity.

### Input
- **Classification results**: `analysis/{benchmark}_classification.json` (Phase 3)
- **Activation files**: `activations/{benchmark}/sample_XXXX.pt` (Phase 1)
- **Paths**:
  - AMBER: `/scratch/fkalghan/circuit_discovery_and_supression/benchmarks_llava/analysis/amber_classification.json`
  - AMBER activations: `/scratch/fkalghan/circuit_discovery_and_supression/benchmarks_llava/activations/amber/`

### Process
1. **Load activations** for GOOD and BAD samples
2. **Compute feature statistics**:
   - For each (layer, feature) pair:
     - `good_freq`: Activation frequency in GOOD samples
     - `bad_freq`: Activation frequency in BAD samples
     - `good_avg`: Average activation strength in GOOD
     - `bad_avg`: Average activation strength in BAD
3. **Calculate influence score**:
   ```
   goodness_score = (good_freq / (bad_freq + epsilon)) * good_avg
   frequency_ratio = good_freq / (bad_freq + epsilon)
   activation_ratio = good_avg / (bad_avg + epsilon)
   ```
4. **Select features with layer diversity**:
   - **Round-robin across layers** (not greedy top-k)
   - Select best feature from each layer iteratively
   - Ensures balanced representation across all 32 layers
   - Example: For 200 features → ~6-7 features per layer

### Output
- `analysis/{benchmark}_good_features.json`
- **Example (AMBER)**: `/scratch/fkalghan/circuit_discovery_and_supression/benchmarks_llava/analysis/amber_good_features.json`

```json
{
  "dataset": "amber",
  "total_good_samples": 65,
  "total_bad_samples": 195,
  "total_features_analyzed": 126976,
  "selected_features_count": 200,
  "layer_distribution": {"0": 7, "1": 7, ..., "30": 6},
  "features": [
    {
      "layer": 8,
      "feature_idx": 3648,
      "goodness_score": 13.53,
      "frequency_ratio": 0.99,
      "good_freq": 0.98,
      "bad_freq": 0.02,
      "good_avg_activation": 2.5,
      "bad_avg_activation": 0.1
    },
    ...
  ]
}
```

### Script
**Path**: `/scratch/fkalghan/circuit_discovery_and_supression/benchmarks_llava/analyze_good_features.py`

**Usage**:
```bash
python analyze_good_features.py --dataset amber --target-features 200 --min-per-layer 3
```

### Key Parameters (Following Circuit-Tracer)
- `--dataset`: Which benchmark (amber, pope)
- `--target-features`: Upper bound on features (default: 200 for analysis, 7500 for full circuits)
- `--min-per-layer`: Minimum features per layer (default: 5)
- **Selection strategy**: Round-robin across layers, NOT greedy top-k
- **Influence metric**: Combines frequency and activation strength

### Circuit-Tracer Methodology
From [paper](https://transformer-circuits.pub/2025/attribution-graphs/methods.html):
- Default: `max_feature_nodes=7500`, `node_threshold=0.8` (cumulative influence)
- Pruning: `node_threshold=0.8`, `edge_threshold=0.98`
- For **visualization/analysis**: 100-500 features is reasonable
- For **steering/intervention**: Use top 10-50 features per behavior
- For **full attribution graphs**: Up to 7500 features

### Our Results (AMBER)
- **Total active features**: 126,976 across all layers
- **Selected**: 200 features with layer diversity
- **Distribution**: 6-7 features per layer (balanced across L0-L30)
- **Top feature**: L8 F3648 (goodness=13.53, freq_ratio=0.99)

---

## PHASE 5: Visualize and Interpret Features

### Goal
Create visualizations to understand feature behavior and validate discoveries.

### Input
- Feature analysis results from Phase 4
- Activation files from Phase 1

### Process
1. **Bar charts**: Top features by goodness score
2. **Layer distribution**: Features per layer histogram
3. **Activation patterns**: GOOD vs BAD activation heatmaps
4. **Frequency ratios**: GOOD/BAD activation frequency comparisons
5. **(Optional)** Feature interpretation: Top activating samples per feature

### Output
- `analysis/visualizations/{benchmark}_top_features.png`
- `analysis/visualizations/{benchmark}_layer_dist.png`
- `analysis/visualizations/{benchmark}_activation_heatmap.png`

### Script
`visualize_top_features.py --dataset {benchmark}`

---

## PHASE 6: Map CLT Features → MLP Neurons

### Goal
Map identified "good" CLT features back to original MLP neurons for practical deployment.

### Why This Matters
CLTs are an **interpretable intermediate representation**. For production deployment, we need to:
1. Identify important CLT features (Phase 4) ✅
2. Map those features → MLP neurons (This phase)
3. Amplify corresponding MLP neurons in original model
4. Deploy production model without CLTs

### Input
- **Top features**: From `analysis/{benchmark}_good_features.json` (Phase 4)
- **CLT decoder weights**: From transcoders (shape: `[hidden_dim=4096, feature_dim=8192]`)
  - Path: `/scratch/fkalghan/circuit_discovery_and_supression/transcoders_llava15_7b/transcoder_L{layer}.pt`

### Process
The mapping is **implicit in the CLT decoder weights**:

```python
# CLT forward pass:
# 1. X (layer output) → encoder → z_pre
# 2. z = ReLU(z_pre)  # Sparse features
# 3. Y_hat = decoder @ z  # Back to MLP space

# Decoder weight: W_dec[hidden_dim=4096, feature_dim=8192]
# Each column i represents: "how feature_i maps to MLP neurons"

# For a selected feature (layer=L, feature_idx=F):
decoder_weights = transcoder.dec.weight  # (4096, 8192)
feature_to_mlp = decoder_weights[:, F]  # (4096,) - MLP neuron importances

# Top MLP neurons for this feature:
top_neurons = torch.topk(feature_to_mlp.abs(), k=50)
```

### Output
- `analysis/{benchmark}_clt_to_mlp_mapping.json`:
```json
{
  "feature": {"layer": 8, "feature_idx": 3648, "goodness_score": 13.53},
  "top_mlp_neurons": [
    {"neuron_idx": 1523, "weight": 0.85},
    {"neuron_idx": 2841, "weight": 0.72},
    ...
  ],
  "amplification_factor": 2.0
}
```

### Script
`map_clt_to_mlp.py --dataset {benchmark}`

### Deployment Strategy
1. **Steering with CLTs** (Research/Validation):
   ```python
   replacement_model.transcoders[8].features[3648] *= 2.0
   ```

2. **Steering with MLPs** (Production):
   ```python
   original_model.layers[8].mlp.neurons[[1523, 2841, ...]] *= 2.0
   ```

---

## PHASE 7 (Optional): Feature Steering/Intervention

### Goal
Validate discovered features by steering model behavior through feature activation manipulation.

### Input
- **Top features**: `analysis/{benchmark}_good_features.json` (Phase 4)
- **Replacement Model**: `LLaVAReplacementModel` with CLT transcoders
- **Path**: Use `llava_replacement_model.py`

### Process
1. **Load replacement model** with intervention capability
2. **Select top 10-20 features** for target behavior
3. **Run inference with feature steering**:
   - Amplify good features: `feature_value *= 2.0` or `+= offset`
   - Suppress bad features: `feature_value *= 0.0`
4. **Measure metric improvement**:
   - AMBER: CHAIR ↓, Cover ↑, HalRate ↓
   - POPE: Accuracy ↑, Precision ↑

### Expected Results
- **Amplifying good features** → Improved metrics
- **Suppressing bad features** → Degraded metrics (validation)
- **Example (from paper)**: Sycophancy reduction 78% → 12%

### Script
`steer_{benchmark}_features.py`

### Steering Implementation
```python
from llava_replacement_model import LLaVAReplacementModel

# Load model
replacement_model = LLaVAReplacementModel.from_pretrained(...)

# Define intervention
def intervention_hook(layer_idx, features):
    """Modify features during forward pass"""
    if layer_idx == 8:
        features[:, :, 3648] *= 2.0  # Amplify feature 3648
    return features

# Apply intervention
# (Requires adding intervention support to LLaVAReplacementModel)
```

---

## PHASE 7 (Optional): Circuit Attribution

### Goal
Build full attribution graph showing how features influence each other and final outputs.

### Input
- Top features from Phase 4
- ReplacementModel

### Process
1. Run circuit-tracer attribution:
   ```python
   graph = run_attribution(
       prompt=prompt,
       model=replacement_model,
       max_feature_nodes=500,
       desired_logit_prob=0.95
   )
   ```
2. Prune graph (influence threshold)
3. Visualize on Neuronpedia or locally

### Output
- Attribution graph (`.pt` file)
- Interactive visualization (JSON for web UI)

---

## Adapting to New Benchmarks

### Steps
1. **Create capture script**: Adapt `capture_amber_activations.py` template
   - Update image paths
   - Update question format
   - Ensure output matches official evaluator format

2. **Run official evaluator**: Use benchmark's evaluation code
   - Identify key metrics
   - Understand metric ranges

3. **Create classification script**: Adapt `classify_amber_simple.py`
   - Define GOOD threshold (strict, high quality)
   - Define BAD threshold
   - Aim for balanced sample sizes

4. **Run feature analysis**: Use `analyze_good_features.py` (no changes needed!)
   - Specify `--dataset {benchmark_name}`
   - Adjust `--target-features` based on goal

5. **Visualize and validate**: Create benchmark-specific plots

### Benchmark Requirements
- ✅ Image + text input pairs
- ✅ Ground truth annotations
- ✅ Official evaluation metrics
- ✅ Clear definition of "good" vs "bad" responses

---

## Summary of Files

| Phase | Script | Input | Output |
|-------|--------|-------|--------|
| 1 | `capture_{benchmark}_activations.py` | Dataset, Model, CLTs | Responses + Activations |
| 2 | Official evaluator (e.g., `AMBER_eval.py`) | Responses, Ground truth | Metrics |
| 3 | `classify_{benchmark}_simple.py` | Metrics, Activations | Classification JSON |
| 4 | `analyze_good_features.py` | Classification, Activations | Feature rankings JSON |
| 5 | `visualize_top_features.py` | Feature rankings | Plots |
| 6 | `steer_{benchmark}_features.py` | Top features, Model | Steering results |
| 7 | `run_attribution_{benchmark}.py` | Top features, Model | Attribution graph |

---

## Key Principles

1. **Use official metrics**: Never rely on heuristics when official evaluators exist
2. **Influence-based selection**: Follow circuit-tracer methodology, not hardcoded limits
3. **Layer diversity**: Ensure balanced representation across all model layers
4. **Strict GOOD threshold**: High-quality positive examples are critical
5. **Reproducibility**: Save all intermediate results for auditing
6. **Generalizability**: Keep scripts modular and adaptable to new benchmarks

