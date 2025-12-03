# VLM CLT Training

**Cross-Layer Transcoder (CLT) training for Vision-Language Models**

This repository adapts [EleutherAI's CLT training](https://github.com/EleutherAI/clt-training) for Vision-Language Models (VLMs), enabling circuit discovery and mechanistic interpretability research on multimodal models.

Based on Anthropic's [Circuit-Tracer methodology](https://transformer-circuits.pub/2025/attribution-graphs/methods.html).

---

## 🎯 What are Cross-Layer Transcoders (CLTs)?

Cross-Layer Transcoders are sparse autoencoders that:
- **Read** from the residual stream at layer L
- **Write** to multiple future MLP layers (L+1, L+2, ..., L+n)
- Enable discovering **cross-layer circuits** in neural networks
- Achieve ~50% token-level match when replacing MLPs during generation
- Enable building attribution graphs for circuit discovery

### Key Features

- **Multi-target decoders**: Each transcoder has multiple decoder heads (n_targets)
- **Cross-layer predictions**: Features predict multiple future layers simultaneously
- **Attribution graphs**: Can trace feature→feature interactions across layers
- **Circuit discovery**: Identify interpretable computational circuits in VLMs

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/KOKOSde/vlm-clt-training
cd vlm-clt-training
pip install -e .
```

### Pre-captured Activations

We provide pre-captured activations from LLaVA-1.5-7B on AMBER dataset (1004 samples, 496MB):

**Option 1: Download from Hugging Face** (Coming soon)
```bash
python scripts/download_activations.py --dataset amber
```

**Option 2: Use existing activations**
If you already have activations at `/scratch/fkalghan/circuit_discovery_and_supression/benchmarks_llava/activations/amber/`:
```bash
ln -s /scratch/fkalghan/circuit_discovery_and_supression/benchmarks_llava/activations/amber ./activations
```

**Option 3: Capture your own**
See "Capture Activations" section below.

### Capture Activations

First, capture activations from your VLM on a multimodal dataset:

```bash
python scripts/capture_activations.py \
  --model llava-hf/llava-1.5-7b-hf \
  --dataset path/to/amber/queries.json \
  --image_dir path/to/images \
  --output_dir ./activations \
  --n_targets 16  # Predict 16 layers ahead
```

This will create:
```
activations/
  L0/
    batch_0_x.pt  # Residual stream at layer 0
    batch_0_y.pt  # Multi-target: [MLP_1, MLP_2, ..., MLP_16]
  L1/
    batch_0_x.pt
    batch_0_y.pt
  ...
```

### Train CLTs

**Single layer** (recommended to start):
```bash
bash scripts/train_single_layer.sh 0  # Train layer 0
```

**Or directly with Python**:

```bash
python scripts/train_llava_clt.py \
  --config config.yaml \
  --layer 0 \
  --clt-mode \
  --steps 5000 \
  --lr 3e-4 \
  --feature-dim 8192 \
  --batch-samples 16 \
  --compile
```

**Outputs**:
- `transcoder_L0.pt` - Trained CLT weights
- `mapping_L0.pt` - MLP→CLT correlation mapping for deployment
- `metrics_L0.json` - Training metrics and dead feature statistics

---

## 📁 Repository Structure

```
vlm-clt-training/
├── sparsify/                 # Core library
│   ├── sparse_coder.py      # CLT implementation (from EleutherAI)
│   ├── config.py            # Configuration
│   ├── vlm_data.py          # VLM-specific data loading
│   ├── vlm_hooks.py         # VLM hookpoint utilities
│   └── trainer.py           # Training loop
├── scripts/                  # Training scripts
│   ├── capture_activations.py  # Capture VLM activations
│   ├── train_llava_clt.sh     # Train CLTs for LLaVA
│   └── evaluate_replacement.py # Test replacement model
├── examples/                 # Example notebooks
│   ├── 01_capture_activations.ipynb
│   ├── 02_train_clt.ipynb
│   └── 03_attribution_graphs.ipynb
└── docs/                     # Documentation
    ├── CLT_ARCHITECTURE.md  # Detailed architecture explanation
    └── PHASE_GUIDE.md       # Full pipeline guide
```

---

## 🔬 Methodology

This implementation follows Anthropic's Circuit-Tracer methodology:

### 1. Architecture

```python
# CLT for layer L
Encoder: [hidden_dim] → [feature_dim]  # Read residual stream
TopK: Select k active features
Decoders: [
    W_dec[0]: [feature_dim] → [hidden_dim]  # → Layer L+1
    W_dec[1]: [feature_dim] → [hidden_dim]  # → Layer L+2
    ...
    W_dec[n-1]: [feature_dim] → [hidden_dim]  # → Layer L+n
]
```

### 2. Training

```python
# Multi-target loss
loss = Σ MSE(W_dec[i](features), target_layer[L+i+1])
       i=0 to n_targets-1
```

### 3. Replacement Model

```python
# During inference, aggregate contributions from all source layers
for source_layer in range(current_layer):
    features = encoder(residual_stream[source_layer])
    decoder_idx = current_layer - source_layer - 1
    mlp_output += decoder[decoder_idx](features)
```


---

## 🛠️ Advanced Usage

### Custom VLM Support

To add support for a new VLM:

```python
# In sparsify/vlm_hooks.py
def get_my_vlm_hookpoints(model, patterns=None):
    """Get hookpoints for MyVLM."""
    if patterns is None:
        patterns = ["model.layers.*.mlp"]  # Adjust to your model
    
    hookpoints = {}
    for name, module in model.named_modules():
        if any(fnmatch(name, p) for p in patterns):
            hookpoints[name] = module
    
    return hookpoints
```

### Distributed Training

```bash
torchrun --nproc_per_node=4 -m sparsify.train_vlm \
  --model llava-hf/llava-1.5-7b-hf \
  --activations_dir ./activations \
  --tp 4  # Tensor parallelism across 4 GPUs
```

### Attribution Graphs

```python
from sparsify.vlm_hooks import VLMReplacementModel

# Create replacement model with CLTs
replacement_model = VLMReplacementModel(
    base_model=model,
    transcoders=trained_clts,
    hookpoints=hookpoints,
)

# Enable feature capture
replacement_model.capture_features = True

# Run inference
outputs = replacement_model.generate(**inputs)

# Extract attribution graph
feature_activations = replacement_model.feature_cache
# Build graph showing which features influenced the output
```

---

## 📚 Documentation

- **[CLT Architecture Analysis](docs/CLT_ARCHITECTURE_ANALYSIS.md)** - Detailed CLT architecture explanation
- **[Phase Guide](docs/PHASE_GUIDE.md)** - Complete pipeline from training to circuit discovery
- **[Anthropic Paper](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)** - Original methodology
- **[EleutherAI CLT Training](https://github.com/EleutherAI/clt-training)** - Text-only CLT implementation

---

## 🎓 Citation

If you use this code in your research, please cite:

```bibtex
@misc{vlm-clt-training,
  author = {KOKOSde},
  title = {Cross-Layer Transcoder Training for Vision-Language Models},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/KOKOSde/vlm-clt-training}
}

@article{anthropic2025attribution,
  title={Circuit Tracing: Revealing Computational Graphs in Language Models},
  author={Anthropic},
  year={2025},
  url={https://transformer-circuits.pub/2025/attribution-graphs/methods.html}
}
```

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

Based on [EleutherAI/clt-training](https://github.com/EleutherAI/clt-training) (MIT License).

---

## 🙏 Acknowledgments

- **Anthropic** for the Circuit-Tracer methodology
- **EleutherAI** for the original CLT training code
- **HuggingFace** for model hosting and infrastructure

---

## 📮 Contact

For questions or issues, please open a GitHub issue at:
https://github.com/KOKOSde/vlm-clt-training/issues

Happy circuit discovering! 🔬✨

