# Pre-captured Activations

This repository includes pre-captured activations from LLaVA-1.5-7B on various benchmarks.

## Available Datasets

### AMBER (Generative)

- **Model**: llava-hf/llava-1.5-7b-hf
- **Samples**: 1004
- **Size**: 496MB
- **Format**: PyTorch tensors (`.pt` files)
- **Location**: `activations_amber/` (symlink to local storage)

**Structure**:
```
activations_amber/
  sample_0000.pt  # Contains CLT activations for all 31 layers
  sample_0001.pt
  ...
  sample_1003.pt
```

Each file contains:
```python
{
    'activations': {
        0: tensor([...]),   # Layer 0 CLT features
        1: tensor([...]),   # Layer 1 CLT features
        ...
        30: tensor([...]),  # Layer 30 CLT features
    }
}
```

## Downloading Activations

Since activations are large (496MB+), they are not included in the Git repository.

### Option 1: Use Existing Local Activations

If you're on the same system where activations were captured:

```bash
ln -s /scratch/fkalghan/circuit_discovery_and_supression/benchmarks_llava/activations/amber ./activations_amber
```

### Option 2: Upload to Hugging Face Hub

To share activations publicly:

```bash
python scripts/upload_activations_to_hf.py \
  --activations_dir /scratch/fkalghan/circuit_discovery_and_supression/benchmarks_llava/activations/amber \
  --repo_id KOKOSde/llava-amber-activations \
  --token YOUR_HF_TOKEN
```

Then others can download:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="KOKOSde/llava-amber-activations",
    repo_type="dataset",
    local_dir="./activations_amber"
)
```

### Option 3: Capture Your Own

See `scripts/capture_activations.py`:

```bash
python scripts/capture_activations.py \
  --model llava-hf/llava-1.5-7b-hf \
  --dataset path/to/amber/queries.json \
  --image_dir path/to/images \
  --output_dir ./activations_amber \
  --n_targets 16
```

## Training Data Format

For CLT training, activations should be organized by layer:

```
activations/
  L0/
    batch_0_x.pt  # Input: residual stream at layer 0
    batch_0_y.pt  # Targets: MLP outputs from layers 1-16
  L1/
    batch_0_x.pt
    batch_0_y.pt
  ...
  L30/
    batch_0_x.pt
    batch_0_y.pt
```

Each `batch_*_y.pt` file for CLT training contains:

```python
{
    'y': tensor([...]),        # Primary target (next layer)
    'targets': [               # All targets for CLT
        tensor([...]),         # MLP output at L+1
        tensor([...]),         # MLP output at L+2
        ...
        tensor([...]),         # MLP output at L+16
    ],
    'n_targets': 16,
}
```

## Notes

- **CLT Format**: Activations should be captured with multi-target format for CLT training
- **Storage**: Keep activations outside Git (use `.gitignore`)
- **Sharing**: Use Hugging Face Hub for large activation datasets

