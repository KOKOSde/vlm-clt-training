#!/usr/bin/env python3
"""
Capture activations from a Vision-Language Model for CLT training.

This script runs inference on a VLM and captures:
- Residual stream activations (input to each MLP layer)
- MLP outputs (targets for transcoder training)

For CLT training, we need multi-target data:
    X_L: residual stream at layer L
    Y_L+1, Y_L+2, ..., Y_L+n: MLP outputs at future layers

Usage:
    python scripts/capture_activations.py \\
        --model llava-hf/llava-1.5-7b-hf \\
        --dataset path/to/multimodal/dataset \\
        --output_dir ./activations \\
        --n_targets 16
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration


def capture_layer_activations(
    model: LlavaForConditionalGeneration,
    inputs: Dict[str, torch.Tensor],
    device: str = "cuda",
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Capture residual stream and MLP outputs for all layers.
    
    Returns:
        Dictionary: {layer_idx: {'residual': tensor, 'mlp_out': tensor}}
    """
    activations = {}
    
    # Register hooks to capture activations
    def make_hook(layer_idx, activation_type):
        def hook_fn(module, input, output):
            if layer_idx not in activations:
                activations[layer_idx] = {}
            
            if activation_type == "residual":
                # Input to MLP (residual stream)
                activations[layer_idx]['residual'] = input[0].detach().cpu()
            else:  # mlp_out
                # Output of MLP
                if isinstance(output, tuple):
                    activations[layer_idx]['mlp_out'] = output[0].detach().cpu()
                else:
                    activations[layer_idx]['mlp_out'] = output.detach().cpu()
        
        return hook_fn
    
    hooks = []
    
    # Install hooks on all language model layers
    for layer_idx, layer in enumerate(model.language_model.model.layers):
        # Hook before MLP (residual stream)
        h1 = layer.mlp.register_forward_hook(
            make_hook(layer_idx, "residual")
        )
        # Hook after MLP (output)
        h2 = layer.mlp.register_forward_hook(
            make_hook(layer_idx, "mlp_out")
        )
        hooks.extend([h1, h2])
    
    # Run forward pass
    with torch.no_grad():
        _ = model(**inputs)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return activations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                       help="HuggingFace model name")
    parser.add_argument("--dataset", type=str, required=True,
                       help="Path to dataset JSON (AMBER format)")
    parser.add_argument("--image_dir", type=str, required=True,
                       help="Directory containing images")
    parser.add_argument("--output_dir", type=str, default="./activations",
                       help="Output directory for activations")
    parser.add_argument("--n_targets", type=int, default=16,
                       help="Number of future layers to save as targets")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to process")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for processing")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    processor = AutoProcessor.from_pretrained(args.model)
    model.eval()
    
    print(f"Loading dataset: {args.dataset}")
    with open(args.dataset) as f:
        dataset = json.load(f)
    
    if args.max_samples:
        dataset = dataset[:args.max_samples]
    
    # Create output directories for each layer
    output_dir = Path(args.output_dir)
    num_layers = len(model.language_model.model.layers)
    
    for layer_idx in range(num_layers):
        (output_dir / f"L{layer_idx}").mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {len(dataset)} samples...")
    print(f"Number of layers: {num_layers}")
    print(f"Targets per source layer: {args.n_targets}")
    
    batch_idx = 0
    for i in tqdm(range(0, len(dataset), args.batch_size)):
        batch_samples = dataset[i:i + args.batch_size]
        
        # Load images and prepare prompts
        images = []
        prompts = []
        for sample in batch_samples:
            image_path = os.path.join(args.image_dir, sample['image'])
            image = Image.open(image_path).convert('RGB')
            images.append(image)
            
            prompt = f"USER: <image>\\n{sample['query']}\\nASSISTANT:"
            prompts.append(prompt)
        
        # Process batch
        inputs = processor(
            text=prompts,
            images=images,
            return_tensors='pt',
            padding=True,
        )
        inputs = {k: v.to(args.device) for k, v in inputs.items()}
        
        # Capture activations
        activations = capture_layer_activations(model, inputs, args.device)
        
        # Save activations for CLT training
        # For each source layer L, save:
        #   - X: residual stream at L
        #   - Y: targets [MLP_L+1, MLP_L+2, ..., MLP_L+n]
        
        for source_layer in range(num_layers):
            if source_layer not in activations:
                continue
            
            # Input: residual stream at source layer
            x = activations[source_layer]['residual']
            
            # Targets: MLP outputs from future layers
            targets = []
            for target_offset in range(1, args.n_targets + 1):
                target_layer = source_layer + target_offset
                if target_layer < num_layers and target_layer in activations:
                    targets.append(activations[target_layer]['mlp_out'])
                else:
                    # Pad with zeros if target layer doesn't exist
                    targets.append(torch.zeros_like(x))
            
            # Save batch
            layer_dir = output_dir / f"L{source_layer}"
            
            # Save input (X)
            torch.save(
                {'x': x},
                layer_dir / f"batch_{batch_idx}_x.pt"
            )
            
            # Save targets (Y)
            # For single-target (PLT), save just first target
            # For multi-target (CLT), save all targets
            if args.n_targets == 1:
                torch.save(
                    {'y': targets[0]},
                    layer_dir / f"batch_{batch_idx}_y.pt"
                )
            else:
                # Save multi-target format for CLT
                torch.save(
                    {
                        'y': targets[0],  # Primary target (next layer)
                        'targets': targets,  # All targets
                        'n_targets': len(targets),
                    },
                    layer_dir / f"batch_{batch_idx}_y.pt"
                )
        
        batch_idx += 1
    
    print(f"\\nSaved {batch_idx} batches to {output_dir}")
    print(f"Format: L{{layer}}/batch_{{idx}}_{{x|y}}.pt")
    print("\\nReady for CLT training!")


if __name__ == "__main__":
    main()

