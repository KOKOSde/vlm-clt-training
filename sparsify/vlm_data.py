"""Data loading for Vision-Language Models (VLMs)."""

import glob
import os
from pathlib import Path
from typing import Iterator, List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset


class VLMActivationDataset(IterableDataset):
    """
    Dataset for loading pre-captured VLM activations for CLT training.
    
    Expected structure:
        activations_dir/
            L0/
                batch_0_x.pt  # Input: residual stream at layer 0
                batch_0_y.pt  # Target: MLP output at layer 1
                batch_1_x.pt
                batch_1_y.pt
                ...
            L1/
                batch_0_x.pt
                batch_0_y.pt
                ...
    
    For CLT training, we need to reorganize this to support multi-target prediction:
        - x: residual stream at layer L
        - targets: [mlp_out_L+1, mlp_out_L+2, ..., mlp_out_L+n_targets]
    """
    
    def __init__(
        self,
        activations_dir: str,
        layer: int,
        n_targets: int = 1,
        shuffle: bool = True,
        max_batches: int | None = None,
    ):
        """
        Args:
            activations_dir: Root directory containing activation files
            layer: Which layer to load activations from (source layer)
            n_targets: Number of future layers to predict (for CLT)
            shuffle: Whether to shuffle batch order
            max_batches: Maximum number of batches to load (None = all)
        """
        self.activations_dir = Path(activations_dir)
        self.layer = layer
        self.n_targets = n_targets
        self.shuffle = shuffle
        self.max_batches = max_batches
        
        # Find all batch files for the source layer
        layer_dir = self.activations_dir / f"L{layer}"
        if not layer_dir.exists():
            raise FileNotFoundError(f"Layer directory not found: {layer_dir}")
        
        # Load batch pairs (x, y)
        x_files = sorted(glob.glob(str(layer_dir / "batch_*_x.pt")))
        y_files = sorted(glob.glob(str(layer_dir / "batch_*_y.pt")))
        
        # Match x and y files by batch index
        self.batch_pairs = self._match_batch_files(x_files, y_files)
        
        if self.max_batches:
            self.batch_pairs = self.batch_pairs[:self.max_batches]
        
        # For CLT: also need to load targets from future layers
        self.target_layers = list(range(layer + 1, min(layer + n_targets + 1, 32)))  # Assume 32 layers max
        
    def _match_batch_files(self, x_files: List[str], y_files: List[str]) -> List[Tuple[str, str]]:
        """Match x and y files by extracting batch indices."""
        x_dict = {}
        for x_path in x_files:
            basename = os.path.basename(x_path)
            # Extract batch index from filename like "batch_0_x.pt"
            batch_idx = basename.split('_')[1]
            x_dict[batch_idx] = x_path
        
        y_dict = {}
        for y_path in y_files:
            basename = os.path.basename(y_path)
            batch_idx = basename.split('_')[1]
            y_dict[batch_idx] = y_path
        
        # Match pairs
        common_idx = sorted(set(x_dict.keys()) & set(y_dict.keys()))
        return [(x_dict[idx], y_dict[idx]) for idx in common_idx]
    
    def _load_batch(self, x_path: str, y_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load a single batch pair."""
        x_data = torch.load(x_path, map_location='cpu')
        y_data = torch.load(y_path, map_location='cpu')
        
        # Handle both dict and direct tensor formats
        x = x_data.get('x', x_data.get('data', x_data)) if isinstance(x_data, dict) else x_data
        y = y_data.get('y', y_data.get('data', y_data)) if isinstance(y_data, dict) else y_data
        
        # Ensure 3D: [N, T, H]
        if x.ndim == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
        if y.ndim == 4 and y.shape[1] == 1:
            y = y.squeeze(1)
        
        return x, y
    
    def _load_multi_target_batch(self, batch_idx: int) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Load a batch with multiple targets for CLT training.
        
        Returns:
            x: Input from source layer [N, T, H]
            targets: List of outputs from future layers, each [N, T, H]
        """
        # Load source layer
        x_path, y_path = self.batch_pairs[batch_idx]
        x, first_target = self._load_batch(x_path, y_path)
        
        # For n_targets > 1, load additional future layers
        if self.n_targets > 1:
            targets = [first_target]
            
            # Load from additional future layers
            for target_layer in self.target_layers[1:]:
                target_dir = self.activations_dir / f"L{target_layer}"
                # Extract batch index from filename
                batch_num = os.path.basename(y_path).split('_')[1]
                target_y_path = target_dir / f"batch_{batch_num}_y.pt"
                
                if target_y_path.exists():
                    _, target_y = self._load_batch(str(x_path), str(target_y_path))
                    targets.append(target_y)
                else:
                    # If target doesn't exist, pad with zeros
                    targets.append(torch.zeros_like(first_target))
            
            return x, targets
        else:
            return x, [first_target]
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Iterate over batches."""
        indices = list(range(len(self.batch_pairs)))
        
        if self.shuffle:
            import random
            random.shuffle(indices)
        
        for idx in indices:
            yield self._load_multi_target_batch(idx)
    
    def __len__(self) -> int:
        return len(self.batch_pairs)


class MultimodalDataset(Dataset):
    """
    Dataset for multimodal (image + text) data.
    
    Used for capturing activations from a VLM during inference.
    """
    
    def __init__(
        self,
        image_paths: List[str],
        prompts: List[str],
        processor,
    ):
        """
        Args:
            image_paths: List of paths to images
            prompts: List of text prompts
            processor: HuggingFace processor for the VLM
        """
        assert len(image_paths) == len(prompts), "Must have equal images and prompts"
        self.image_paths = image_paths
        self.prompts = prompts
        self.processor = processor
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int):
        """Load and process a single sample."""
        image = Image.open(self.image_paths[idx]).convert('RGB')
        prompt = self.prompts[idx]
        
        # Process with VLM processor
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors='pt',
            padding=True,
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'attention_mask': inputs.get('attention_mask', torch.ones_like(inputs['input_ids'])).squeeze(0),
        }


def collate_multimodal(batch):
    """Collate function for multimodal batches."""
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'pixel_values': torch.stack([item['pixel_values'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
    }

