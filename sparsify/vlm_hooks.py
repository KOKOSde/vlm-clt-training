"""Hook utilities for Vision-Language Models."""

from fnmatch import fnmatch
from typing import Dict, List

import torch
import torch.nn as nn
from transformers import LlavaForConditionalGeneration, PreTrainedModel


def get_vlm_hookpoints(
    model: PreTrainedModel,
    patterns: List[str] | None = None,
    model_type: str = "llava",
) -> Dict[str, nn.Module]:
    """
    Get hookpoints for a Vision-Language Model.
    
    Args:
        model: The VLM model
        patterns: List of patterns to match (Unix-style, e.g., "*.mlp")
                 If None, defaults to all MLP layers in language model
        model_type: Type of VLM ("llava", "qwen", etc.)
    
    Returns:
        Dictionary mapping hookpoint names to modules
    """
    if model_type == "llava":
        return get_llava_hookpoints(model, patterns)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def get_llava_hookpoints(
    model: LlavaForConditionalGeneration,
    patterns: List[str] | None = None,
) -> Dict[str, nn.Module]:
    """
    Get hookpoints for LLaVA model.
    
    LLaVA structure:
        model.vision_tower (CLIP vision encoder) - usually not transcoded
        model.multi_modal_projector - usually not transcoded
        model.language_model.model.layers[i].mlp - TARGET for CLT
        model.language_model.model.layers[i].self_attn - not transcoded
    
    Args:
        model: LLaVA model
        patterns: Patterns to match hookpoint names
                 Default: ["language_model.model.layers.*.mlp"]
    
    Returns:
        Dictionary of {hookpoint_name: module}
    """
    if patterns is None:
        # Default: all MLP layers in language model
        patterns = ["language_model.model.layers.*.mlp"]
    
    hookpoints = {}
    
    # Iterate through all named modules
    for name, module in model.named_modules():
        # Check if name matches any pattern
        for pattern in patterns:
            if fnmatch(name, pattern):
                hookpoints[name] = module
                break
    
    return hookpoints


def get_layer_indices_from_hookpoints(hookpoints: Dict[str, nn.Module]) -> List[int]:
    """
    Extract layer indices from hookpoint names.
    
    Example:
        "language_model.model.layers.10.mlp" -> 10
    
    Args:
        hookpoints: Dictionary of hookpoint names to modules
    
    Returns:
        Sorted list of layer indices
    """
    indices = []
    for name in hookpoints.keys():
        # Extract number from patterns like "layers.10.mlp"
        parts = name.split('.')
        for i, part in enumerate(parts):
            if part == 'layers' and i + 1 < len(parts):
                try:
                    idx = int(parts[i + 1])
                    indices.append(idx)
                    break
                except ValueError:
                    continue
    
    return sorted(set(indices))


def register_activation_hooks(
    model: nn.Module,
    hookpoints: Dict[str, nn.Module],
    storage: Dict[str, torch.Tensor],
    hook_type: str = "input",
) -> List:
    """
    Register hooks to capture activations.
    
    Args:
        model: The model
        hookpoints: Dictionary of hookpoint names to modules
        storage: Dictionary to store captured activations
        hook_type: "input" or "output"
    
    Returns:
        List of hook handles (call .remove() to unregister)
    """
    handles = []
    
    for name, module in hookpoints.items():
        def make_hook(hook_name):
            def hook_fn(module, input, output):
                if hook_type == "input":
                    # Store input (typically a tuple, take first element)
                    storage[hook_name] = input[0].detach().cpu()
                else:
                    # Store output
                    if isinstance(output, tuple):
                        storage[hook_name] = output[0].detach().cpu()
                    else:
                        storage[hook_name] = output.detach().cpu()
            return hook_fn
        
        handle = module.register_forward_hook(make_hook(name))
        handles.append(handle)
    
    return handles


class VLMReplacementModel(nn.Module):
    """
    Wrapper for VLM with CLT-replaced MLPs.
    
    This implements the Anthropic attribution graph methodology:
    - Each layer's features contribute to multiple future layers
    - Attention patterns and layer norms are frozen during attribution
    """
    
    def __init__(
        self,
        base_model: PreTrainedModel,
        transcoders: Dict[str, nn.Module],
        hookpoints: Dict[str, nn.Module],
    ):
        super().__init__()
        self.base_model = base_model
        self.transcoders = transcoders
        self.hookpoints = hookpoints
        self.hooks = []
        
        # Cache for feature activations (for attribution graphs)
        self.feature_cache = {}
        self.capture_features = False
        
        self._install_hooks()
    
    def _install_hooks(self):
        """Install forward hooks to replace MLP outputs with CLT outputs."""
        for name, module in self.hookpoints.items():
            # Extract layer index
            layer_idx = int(name.split('layers.')[1].split('.')[0])
            
            def make_hook(layer_idx, hook_name):
                def forward_hook(module, input, output):
                    # Get input to MLP (residual stream)
                    hidden_states = input[0]
                    
                    # Get transcoder for this layer
                    transcoder = self.transcoders.get(f"L{layer_idx}")
                    if transcoder is None:
                        return output  # No replacement
                    
                    # Forward through transcoder
                    # For multi-target CLT, this returns a MidDecoder object
                    result = transcoder(hidden_states)
                    
                    # Cache features if requested
                    if self.capture_features:
                        # Extract features from result
                        if hasattr(result, 'latent_acts'):
                            self.feature_cache[layer_idx] = result.latent_acts.detach()
                    
                    # Return replacement output
                    if isinstance(output, tuple):
                        return (result.sae_out,) + output[1:]
                    else:
                        return result.sae_out
                
                return forward_hook
            
            handle = module.register_forward_hook(make_hook(layer_idx, name))
            self.hooks.append(handle)
    
    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def forward(self, **kwargs):
        """Forward pass through VLM with CLT replacements."""
        return self.base_model(**kwargs)
    
    def generate(self, **kwargs):
        """Generate text using VLM with CLT replacements."""
        return self.base_model.generate(**kwargs)

