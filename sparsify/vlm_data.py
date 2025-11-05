"""
VLM dataset handling for LLaVA-style models.
Loads JSONL with image paths, processes through LlavaProcessor.
"""

import json
import os
from pathlib import Path
from typing import Iterator, Dict, Any

import torch
from datasets import Dataset
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer
from transformers.models.clip.image_processing_clip import CLIPImageProcessor


def load_jsonl_with_images(
    jsonl_path: str,
    image_dir: str | None = None,
    text_column: str = "prompt",
    image_column: str = "image",
    max_samples: int | None = None
) -> Dataset:
    """
    Load a JSONL file with image references into a HuggingFace Dataset.
    
    Args:
        jsonl_path: Path to JSONL file
        image_dir: Base directory for images (if not absolute in JSONL)
        text_column: Column name for text/prompt
        image_column: Column name for image paths
        max_samples: Limit number of samples
    
    Returns:
        HuggingFace Dataset with 'text' and 'image_path' columns
    """
    data = []
    
    with open(jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            
            line = line.strip()
            if not line:
                continue
            
            try:
                sample = json.loads(line)
                text = sample.get(text_column, "")
                image_path = sample.get(image_column, "")
                
                # Resolve image path
                if image_path and not os.path.isabs(image_path):
                    if image_dir:
                        image_path = os.path.join(image_dir, image_path)
                    else:
                        # Try relative to JSONL location
                        jsonl_dir = os.path.dirname(jsonl_path)
                        image_path = os.path.join(jsonl_dir, image_path)
                
                # Verify image exists
                if image_path and os.path.exists(image_path):
                    data.append({
                        'text': text,
                        'image_path': image_path
                    })
            except json.JSONDecodeError:
                continue
    
    return Dataset.from_list(data)


def process_vlm_batch(batch: Dict[str, Any], processor, max_length: int = 512) -> Dict[str, Any]:
    """
    Process a batch of VLM data (images + text) through the processor.
    
    Args:
        batch: Dict with 'text' and 'image_path' keys (lists)
        processor: LlavaProcessor or similar VLM processor
        max_length: Max sequence length
    
    Returns:
        Processed batch with 'input_ids', 'attention_mask', 'pixel_values'
    """
    # Import here to avoid issues with multiprocessing
    import torch
    from PIL import Image
    
    images = []
    texts = batch['text']
    
    # Load images with better error handling
    for img_path in batch['image_path']:
        try:
            if img_path and os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB')
                images.append(img)
            else:
                # Missing image: create a blank image
                images.append(Image.new('RGB', (336, 336), color='black'))
        except Exception:
            # Fallback: create a blank image
            images.append(Image.new('RGB', (336, 336), color='black'))
    
    # Process through VLM processor
    try:
        processed = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
        return {
            'input_ids': processed['input_ids'],
            'attention_mask': processed.get('attention_mask', torch.ones_like(processed['input_ids'])),
            'pixel_values': processed['pixel_values']
        }
    except Exception as e:
        # Fallback: return dummy data
        batch_size = len(texts)
        return {
            'input_ids': torch.zeros((batch_size, max_length), dtype=torch.long),
            'attention_mask': torch.zeros((batch_size, max_length), dtype=torch.long),
            'pixel_values': torch.zeros((batch_size, 3, 336, 336))
        }


def create_vlm_processor(model_id: str, hf_token: str | None = None):
    """
    Create a VLM processor (LlavaProcessor) for the given model.
    Falls back to manual creation if AutoProcessor fails.
    """
    try:
        # Try AutoProcessor first
        processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
        return processor
    except Exception:
        # Manual fallback for LLaVA
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, use_fast=True)
        
        from transformers import LlavaProcessor
        image_processor = CLIPImageProcessor(
            size={"shortest_edge": 336},
            crop_size={"height": 336, "width": 336},
            do_center_crop=True,
            do_normalize=True,
            do_resize=True,
            image_mean=[0.48145466, 0.4578275, 0.40821073],
            image_std=[0.26862954, 0.26130258, 0.27577711],
            resample=3,
            do_convert_rgb=True,
        )
        
        processor = LlavaProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            patch_size=14
        )
        return processor
