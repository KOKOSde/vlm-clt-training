#!/usr/bin/env python3
"""
Upload captured activations to Hugging Face Hub.

Activations are too large for GitHub (496MB+), so we upload to HF Hub instead.

Usage:
    python scripts/upload_activations_to_hf.py \\
        --activations_dir ./activations \\
        --repo_id YOUR_USERNAME/vlm-clt-activations \\
        --token YOUR_HF_TOKEN
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations_dir", type=str, required=True,
                       help="Directory containing activation files")
    parser.add_argument("--repo_id", type=str, required=True,
                       help="HuggingFace repo ID (e.g., username/repo-name)")
    parser.add_argument("--token", type=str, default=None,
                       help="HuggingFace API token (or set HF_TOKEN env var)")
    parser.add_argument("--private", action="store_true",
                       help="Make the repository private")
    
    args = parser.parse_args()
    
    api = HfApi(token=args.token)
    
    # Create repository
    print(f"Creating repository: {args.repo_id}")
    try:
        create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=args.private,
            token=args.token,
        )
        print("✅ Repository created")
    except Exception as e:
        print(f"Repository may already exist: {e}")
    
    # Upload activations
    print(f"\\nUploading activations from: {args.activations_dir}")
    activations_path = Path(args.activations_dir)
    
    api.upload_folder(
        folder_path=str(activations_path),
        repo_id=args.repo_id,
        repo_type="dataset",
        token=args.token,
    )
    
    print(f"\\n✅ Activations uploaded successfully!")
    print(f"📦 View at: https://huggingface.co/datasets/{args.repo_id}")
    print(f"\\n📥 Download with:")
    print(f"   from huggingface_hub import snapshot_download")
    print(f"   snapshot_download(repo_id='{args.repo_id}', repo_type='dataset')")


if __name__ == "__main__":
    main()

