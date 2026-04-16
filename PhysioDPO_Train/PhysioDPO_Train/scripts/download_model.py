#!/usr/bin/env python3
"""
Download a model from HuggingFace.
Includes a progress bar and download speed reporting.

Usage:
    python scripts/download_model.py
    python scripts/download_model.py --model_id hugohrban/progen2-xlarge --local_dir ./models/progen2-xlarge
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Configure the HuggingFace endpoint
os.environ.setdefault('HF_ENDPOINT', 'https://huggingface.co')

from huggingface_hub import snapshot_download, hf_hub_download, HfApi
from huggingface_hub.utils import tqdm as hf_tqdm
from tqdm import tqdm
import requests


def get_model_info(repo_id: str):
    """Fetch model metadata."""
    try:
        api = HfApi()
        model_info = api.model_info(repo_id)
        return model_info
    except Exception as e:
        print(f"Failed to fetch model info: {e}")
        return None


def format_size(size_bytes):
    """Format a file size into a human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"


def download_with_progress(repo_id: str, local_dir: str, use_mirror: bool = False):
    """
    Download a model and display progress and speed information.
    """
    # Configure endpoint
    if use_mirror:
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        print("Using mirror endpoint: https://hf-mirror.com")
    else:
        os.environ['HF_ENDPOINT'] = 'https://huggingface.co'
        print("Using official endpoint: https://huggingface.co")
    
    # Create the destination directory
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Model ID: {repo_id}")
    print(f"Download directory: {local_path.absolute()}")
    print(f"{'='*60}\n")
    
    # Fetch model information
    print("Fetching model information...")
    try:
        api = HfApi()
        model_info = api.model_info(repo_id)
        
        # List repository files
        files = api.list_repo_files(repo_id)
        print(f"Model contains {len(files)} files:")
        for f in files[:10]:
            print(f"  - {f}")
        if len(files) > 10:
            print(f"  ... and {len(files) - 10} more files")
        print()
    except Exception as e:
        print(f"Failed to fetch model information: {e}")
        print("Continuing with the download attempt...\n")
    
    # Start download
    print("Starting model download...")
    print("(The progress bar will appear below)\n")
    
    start_time = time.time()
    
    try:
        # Download the full repository with a built-in tqdm progress bar
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_path),
            local_dir_use_symlinks=False,
            resume_download=True,  # Resume partial downloads when possible
        )
        
        elapsed_time = time.time() - start_time
        
        # Compute total downloaded size
        total_size = sum(
            f.stat().st_size 
            for f in Path(downloaded_path).rglob('*') 
            if f.is_file()
        )
        
        print(f"\n{'='*60}")
        print(f"✓ Download complete!")
        print(f"  Download directory: {downloaded_path}")
        print(f"  Total size: {format_size(total_size)}")
        print(f"  Elapsed time: {elapsed_time:.1f} seconds")
        if elapsed_time > 0:
            print(f"  Average speed: {format_size(total_size / elapsed_time)}/s")
        print(f"{'='*60}\n")
        
        return downloaded_path
        
    except KeyboardInterrupt:
        print("\n\n⚠ Download interrupted by user")
        print("Downloaded files were kept and the next run will resume automatically when possible")
        return None
        
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        
        # If the mirror fails, retry with the official site
        if use_mirror:
            print("\nMirror download failed. Retrying with the official HuggingFace endpoint...")
            return download_with_progress(repo_id, local_dir, use_mirror=False)
        
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Download a model from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download the default model (progen2-xlarge)
    python scripts/download_model.py
    
    # Download a specific model
    python scripts/download_model.py --model_id facebook/opt-125m --local_dir ./models/opt-125m
    
    # Download via the mirror endpoint
    python scripts/download_model.py --mirror
"""
    )
    
    parser.add_argument(
        "--model_id", 
        type=str, 
        default="hugohrban/progen2-xlarge",
        help="HuggingFace model ID (default: hugohrban/progen2-xlarge)"
    )
    
    parser.add_argument(
        "--local_dir", 
        type=str, 
        default="./models/progen2-xlarge",
        help="Local output directory (default: ./models/progen2-xlarge)"
    )
    
    parser.add_argument(
        "--mirror", 
        action="store_true",
        help="Use the mirror endpoint (hf-mirror.com)"
    )
    
    args = parser.parse_args()
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║              HuggingFace Model Downloader                   ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    result = download_with_progress(
        repo_id=args.model_id,
        local_dir=args.local_dir,
        use_mirror=args.mirror
    )
    
    if result:
        print("Next step: update the training script to use the local model path")
        print(f"  --model_id {args.local_dir}")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())

