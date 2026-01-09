#!/usr/bin/env python3
"""Download models from HuggingFace Hub using domestic mirrors"""

import argparse
import os
import sys

HF_MIRROR = "https://hf-mirror.com"


def download_from_hf_mirror(model_id: str, local_dir: str = None, revision: str = None):
    """Download model using HF-Mirror (recommended)"""
    os.environ["HF_ENDPOINT"] = HF_MIRROR
    
    from huggingface_hub import snapshot_download
    
    if local_dir is None:
        local_dir = os.path.join("./models", model_id.replace("/", "_"))
    
    print(f"Downloading from HF-Mirror: {model_id}")
    print(f"Target directory: {local_dir}")
    
    path = snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        revision=revision,
        resume_download=True,
    )
    return path


def download_from_modelscope(model_id: str, local_dir: str = None, revision: str = None):
    """Download model from ModelScope"""
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError:
        print("Error: modelscope not installed. Run: pip install modelscope")
        sys.exit(1)
    
    if local_dir is None:
        local_dir = "./models"
    
    print(f"Downloading from ModelScope: {model_id}")
    print(f"Cache directory: {local_dir}")
    print("Note: Model ID on ModelScope may differ from HuggingFace")
    
    path = snapshot_download(
        model_id=model_id,
        cache_dir=local_dir,
        revision=revision,
    )
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Download models using domestic mirrors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s BAAI/bge-large-zh-v1.5
  %(prog)s Qwen/Qwen2.5-7B-Instruct --dir ./models/qwen
  %(prog)s BAAI/bge-large-zh-v1.5 --source modelscope

Sources:
  hf-mirror   - HuggingFace Mirror (default, recommended)
  modelscope  - Alibaba ModelScope
        """,
    )
    parser.add_argument(
        "model_id",
        help="Model ID (e.g., BAAI/bge-large-zh-v1.5)",
    )
    parser.add_argument(
        "--source",
        choices=["hf-mirror", "modelscope"],
        default="hf-mirror",
        help="Download source (default: hf-mirror)",
    )
    parser.add_argument(
        "--dir", "-d",
        dest="local_dir",
        help="Local directory to save the model",
    )
    parser.add_argument(
        "--revision", "-r",
        help="Model revision/branch (default: main)",
    )
    
    args = parser.parse_args()
    
    try:
        if args.source == "hf-mirror":
            path = download_from_hf_mirror(
                args.model_id,
                args.local_dir,
                args.revision,
            )
        else:
            path = download_from_modelscope(
                args.model_id,
                args.local_dir,
                args.revision,
            )
        
        print(f"\nDownload completed!")
        print(f"Model saved to: {path}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
