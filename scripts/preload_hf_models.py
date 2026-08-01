"""Pre-download Hugging Face models to local cache to avoid runtime timeouts.

Usage:
    python scripts/preload_hf_models.py

This script downloads `sentence-transformers/all-MiniLM-L6-v2` into a local
cache directory and sets `HF_HOME` so subsequent runs use the cached files.
"""
from pathlib import Path
import os
import sys

MODEL_REPO = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    workspace_root = Path(__file__).resolve().parents[1]
    default_cache = workspace_root / "data" / "hf_cache"
    cache_dir = Path(os.environ.get("HF_HOME", default_cache))
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Export HF_HOME for the current process so HF libraries use this cache.
    os.environ["HF_HOME"] = str(cache_dir)

    print(f"Using HF cache directory: {cache_dir}")

    # Try to use huggingface_hub.snapshot_download first (reliable).
    try:
        from huggingface_hub import snapshot_download

        print(f"Downloading {MODEL_REPO} via huggingface_hub.snapshot_download...")
        snapshot_download(repo_id=MODEL_REPO, cache_dir=str(cache_dir), allow_regex=".*")
        print("Download complete.")
        return 0
    except Exception as e:
        print("huggingface_hub not available or failed:", e)

    # Fallback: try sentence-transformers to trigger a download.
    try:
        from sentence_transformers import SentenceTransformer

        print(f"Downloading {MODEL_REPO} via SentenceTransformer(...)")
        SentenceTransformer(MODEL_REPO)
        print("Download complete.")
        return 0
    except Exception as e:
        print("sentence-transformers fallback failed:", e)

    print("Failed to download model. Install huggingface_hub or sentence-transformers and try again.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
