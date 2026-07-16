Preload Hugging Face models
==========================

Run `preload_hf_models.py` to download the `sentence-transformers/all-MiniLM-L6-v2` model
into a local cache directory. This helps avoid transient network timeouts during runtime.

Usage
-----

From the repository root:

```bash
python scripts/preload_hf_models.py
```

Environment variables
---------------------

- `HF_HOME`: (optional) set this to control where the Hugging Face cache is stored.
  If not set, the script will use `data/hf_cache` in the repository.

If you still see `huggingface_hub` read timeouts, consider re-running the script
on a more stable connection or installing `huggingface_hub` and `sentence-transformers`:

```bash
pip install huggingface_hub sentence-transformers
```
