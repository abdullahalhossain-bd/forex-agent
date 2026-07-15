#!/usr/bin/env python3
"""
scripts/regenerate_pickle_metadata.py
======================================

FIX (2026-07-15): trader.log showed 12x
  "[SafePickle] No metadata file for ...pkl — cannot verify integrity"
for every existing model under memory/ml_models/ (EURUSD, GBPUSD, USDJPY,
USDCAD, AUDUSD, XAUUSD).

Those models were saved before utils/safe_pickle.py's hash-metadata
convention existed, so they never got a `.meta` sidecar file. This script
back-fills a `.meta` file for every `.pkl` under memory/ml_models/ that is
missing one, computing the hash from the file as it exists *right now*.

IMPORTANT HONESTY NOTE: this does NOT prove the files haven't already
been tampered with — it just establishes a fresh, verifiable baseline
from this point forward (the whole point of hash-checking is to detect
*future* tampering). The metadata's `regenerated: true` flag makes this
explicit so nobody mistakes it for an original training-time hash.

Usage:
    python scripts/regenerate_pickle_metadata.py
    python scripts/regenerate_pickle_metadata.py --dry-run
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)


def compute_hash(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="List what would be created, don't write anything")
    parser.add_argument("--dir", default=os.path.join(PROJECT_ROOT, "memory", "ml_models"))
    args = parser.parse_args()

    root = Path(args.dir)
    if not root.exists():
        print(f"Directory not found: {root}")
        return 1

    pkls = sorted(root.rglob("*.pkl"))
    missing = [p for p in pkls if not Path(str(p) + ".meta").exists()]

    print(f"Found {len(pkls)} .pkl files, {len(missing)} missing .meta sidecars.")

    for pkl_path in missing:
        meta_path = Path(str(pkl_path) + ".meta")
        if args.dry_run:
            print(f"  [dry-run] would create {meta_path}")
            continue
        file_hash = compute_hash(pkl_path)
        meta = {
            "hash": file_hash,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "regenerated": True,
            "metadata": {
                "note": (
                    "Metadata back-filled post-hoc; this file predates the "
                    "safe_pickle .meta convention. Hash reflects file content "
                    "at regeneration time, not at original training time."
                ),
                "pair_timeframe_dir": pkl_path.parent.name,
                "model_file": pkl_path.name,
            },
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"  created {meta_path} (hash={file_hash[:16]}...)")

    if args.dry_run:
        print("\nDry run — nothing written. Re-run without --dry-run to apply.")
    else:
        print(f"\nDone. {len(missing)} .meta files created.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
