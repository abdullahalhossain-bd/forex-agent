"""
utils/safe_pickle.py — Safe pickle loading with integrity verification
======================================================================

pickle.load is a security risk — arbitrary code execution if file is tampered.
This module provides a safe wrapper that:
  1. Verifies file hash before loading (detects tampering)
  2. Restricts unpickler to known-safe classes (whitelist)
  3. Logs all pickle operations for audit

Usage:
    from utils.safe_pickle import safe_pickle_load, safe_pickle_dump

    # Save (with integrity hash)
    safe_pickle_dump(model, "model.pkl")

    # Load (verifies hash, restricts classes)
    model = safe_pickle_load("model.pkl")
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any, Optional, Set

from utils.logger import get_logger

log = get_logger("safe_pickle")


# Whitelist of allowed classes for unpickling
# Co-founder fix: expanded to include xgboost/lightgbm/catboost + more
ALLOWED_CLASSES: Set[str] = {
    # Builtins
    "builtins.dict", "builtins.list", "builtins.tuple", "builtins.set",
    "builtins.int", "builtins.float", "builtins.str", "builtins.bool",
    "builtins.bytes", "builtins.NoneType",
    "builtins.complex", "builtins.range", "builtins.frozenset",
    "builtins.bytearray", "builtins.memoryview", "builtins.slice",
    # Collections
    "collections.OrderedDict", "collections.defaultdict",
    "collections.Counter",
    # Numpy (comprehensive)
    "numpy.ndarray", "numpy.dtype", "numpy.float64", "numpy.int64",
    "numpy.float32", "numpy.int32", "numpy.float16", "numpy.int16",
    "numpy.uint8", "numpy.uint16", "numpy.uint32", "numpy.uint64",
    "numpy.bool_", "numpy.complex128", "numpy.complex64",
    "numpy.core.multiarray._reconstruct",
    "numpy.core.multiarray.scalar",
    "numpy._core.multiarray._reconstruct",
    "numpy._core.multiarray.scalar",
    "numpy.ma.core.MaskedArray",
    # sklearn (comprehensive)
    "sklearn.linear_model._base.LinearRegression",
    "sklearn.linear_model._logistic.LogisticRegression",
    "sklearn.ensemble._forest.RandomForestClassifier",
    "sklearn.ensemble._forest.RandomForestRegressor",
    "sklearn.ensemble._forest.ForestClassifier",
    "sklearn.ensemble._forest.ForestRegressor",
    "sklearn.tree._classes.DecisionTreeClassifier",
    "sklearn.tree._classes.DecisionTreeRegressor",
    "sklearn.ensemble._gb.GradientBoostingClassifier",
    "sklearn.ensemble._gb.GradientBoostingRegressor",
    "sklearn.svm._classes.SVC", "sklearn.svm._classes.SVR",
    "sklearn.pipeline.Pipeline",
    "sklearn.preprocessing._data.StandardScaler",
    "sklearn.preprocessing._data.MinMaxScaler",
    "sklearn.preprocessing._data.RobustScaler",
    "sklearn.preprocessing._label.LabelEncoder",
    # XGBoost
    "xgboost.sklearn.XGBClassifier",
    "xgboost.sklearn.XGBRegressor",
    "xgboost.sklearn.XGBRanker",
    "xgboost.core.Booster",
    "xgboost.core.DMatrix",
    # LightGBM
    "lightgbm.sklearn.LGBMClassifier",
    "lightgbm.sklearn.LGBMRegressor",
    "lightgbm.sklearn.LGBMRanker",
    "lightgbm.basic.Booster",
    "lightgbm.basic.Dataset",
    # CatBoost
    "catboost.core.CatBoostClassifier",
    "catboost.core.CatBoostRegressor",
    "catboost.core.CatBoostRanker",
    "catboost.core.CatBoost",
    # Pandas
    "pandas.core.frame.DataFrame", "pandas.core.series.Series",
    "pandas.core.indexes.base.Index",
    "pandas.core.indexes.range.RangeIndex",
    "pandas.core.indexes.datetimes.DatetimeIndex",
    # Joblib
    "joblib.numpy_pickle.NumpyArrayWrapper",
    "joblib.numpy_pickle.NumpyPickle",
    # scipy
    "scipy.sparse._csr.csr_matrix",
    "scipy.sparse._csc.csc_matrix",
    "scipy.sparse._base.spmatrix",
}

# Module prefixes that are always safe (for ML model loading)
# Necessary because ML libraries add new internal classes across versions
SAFE_MODULE_PREFIXES: tuple = (
    "numpy.",
    "sklearn.",
    "xgboost.",
    "lightgbm.",
    "catboost.",
    "pandas.",
    "scipy.",
    "joblib.",
)


class RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that only allows whitelisted classes + safe prefixes."""

    def find_class(self, module: str, name: str) -> Any:
        full_name = f"{module}.{name}"
        # Check explicit whitelist first
        if full_name in ALLOWED_CLASSES:
            return super().find_class(module, name)
        # Check safe module prefixes (for ML library internals)
        for prefix in SAFE_MODULE_PREFIXES:
            if full_name.startswith(prefix):
                return super().find_class(module, name)
        # Block everything else
        raise pickle.UnpicklingError(
            f"Forbidden class: {full_name} — not in whitelist or safe prefixes. "
            f"Add it to ALLOWED_CLASSES or SAFE_MODULE_PREFIXES."
        )


def _compute_file_hash(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_pickle_dump(obj: Any, filepath: str, metadata: Optional[dict] = None) -> None:
    """
    Save object to pickle file with integrity hash.

    Creates two files:
      - {filepath} — the pickle file
      - {filepath}.meta — JSON with hash, timestamp, metadata

    Args:
        obj: object to save
        filepath: path to save to
        metadata: optional metadata dict (e.g., model version, training data info)
    """
    filepath = str(filepath)
    # Write pickle
    with open(filepath, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Compute hash
    file_hash = _compute_file_hash(filepath)

    # Write metadata
    from datetime import datetime, timezone
    meta = {
        "hash": file_hash,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
    }
    meta_path = filepath + ".meta"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    log.info(f"[SafePickle] Saved {filepath} (hash: {file_hash[:16]}...)")


def safe_pickle_load(filepath: str, verify_hash: bool = True) -> Any:
    """
    Load object from pickle file with integrity verification.

    Args:
        filepath: path to load from
        verify_hash: if True, verify file hash matches metadata

    Returns:
        The loaded object

    Raises:
        FileNotFoundError: if file doesn't exist
        pickle.UnpicklingError: if class not in whitelist
        ValueError: if hash mismatch (file tampered)
    """
    filepath = str(filepath)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Pickle file not found: {filepath}")

    # Verify hash if metadata exists
    if verify_hash:
        meta_path = filepath + ".meta"
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                expected_hash = meta.get("hash", "")
                actual_hash = _compute_file_hash(filepath)
                if expected_hash and actual_hash != expected_hash:
                    raise ValueError(
                        f"HASH MISMATCH for {filepath} — file may be tampered!\n"
                        f"Expected: {expected_hash[:16]}...\n"
                        f"Actual:   {actual_hash[:16]}..."
                    )
                log.debug(f"[SafePickle] Hash verified for {filepath}")
            except json.JSONDecodeError:
                raise ValueError(
                    f"Corrupt metadata for {filepath} — file may be tampered"
                )
        else:
            raise ValueError(
                f"No metadata file for {filepath} — cannot verify integrity"
            )

    # Load with restricted unpickler
    with open(filepath, "rb") as f:
        obj = RestrictedUnpickler(f).load()

    log.info(f"[SafePickle] Loaded {filepath}")
    return obj


# ════════════════════════════════════════════════════════════════
#  CLI ENTRY (smoke test)
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import tempfile

    print("=" * 60)
    print("  SAFE PICKLE — Smoke Test")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save a dict
        test_data = {"model": "test", "weights": [1.0, 2.0, 3.0]}
        filepath = f"{tmpdir}/test.pkl"
        safe_pickle_dump(test_data, filepath, metadata={"version": "1.0"})
        print(f"\n  Saved: {filepath}")

        # Load it back
        loaded = safe_pickle_load(filepath)
        print(f"  Loaded: {loaded}")
        assert loaded == test_data

        # Test tamper detection
        with open(filepath, "ab") as f:
            f.write(b"TAMPERED")
        print(f"\n  Tampered with file...")
        try:
            safe_pickle_load(filepath)
            print(f"  ✗ FAILED — tamper not detected")
        except ValueError as e:
            print(f"  ✓ Tamper detected: {str(e)[:60]}...")

    print("\n" + "=" * 60)
