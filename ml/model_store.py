"""
ml/model_store.py — Model version control + persistence (Day 69)
==================================================================

Manages ML model artifacts on disk with full version control:
  - Save models with semantic versioning (v1, v2, v3, ...)
  - Load the latest version OR a specific version
  - Rollback to a previous version if the current one underperforms
  - Track model metadata (accuracy, AUC, trained_at, training_size)

Directory layout:
    memory/ml_models/
    ├── EURUSD_15m/
    │   ├── xgboost_v1.pkl + xgboost_v1_meta.json
    │   ├── xgboost_v2.pkl + xgboost_v2_meta.json
    │   ├── random_forest_v1.pkl + ...
    │   └── lstm_v1.keras + ...
    ├── GBPUSD_15m/
    │   └── ...
    └── _registry.json  (global version index)
"""

from __future__ import annotations

import json
import logging
import pickle
import shutil
from datetime import datetime, timezone
from pathlib import Path
from utils.safe_pickle import safe_pickle_load as _safe_load, safe_pickle_dump as _safe_dump
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

log = get_logger("model_store")

from config import PROJECT_ROOT

MODELS_DIR = PROJECT_ROOT / "memory" / "ml_models"
REGISTRY_PATH = MODELS_DIR / "_registry.json"


class ModelStore:
    """Versioned model persistence with rollback support."""

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or MODELS_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.base_dir / "_registry.json"
        self._registry = self._load_registry()

    def _load_registry(self) -> Dict[str, Any]:
        if self.registry_path.exists():
            try:
                return json.loads(self.registry_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"models": {}}

    def _save_registry(self) -> None:
        try:
            self.registry_path.write_text(json.dumps(self._registry, indent=2, default=str), encoding="utf-8")
        except Exception as e:
            log.warning(f"[ModelStore] registry save failed: {e}")

    def _pair_dir(self, pair: str, timeframe: str) -> Path:
        d = self.base_dir / f"{pair.upper()}_{timeframe}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_model(
        self,
        model: Any,
        pair: str,
        timeframe: str,
        model_type: str,
        metrics: Dict[str, Any],
        is_keras: bool = False,
        feature_names: Optional[List[str]] = None,
    ) -> str:
        """Save a model with versioning. Returns the version label (e.g. 'v3')."""
        # Determine next version
        key = f"{pair.upper()}_{timeframe}_{model_type}"
        versions = self._registry["models"].get(key, {}).get("versions", [])
        next_version = len(versions) + 1
        version_label = f"v{next_version}"

        pair_dir = self._pair_dir(pair, timeframe)
        model_path = pair_dir / f"{model_type}_{version_label}.pkl"
        meta_path = pair_dir / f"{model_type}_{version_label}_meta.json"

        # Save model
        try:
            if is_keras:
                model.save(str(model_path).replace(".pkl", ".keras"))
                model_path = pair_dir / f"{model_type}_{version_label}.keras"
            else:
                # Keep the model hash beside every new pickle.  Loading an
                # unverified model is unsafe and used to generate the
                # recurring "No metadata file" production warning.
                _safe_dump(model, str(model_path), metadata={
                    "pair": pair.upper(), "timeframe": timeframe,
                    "model_type": model_type, "version": version_label,
                    "feature_names": list(feature_names or []),
                })
        except Exception as e:
            log.error(f"[ModelStore] model save failed: {e}")
            return ""

        # Save metadata
        meta = {
            "pair": pair.upper(),
            "timeframe": timeframe,
            "model_type": model_type,
            "version": version_label,
            "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "metrics": metrics,
            "model_path": str(model_path),
            "is_keras": is_keras,
            "feature_names": list(feature_names or []),
        }
        try:
            meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        except Exception as e:
            log.warning(f"[ModelStore] meta save failed: {e}")

        # Update registry
        versions.append({
            "version": version_label,
            "saved_at": meta["saved_at"],
            "metrics": metrics,
            "is_keras": is_keras,
            "model_path": str(model_path),
            "meta_path": str(meta_path),
            "feature_names": list(feature_names or []),
        })
        self._registry["models"][key] = {
            "pair": pair.upper(),
            "timeframe": timeframe,
            "model_type": model_type,
            "versions": versions,
            "latest": version_label,
        }
        self._save_registry()

        log.info(f"[ModelStore] saved {key} {version_label} | acc={metrics.get('accuracy', 0):.1%}")
        return version_label

    def load_model(
        self,
        pair: str,
        timeframe: str,
        model_type: str,
        version: Optional[str] = None,
    ) -> Optional[Any]:
        """Load a model. If version=None, loads the latest."""
        key = f"{pair.upper()}_{timeframe}_{model_type}"
        entry = self._registry["models"].get(key)
        if not entry or not entry.get("versions"):
            return None

        if version is None:
            version = entry.get("latest")
        if version is None:
            return None

        # Find the version entry
        ver_entry = None
        for v in entry["versions"]:
            if v["version"] == version:
                ver_entry = v
                break
        if ver_entry is None:
            return None

        model_path = Path(ver_entry["model_path"])
        # Co-founder fix: path re-resolution for stale registry entries
        if not model_path.exists():
            try:
                _filename = model_path.name
                _pair_dir_name = model_path.parent.name
                _resolved = self.base_dir / _pair_dir_name / _filename
                if _resolved.exists():
                    log.info(f"[ModelStore] path re-resolved: {model_path} → {_resolved}")
                    model_path = _resolved
                    ver_entry["model_path"] = str(_resolved)
                    self._save_registry()
                else:
                    log.warning(f"[ModelStore] model file missing: {model_path} (also tried {_resolved})")
                    return None
            except Exception as e:
                log.warning(f"[ModelStore] model file missing: {model_path} ({e})")
                return None

        try:
            if ver_entry.get("is_keras"):
                try:
                    from tensorflow import keras
                    return keras.models.load_model(str(model_path))
                except ImportError:
                    log.warning("[ModelStore] tensorflow not installed — cannot load keras model")
                    return None
            else:
                # Co-founder fix: safe_pickle_load expects a filepath STRING, not a file object
                return _safe_load(str(model_path))
        except Exception as e:
            log.error(f"[ModelStore] model load failed: {e}")
            return None

    def rollback(
        self,
        pair: str,
        timeframe: str,
        model_type: str,
        to_version: str,
    ) -> bool:
        """Roll back to a previous version (sets it as 'latest')."""
        key = f"{pair.upper()}_{timeframe}_{model_type}"
        entry = self._registry["models"].get(key)
        if not entry:
            return False
        # Verify version exists
        found = any(v["version"] == to_version for v in entry["versions"])
        if not found:
            log.warning(f"[ModelStore] version {to_version} not found for {key}")
            return False
        entry["latest"] = to_version
        self._save_registry()
        log.info(f"[ModelStore] rolled back {key} to {to_version}")
        return True

    def list_models(self, pair: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all models (optionally filtered by pair)."""
        result = []
        for key, entry in self._registry["models"].items():
            if pair and not key.startswith(pair.upper()):
                continue
            result.append({
                "key": key,
                "pair": entry["pair"],
                "timeframe": entry["timeframe"],
                "model_type": entry["model_type"],
                "latest": entry.get("latest"),
                "versions": len(entry.get("versions", [])),
                "latest_metrics": entry["versions"][-1]["metrics"] if entry.get("versions") else {},
            })
        return result

    def get_latest_metrics(self, pair: str, timeframe: str, model_type: str) -> Optional[Dict]:
        """Get the metrics of the latest model version."""
        key = f"{pair.upper()}_{timeframe}_{model_type}"
        entry = self._registry["models"].get(key)
        if not entry or not entry.get("versions"):
            return None
        return entry["versions"][-1].get("metrics")

    def get_feature_names(
        self, pair: str, timeframe: str, model_type: str, version: Optional[str] = None,
    ) -> List[str]:
        """Return the feature schema saved with a model version.

        An empty list denotes a legacy artifact with no schema; callers must
        not guess an ordering for it.
        """
        key = f"{pair.upper()}_{timeframe}_{model_type}"
        entry = self._registry["models"].get(key, {})
        version = version or entry.get("latest")
        for item in entry.get("versions", []):
            if item.get("version") == version:
                return list(item.get("feature_names") or [])
        return []


# ── Singleton ───────────────────────────────────────────────────────

_STORE: Optional[ModelStore] = None


def get_model_store() -> ModelStore:
    global _STORE
    if _STORE is None:
        _STORE = ModelStore()
    return _STORE
