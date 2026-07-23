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

--------------------------------------------------------------------
FIX LOG (this revision)
--------------------------------------------------------------------
Bug #A — registry stored ABSOLUTE, OS-specific paths (e.g.
    "D:\\Projects\\forex\\memory\\ml_models\\EURUSD_15m\\xgboost_v6.pkl").
    The old "path re-resolution" fallback tried to recover from a moved
    file via `Path(raw).parent.name` / `Path(raw).name`, but that only
    works when the path was written on the SAME OS family that's reading
    it. A Windows-style backslash path handed to Python's PosixPath (on
    Linux) doesn't split on '\\' at all — the whole string becomes a
    single opaque "filename", parent becomes ".", and the resolved
    candidate path is garbage. Every model trained on Windows and then
    audited/loaded on Linux (or vice versa) was reported "missing" even
    when the file was sitting exactly where expected. This was very
    likely the majority cause of a "139 of 189 model files missing"
    audit result — and it in turn triggered unnecessary emergency
    retraining for pairs that were never actually broken.

    Fix: the registry now stores a PORTABLE RELATIVE path
    ("{PAIR}_{tf}/{model_type}_{version}.pkl") instead of an absolute
    one. `_resolve_path()` also understands legacy absolute entries
    written by the old code (Windows OR POSIX style, regardless of which
    OS is running now) so existing registries keep working without a
    migration script.

Bug #B — concurrent `save_model()` calls could compute the same
    "next_version" and race on `_save_registry()` (classic
    read-modify-write lost update; two processes could also overwrite
    the same version's model file). Fix: all registry mutations
    (save/rollback) now go through a cross-process file lock.

Bug #C — `rollback()` only checked that the version LABEL existed in
    the registry, not that its model file still exists on disk. Fix:
    rollback now verifies the target file resolves to a real file
    before switching `latest`.

Bug #D — `list_models()` used direct dict-key access on registry
    entries; a partially-written or hand-edited entry would raise
    KeyError and take down the whole listing. Fix: defensive `.get()`
    access with sane defaults.
"""

from __future__ import annotations
import os
import json
import time
import errno
import contextlib
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath, PurePosixPath
from utils.safe_pickle import safe_pickle_load as _safe_load, safe_pickle_dump as _safe_dump
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

log = get_logger("model_store")

from config import PROJECT_ROOT

MODELS_DIR = PROJECT_ROOT / "memory" / "ml_models"
REGISTRY_PATH = MODELS_DIR / "_registry.json"


class _RegistryLock:
    """Simple cross-process mutex via an exclusive-create lock file.

    Not a perfect distributed lock, but it's dependency-free and good
    enough to serialize registry read-modify-write cycles across
    multiple local training/trading processes on the same machine —
    which is the actual concurrency risk here (parallel retrain jobs,
    trading process + retrain script, etc). Stale locks (owner crashed)
    are broken automatically after `stale_after` seconds.
    """

    def __init__(self, path: Path, timeout: float = 30.0, stale_after: float = 120.0):
        self.lock_path = path
        self.timeout = timeout
        self.stale_after = stale_after
        self._fd: Optional[int] = None

    def __enter__(self) -> "_RegistryLock":
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                self._fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self._fd, str(os.getpid()).encode("utf-8"))
                return self
            except FileExistsError:
                try:
                    age = time.time() - self.lock_path.stat().st_mtime
                    if age > self.stale_after:
                        log.warning(f"[ModelStore] breaking stale registry lock (age={age:.0f}s)")
                        with contextlib.suppress(FileNotFoundError):
                            self.lock_path.unlink()
                        continue
                except FileNotFoundError:
                    continue
                if time.monotonic() >= deadline:
                    log.warning("[ModelStore] registry lock timed out — proceeding without lock")
                    return self
                time.sleep(0.05)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    log.warning(f"[ModelStore] lock error, proceeding without lock: {e}")
                    return self

    def __exit__(self, *exc) -> None:
        if self._fd is not None:
            with contextlib.suppress(Exception):
                os.close(self._fd)
        with contextlib.suppress(FileNotFoundError):
            self.lock_path.unlink()


class ModelStore:
    """Versioned model persistence with rollback support."""

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or MODELS_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.base_dir / "_registry.json"
        self._lock_path = self.base_dir / "_registry.lock"
        self._registry = self._load_registry()

    def _load_registry(self) -> Dict[str, Any]:
        if self.registry_path.exists():
            try:
                return json.loads(self.registry_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"models": {}}

    def _save_registry(self) -> None:
        """Bug #21 fix: atomic write using temp file + os.replace().

        Previously wrote directly with write_text() — a crash mid-write
        would corrupt _registry.json and prevent ALL model loading.
        Caller must hold `_RegistryLock` for multi-process safety.
        """
        import tempfile
        tmp_path = None
        try:
            dir_name = str(self.registry_path.parent)
            with tempfile.NamedTemporaryFile(
                mode="w", dir=dir_name, suffix=".tmp",
                prefix="registry_", delete=False, encoding="utf-8"
            ) as tmp_f:
                tmp_f.write(json.dumps(self._registry, indent=2, default=str))
                tmp_path = tmp_f.name
            os.replace(tmp_path, str(self.registry_path))
        except Exception as e:
            if tmp_path:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_path)
            log.warning(f"[ModelStore] registry save failed: {e}")

    def _pair_dir(self, pair: str, timeframe: str) -> Path:
        d = self.base_dir / f"{pair.upper()}_{timeframe}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def _relative_model_path(pair: str, timeframe: str, filename: str) -> str:
        """Portable, OS-independent path stored in the registry.

        Always POSIX-style ('/') regardless of the OS writing it, so the
        registry can move between Windows and Linux machines untouched.
        """
        return f"{pair.upper()}_{timeframe}/{filename}"

    def _resolve_path(self, raw_path: str) -> Optional[Path]:
        """Resolve a registry-stored path (new relative style OR legacy
        absolute style, written by either Windows or POSIX Python) to a
        real file under `self.base_dir`.

        Returns the resolved Path if the file exists, else None.
        """
        if not raw_path:
            return None

        # 1) New-style: already relative, POSIX-separated.
        candidate = self.base_dir / raw_path
        if candidate.exists():
            return candidate

        # 2) Legacy absolute path written on THIS OS — try as-is first.
        direct = Path(raw_path)
        if direct.exists():
            return direct

        # 3) Legacy absolute path written on a DIFFERENT OS. Parse with
        #    both Windows and POSIX flavors (whichever actually splits
        #    the string into more than one part) to recover the last two
        #    components: "{PAIR}_{tf}/{filename}".
        parts: List[str] = []
        for flavor in (PureWindowsPath, PurePosixPath):
            try:
                p = flavor(raw_path)
                if len(p.parts) > 1:
                    parts = list(p.parts)
                    break
            except Exception:
                continue
        if len(parts) >= 2:
            pair_dir_name, filename = parts[-2], parts[-1]
            resolved = self.base_dir / pair_dir_name / filename
            if resolved.exists():
                return resolved

        return None

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
        with _RegistryLock(self._lock_path):
            # Re-load registry under the lock so we see any update made by
            # another process since __init__ / the last save.
            self._registry = self._load_registry()

            key = f"{pair.upper()}_{timeframe}_{model_type}"
            versions = self._registry["models"].get(key, {}).get("versions", [])
            existing_labels = {v["version"] for v in versions}
            next_num = len(versions) + 1
            version_label = f"v{next_num}"
            while version_label in existing_labels:  # defensive, handles gaps/dupes
                next_num += 1
                version_label = f"v{next_num}"

            pair_dir = self._pair_dir(pair, timeframe)
            filename = f"{model_type}_{version_label}.pkl"
            model_path = pair_dir / filename

            try:
                if is_keras:
                    keras_filename = f"{model_type}_{version_label}.keras"
                    model.save(str(pair_dir / keras_filename))
                    filename = keras_filename
                    model_path = pair_dir / filename
                else:
                    # Keep the model hash beside every new pickle. Loading an
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

            rel_path = self._relative_model_path(pair, timeframe, filename)

            meta = {
                "pair": pair.upper(),
                "timeframe": timeframe,
                "model_type": model_type,
                "version": version_label,
                "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "metrics": metrics,
                "model_path": rel_path,
                "is_keras": is_keras,
                "feature_names": list(feature_names or []),
            }
            meta_path = pair_dir / f"{model_type}_{version_label}_meta.json"
            try:
                meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
            except Exception as e:
                log.warning(f"[ModelStore] meta save failed: {e}")

            versions.append({
                "version": version_label,
                "saved_at": meta["saved_at"],
                "metrics": metrics,
                "is_keras": is_keras,
                "model_path": rel_path,
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

        ver_entry = None
        for v in entry["versions"]:
            if v["version"] == version:
                ver_entry = v
                break
        if ver_entry is None:
            return None

        raw_path = ver_entry.get("model_path", "")
        resolved = self._resolve_path(raw_path)
        if resolved is None:
            log.warning(f"[ModelStore] model file missing: {raw_path} (pair={pair}, tf={timeframe}, type={model_type}, version={version})")
            return None

        # Migrate the entry to the portable relative form once we've
        # successfully resolved it, so future lookups skip the fallback.
        rel = self._relative_model_path(pair, timeframe, resolved.name)
        if ver_entry.get("model_path") != rel:
            ver_entry["model_path"] = rel
            with _RegistryLock(self._lock_path):
                self._registry = self._load_registry()
                # re-apply the same update against the freshly loaded registry
                for v in self._registry.get("models", {}).get(key, {}).get("versions", []):
                    if v.get("version") == version:
                        v["model_path"] = rel
                        break
                self._save_registry()

        try:
            if ver_entry.get("is_keras"):
                try:
                    from tensorflow import keras
                    return keras.models.load_model(str(resolved))
                except ImportError:
                    log.warning("[ModelStore] tensorflow not installed — cannot load keras model")
                    return None
            else:
                # safe_pickle_load expects a filepath STRING, not a file object
                return _safe_load(str(resolved))
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
        """Roll back to a previous version (sets it as 'latest').

        Verifies the target version's model file actually resolves on
        disk before switching — rolling back to a version whose artifact
        is missing would just move the NOT_READY failure to a different
        version label instead of preventing it.
        """
        key = f"{pair.upper()}_{timeframe}_{model_type}"
        entry = self._registry["models"].get(key)
        if not entry:
            log.warning(f"[ModelStore] rollback failed: no entry for {key}")
            return False

        target = next((v for v in entry["versions"] if v["version"] == to_version), None)
        if target is None:
            log.warning(f"[ModelStore] version {to_version} not found for {key}")
            return False

        if self._resolve_path(target.get("model_path", "")) is None:
            log.warning(
                f"[ModelStore] rollback aborted: {key} {to_version} "
                f"model file does not exist on disk (expected {target.get('model_path')})"
            )
            return False

        with _RegistryLock(self._lock_path):
            self._registry = self._load_registry()
            live_entry = self._registry["models"].get(key)
            if live_entry is None:
                return False
            live_entry["latest"] = to_version
            self._save_registry()

        log.info(f"[ModelStore] rolled back {key} to {to_version}")
        return True

    def audit_registry_vs_disk(self) -> Dict[str, Any]:
        """Startup consistency check: verify every model file the registry
        claims to have actually exists on disk.

        Uses `_resolve_path()`, which understands both the new portable
        relative paths and legacy absolute paths written by either
        Windows or POSIX Python — so an entry saved on one OS is no
        longer misreported as "missing" when audited from another.

        Returns:
            {
                "checked": int,          # total (pair, tf, model_type, version) entries examined
                "ok": int,               # entries whose file exists (directly or after re-resolve)
                "missing": [             # entries whose file could not be found at all
                    {"key": str, "pair": str, "timeframe": str,
                     "model_type": str, "version": str, "expected_path": str},
                    ...
                ],
            }
        """
        checked = 0
        ok = 0
        missing: List[Dict[str, Any]] = []

        for key, entry in self._registry.get("models", {}).items():
            pair = entry.get("pair", "")
            timeframe = entry.get("timeframe", "")
            model_type = entry.get("model_type", "")
            for ver_entry in entry.get("versions", []):
                checked += 1
                raw_path = ver_entry.get("model_path", "")
                if self._resolve_path(raw_path) is not None:
                    ok += 1
                else:
                    missing.append({
                        "key": key,
                        "pair": pair,
                        "timeframe": timeframe,
                        "model_type": model_type,
                        "version": ver_entry.get("version"),
                        "expected_path": raw_path,
                    })

        return {"checked": checked, "ok": ok, "missing": missing}

    def list_models(self, pair: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all models (optionally filtered by pair)."""
        result = []
        for key, entry in self._registry["models"].items():
            if pair and not key.startswith(pair.upper()):
                continue
            versions = entry.get("versions", [])
            result.append({
                "key": key,
                "pair": entry.get("pair", key.split("_")[0] if "_" in key else key),
                "timeframe": entry.get("timeframe"),
                "model_type": entry.get("model_type"),
                "latest": entry.get("latest"),
                "versions": len(versions),
                "latest_metrics": versions[-1].get("metrics", {}) if versions else {},
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