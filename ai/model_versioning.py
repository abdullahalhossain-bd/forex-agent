"""
Model Versioning System
Handles model versioning, storage, and deployment

NOTE: TensorFlow and MLflow are OPTIONAL. The system works without them.
"""
import os
import json
import logging
import pickle
import tempfile
import threading
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

# Optional ML dependencies
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    # BUGFIX: previously `tf`/`keras` were left completely unbound here.
    # save_model_version()/load_model_version() reference `keras.Model` /
    # `tf.keras.Model` unconditionally in isinstance() checks below, which
    # raised NameError as soon as this module was used without TensorFlow
    # installed -- even for a plain sklearn model. That defeated the
    # documented "works without TensorFlow" guarantee. Binding both names
    # to None lets the isinstance() checks evaluate to False (as intended)
    # instead of crashing.
    tf = None
    keras = None
    TF_AVAILABLE = False

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MLFLOW_AVAILABLE = False

# Tracking general model metadata does not require TensorFlow.  Import the
# TensorFlow flavour separately so a missing optional TensorFlow installation
# cannot incorrectly downgrade an installed MLflow package to "unavailable".
try:
    import mlflow.tensorflow
    MLFLOW_TENSORFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_TENSORFLOW_AVAILABLE = False

from config import Config
from utils.safe_pickle import safe_pickle_load as _safe_load


def _atomic_write_json(path: str, data: Any) -> None:
    """Write JSON atomically: write to a temp file in the same directory,
    flush + fsync, then os.replace() onto the final path.

    BUGFIX (audit follow-up): previously metadata / production / candidate
    JSON files were written with a plain `open(path, 'w')`. If the process
    crashed or was killed mid-write (e.g. OOM-killed during a live trading
    session), the file could be left truncated or half-written, corrupting
    the model registry. os.replace() is atomic on POSIX and Windows, so
    readers only ever see the fully-old or fully-new file, never a partial
    one.
    """
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", dir=directory)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        # Clean up the temp file if something went wrong before replace()
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def _atomic_write_bytes(path: str, write_fn) -> None:
    """Atomically write a binary/model file.

    `write_fn(tmp_path)` must write the complete file to `tmp_path`
    (used for pickle.dump and Keras' model.save(), both of which accept
    a plain filesystem path).
    """
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp_path = os.path.join(directory, f".tmp_{os.path.basename(path)}_{os.getpid()}")
    try:
        write_fn(tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise


class ModelVersionManager:
    """Manages model versions, storage, and deployment"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = Config()
        self.model_dir = os.path.join(self.config.MODEL_DIR, 'versions')
        os.makedirs(self.model_dir, exist_ok=True)

        # BUGFIX (audit follow-up): all file operations below were
        # unsynchronized. save/load/list/delete/compare can all be called
        # concurrently (retraining thread, CLI promote/reject commands,
        # a status/health endpoint listing versions, etc.) and were racing
        # on the same directory tree with no coordination. A single
        # in-process RLock serializes access to this manager. It doesn't
        # protect against a second *process* writing the same model_dir,
        # but combined with the atomic writes below it eliminates the
        # torn-read/torn-write failure mode within this process, which is
        # the realistic threat here (retraining runs in a background
        # thread of the same process per start_scheduled_retraining()).
        self._lock = threading.RLock()
        
        # Initialize MLflow — guarded so the module loads even when mlflow
        # is not installed. (Previously this ran unconditionally and broke
        # the module-level singleton `model_manager = ModelVersionManager()`.)
        if MLFLOW_AVAILABLE:
            try:
                mlflow.set_tracking_uri(f"file://{os.path.join(self.config.MODEL_DIR, 'mlruns')}")
                if MLFLOW_TENSORFLOW_AVAILABLE:
                    mlflow.tensorflow.autolog()
                else:
                    self.logger.info("MLflow tracking enabled; TensorFlow autologging unavailable.")
            except Exception as e:
                self.logger.warning("MLflow init failed (continuing without): %s", e)
        else:
            self.logger.info("MLflow not installed — model_versioning running in degraded mode.")
    
    def _resolve_version_dir(self, version: str) -> str:
        """Resolve a `version` identifier to a path under self.model_dir,
        rejecting anything that could escape it.

        BUGFIX (audit follow-up): save/load/delete/compare all previously
        did `os.path.join(self.model_dir, version)` directly with zero
        validation on `version`. `delete_model_version` then calls
        `shutil.rmtree()` on the result — so a version string containing
        `..` path components (from a bug upstream, a hand-edited call
        site, or this manager ever being exposed behind an API/CLI that
        takes a version as external input) could delete or read/write
        files anywhere the process has permissions, not just inside
        model_dir. `version` is internally-generated today (see
        automated_retraining.py's `f"{pair}_{timestamp}"`), but this
        class has no way to enforce that invariant from its own call
        sites, so it enforces it here instead.
        """
        if not isinstance(version, str) or not version.strip():
            raise ValueError("version must be a non-empty string")
        if os.path.isabs(version) or os.sep in version or (os.altsep and os.altsep in version):
            raise ValueError(f"Invalid version identifier: {version!r}")
        model_dir_abs = os.path.normpath(os.path.abspath(self.model_dir))
        version_dir = os.path.normpath(os.path.abspath(os.path.join(self.model_dir, version)))
        if os.path.commonpath([version_dir, model_dir_abs]) != model_dir_abs:
            raise ValueError(f"Invalid version identifier (path traversal attempt): {version!r}")
        return version_dir

    def save_model_version(self, model: Any, version: str, metrics: Dict, 
                          params: Dict, notes: str = "") -> str:
        """Save a model version with metadata"""
        with self._lock:
            try:
                # Create version directory
                version_dir = self._resolve_version_dir(version)
                os.makedirs(version_dir, exist_ok=True)

                # Save model
                # BUGFIX: guard with TF_AVAILABLE first -- isinstance(model, None)
                # raises TypeError, so we must not evaluate these branches at all
                # when TensorFlow isn't installed.
                # BUGFIX (audit follow-up): model files are now written
                # atomically (write to temp file in the same dir, then
                # os.replace()) so a crash mid-save can never leave a
                # truncated/corrupt model.keras / model.pkl behind that
                # load_model_version() would later choke on.
                #
                # BUGFIX (audit follow-up): this used to have a second
                # `elif isinstance(model, tf.keras.Model): ...model.h5...`
                # branch for "legacy" saves. In TF2, `from tensorflow
                # import keras` and `tf.keras` are the SAME module object,
                # so `keras.Model is tf.keras.Model` — any model that
                # fails the first isinstance check also fails the second.
                # That branch was dead code that could never execute,
                # which silently implied legacy .h5 support that didn't
                # actually exist. Removed; load_model_version still
                # supports reading a pre-existing .h5 file if one is ever
                # placed in a version directory by hand.
                if TF_AVAILABLE and isinstance(model, keras.Model):
                    model_path = os.path.join(version_dir, 'model.keras')
                    _atomic_write_bytes(model_path, lambda tmp: model.save(tmp))
                else:
                    # For sklearn models
                    model_path = os.path.join(version_dir, 'model.pkl')

                    def _dump_pickle(tmp_path, _model=model):
                        with open(tmp_path, 'wb') as f:
                            pickle.dump(_model, f)

                    _atomic_write_bytes(model_path, _dump_pickle)

                # Save metadata
                metadata = {
                    'version': version,
                    'created_at': datetime.now(timezone.utc).isoformat(),
                    'metrics': metrics,
                    'params': params,
                    'notes': notes,
                    'model_type': type(model).__name__,
                    'model_path': model_path
                }

                metadata_path = os.path.join(version_dir, 'metadata.json')
                _atomic_write_json(metadata_path, metadata)

                # Log to MLflow (only if available)
                if MLFLOW_AVAILABLE:
                    try:
                        with mlflow.start_run(run_name=version):
                            mlflow.log_params(params)
                            mlflow.log_metrics(metrics)
                            mlflow.log_artifact(metadata_path)
                    except Exception as mlflow_err:
                        self.logger.warning(f"MLflow logging failed (non-critical): {mlflow_err}")

                self.logger.info(f"Model version {version} saved successfully")
                return version

            except Exception as e:
                self.logger.error(f"Error saving model version {version}: {e}")
                raise
    
    def load_model_version(self, version: str) -> tuple:
        """Load a model version with metadata"""
        with self._lock:
            try:
                version_dir = self._resolve_version_dir(version)

                if not os.path.exists(version_dir):
                    raise ValueError(f"Model version {version} not found")

                # BUGFIX (audit follow-up): metadata.json existence was not
                # validated before opening it -- a missing file raised a
                # bare FileNotFoundError instead of a clear, actionable
                # error, and a version_dir that exists but is mid-write
                # (see _atomic_write_json above) would previously have
                # been readable half-written. The atomic write eliminates
                # the half-written case; this explicit check gives a clear
                # error for the "directory exists but metadata missing"
                # case (e.g. a crash between os.makedirs and the metadata
                # write completing).
                metadata_path = os.path.join(version_dir, 'metadata.json')
                if not os.path.exists(metadata_path):
                    raise ValueError(
                        f"Model version {version} is missing metadata.json "
                        f"(incomplete or corrupted save at {version_dir})"
                    )

                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                # Load model
                model_path = metadata.get('model_path')
                if not model_path:
                    raise ValueError(
                        f"Model version {version} metadata has no 'model_path' entry"
                    )
                if not os.path.exists(model_path):
                    raise ValueError(
                        f"Model version {version} metadata references "
                        f"'{model_path}' but that file does not exist "
                        f"(incomplete or corrupted save)"
                    )

                if model_path.endswith('.keras'):
                    if not TF_AVAILABLE:
                        raise RuntimeError(
                            f"Model version {version} is a Keras model but "
                            f"TensorFlow is not installed in this environment."
                        )
                    model = keras.models.load_model(model_path)
                elif model_path.endswith('.h5'):
                    if not TF_AVAILABLE:
                        raise RuntimeError(
                            f"Model version {version} is a Keras model but "
                            f"TensorFlow is not installed in this environment."
                        )
                    model = tf.keras.models.load_model(model_path)
                elif model_path.endswith('.pkl'):
                    with open(model_path, 'rb') as f:
                        model = _safe_load(f)
                else:
                    raise ValueError(f"Unknown model format: {model_path}")

                self.logger.info(f"Model version {version} loaded successfully")
                return model, metadata

            except Exception as e:
                self.logger.error(f"Error loading model version {version}: {e}")
                raise
    
    def list_model_versions(self) -> List[Dict]:
        """List all available model versions"""
        with self._lock:
            versions = []

            if not os.path.exists(self.model_dir):
                return versions

            for version in os.listdir(self.model_dir):
                version_dir = os.path.join(self.model_dir, version)
                # Skip the temp files atomic writes create/clean up
                # (.tmp_* prefix) if one is ever left behind by a crash.
                if version.startswith('.tmp_'):
                    continue
                if os.path.isdir(version_dir):
                    metadata_path = os.path.join(version_dir, 'metadata.json')
                    if os.path.exists(metadata_path):
                        try:
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                            # BUGFIX (audit follow-up): the final sort below
                            # keys on metadata['created_at'] with no
                            # fallback. One legacy/hand-edited metadata.json
                            # missing that field used to raise an uncaught
                            # KeyError from inside .sort(), crashing this
                            # method (and therefore get_latest_model_version)
                            # entirely instead of just skipping the one bad
                            # entry — the same defensive pattern already
                            # used for corrupted JSON below.
                            if 'created_at' not in metadata:
                                self.logger.error(f"Metadata for {version} missing 'created_at' — skipping")
                                continue
                            versions.append(metadata)
                        except Exception as e:
                            self.logger.error(f"Error reading metadata for {version}: {e}")

            # Sort by creation date
            versions.sort(key=lambda x: x['created_at'], reverse=True)
            return versions
    
    def get_latest_model_version(self) -> Optional[Dict]:
        """Get the latest model version"""
        versions = self.list_model_versions()
        return versions[0] if versions else None
    
    def compare_model_versions(self, version1: str, version2: str) -> Dict:
        """Compare two model versions"""
        with self._lock:
            try:
                _, metadata1 = self.load_model_version(version1)
                _, metadata2 = self.load_model_version(version2)

                comparison = {
                    'version1': version1,
                    'version2': version2,
                    'metrics_comparison': {},
                    'params_comparison': {},
                    'performance_diff': {}
                }

                # Compare metrics
                all_metrics = set(metadata1['metrics'].keys()) | set(metadata2['metrics'].keys())
                for metric in all_metrics:
                    val1 = metadata1['metrics'].get(metric, None)
                    val2 = metadata2['metrics'].get(metric, None)

                    # BUGFIX (audit follow-up): a NaN metric (e.g. from a
                    # training run that diverged) previously passed the
                    # `is not None` check and silently produced NaN diffs
                    # / percent_change, which then look like valid numbers
                    # to any downstream consumer of this comparison.
                    def _is_valid_number(v):
                        return isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))

                    if _is_valid_number(val1) and _is_valid_number(val2):
                        diff = val2 - val1
                        # BUGFIX (audit follow-up): val1 == 0 previously
                        # reported percent_change as a literal 0, which
                        # reads as "no change" — but a metric going from 0
                        # to any nonzero value is an undefined/infinite
                        # percent change, not 0%. Reporting 0 here actively
                        # misleads anyone reading the comparison. Report
                        # None with a note instead, same as the
                        # non-numeric/NaN case below.
                        if val1 != 0:
                            comparison['metrics_comparison'][metric] = {
                                'version1': val1,
                                'version2': val2,
                                'difference': diff,
                                'percent_change': (diff / val1) * 100,
                            }
                        else:
                            comparison['metrics_comparison'][metric] = {
                                'version1': val1,
                                'version2': val2,
                                'difference': diff,
                                'percent_change': None,
                                'note': 'percent_change undefined: version1 value is 0',
                            }
                    elif val1 is not None or val2 is not None:
                        comparison['metrics_comparison'][metric] = {
                            'version1': val1,
                            'version2': val2,
                            'difference': None,
                            'percent_change': None,
                            'note': 'skipped: non-numeric or NaN value',
                        }

                # Compare parameters
                all_params = set(metadata1['params'].keys()) | set(metadata2['params'].keys())
                for param in all_params:
                    val1 = metadata1['params'].get(param, None)
                    val2 = metadata2['params'].get(param, None)

                    comparison['params_comparison'][param] = {
                        'version1': val1,
                        'version2': val2
                    }

                return comparison

            except Exception as e:
                self.logger.error(f"Error comparing model versions: {e}")
                raise
    
    def delete_model_version(self, version: str) -> bool:
        """Delete a model version"""
        with self._lock:
            try:
                version_dir = self._resolve_version_dir(version)

                if not os.path.exists(version_dir):
                    self.logger.warning(f"Model version {version} not found")
                    return False

                # Delete directory and all contents
                import shutil
                shutil.rmtree(version_dir)

                self.logger.info(f"Model version {version} deleted successfully")
                return True

            except Exception as e:
                self.logger.error(f"Error deleting model version {version}: {e}")
                return False

# Singleton instance
model_manager = ModelVersionManager()