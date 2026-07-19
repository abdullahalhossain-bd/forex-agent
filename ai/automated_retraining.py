"""
Automated Model Retraining System
Handles scheduled retraining, performance monitoring, and automatic updates

NOTE: TensorFlow, sklearn, and schedule are OPTIONAL dependencies.
      If they are not installed, this module gracefully degrades.
"""
# BUGFIX (audit follow-up): _build_model()'s signature below has a
# `-> keras.Model` return annotation. Without postponed evaluation,
# Python evaluates that annotation the moment the `def` statement runs
# (at class-body execution / import time) -- so if TensorFlow isn't
# installed and `keras` is None (see the try/except below), importing
# this module at all raised AttributeError, regardless of whether any
# TF-dependent method was ever called. That directly contradicted this
# module's own docstring ("gracefully degrades"). `from __future__
# import annotations` makes all annotations lazy strings instead, so
# they're never evaluated unless something explicitly inspects them.
from __future__ import annotations

import os
import json
import logging
import math
import tempfile
import threading
import time
from datetime import datetime, timezone
from typing import Dict, Optional
import pandas as pd
import numpy as np

# Optional ML dependencies — the system works without them
try:
    import schedule
    SCHEDULE_AVAILABLE = True
except ImportError:
    SCHEDULE_AVAILABLE = False

try:
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# `tensorflow` itself is unused in this module — only `keras` is.
# Importing tensorflow at module load added ~3-5s of startup cost
# (TF prints verbose GPU/CPU discovery logs even when not used).
# We import keras directly; TF_AVAILABLE flag stays for downstream
# feature detection.
try:
    from tensorflow import keras
    TF_AVAILABLE = True
except (ImportError, RuntimeError):
    keras = None
    TF_AVAILABLE = False

from config import Config
from data.automated_updater import data_updater
from ai.model_versioning import model_manager


def _atomic_write_json(path: str, data: Dict) -> None:
    """Write JSON atomically (temp file in the same dir + os.replace()).

    BUGFIX (audit follow-up): production_models.json and
    candidate_models.json were both previously written with a plain
    `open(path, 'w')`. A crash/kill mid-write (very plausible for a
    long-running trading process) could leave either file truncated,
    which would then raise on the next json.load() and take down the
    retraining/promotion path -- or worse, silently look like "no
    production models" if the truncation happened to leave valid-but-
    incomplete JSON. os.replace() is atomic, so readers only ever see
    the fully-old or fully-new file.
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
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


class AutomatedRetrainingSystem:
    """Automated model retraining with performance monitoring"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = Config()
        self.retraining_interval = self.config.RETRAINING_INTERVAL  # e.g., 7 days
        self.performance_threshold = self.config.PERFORMANCE_THRESHOLD
        self.min_training_samples = self.config.MIN_TRAINING_SAMPLES

        # BUGFIX (audit follow-up): guards production_models.json /
        # candidate_models.json read-modify-write cycles (_stage_candidate,
        # _deploy_model, promote_candidate, reject_candidate) against
        # concurrent access -- e.g. the scheduled retraining thread staging
        # a candidate at the same moment an operator calls
        # promote_candidate() from a CLI/API in the same process.
        self._registry_lock = threading.RLock()
        
    def start_scheduled_retraining(self):
        """Start scheduled retraining job"""
        if not SCHEDULE_AVAILABLE:
            self.logger.warning("schedule package not installed — scheduled retraining disabled")
            return
        schedule.every(self.retraining_interval).days.do(self._retrain_models)
        
        self.logger.info(f"Scheduled model retraining every {self.retraining_interval} days")
        
        # Run in a separate thread (threading already imported at module top)
        thread = threading.Thread(target=self._run_scheduler, daemon=True)
        thread.start()
    
    def _run_scheduler(self):
        """Run the scheduler in a loop.

        BUGFIX (audit follow-up): this loop had no exception handling. If
        `schedule.run_pending()` (or the `schedule` library itself) ever
        raised, the exception would propagate out of this daemon thread's
        target function, silently killing the thread — scheduled
        retraining would stop forever with no crash, no log, and no
        visibility for an operator that it had happened. Since
        `_retrain_models` already catches its own exceptions internally,
        anything that reaches here is unexpected; log it and keep the
        scheduler loop alive rather than let it die silently.
        """
        while True:
            try:
                schedule.run_pending()
            except Exception as e:
                self.logger.error(f"Scheduler loop error (continuing): {e}", exc_info=True)
            time.sleep(60)  # Check every minute
    
    def _retrain_models(self):
        """Retrain all models with latest data"""
        self.logger.info("Starting automated model retraining")
        
        try:
            # Update data first
            data_status = data_updater.update_all_pairs()

            # BUGFIX (audit follow-up): previously `if not all(data_status
            # .values())` aborted retraining for EVERY pair the moment a
            # single pair's data update failed (e.g. one flaky broker/API
            # call for GBPJPY blocked EURUSD, USDJPY, etc. too) — this
            # contradicted the resilience pattern used everywhere else in
            # this file (_load_all_forex_data already skips individual
            # pairs with missing data; the per-pair training loop below
            # already isolates exceptions per pair). Only treat this as
            # fatal when NO pair updated successfully, which is the actual
            # signal of a systemic failure (e.g. broker/API entirely down)
            # rather than one pair having a bad day.
            failed_pairs = [p for p, ok in data_status.items() if not ok]
            if failed_pairs:
                self.logger.warning(
                    f"Data update failed for {failed_pairs} — "
                    f"continuing retraining with the remaining pairs"
                )
            if data_status and not any(data_status.values()):
                self.logger.error("Data update failed for ALL pairs — skipping retraining entirely")
                return False
            
            # Get latest data
            all_data = self._load_all_forex_data()
            
            if all_data.empty:
                self.logger.error("No data available for retraining")
                return False
            
            # Train models
            model_results = {}
            
            for pair in self.config.FOREX_PAIRS:
                try:
                    result = self._train_model_for_pair(pair, all_data)
                    model_results[pair] = result
                except Exception as e:
                    self.logger.error(f"Error training model for {pair}: {e}")
                    model_results[pair] = {'success': False, 'error': str(e)}
            
            # Evaluate and deploy best models
            self._evaluate_and_deploy_models(model_results)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in automated retraining: {e}")
            return False
    
    def _load_all_forex_data(self) -> pd.DataFrame:
        """Load all available forex data"""
        all_data = {}

        for pair in self.config.FOREX_PAIRS:
            data = data_updater.load_existing_data(pair)
            if data is None or data.empty:
                # BUGFIX (audit follow-up): previously silent -- a pair
                # with no data just quietly disappeared from `combined`,
                # which then made every downstream `{pair}_Close not in
                # columns` check in _train_model_for_pair raise a
                # ValueError with no indication *why* the column was
                # missing. Logging here makes a bad data feed visible
                # immediately instead of several stack frames later.
                self.logger.warning(f"No data available for {pair} — excluding from this retraining run")
                continue
            all_data[pair] = data

        # BUGFIX (audit follow-up): explicit empty-dataset guard at the
        # source, in addition to the caller's `all_data.empty` check in
        # _retrain_models. Keeping the check here too means any other
        # caller of this method (tests, CLI tools, a future inference
        # pipeline) also gets a clean empty DataFrame instead of relying
        # on every caller remembering to check.
        if not all_data:
            self.logger.error("No forex data available for any configured pair")
            return pd.DataFrame()

        # Combine all data
        combined = pd.DataFrame()
        
        for pair, data in all_data.items():
            # Add prefix to columns
            prefixed_data = data.add_prefix(f"{pair.replace('/', '_')}_")
            combined = pd.concat([combined, prefixed_data], axis=1)

        # BUGFIX (audit follow-up): pd.concat(axis=1) does an implicit
        # OUTER join on the index. If any pair's data has a slightly
        # different index than the others (a missed candle, a different
        # history start date, a broker feed hiccup on just one symbol),
        # that timestamp's row becomes NaN for every OTHER pair's columns
        # too — and since _create_features builds cross-pair columns
        # (f'{other}_returns') straight out of this combined frame, a
        # single misaligned pair can silently shrink the usable training
        # window for every pair, not just itself. There was previously no
        # visibility into this at all. This doesn't change behavior (the
        # downstream notna()/isfinite() filtering already handles it
        # correctly) — it just makes a real but silent data-quality issue
        # observable instead of invisible.
        if not combined.empty:
            rows_with_any_gap = int(combined.isna().any(axis=1).sum())
            if rows_with_any_gap:
                self.logger.warning(
                    f"_load_all_forex_data: {rows_with_any_gap}/{len(combined)} rows have a "
                    f"gap in at least one pair's columns after alignment across "
                    f"{len(all_data)} pairs — these rows will be dropped downstream"
                )

        return combined
    
    @staticmethod
    def _make_windows(X: np.ndarray, y: np.ndarray, seq_len: int):
        """Build sliding-window sequences for LSTM input.

        CRITICAL FIX (audit follow-up): the previous implementation reshaped
        each row into a sequence of length 1 (`timesteps=1`), which makes
        the LSTM mathematically equivalent to a single dense layer applied
        per-timestep -- it never actually sees any temporal context and
        "sequence learning" was effectively disabled despite the LSTM
        architecture. This builds true overlapping windows of `seq_len`
        consecutive rows so each training example is a real time-ordered
        sequence, and returns the original-array index of each window's
        last row (`end_idx`) so callers can align windows back to a
        TimeSeriesSplit train/val partition.

        Returns:
            X_seq: (n_windows, seq_len, n_features)
            y_seq: (n_windows,) -- target aligned to each window's last row
            end_idx: (n_windows,) -- row index (into X/y) of each window's last element
        """
        n_windows = len(X) - seq_len + 1
        if n_windows <= 0:
            return (np.empty((0, seq_len, X.shape[1] if X.ndim > 1 else 0)),
                    np.empty((0,)), np.empty((0,), dtype=int))
        X_seq = np.stack([X[i:i + seq_len] for i in range(n_windows)])
        y_seq = y[seq_len - 1:]
        end_idx = np.arange(seq_len - 1, len(X))
        return X_seq, y_seq, end_idx

    def _fit_fold(self, X_train_seq, y_train_seq, X_val_seq, y_val_seq,
                  seq_len: int, n_features: int, epochs: int, checkpoint_path: str):
        """Build, train, and evaluate one LSTM fold. Shared by the
        walk-forward evaluation loop and the final production fit."""
        model = self._build_model(input_shape=(seq_len, n_features))

        callbacks = []
        try:
            from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
            # BUGFIX (audit follow-up): no EarlyStopping meant training
            # always ran the full `epochs` even after validation loss
            # stopped improving, wasting compute and risking overfitting
            # on the tail epochs.
            callbacks.append(EarlyStopping(monitor='val_loss', patience=8,
                                            restore_best_weights=True))
            # BUGFIX (audit follow-up): no ModelCheckpoint meant the
            # in-memory model at the end of `fit()` was saved even if a
            # later epoch had degraded — combined with restore_best_weights
            # above this is largely redundant, but the checkpoint file also
            # gives us a crash-safe snapshot of the best weights on disk
            # during long training runs.
            callbacks.append(ModelCheckpoint(checkpoint_path, monitor='val_loss',
                                              save_best_only=True, save_weights_only=True))
            # BUGFIX (audit follow-up): no LR scheduler — a fixed Adam LR
            # for the whole run makes it easy to plateau or oscillate near
            # a minimum. Reduce LR when validation loss stalls.
            callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                patience=4, min_lr=1e-6))
        except ImportError:
            self.logger.warning("Keras callbacks unavailable — training without EarlyStopping/checkpoint/LR schedule")

        model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=epochs,
            batch_size=32,
            # BUGFIX (audit follow-up): Keras defaults to shuffle=True,
            # which for time-series data shuffles the *order in which
            # windows are presented within an epoch*. That's not fatal on
            # its own (each window is still internally time-ordered), but
            # it removes the mild curriculum/locality signal from training
            # in temporal order and makes runs non-reproducible batch-to-
            # batch. Time-series models should not shuffle.
            shuffle=False,
            callbacks=callbacks,
            verbose=0,
        )

        val_pred = model.predict(X_val_seq, verbose=0)
        val_mse = float(mean_squared_error(y_val_seq, val_pred))
        val_mae = float(mean_absolute_error(y_val_seq, val_pred))

        residuals = np.asarray(y_val_seq).reshape(-1) - np.asarray(val_pred).reshape(-1)
        squared_errors = residuals ** 2
        n_val = len(squared_errors)
        val_mse_se = float(np.std(squared_errors, ddof=1) / np.sqrt(n_val)) if n_val > 1 else float('inf')

        return model, val_mse, val_mae, val_mse_se, n_val

    def _train_model_for_pair(self, pair: str, all_data: pd.DataFrame) -> Dict:
        """Train a model for a specific currency pair"""
        scaler_path = None
        try:
            if not SKLEARN_AVAILABLE:
                raise RuntimeError("scikit-learn is required for training (TimeSeriesSplit/scaling/metrics)")
            if not TF_AVAILABLE:
                raise RuntimeError("TensorFlow/Keras is required for training")

            # Prepare features and target
            pair_col = f"{pair.replace('/', '_')}_Close"
            
            if pair_col not in all_data.columns:
                raise ValueError(f"Data for {pair} not found")
            
            # Create features
            features = self._create_features(all_data, pair)
            
            # Create target (next day's return)
            target = all_data[pair_col].pct_change().shift(-1)
            
            # Drop NaN values
            valid_idx = features.notna().all(axis=1) & target.notna()
            features = features[valid_idx]
            target = target[valid_idx]

            # BUGFIX (audit follow-up): also drop rows with +/-inf, which
            # notna() does NOT catch (inf is not NaN). These can come from
            # e.g. log(0) in log_returns or a zero-average-loss RSI division
            # before the _create_features/_calculate_rsi fixes below, or
            # from any future feature added without an inf-guard.
            finite_idx = np.isfinite(features.to_numpy(dtype=float)).all(axis=1) & np.isfinite(target.to_numpy(dtype=float))
            if not finite_idx.all():
                n_dropped = int((~finite_idx).sum())
                self.logger.warning(f"{pair}: dropping {n_dropped} rows with non-finite feature/target values")
            features = features[finite_idx]
            target = target[finite_idx]

            seq_len = int(getattr(self.config, 'LSTM_SEQUENCE_LENGTH', 10))
            n_splits = int(getattr(self.config, 'RETRAINING_CV_SPLITS', 5))

            # BUGFIX (audit follow-up): the old check only verified
            # `len(features) < min_training_samples`, which said nothing
            # about whether there were enough rows to actually form
            # `n_splits` non-overlapping TimeSeriesSplit folds *and* still
            # have `seq_len` rows left over to build even one LSTM window
            # per fold. Undersized data used to reach TimeSeriesSplit/the
            # reshape and fail there with a much less clear error.
            min_required = max(self.min_training_samples, (n_splits + 1) * seq_len)
            if len(features) < min_required:
                raise ValueError(
                    f"Insufficient samples for {pair}: {len(features)} "
                    f"(need >= {min_required} for {n_splits} CV folds with "
                    f"sequence length {seq_len})"
                )

            # Time series split for training/validation
            tscv = TimeSeriesSplit(n_splits=n_splits)
            splits = list(tscv.split(features))

            X_all = features.to_numpy(dtype=float)
            y_all = target.to_numpy(dtype=float)

            # ── Walk-forward validation ──────────────────────────────
            # BUGFIX (audit follow-up): previously only the *last* CV fold
            # was ever used, and only for the final model — there was no
            # walk-forward evaluation at all, so a model could look great
            # on one lucky fold with no visibility into how stable that
            # performance is across earlier time periods. We now train and
            # evaluate a fold-local model for every split except the last
            # (reserved for the production model below) and report the
            # mean/std of out-of-fold MSE as a robustness signal alongside
            # the final fold's metrics used by the deployment gate.
            walk_forward_mse = []
            walk_forward_mae = []
            tmp_ckpt_dir = tempfile.mkdtemp(prefix=f"wf_{pair}_")
            try:
                for fold_i, (train_idx, val_idx) in enumerate(splits[:-1]):
                    scaler_fold = StandardScaler()
                    scaler_fold.fit(X_all[train_idx])
                    X_scaled_fold = scaler_fold.transform(X_all[:val_idx[-1] + 1])

                    X_seq, y_seq, end_idx = self._make_windows(X_scaled_fold, y_all[:val_idx[-1] + 1], seq_len)
                    train_mask = np.isin(end_idx, train_idx)
                    val_mask = np.isin(end_idx, val_idx)

                    if train_mask.sum() < 2 or val_mask.sum() < 2:
                        continue  # not enough windows in this fold to train/evaluate

                    _, fold_mse, fold_mae, _, _ = self._fit_fold(
                        X_seq[train_mask], y_seq[train_mask],
                        X_seq[val_mask], y_seq[val_mask],
                        seq_len, X_all.shape[1],
                        epochs=int(getattr(self.config, 'RETRAINING_WF_EPOCHS', 20)),
                        checkpoint_path=os.path.join(tmp_ckpt_dir, f"fold_{fold_i}.weights.h5"),
                    )
                    walk_forward_mse.append(fold_mse)
                    walk_forward_mae.append(fold_mae)
            finally:
                import shutil as _shutil
                _shutil.rmtree(tmp_ckpt_dir, ignore_errors=True)

            # ── Final production model: trained on the last fold ────
            train_idx, val_idx = splits[-1]

            scaler = StandardScaler()
            scaler.fit(X_all[train_idx])
            X_scaled = scaler.transform(X_all[:val_idx[-1] + 1])

            X_seq, y_seq, end_idx = self._make_windows(X_scaled, y_all[:val_idx[-1] + 1], seq_len)
            train_mask = np.isin(end_idx, train_idx)
            val_mask = np.isin(end_idx, val_idx)

            X_train_seq, y_train_seq = X_seq[train_mask], y_seq[train_mask]
            X_val_seq, y_val_seq = X_seq[val_mask], y_seq[val_mask]

            if len(X_train_seq) < 2 or len(X_val_seq) < 2:
                raise ValueError(f"Insufficient windowed samples for {pair} after sequence construction")

            # NaN/Inf guard immediately before fitting — belt-and-braces on
            # top of the row-level filtering above, since scaling/windowing
            # could in principle introduce non-finite values (e.g. a
            # constant feature column producing 0/0 in StandardScaler).
            if not (np.isfinite(X_train_seq).all() and np.isfinite(y_train_seq).all()
                    and np.isfinite(X_val_seq).all() and np.isfinite(y_val_seq).all()):
                raise ValueError(f"Non-finite values present in training/validation data for {pair} after preprocessing")

            final_ckpt = tempfile.NamedTemporaryFile(suffix='.weights.h5', delete=False)
            final_ckpt.close()
            try:
                model, val_mse, val_mae, val_mse_se, n_val = self._fit_fold(
                    X_train_seq, y_train_seq, X_val_seq, y_val_seq,
                    seq_len, X_all.shape[1],
                    epochs=int(getattr(self.config, 'RETRAINING_EPOCHS', 50)),
                    checkpoint_path=final_ckpt.name,
                )
            finally:
                try:
                    os.remove(final_ckpt.name)
                except OSError:
                    pass

            # Save model version
            version = f"{pair}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            metrics = {
                'val_mse': val_mse,
                'val_mae': val_mae,
                'val_rmse': float(np.sqrt(val_mse)),
                'val_mse_se': val_mse_se,      # standard error, used for significance check
                'n_val_samples': n_val,
                'walk_forward_val_mse_mean': float(np.mean(walk_forward_mse)) if walk_forward_mse else None,
                'walk_forward_val_mse_std': float(np.std(walk_forward_mse)) if walk_forward_mse else None,
                'walk_forward_val_mae_mean': float(np.mean(walk_forward_mae)) if walk_forward_mae else None,
                'walk_forward_folds': len(walk_forward_mse),
            }
            params = {
                'model_type': 'LSTM',
                'layers': [64, 32],
                'epochs': int(getattr(self.config, 'RETRAINING_EPOCHS', 50)),
                'batch_size': 32,
                'sequence_length': seq_len,
                'cv_splits': n_splits,
                'scaler': 'StandardScaler',
            }
            
            model_manager.save_model_version(model, version, metrics, params, 
                                           f"Automated retraining for {pair}")

            # Persist the fitted scaler alongside the model version so
            # inference can apply the identical transform. save_model_version()
            # creates the version directory as a side effect, so this must
            # run after it.
            try:
                import pickle
                version_dir = os.path.join(model_manager.model_dir, version)
                scaler_path = os.path.join(version_dir, 'scaler.pkl')
                tmp_scaler = scaler_path + f".tmp_{os.getpid()}"
                with open(tmp_scaler, 'wb') as f:
                    pickle.dump(scaler, f)
                os.replace(tmp_scaler, scaler_path)
            except Exception as scaler_err:
                self.logger.warning(f"Could not persist scaler for {version}: {scaler_err}")

            return {
                'success': True,
                'version': version,
                'metrics': metrics
            }
            
        except Exception as e:
            self.logger.error(f"Error training model for {pair}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _create_features(self, data: pd.DataFrame, pair: str) -> pd.DataFrame:
        """Create features for model training"""
        pair_col = f"{pair.replace('/', '_')}_Close"
        
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        features['returns'] = data[pair_col].pct_change()

        # BUGFIX (audit follow-up): np.log() of a zero, negative, or
        # missing price ratio produces -inf/NaN/complex-cast-NaN, which
        # then propagates into every downstream feature and eventually the
        # LSTM's loss (a single -inf gradient can NaN out the whole
        # network's weights). Forex Close prices should never be <= 0, but
        # a bad data feed / gap-fill artifact can produce exactly that. We
        # compute the ratio, mask out non-positive values before taking
        # the log, and rely on the finite-value filtering in
        # _train_model_for_pair to drop the resulting NaN rows.
        price_ratio = data[pair_col] / data[pair_col].shift(1)
        price_ratio = price_ratio.where(price_ratio > 0, np.nan)
        features['log_returns'] = np.log(price_ratio)
        
        # Technical indicators
        features['sma_10'] = data[pair_col].rolling(window=10).mean()
        features['sma_30'] = data[pair_col].rolling(window=30).mean()
        features['sma_50'] = data[pair_col].rolling(window=50).mean()
        
        features['rsi'] = self._calculate_rsi(data[pair_col])
        features['macd'], features['macd_signal'] = self._calculate_macd(data[pair_col])
        
        # Volatility features
        features['volatility_10'] = features['returns'].rolling(window=10).std()
        features['volatility_30'] = features['returns'].rolling(window=30).std()
        
        # Momentum features
        features['momentum_5'] = data[pair_col] / data[pair_col].shift(5) - 1
        features['momentum_10'] = data[pair_col] / data[pair_col].shift(10) - 1
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            features[f'return_lag_{lag}'] = features['returns'].shift(lag)
        
        # Cross-pair features (if other pairs available)
        other_pairs = [p for p in self.config.FOREX_PAIRS if p != pair]
        for other in other_pairs[:3]:  # Use top 3 correlated pairs
            other_col = f"{other.replace('/', '_')}_Close"
            if other_col in data.columns:
                features[f'{other}_returns'] = data[other_col].pct_change()
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

        # BUGFIX (audit follow-up): when the average loss over the window
        # is exactly 0 (price only went up, or was flat), `gain / loss`
        # divides by zero. Pandas turns that into +inf (or NaN when gain
        # is also 0), which then fails the np.isfinite() check on the
        # whole feature matrix. Handle both cases explicitly per RSI's own
        # definition: all gains, no losses -> RSI = 100 (maximally
        # overbought); no gains and no losses (flat price) -> RSI = 50
        # (neutral), rather than propagating inf/NaN.
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

        rsi = rsi.where(loss != 0, np.where(gain > 0, 100.0, 50.0))

        return rsi
    
    def _calculate_macd(self, prices: pd.Series) -> tuple:
        """Calculate MACD and signal line"""
        ema_12 = prices.ewm(span=12, adjust=False).mean()
        ema_26 = prices.ewm(span=26, adjust=False).mean()
        
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, adjust=False).mean()
        
        return macd, signal
    
    def _build_model(self, input_shape: tuple) -> keras.Model:
        """Build LSTM model for time series prediction"""
        model = keras.Sequential([
            keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(32),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return model
    
    def _evaluate_and_deploy_models(self, model_results: Dict):
        """Evaluate newly trained models and stage the best performers as
        deployment candidates.

        BUGFIX (audit follow-up): this previously called _deploy_model()
        directly, which overwrote production_models.json immediately —
        the model actually used for live trading decisions could change
        with zero human visibility, based solely on one validation fold
        clearing a 5% MSE/MAE threshold. That is a real risk for a system
        trading live capital: an overfit or spuriously-lucky model could
        silently become "production" mid-week.

        Now this method only stages a candidate (via _stage_candidate).
        Going live requires an explicit call to promote_candidate(pair),
        which an operator/CLI/approval flow must trigger deliberately —
        analogous to how approval_mode.py already requires a human step
        for Mode 2 trade execution. This keeps retraining itself fully
        automatic (nothing here needs to change on a schedule) while
        removing the silent-auto-deploy risk.
        """
        successful_models = {k: v for k, v in model_results.items() if v['success']}
        
        if not successful_models:
            self.logger.error("No models trained successfully")
            return
        
        # Compare with current production models
        current_models = self._get_current_production_models()
        
        for pair, result in successful_models.items():
            new_version = result['version']
            new_metrics = result['metrics']
            
            if pair in current_models:
                current_version = current_models[pair]['version']
                # .get() not [] -- tolerate production records written
                # before metrics persistence was added, or edited by hand.
                current_metrics = current_models[pair].get('metrics', {})
                
                # Compare performance
                if self._is_better_performance(new_metrics, current_metrics):
                    self.logger.info(
                        f"Candidate staged for {pair}: {new_version} "
                        f"(beats current production {current_version} — "
                        f"awaiting promote_candidate() to go live)"
                    )
                    self._stage_candidate(pair, new_version, new_metrics, current_version)
                else:
                    self.logger.info(f"Keeping current model for {pair}: {current_version}")
            else:
                self.logger.info(
                    f"Candidate staged for {pair}: {new_version} "
                    f"(no existing production model — awaiting promote_candidate() to go live)"
                )
                self._stage_candidate(pair, new_version, new_metrics, current_version=None)
    
    def _get_current_production_models(self) -> Dict:
        """Get currently deployed production models"""
        # This would typically check a model registry or database
        # For now, we'll use a simple file-based approach
        prod_file = os.path.join(self.config.MODEL_DIR, 'production_models.json')

        with self._registry_lock:
            if os.path.exists(prod_file):
                try:
                    with open(prod_file, 'r') as f:
                        return json.load(f)
                except (json.JSONDecodeError, OSError) as e:
                    # BUGFIX (audit follow-up): a corrupted registry file
                    # (partial write from before atomic writes were added,
                    # or external tampering) previously raised straight out
                    # of this method and crashed the retraining/evaluation
                    # path. Treat it as "no known production models" and
                    # log loudly instead — losing a stale comparison
                    # baseline is far safer than crashing the retraining
                    # loop.
                    self.logger.error(f"production_models.json is corrupted/unreadable ({e}) — treating as empty")
                    return {}
            else:
                return {}

    def _candidates_file(self) -> str:
        return os.path.join(self.config.MODEL_DIR, 'candidate_models.json')

    def _stage_candidate(self, pair: str, version: str, metrics: Dict,
                         current_version: Optional[str]) -> None:
        """Record a newly trained model as a promotion candidate.

        This does NOT touch production_models.json — the live model used
        for trading decisions is unchanged until promote_candidate() is
        called explicitly.
        """
        candidates_file = self._candidates_file()
        with self._registry_lock:
            if os.path.exists(candidates_file):
                try:
                    with open(candidates_file, 'r') as f:
                        candidates = json.load(f)
                except (json.JSONDecodeError, OSError) as e:
                    self.logger.error(f"candidate_models.json is corrupted/unreadable ({e}) — starting fresh")
                    candidates = {}
            else:
                candidates = {}

            candidates[pair] = {
                'version': version,
                'metrics': metrics,
                'staged_at': datetime.now(timezone.utc).isoformat(),
                'replaces_version': current_version,
                'status': 'PENDING_PROMOTION',
            }

            # BUGFIX (audit follow-up): atomic write instead of a plain
            # `open(..., 'w')` — see _atomic_write_json docstring above.
            _atomic_write_json(candidates_file, candidates)

    def list_candidates(self) -> Dict:
        """Return all models currently staged and awaiting promotion."""
        candidates_file = self._candidates_file()
        with self._registry_lock:
            if os.path.exists(candidates_file):
                try:
                    with open(candidates_file, 'r') as f:
                        return json.load(f)
                except (json.JSONDecodeError, OSError) as e:
                    self.logger.error(f"candidate_models.json is corrupted/unreadable ({e}) — treating as empty")
                    return {}
            return {}

    def promote_candidate(self, pair: str, version: Optional[str] = None) -> bool:
        """Explicitly promote a staged candidate to production.

        This is the ONLY code path that writes production_models.json.
        Call this deliberately (operator action, CLI command, or an
        approval-gated workflow) after reviewing the candidate's metrics —
        it is intentionally never called automatically by the retraining
        scheduler.

        Args:
            pair: currency pair, e.g. "EURUSD".
            version: specific candidate version to promote. If omitted,
                     promotes whatever candidate is currently staged for
                     this pair.

        Returns:
            True if a candidate was promoted, False if no matching
            candidate was found.
        """
        with self._registry_lock:
            candidates = self.list_candidates()
            candidate = candidates.get(pair)
            if not candidate:
                self.logger.warning(f"No staged candidate for {pair} to promote")
                return False
            if version is not None and candidate.get('version') != version:
                self.logger.warning(
                    f"Staged candidate for {pair} is {candidate.get('version')}, "
                    f"not requested version {version} — not promoting"
                )
                return False

            self._deploy_model(pair, candidate['version'], metrics=candidate.get('metrics'))

            # Remove from the candidate queue now that it's live
            candidates.pop(pair, None)
            _atomic_write_json(self._candidates_file(), candidates)

            self.logger.info(f"Promoted {pair} candidate {candidate['version']} to production")
            return True

    def reject_candidate(self, pair: str) -> bool:
        """Discard a staged candidate without promoting it."""
        with self._registry_lock:
            candidates = self.list_candidates()
            if pair not in candidates:
                return False
            rejected = candidates.pop(pair)
            _atomic_write_json(self._candidates_file(), candidates)
            self.logger.info(f"Rejected candidate {rejected.get('version')} for {pair}")
            return True
    
    def _deploy_model(self, pair: str, version: str, metrics: Optional[Dict] = None):
        """Deploy a model to production.

        Args:
            metrics: the promoted model's validation metrics. BUGFIX: this
                was previously never stored, so _evaluate_and_deploy_models'
                `current_models[pair]['metrics']` lookup would raise
                KeyError the next time this same pair was compared against
                its own (metrics-less) production record -- i.e. every
                pair crashed on its second retraining cycle. If not passed
                explicitly, falls back to the metrics of the matching
                staged/just-promoted candidate when available.
        """
        # Update production model registry
        prod_file = os.path.join(self.config.MODEL_DIR, 'production_models.json')

        with self._registry_lock:
            # Load current production models
            prod_models = self._get_current_production_models()

            # Update with new model
            prod_models[pair] = {
                'version': version,
                'metrics': metrics or {},
                'deployed_at': datetime.now(timezone.utc).isoformat()
            }

            # BUGFIX (audit follow-up): atomic write instead of a plain
            # `open(..., 'w')` — the production registry is the single
            # source of truth for "what model makes live trading
            # decisions right now"; a torn write here is the worst place
            # in this module for one to happen.
            _atomic_write_json(prod_file, prod_models)

            self.logger.info(f"Model {version} deployed for {pair}")
    
    def _is_better_performance(self, new_metrics: Dict, current_metrics: Dict) -> bool:
        """Check if new model performs better than current.

        BUGFIX (audit follow-up): the original check only required the new
        model's MSE/MAE to be 5% lower than the current model's on a
        single validation fold. A single fold is noisy — a model can
        clear a 5% bar purely by chance, and the system would then
        silently swap the live model used for real trading decisions.

        This now additionally requires the improvement to exceed one
        standard error of the new model's validation MSE (a lightweight
        significance check — is the improvement bigger than the model's
        own measurement noise?), on top of the existing 5% relative bar.
        Both conditions must hold. If standard-error data isn't available
        on either side (e.g. current_metrics was saved by an older version
        of this module, before val_mse_se existed), this degrades to the
        original relative-only check so existing production_models.json
        entries keep working without needing to be regenerated.
        """
        # Use validation MSE as primary metric
        new_mse = new_metrics.get('val_mse', float('inf'))
        current_mse = current_metrics.get('val_mse', float('inf'))
        new_mse_se = new_metrics.get('val_mse_se')

        # BUGFIX (audit follow-up): a NaN val_mse (e.g. from a fold that
        # diverged) makes every `<` comparison below evaluate to False,
        # which *happens* to be safe for the primary MSE gate (a NaN
        # candidate silently never looks "better") but is fragile and
        # relies on that implicit behavior rather than an explicit check.
        # Reject explicitly and up front so this doesn't depend on Python's
        # NaN comparison semantics, and so the reason is logged.
        def _bad(x):
            return x is None or (isinstance(x, float) and math.isnan(x))

        if _bad(new_mse):
            self.logger.warning("New model has NaN/missing val_mse — rejecting as candidate")
            return False
        if _bad(current_mse):
            # Current production metrics are unusable for comparison; treat
            # current_mse as +inf (i.e. "unknown baseline") rather than
            # silently comparing against NaN.
            current_mse = float('inf')

        relative_improvement_ok = new_mse < current_mse * 0.95

        if new_mse_se is not None:
            # Statistically-aware path: the drop in MSE must exceed the
            # new model's own standard error, i.e. the improvement is
            # unlikely to be noise from this particular validation split.
            significant_improvement_ok = (current_mse - new_mse) > new_mse_se
            if relative_improvement_ok and significant_improvement_ok:
                return True
        else:
            # Backward-compatible path: no SE data available to compare
            # against (older metrics format) — fall back to the original
            # relative-only check.
            if relative_improvement_ok:
                return True

        # Also consider MAE (kept as-is: same relative-only bar as before,
        # MAE is a secondary/tie-breaker metric here, not the primary gate)
        #
        # BUGFIX (audit follow-up): as written, this MAE check ran
        # unconditionally whenever the MSE gate above didn't return True —
        # including when MSE was NOT significant, or even when the new
        # model's MSE was outright WORSE than production's. That let a
        # regressed-on-its-primary-metric model still get promoted purely
        # because MAE happened to clear the 5% bar, completely bypassing
        # the significance check this function's own docstring says is
        # required ("Both conditions must hold"). A tie-breaker must not
        # be able to override the primary gate on its own — it should only
        # settle genuinely close calls. Guard: MAE can only promote when
        # MSE is at least not worse than the current production model.
        new_mae = new_metrics.get('val_mae', float('inf'))
        current_mae = current_metrics.get('val_mae', float('inf'))

        if _bad(new_mae):
            return False
        if _bad(current_mae):
            current_mae = float('inf')

        if new_mse <= current_mse and new_mae < current_mae * 0.95:
            return True

        return False

# Singleton instance
retraining_system = AutomatedRetrainingSystem()