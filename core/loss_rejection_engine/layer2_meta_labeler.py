"""Layer 2: ML-based Meta Labeler - Binary accept/reject classifier."""
from __future__ import annotations
import logging, os, pickle, json, time, datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import numpy as np
except ImportError:
    np = None
try:
    import pandas as pd
except ImportError:
    pd = None

log = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent.parent / "memory" / "lre_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

META_REJECT_PROB = float(os.getenv("LRE_META_REJECT", "0.65"))
META_WARN_PROB = float(os.getenv("LRE_META_WARN", "0.50"))
META_MIN_TRADES = int(os.getenv("LRE_META_MIN_TRADES", "50"))

@dataclass
class MetaLabelerOutput:
    prediction: int = 1
    loss_probability: float = 0.0
    confidence: float = 0.0
    verdict: str = "PASS"
    pass_through: bool = True
    features_used: Dict[str, float] = field(default_factory=dict)
    model_loaded: bool = False
    reason: str = ""


def _atr_from_ctx(ind):
    v = None
    if isinstance(ind, dict):
        atr = ind.get("atr")
        if isinstance(atr, dict): v = atr.get("value")
        elif isinstance(atr, (int, float)): v = atr
        if v is None: v = ind.get("ATR")
    try: return float(v)
    except: return None


def _rsi_from_ctx(ind):
    v = None
    if isinstance(ind, dict):
        rsi = ind.get("rsi")
        if isinstance(rsi, dict): v = rsi.get("value")
        elif isinstance(rsi, (int, float)): v = rsi
        if v is None: v = ind.get("RSI")
    try: return float(v)
    except: return None


class MetaLabeler:
    """ML binary classifier: accept or reject a given trade signal.
    Trains on historical (features, win/loss) pairs using LightGBM/sklearn."""

    def __init__(self):
        self._model = None
        self._scaler = None
        self._feature_names: List[str] = []
        self._is_trained = False
        self._n_trades_seen = 0
        self._feature_buffer: List[Dict] = []
        self._label_buffer: List[int] = []
        self._buffer_max = 200
        self._model_path = MODEL_DIR / "meta_labeler.pkl"
        self._metadata_path = MODEL_DIR / "meta_labeler_meta.json"
        self._load_model()

    def _extract_features(self, dec_out, analysis_out, market_out) -> Dict[str, float]:
        features = {}
        ind = market_out.get("ind_ctx", {}) or {}
        features["confidence"] = float(dec_out.get("confidence", 60))
        features["rr"] = float(dec_out.get("rr", 0) or 2.0)
        features["sl_pips"] = float(dec_out.get("sl_pips", 0) or 20.0)
        features["tp_pips"] = float(dec_out.get("tp_pips", 0) or 40.0)
        atr = _atr_from_ctx(ind)
        rsi = _rsi_from_ctx(ind)
        features["atr"] = atr if atr else 0.0
        features["rsi"] = rsi if rsi else 50.0
        if atr and atr > 0 and features["sl_pips"] > 0:
            features["sl_atr_ratio"] = features["sl_pips"] / (atr * 10000)
        else:
            features["sl_atr_ratio"] = 1.0
        regime = market_out.get("regime") or {}
        if isinstance(regime, dict):
            features["regime_confidence"] = float(regime.get("confidence", 0.5))
            features["trend_strength"] = float(regime.get("trend_strength", 0.5))
            rl = str(regime.get("regime", regime.get("label", "unknown")))
            features["regime_trending"] = 1.0 if "trend" in rl.lower() else 0.0
            features["regime_volatile"] = 1.0 if "volat" in rl.lower() else 0.0
        else:
            features["regime_confidence"] = 0.5
            features["trend_strength"] = 0.5
            features["regime_trending"] = 0.0
            features["regime_volatile"] = 0.0
        smc = analysis_out.get("smc") or analysis_out.get("smc_ctx") or {}
        features["smc_score"] = float(smc.get("score", smc.get("total_score", 0)))
        features["smc_bos"] = 1.0 if smc.get("bos") else 0.0
        features["smc_ob"] = 1.0 if smc.get("order_block") else 0.0
        features["smc_fvg"] = 1.0 if smc.get("fvg") else 0.0
        features["smc_sweep"] = 1.0 if smc.get("sweep_detected") or smc.get("liquidity_sweep") else 0.0
        session = analysis_out.get("session") or analysis_out.get("session_ctx") or {}
        sq = str(session.get("quality", session.get("session_quality", "MEDIUM")))
        features["session_quality"] = 1.0 if "HIGH" in sq.upper() else (0.5 if "MEDIUM" in sq.upper() else 0.0)
        sent = analysis_out.get("sentiment") or analysis_out.get("sentiment_ctx") or {}
        features["sentiment_agree"] = float(sent.get("agreement", sent.get("sentiment_agreement", 0)))
        hour = datetime.datetime.now().hour
        features["hour_sin"] = np.sin(2 * np.pi * hour / 24) if np else 0.0
        features["hour_cos"] = np.cos(2 * np.pi * hour / 24) if np else 0.0
        mtf = market_out.get("mtf_bias")
        if isinstance(mtf, dict):
            bias = str(mtf.get("bias", "")).upper()
            direction = (dec_out.get("decision") or "WAIT").upper()
            features["mtf_aligned"] = 1.0 if direction in ("BUY", "SELL") and direction in bias else 0.0
        else:
            features["mtf_aligned"] = 0.5
        news = analysis_out.get("news") or {}
        features["news_clear"] = 0.0 if news.get("high_impact_nearby") else 1.0
        l1 = dec_out.get("_lre_l1_score")
        features["l1_composite"] = float(l1) if l1 else 0.0
        return features

    def _load_model(self):
        if not self._model_path.exists():
            log.info("[LRE-L2] No trained model, starting in collection mode")
            return
        try:
            with open(self._model_path, "rb") as f:
                saved = pickle.load(f)
            self._model = saved.get("model")
            self._scaler = saved.get("scaler")
            self._feature_names = saved.get("feature_names", [])
            self._n_trades_seen = saved.get("n_trades", 0)
            self._is_trained = True
            log.info(f"[LRE-L2] Model loaded: {self._n_trades_seen} trades")
        except Exception as e:
            log.warning(f"[LRE-L2] Failed to load model: {e}")

    def _save_model(self):
        if not self._is_trained or self._model is None: return
        try:
            with open(self._model_path, "wb") as f:
                pickle.dump({
                    "model": self._model, "scaler": self._scaler,
                    "feature_names": self._feature_names, "n_trades": self._n_trades_seen,
                }, f)
            with open(self._metadata_path, "w") as f:
                json.dump({"trained_at": time.time(), "n_trades": self._n_trades_seen, "version": 1}, f, indent=2)
            log.info(f"[LRE-L2] Model saved ({self._n_trades_seen} trades)")
        except Exception as e:
            log.warning(f"[LRE-L2] Save failed: {e}")

    def _build_model(self):
        try:
            from lightgbm import LGBMClassifier
            return LGBMClassifier(n_estimators=50, max_depth=4, learning_rate=0.05,
                                 min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
                                 random_state=42, verbose=-1)
        except ImportError: pass
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(n_estimators=50, max_depth=4, learning_rate=0.05,
                                             min_samples_leaf=10, random_state=42)
        except ImportError: pass
        return None

    def _train(self):
        if len(self._feature_buffer) < 30 or pd is None or np is None: return
        try:
            df = pd.DataFrame(self._feature_buffer)
            labels = np.array(self._label_buffer)
            pos = np.sum(labels == 1); neg = np.sum(labels == 0)
            if min(pos, neg) < 5: return
            model = self._build_model()
            if model is None: return
            means = df.mean(); stds = df.std().replace(0, 1)
            df_norm = (df - means) / stds
            self._scaler = {"means": means.to_dict(), "stds": stds.to_dict()}
            model.fit(df_norm, labels)
            self._model = model
            self._feature_names = list(df.columns)
            self._is_trained = True
            self._feature_buffer = self._feature_buffer[-20:]
            self._label_buffer = self._label_buffer[-20:]
            self._save_model()
            log.info("[LRE-L2] Model retrained")
        except Exception as e:
            log.warning(f"[LRE-L2] Training failed: {e}")

    def record_outcome(self, features: Dict[str, float], pnl: float):
        self._feature_buffer.append(features)
        self._label_buffer.append(1 if pnl > 0 else 0)
        self._n_trades_seen += 1
        if len(self._feature_buffer) >= self._buffer_max:
            self._train()

    def evaluate(self, dec_out, analysis_out, market_out, **kwargs) -> MetaLabelerOutput:
        direction = (dec_out.get("decision") or "WAIT").upper()
        if direction not in ("BUY", "SELL"):
            return MetaLabelerOutput(verdict="PASS", reason="No trade signal")
        features = self._extract_features(dec_out, analysis_out, market_out)
        dec_out["_lre_meta_features"] = features
        if not self._is_trained or self._model is None:
            return MetaLabelerOutput(model_loaded=False, verdict="PASS",
                                   reason=f"Collecting ({self._n_trades_seen}/{META_MIN_TRADES})",
                                   features_used=features)
        try:
            if pd is None: return MetaLabelerOutput(verdict="PASS", reason="No pandas")
            df = pd.DataFrame([features])
            for fname in self._feature_names:
                if fname not in df.columns: df[fname] = 0.0
            df = df[self._feature_names]
            if self._scaler:
                means = self._scaler.get("means", {})
                stds = self._scaler.get("stds", {})
                for fname in self._feature_names:
                    m = means.get(fname, 0); s = stds.get(fname, 1)
                    df[fname] = (df[fname] - m) / s
            proba = self._model.predict_proba(df)[0]
            win_prob = proba[1] if len(proba) > 1 else proba[0]
            loss_prob = 1.0 - win_prob
            confidence = abs(win_prob - 0.5) * 2
            if loss_prob >= META_REJECT_PROB:
                verdict, pt = "REJECT", False
                reason = f"P(loss)={loss_prob:.2f} >= {META_REJECT_PROB}"
            elif loss_prob >= META_WARN_PROB:
                verdict, pt = "WARN", True
                reason = f"P(loss)={loss_prob:.2f} warning"
            else:
                verdict, pt = "PASS", True
                reason = f"P(loss)={loss_prob:.2f}"
            if loss_prob >= META_REJECT_PROB:
                log.info(f"[LRE-L2] {verdict} loss_prob={loss_prob:.3f}")
            return MetaLabelerOutput(prediction=1 if win_prob > 0.5 else 0,
                                   loss_probability=round(loss_prob, 4),
                                   confidence=round(confidence, 4), verdict=verdict,
                                   pass_through=pt, model_loaded=True, reason=reason,
                                   features_used=features)
        except Exception as e:
            log.warning(f"[LRE-L2] Inference error: {e}")
            return MetaLabelerOutput(verdict="PASS", reason=f"Error: {e}")
