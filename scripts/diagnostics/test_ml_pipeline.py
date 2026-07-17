#!/usr/bin/env python3
"""
test_ml_pipeline.py — ML Pipeline Diagnostic
=============================================
Read-only diagnostic that validates the entire ML pipeline from data
fetch through EnsembleEngine.decide() to the dict shape that
DecisionAgent reads (analysis_out["ensemble"]["confidence"]).

No trading side effects. Safe to run anytime.

Usage:
    cd /path/to/forex-agent
    python scripts/diagnostics/test_ml_pipeline.py
"""

import os
import sys
import traceback

# ── Ensure project root is on sys.path ──────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

results: list[tuple[str, bool, str]] = []


def step(name: str, passed: bool, detail: str = ""):
    tag = "[PASS]" if passed else "[FAIL]"
    msg = f"  {tag} {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    results.append((name, passed, detail))


def main():
    print("=" * 64)
    print("  ML PIPELINE DIAGNOSTIC")
    print("=" * 64)

    # ── Step 1: Data fetcher ─────────────────────────────────────
    print("\n── Step 1: Data Fetcher ──")
    try:
        from data.fetcher import get_data_fetcher
        fetcher = get_data_fetcher()
        step("DataFetcher import + singleton", True)
    except Exception as e:
        step("DataFetcher import + singleton", False, str(e))
        _summary()
        return

    try:
        df = fetcher.fetch_ohlcv("EURUSD", "15m", limit=200)
        if df is not None and not df.empty:
            step("Fetch OHLCV data", True, f"{len(df)} bars")
        else:
            step("Fetch OHLCV data", False, "returned None or empty (offline / no MT5?)")
    except Exception as e:
        step("Fetch OHLCV data", False, str(e))

    # ── Step 2: ModelStore registry ──────────────────────────────
    print("\n── Step 2: ModelStore Registry ──")
    try:
        from ml.model_store import get_model_store
        store = get_model_store()
        step("ModelStore import + singleton", True)
    except Exception as e:
        step("ModelStore import + singleton", False, str(e))
        _summary()
        return

    try:
        models = store.list_models(pair="EURUSD")
        model_count = len(models) if models else 0
        if model_count > 0:
            step("ModelStore.list_models()", True,
                 f"{model_count} model(s) for EURUSD: "
                 + ", ".join(f"{m.get('model_type','?')}:{m.get('version','?')}" for m in models[:5]))
        else:
            step("ModelStore.list_models()", False,
                 "0 models for EURUSD — model files may not be trained yet")
    except Exception as e:
        step("ModelStore.list_models()", False, str(e))

    # ── Step 3: Model load + feature schema match ────────────────
    print("\n── Step 3: Model Load + Feature Schema ──")
    loaded_model = None
    loaded_meta = None
    try:
        loaded_model = store.load_model("EURUSD", "15m", "xgboost")
        if loaded_model is not None:
            step("Load XGBoost model", True, f"type={type(loaded_model).__name__}")
        else:
            step("Load XGBoost model", False, "None returned (model not trained)")
    except Exception as e:
        step("Load XGBoost model", False, str(e))

    # Try RandomForest if XGBoost failed
    if loaded_model is None:
        try:
            loaded_model = store.load_model("EURUSD", "15m", "random_forest")
            if loaded_model is not None:
                step("Load RandomForest model (fallback)", True,
                     f"type={type(loaded_model).__name__}")
        except Exception as e:
            step("Load RandomForest model (fallback)", False, str(e))

    if loaded_model is not None:
        try:
            feature_names = store.get_feature_names("EURUSD", "15m",
                                                     "xgboost" if store.load_model("EURUSD", "15m", "xgboost") is not None else "random_forest")
            if feature_names:
                step("Feature schema from meta", True, f"{len(feature_names)} features")
            else:
                step("Feature schema from meta", False, "empty list (legacy model without meta)")
        except Exception as e:
            step("Feature schema from meta", False, str(e))

    # ── Step 4: Feature engineering ──────────────────────────────
    print("\n── Step 4: Feature Engineering ──")
    try:
        from ml.feature_engineer import get_feature_engineer
        fe = get_feature_engineer()
        step("FeatureEngineer import + singleton", True)
    except Exception as e:
        step("FeatureEngineer import + singleton", False, str(e))
        _summary()
        return

    features = {}
    if df is not None and not df.empty:
        try:
            # Build features with the real dataframe
            features = fe.build_feature_vector(df, pair="EURUSD", timeframe="15m")
            if features and len(features) > 0:
                step("Feature vector build", True, f"{len(features)} features")
            else:
                step("Feature vector build", False, "empty dict returned")
        except Exception as e:
            step("Feature vector build", False, str(e))

    # Check feature-count match against model's saved schema
    if loaded_model is not None and features:
        try:
            feature_names = store.get_feature_names("EURUSD", "15m", "xgboost") or \
                            store.get_feature_names("EURUSD", "15m", "random_forest") or []
            if feature_names:
                # Check overlap
                model_set = set(feature_names)
                built_set = set(features.keys())
                overlap = model_set & built_set
                missing = model_set - built_set
                extra = built_set - model_set
                if missing:
                    step("Feature-count match", False,
                         f"{len(overlap)}/{len(model_set)} match, "
                         f"{len(missing)} missing: {list(missing)[:5]}")
                else:
                    step("Feature-count match", True,
                         f"all {len(model_set)} schema features present "
                         f"(built {len(built_set)} total)")
            else:
                # No schema — check if model can still accept the features
                step("Feature-count match", None,
                     "no saved schema (legacy model) — skipping comparison")
        except Exception as e:
            step("Feature-count match", False, str(e))

    # ── Step 5: ModelPredictor.predict() ─────────────────────────
    print("\n── Step 5: ModelPredictor.predict() ──")
    try:
        from ml.model_predictor import get_model_predictor
        predictor = get_model_predictor()
        step("ModelPredictor import + singleton", True)
    except Exception as e:
        step("ModelPredictor import + singleton", False, str(e))
        _summary()
        return

    try:
        is_ready = predictor.is_ready("EURUSD", "15m")
        step("ModelPredictor.is_ready()", True, f"ready={is_ready}")
    except Exception as e:
        step("ModelPredictor.is_ready()", False, str(e))

    ml_prediction = None
    if features:
        try:
            ml_prediction = predictor.predict(features, "EURUSD", "15m")
            if ml_prediction:
                step("ModelPredictor.predict()", True,
                     f"prediction={ml_prediction.get('prediction')}, "
                     f"prob={ml_prediction.get('probability', 0):.3f}, "
                     f"agreement={ml_prediction.get('model_agreement')}, "
                     f"ml_available={ml_prediction.get('ml_available')}")
            else:
                step("ModelPredictor.predict()", False, "returned None")
        except Exception as e:
            step("ModelPredictor.predict()", False, str(e))

    # ── Step 6: EnsembleEngine.decide() ──────────────────────────
    print("\n── Step 6: EnsembleEngine.decide() ──")
    try:
        from ml.ensemble import get_ensemble_engine
        ensemble = get_ensemble_engine()
        step("EnsembleEngine import + singleton", True)
    except Exception as e:
        step("EnsembleEngine import + singleton", False, str(e))
        _summary()
        return

    ensemble_decision = None
    try:
        ensemble_decision = ensemble.decide(
            pair="EURUSD",
            timeframe="15m",
            ml_prediction=ml_prediction,
            rule_signal="BUY",
            rule_confidence=65.0,
            master_signal="BUY",
            master_confidence=70.0,
            regime="RANGING",
        )
        if ensemble_decision:
            d = ensemble_decision.to_dict() if hasattr(ensemble_decision, 'to_dict') else ensemble_decision
            step("EnsembleEngine.decide()", True,
                 f"decision={d.get('decision')}, confidence={d.get('confidence', 0):.1f}, "
                 f"agreement={d.get('agreement')}, position={d.get('position_size')}, "
                 f"ml_available={d.get('ml_available')}")
        else:
            step("EnsembleEngine.decide()", False, "returned None")
    except Exception as e:
        step("EnsembleEngine.decide()", False, str(e))

    # ── Step 7: Dict shape verification ──────────────────────────
    print("\n── Step 7: Dict Shape Verification (DecisionAgent reads) ──")
    if ensemble_decision:
        d = ensemble_decision.to_dict() if hasattr(ensemble_decision, 'to_dict') else ensemble_decision
        required_keys = [
            "decision", "confidence", "agreement", "agreement_count",
            "total_models", "position_size", "position_multiplier",
            "models", "has_conflict", "regime", "ml_available",
            "analysis_signal", "analysis_confidence",
        ]
        missing = [k for k in required_keys if k not in d]
        if not missing:
            step("EnsembleDecision.to_dict() keys", True,
                 f"all {len(required_keys)} required keys present")
        else:
            step("EnsembleDecision.to_dict() keys", False,
                 f"missing: {missing}")

        # Verify confidence is the right type and range
        conf = d.get("confidence")
        if isinstance(conf, (int, float)):
            step("Confidence type + range", 0 <= conf <= 100,
                 f"type={type(conf).__name__}, value={conf:.1f}")
        else:
            step("Confidence type + range", False,
                 f"expected numeric, got {type(conf).__name__}")

        # Verify the round-trip shape that DecisionAgent reads:
        #   analysis_out["ensemble"]["confidence"]
        simulated_analysis_out = {"ensemble": d}
        readback = simulated_analysis_out["ensemble"]["confidence"]
        if isinstance(readback, (int, float)):
            step("Round-trip analysis_out['ensemble']['confidence']", True,
                 f"value={readback:.1f}")
        else:
            step("Round-trip analysis_out['ensemble']['confidence']", False,
                 f"got {type(readback).__name__}")
    else:
        step("Dict shape verification", False, "EnsembleEngine.decide() returned None — skip")

    _summary()


def _summary():
    print("\n" + "=" * 64)
    passed = sum(1 for _, p, _ in results if p is True)
    failed = sum(1 for _, p, _ in results if p is False)
    skipped = sum(1 for _, p, _ in results if p is None)
    total = len(results)

    tag = "PASS" if failed == 0 and passed > 0 else "FAIL"
    print(f"  OVERALL: {tag}  ({passed} passed, {failed} failed, {skipped} skipped / {total} total)")

    if failed > 0:
        print("\n  Failed steps:")
        for name, p, detail in results:
            if p is False:
                print(f"    [FAIL] {name}: {detail}")

    print("=" * 64)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()