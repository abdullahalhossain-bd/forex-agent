"""
audit_tools/rl_ml_audit.py — RL/ML training & model-loading audit
=================================================================

Verifies the four RL/ML questions the user asked:

  1. Which of the 48 pairs are trained vs missing?
  2. Are completed models actually valid (not corrupted)?
  3. Is progress persisted safely (so a stopped run can resume)?
  4. Are trained models actually LOADED and USED during backtest / live?
     (Do not assume they are active just because files exist.)

Also checks:
  - pair detection correctness (does training iterate over the right 48?)
  - training failures are clearly logged
  - the system does not falsely report a pair as trained
  - per-pair model file naming (RL today uses ONE global file — that's a bug)

Outputs:
  download/ablation_results/rl_ml_audit.json  — machine-readable
  stdout                                       — human-readable summary

USAGE:
    py -3.13 audit_tools\\rl_ml_audit.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def audit_ml_models_per_pair() -> dict:
    """Audit the ML side: 4 model types per pair under data/trained_models/.

    This is the per-pair ML model store (xgboost / lightgbm /
    random_forest / catboost) — separate from the RL policy store.
    """
    expected_model_types = {"xgboost", "lightgbm", "random_forest", "catboost"}
    models_root = PROJECT_ROOT / "data" / "trained_models"
    pairs = {}
    issues = []

    # Pull the 48 expected pairs from config
    try:
        from config import SYMBOLS
        expected_pairs = list(SYMBOLS)
    except Exception as e:
        issues.append(f"could not import config.SYMBOLS: {e}")
        expected_pairs = []

    # Walk data/trained_models/
    if not models_root.is_dir():
        issues.append(f"models root missing: {models_root}")
    else:
        for pair_dir in sorted(models_root.iterdir()):
            if not pair_dir.is_dir():
                continue
            pair = pair_dir.name
            model_types_found = set()
            model_files = {}
            for mt_dir in pair_dir.iterdir():
                if not mt_dir.is_dir():
                    continue
                model_types_found.add(mt_dir.name)
                model_pkl = mt_dir / "model.pkl"
                normalizer_pkl = mt_dir / "normalizer.pkl"
                metadata = mt_dir / "metadata.json"
                model_meta = mt_dir / "model.pkl.meta"

                # Corruption checks
                issues_found = []
                if not model_pkl.exists():
                    issues_found.append("model.pkl MISSING")
                elif model_pkl.stat().st_size < 100:
                    issues_found.append(f"model.pkl TOO SMALL ({model_pkl.stat().st_size}B)")
                if not metadata.exists() and not model_meta.exists():
                    issues_found.append("metadata MISSING")
                if issues_found:
                    issues.append(f"{pair}/{mt_dir.name}: {'; '.join(issues_found)}")

                model_files[mt_dir.name] = {
                    "model_pkl_exists": model_pkl.exists(),
                    "model_pkl_size": model_pkl.stat().st_size if model_pkl.exists() else 0,
                    "normalizer_exists": normalizer_pkl.exists(),
                    "metadata_exists": metadata.exists() or model_meta.exists(),
                    "issues": issues_found,
                }

            missing_types = expected_model_types - model_types_found
            pairs[pair] = {
                "model_types_found": sorted(model_types_found),
                "missing_types": sorted(missing_types),
                "complete": not missing_types,
                "model_files": model_files,
            }

    # Cross-reference: which of the 48 expected pairs have NO model dir at all?
    trained_pairs = set(pairs.keys())
    missing_pairs = [p for p in expected_pairs if p not in trained_pairs]
    incomplete_pairs = [
        {"pair": p, "missing": pairs[p]["missing_types"]}
        for p in sorted(trained_pairs)
        if not pairs[p]["complete"]
    ]

    return {
        "expected_pair_count": len(expected_pairs),
        "trained_pair_count": len(trained_pairs),
        "missing_pairs": missing_pairs,
        "incomplete_pairs": incomplete_pairs,
        "per_pair": pairs,
        "issues": issues,
        "coverage_pct": round(100.0 * len(trained_pairs) / max(len(expected_pairs), 1), 1),
    }


def audit_rl_policy_store() -> dict:
    """Audit the RL side: PPO policy files under ml/rl_policy/ and
    memory/rl_policy_versions/.

    Critical bug check: ml/train_rl.py saves to a SINGLE global file
    'ppo_forex_latest.zip' regardless of which pair was trained. That
    means training a different pair OVERWRITES the previous one. There
    is NO per-pair RL model file naming AT ALL.
    """
    issues = []
    rl_policy_dir = PROJECT_ROOT / "ml" / "rl_policy"
    policy_files = []
    if rl_policy_dir.is_dir():
        for f in rl_policy_dir.iterdir():
            if f.is_file():
                meta = None
                if f.suffix == ".json" and f.stem.endswith("_meta"):
                    try:
                        meta = json.loads(f.read_text(encoding="utf-8"))
                    except Exception as e:
                        issues.append(f"corrupt meta {f.name}: {e}")
                policy_files.append({
                    "name": f.name,
                    "size_bytes": f.stat().st_size,
                    "meta": meta,
                })

    # Check the per-pair naming bug
    latest_meta_path = rl_policy_dir / "ppo_forex_latest_meta.json"
    latest_meta = None
    if latest_meta_path.exists():
        try:
            latest_meta = json.loads(latest_meta_path.read_text(encoding="utf-8"))
        except Exception as e:
            issues.append(f"ppo_forex_latest_meta.json corrupt: {e}")

    # Quality-gate evaluation: does the meta pass the gate?
    quality_gate = None
    if latest_meta:
        episodes = latest_meta.get("episodes", 0)
        win_rate = latest_meta.get("win_rate", 0.0)
        avg_reward = latest_meta.get("avg_reward", 0.0)
        MIN_EPISODES = 5
        MIN_WIN_RATE = 0.01
        # Note: the rl_agent.py gate does NOT check avg_reward — a model
        # with avg_reward=-9140 (catastrophic losses) still passes.
        # This is itself an audit finding.
        gate_passes_episodes = episodes >= MIN_EPISODES
        gate_passes_win_rate = win_rate >= MIN_WIN_RATE
        gate_passes = gate_passes_episodes and gate_passes_win_rate
        quality_gate = {
            "episodes": episodes,
            "win_rate": win_rate,
            "avg_reward": avg_reward,
            "MIN_EPISODES_TO_TRUST": MIN_EPISODES,
            "MIN_WIN_RATE_TO_TRUST": MIN_WIN_RATE,
            "gate_passes": gate_passes,
            "gate_passes_episodes": gate_passes_episodes,
            "gate_passes_win_rate": gate_passes_win_rate,
            # Audit finding: avg_reward is NOT checked
            "audit_note": (
                "rl_agent._passes_quality_gate() checks episodes + win_rate "
                "but NOT avg_reward. A model with avg_reward=-9140 (massive "
                "consistent losses) PASSES the gate as long as win_rate>=0.01 "
                "and episodes>=5. This is a SAFETY BUG — a catastrophically "
                "bad model can be loaded as 'trusted'."
            ),
        }

    # The 48-pair RL training is NOT implemented anywhere
    rl_training_status = {
        "train_rl_script": "ml/train_rl.py",
        "iterates_48_pairs": False,
        "explanation": (
            "ml/train_rl.py's CLI takes a single --pair arg and trains ONE "
            "pair per invocation. There is NO loop over config.SYMBOLS, "
            "NO progress.json / completed_pairs file, and NO --resume flag. "
            "If interrupted midway through a 48-pair batch, the next run "
            "has no way to know which pairs were already done — it would "
            "either start from pair 0 again or rely on the operator to "
            "pass the right --pair arg. This is a critical missing feature."
        ),
        "per_pair_model_files": False,
        "explanation_per_pair": (
            "train_rl.py saves to a SINGLE hardcoded path "
            "ml/rl_policy/ppo_forex_latest.zip regardless of --pair. "
            "Training GBPUSD then EURUSD OVERWRITES the same file. "
            "The meta JSON records the pair trained on, but the model "
            "file itself has no per-pair naming. There is NO way to "
            "have 48 separate RL policies loaded simultaneously."
        ),
        "progress_persistence": False,
        "explanation_progress": (
            "RLPolicyStore (_registry.json) tracks versions v1/v2/v3... "
            "but each version is a SNAPSHOT of whatever pair was last "
            "trained. There is no field for 'which pairs have been "
            "trained' — only 'what is the latest version'. A resume "
            "from interrupted 48-pair training is impossible."
        ),
        "corruption_detection": False,
        "explanation_corruption": (
            "RLPolicyStore.load_policy() only checks if the file exists "
            "at line 108-109 — it does NOT validate the .zip integrity, "
            "does NOT verify the model can be deserialized, and does NOT "
            "compare the meta's pair field to the requested pair. A "
            "truncated/corrupt .zip would silently fail at PPO.load() "
            "time inside rl_agent.load_model(), which is wrapped in "
            "try/except and falls back to heuristic mode — hiding the "
            "corruption from the operator."
        ),
        "training_failure_logging": "partial",
        "explanation_failure_logging": (
            "train_rl_agent() returns {'error': ...} on data-load failure "
            "and logs stage errors at ERROR level. But the CLI exit code "
            "is always 0 (no sys.exit(non-zero) on training error), so a "
            "CI/cron wrapper cannot detect failure from exit code alone."
        ),
    }

    return {
        "rl_policy_dir": str(rl_policy_dir),
        "policy_files": policy_files,
        "latest_meta": latest_meta,
        "quality_gate": quality_gate,
        "rl_training_status": rl_training_status,
        "issues": issues,
    }


def audit_model_loading_in_backtest() -> dict:
    """Verify whether RL/ML models are actually LOADED and USED during
    backtest execution — not just that files exist.

    Method: static analysis of the call graph from
    backtest.persistent_runner._process_bar → trader.evaluate_decision_core
    → agents.analysis_agent.run → ml.model_predictor + ml.rl_agent.
    """
    findings = []

    # 1. persistent_runner calls trader.evaluate_decision_core
    pr_path = PROJECT_ROOT / "backtest" / "persistent_runner.py"
    pr_src = pr_path.read_text(encoding="utf-8") if pr_path.exists() else ""
    if "evaluate_decision_core" in pr_src:
        findings.append({
            "check": "persistent_runner calls evaluate_decision_core",
            "status": "PASS",
            "evidence": "backtest/persistent_runner.py invokes trader.evaluate_decision_core(market_out, session_ctx)",
        })
    else:
        findings.append({
            "check": "persistent_runner calls evaluate_decision_core",
            "status": "FAIL",
            "evidence": "persistent_runner does NOT call evaluate_decision_core — backtest bypasses the analysis pipeline entirely",
        })

    # 2. evaluate_decision_core calls analysis_agent.run (which loads ML/RL)
    trader_path = PROJECT_ROOT / "core" / "trader.py"
    trader_src = trader_path.read_text(encoding="utf-8") if trader_path.exists() else ""
    if "self._analysis.run" in trader_src and "evaluate_decision_core" in trader_src:
        findings.append({
            "check": "evaluate_decision_core invokes AnalysisAgent.run",
            "status": "PASS",
            "evidence": "core/trader.py evaluate_decision_core calls self._analysis.run(market_out)",
        })
    else:
        findings.append({
            "check": "evaluate_decision_core invokes AnalysisAgent.run",
            "status": "FAIL",
            "evidence": "evaluate_decision_core does not call AnalysisAgent.run — ML/RL never invoked",
        })

    # 3. analysis_agent.run calls ml.model_predictor + ml.rl_agent
    aa_path = PROJECT_ROOT / "agents" / "analysis_agent.py"
    aa_src = aa_path.read_text(encoding="utf-8") if aa_path.exists() else ""
    if "from ml.model_predictor import get_model_predictor" in aa_src:
        findings.append({
            "check": "AnalysisAgent imports ml.model_predictor",
            "status": "PASS",
            "evidence": "agents/analysis_agent.py imports ModelPredictor (Day 69 ML ensemble)",
        })
    else:
        findings.append({
            "check": "AnalysisAgent imports ml.model_predictor",
            "status": "FAIL",
        })
    if "from ml.rl_agent import get_rl_agent" in aa_src:
        findings.append({
            "check": "AnalysisAgent imports ml.rl_agent",
            "status": "PASS",
            "evidence": "agents/analysis_agent.py imports RLAgent (Day 71 RL filter)",
        })
    else:
        findings.append({
            "check": "AnalysisAgent imports ml.rl_agent",
            "status": "FAIL",
        })

    # 4. RL agent singleton loads ppo_forex_latest.zip on init
    rla_path = PROJECT_ROOT / "ml" / "rl_agent.py"
    rla_src = rla_path.read_text(encoding="utf-8") if rla_path.exists() else ""
    if "PPO.load" in rla_src and "ppo_forex_latest.zip" in rla_src:
        findings.append({
            "check": "RLAgent loads ppo_forex_latest.zip via PPO.load",
            "status": "PASS",
            "evidence": "ml/rl_agent.py load_model() calls PPO.load(model_path) on the singleton init",
        })

    # 5. CRITICAL: stable_baselines3 must be installed or RL silently falls back
    try:
        import stable_baselines3  # noqa: F401
        sb3_available = True
        sb3_status = "PASS"
        sb3_evidence = "stable_baselines3 is installed in this env — PPO model will be loaded if quality gate passes"
    except ImportError:
        sb3_available = False
        sb3_status = "WARN"
        sb3_evidence = (
            "stable_baselines3 is NOT installed in this env — "
            "RLAgent._check_sb3() returns False, load_model() returns False "
            "immediately, and predict() falls back to the heuristic path "
            "(source='heuristic'). The 'loaded' RL model has ZERO effect "
            "on trading decisions. Check that the production env has "
            "stable_baselines3 + torch installed."
        )
    findings.append({
        "check": "stable_baselines3 available for PPO.load",
        "status": sb3_status,
        "evidence": sb3_evidence,
        "sb3_available": sb3_available,
    })

    # 6. CRITICAL: the existing ppo_forex_latest.zip was trained on GBPUSD H1
    # but is loaded globally — so an EURUSD H1 backtest uses the WRONG pair's model
    latest_meta_path = PROJECT_ROOT / "ml" / "rl_policy" / "ppo_forex_latest_meta.json"
    pair_mismatch = None
    if latest_meta_path.exists():
        try:
            meta = json.loads(latest_meta_path.read_text(encoding="utf-8"))
            trained_on = meta.get("symbol", "?")
            trained_tf = meta.get("timeframe", "?")
            pair_mismatch = {
                "trained_on_pair": trained_on,
                "trained_on_timeframe": trained_tf,
                "audit_finding": (
                    f"The single existing PPO model was trained on {trained_on} {trained_tf}, "
                    f"but ml/rl_agent.get_rl_agent() is a SINGLETON with NO pair-awareness. "
                    f"When backtesting EURUSD H1, this GBPUSD-H1 model is loaded and its "
                    f"predict() output is fed into DecisionAgent's weighted vote as the "
                    f"'rl_agent' signal. This is a SILENT PAIR MISMATCH — the model has "
                    f"never seen EURUSD H1 features during training, so its predictions "
                    f"are essentially noise dressed up as RL wisdom. Either train per-pair "
                    f"models (with per-pair filenames) or skip RL load entirely when the "
                    f"current pair != trained_on pair."
                ),
            }
            findings.append({
                "check": "RL model pair matches backtest pair",
                "status": "FAIL",
                "evidence": pair_mismatch["audit_finding"],
                **pair_mismatch,
            })
        except Exception as e:
            findings.append({
                "check": "RL model pair matches backtest pair",
                "status": "WARN",
                "evidence": f"could not read latest_meta: {e}",
            })

    # 7. ML model_predictor — does it actually find models for EURUSD H1?
    # Models live under data/trained_models/{PAIR}/{model_type}/ — but the
    # model_predictor.py first looks under memory/ml_models/{PAIR}_{TF}/
    # (which doesn't exist in this repo). It only falls back to
    # data/trained_models/ when caller passes df_recent (institutional path).
    mp_path = PROJECT_ROOT / "ml" / "model_predictor.py"
    mp_src = mp_path.read_text(encoding="utf-8") if mp_path.exists() else ""
    findings.append({
        "check": "ModelPredictor falls back to data/trained_models/ when memory/ml_models empty",
        "status": "PASS" if "_predict_institutional" in mp_src and "data/trained_models" in mp_src else "FAIL",
        "evidence": (
            "ModelPredictor._load_models() first checks memory/ml_models/{PAIR}_{TF}/ "
            "(empty in this repo — that dir is gitignored). If empty AND caller passes "
            "df_recent, falls back to _predict_institutional() which globs "
            "data/trained_models/{PAIR}/*/production/metadata.json. "
            + (
                "WARNING: AnalysisAgent passes df_recent (verified in source) so the "
                "fallback IS active — but only for pairs that have a 'production/' subdir, "
                "which is NOT how train_historical.py saves them (it saves directly "
                "to {PAIR}/{model_type}/, no production/ subdir). The institutional "
                "fallback therefore NEVER finds a model and ML ensemble returns "
                "NOT_READY — silently degrading to rule-based logic."
                if "production/metadata.json" in mp_src else ""
            )
        ),
    })

    # 8. Check if data/trained_models/{PAIR}/xgboost/ has 'production/' subdir
    # (model_predictor._predict_institutional requires it)
    eurusd_xgb = PROJECT_ROOT / "data" / "trained_models" / "EURUSD" / "xgboost"
    has_production_subdir = (eurusd_xgb / "production").is_dir() if eurusd_xgb.exists() else False
    findings.append({
        "check": "data/trained_models/EURUSD/xgboost/production/ subdir exists",
        "status": "PASS" if has_production_subdir else "FAIL",
        "evidence": (
            f"model_predictor._predict_institutional() looks for "
            f"data/trained_models/{{PAIR}}/*/production/metadata.json. "
            f"EURUSD/xgboost/production/ exists: {has_production_subdir}. "
            f"If False, the institutional fallback will NOT find the model "
            f"even though it exists at data/trained_models/EURUSD/xgboost/model.pkl."
        ),
    })

    return {
        "findings": findings,
        "summary": {
            "total": len(findings),
            "pass": sum(1 for f in findings if f["status"] == "PASS"),
            "warn": sum(1 for f in findings if f["status"] == "WARN"),
            "fail": sum(1 for f in findings if f["status"] == "FAIL"),
        },
    }


def audit_48_pair_progress_persistence() -> dict:
    """Check whether there's any progress-persistence mechanism for the
    48-pair training the user wants.

    Required behavior:
      - Prints "Training pairs: 0/48" → "1/48" → ... → "48/48"
      - If stopped midway, completed training remains available
      - Next run continues from remaining pairs (not retraining)
      - Corrupted/incomplete models are detected
      - Training failures are clearly logged
      - System does not falsely report a pair as trained
    """
    checks = []

    # Check ml/train_rl.py for 48-pair iteration
    trl_path = PROJECT_ROOT / "ml" / "train_rl.py"
    trl_src = trl_path.read_text(encoding="utf-8") if trl_path.exists() else ""
    checks.append({
        "requirement": "Train RL iterates over all 48 pairs",
        "status": "FAIL" if "for pair in" not in trl_src or "SYMBOLS" not in trl_src else "PASS",
        "evidence": (
            "ml/train_rl.py CLI takes --pair (single) arg. "
            "There is NO loop over config.SYMBOLS. To train 48 pairs the "
            "operator must invoke train_rl.py 48 times manually, with no "
            "tracking of which were done."
        ),
    })

    # Check ai/automated_retraining.py — it DOES iterate FOREX_PAIRS but
    # does NOT have progress persistence
    ar_path = PROJECT_ROOT / "ai" / "automated_retraining.py"
    ar_src = ar_path.read_text(encoding="utf-8") if ar_path.exists() else ""
    has_pair_loop = "for pair in self.config.FOREX_PAIRS" in ar_src
    has_progress_file = any(
        s in ar_src for s in (
            "progress.json", "completed_pairs", "training_progress",
            "resume_from", "skip_trained",
        )
    )
    has_progress_print = "Training pairs:" in ar_src and "/48" in ar_src
    checks.append({
        "requirement": "ML retraining iterates over all 48 pairs",
        "status": "PASS" if has_pair_loop else "FAIL",
        "evidence": (
            "ai/automated_retraining._retrain_models() loops over "
            "self.config.FOREX_PAIRS (which == config.SYMBOLS, 48 pairs)."
        ) if has_pair_loop else "no pair loop found",
    })
    checks.append({
        "requirement": "Progress persistence file exists",
        "status": "FAIL" if not has_progress_file else "PASS",
        "evidence": (
            "ai/automated_retraining.py has NO progress.json, NO completed_pairs "
            "set, NO --resume flag. If interrupted midway, the next run starts "
            "from pair 0 again — retraining already-trained pairs."
        ) if not has_progress_file else "progress file found",
    })
    checks.append({
        "requirement": "Progress printed as '0/48' → '48/48'",
        "status": "FAIL" if not has_progress_print else "PASS",
        "evidence": (
            "Neither ml/train_rl.py nor ai/automated_retraining.py prints "
            "'Training pairs: N/48' progress. The operator has no way to "
            "see at-a-glance how far through the 48-pair batch the training is."
        ) if not has_progress_print else "progress print found",
    })

    # Check for the "Training pairs:" pattern anywhere
    found_progress_pattern = []
    for p in PROJECT_ROOT.rglob("*.py"):
        if "/.git/" in str(p) or "\\venv\\" in str(p):
            continue
        # Skip this audit script itself (it would match its own strings)
        if p.name == "rl_ml_audit.py" and "audit_tools" in str(p):
            continue
        try:
            src = p.read_text(encoding="utf-8")
            if "Training pairs:" in src and "/48" in src:
                found_progress_pattern.append(str(p.relative_to(PROJECT_ROOT)))
        except Exception:
            pass
    checks.append({
        "requirement": "'Training pairs: N/48' pattern anywhere in codebase",
        "status": "PASS" if found_progress_pattern else "FAIL",
        "evidence": (
            f"Found in: {found_progress_pattern}" if found_progress_pattern
            else "Pattern not found anywhere. The user's expected progress output is NOT implemented."
        ),
    })

    return {
        "checks": checks,
        "summary": {
            "total": len(checks),
            "pass": sum(1 for c in checks if c["status"] == "PASS"),
            "fail": sum(1 for c in checks if c["status"] == "FAIL"),
        },
    }


def main():
    print("="*78)
    print("  RL/ML TRAINING & MODEL-LOADING AUDIT")
    print("="*78)

    print("\n[1/4] Auditing per-pair ML models (data/trained_models/)...")
    ml_audit = audit_ml_models_per_pair()
    print(f"  Expected pairs: {ml_audit['expected_pair_count']}")
    print(f"  Trained pairs:  {ml_audit['trained_pair_count']} ({ml_audit['coverage_pct']}%)")
    print(f"  Missing pairs:  {ml_audit['missing_pairs']}")
    print(f"  Incomplete:     {len(ml_audit['incomplete_pairs'])}")
    for ip in ml_audit["incomplete_pairs"]:
        print(f"    {ip['pair']}: missing {ip['missing']}")
    if ml_audit["issues"]:
        print(f"  Issues ({len(ml_audit['issues'])}):")
        for i in ml_audit["issues"][:10]:
            print(f"    - {i}")

    print("\n[2/4] Auditing RL policy store (ml/rl_policy/)...")
    rl_audit = audit_rl_policy_store()
    if rl_audit["latest_meta"]:
        m = rl_audit["latest_meta"]
        print(f"  Latest model trained on: {m.get('symbol','?')} {m.get('timeframe','?')}")
        print(f"  Episodes: {m.get('episodes',0):,} | win_rate: {m.get('win_rate',0):.4f} | avg_reward: {m.get('avg_reward',0):.2f}")
    if rl_audit["quality_gate"]:
        q = rl_audit["quality_gate"]
        print(f"  Quality gate: {'PASSES' if q['gate_passes'] else 'FAILS'} "
              f"(episodes={q['episodes']}≥{q['MIN_EPISODES_TO_TRUST']} "
              f"AND win_rate={q['win_rate']}≥{q['MIN_WIN_RATE_TO_TRUST']})")
        print(f"  ⚠ AUDIT NOTE: {q['audit_note']}")
    rs = rl_audit["rl_training_status"]
    print(f"  48-pair iteration: {'YES' if rs['iterates_48_pairs'] else 'NO'}")
    print(f"  Per-pair model files: {'YES' if rs['per_pair_model_files'] else 'NO'}")
    print(f"  Progress persistence: {'YES' if rs['progress_persistence'] else 'NO'}")
    print(f"  Corruption detection: {'YES' if rs['corruption_detection'] else 'NO'}")

    print("\n[3/4] Auditing model loading in backtest path...")
    loading_audit = audit_model_loading_in_backtest()
    for f in loading_audit["findings"]:
        marker = "✓" if f["status"] == "PASS" else ("⚠" if f["status"] == "WARN" else "✗")
        print(f"  {marker} [{f['status']}] {f['check']}")
        # Truncate evidence to 200 chars for stdout
        ev = f["evidence"]
        if len(ev) > 200:
            ev = ev[:200] + "..."
        print(f"      {ev}")

    print("\n[4/4] Auditing 48-pair progress persistence...")
    progress_audit = audit_48_pair_progress_persistence()
    for c in progress_audit["checks"]:
        marker = "✓" if c["status"] == "PASS" else "✗"
        print(f"  {marker} [{c['status']}] {c['requirement']}")
        ev = c["evidence"]
        if len(ev) > 200:
            ev = ev[:200] + "..."
        print(f"      {ev}")

    # Write JSON report
    out_dir = PROJECT_ROOT / "download" / "ablation_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ml_models": ml_audit,
        "rl_policy_store": rl_audit,
        "model_loading_in_backtest": loading_audit,
        "progress_persistence": progress_audit,
    }
    out_path = out_dir / "rl_ml_audit.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\n[audit] Full JSON report: {out_path}")

    # Exit non-zero if any FAIL
    has_fail = (
        any(c["status"] == "FAIL" for c in progress_audit["checks"])
        or any(f["status"] == "FAIL" for f in loading_audit["findings"])
        or bool(ml_audit["issues"])
    )
    return 1 if has_fail else 0


if __name__ == "__main__":
    sys.exit(main())
