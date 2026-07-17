#!/usr/bin/env python3
"""
test_rl_pipeline.py — RL Pipeline Diagnostic
==============================================
Read-only diagnostic that validates the RL agent pipeline:
  stable-baselines3 install -> PPO file exists -> loads ->
  observation-space dims -> RLAction mapping -> predict() confidence ->
  shape check against analysis_out["rl_agent"]["confidence"].

No trading side effects. Safe to run anytime.

Usage:
    cd /path/to/forex-agent
    python scripts/diagnostics/test_rl_pipeline.py
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
    print("  RL PIPELINE DIAGNOSTIC")
    print("=" * 64)

    # ── Step 1: stable-baselines3 install check ──────────────────
    print("\n── Step 1: stable-baselines3 Install ──")
    try:
        import stable_baselines3
        sb3_version = getattr(stable_baselines3, "__version__", "unknown")
        step("stable_baselines3 import", True, f"version={sb3_version}")
    except ImportError:
        step("stable_baselines3 import", False,
             "not installed — RL agent will use heuristic fallback")
    except Exception as e:
        step("stable_baselines3 import", False, str(e))

    has_sb3 = False
    try:
        from stable_baselines3 import PPO
        has_sb3 = True
        step("PPO class available", True)
    except ImportError:
        step("PPO class available", False, "not in stable_baselines3 (unexpected)")
    except Exception as e:
        step("PPO class available", False, str(e))

    # ── Step 2: PPO model file ───────────────────────────────────
    print("\n── Step 2: PPO Model File ──")
    try:
        from config import PROJECT_ROOT
        ppo_dir = os.path.join(PROJECT_ROOT, "ml", "rl_policy")
    except Exception:
        ppo_dir = os.path.join(_PROJECT_ROOT, "ml", "rl_policy")

    ppo_latest = os.path.join(ppo_dir, "ppo_forex_latest.zip")
    if os.path.isfile(ppo_latest):
        size_kb = os.path.getsize(ppo_latest) / 1024
        step("PPO model file exists", True, f"{ppo_latest} ({size_kb:.0f} KB)")
    else:
        step("PPO model file exists", False, f"{ppo_latest} not found")
        # List what IS in the directory
        if os.path.isdir(ppo_dir):
            files = os.listdir(ppo_dir)
            step("rl_policy directory contents", True,
                 f"{len(files)} files: {files[:8]}")
        else:
            step("rl_policy directory", False, f"{ppo_dir} does not exist")

    # Check policy versioning store
    ppo_versions_dir = os.path.join(_PROJECT_ROOT, "memory", "rl_policy_versions")
    if os.path.isdir(ppo_versions_dir):
        version_files = [f for f in os.listdir(ppo_versions_dir) if f.endswith(".zip")]
        step("RL policy version store", True,
             f"{len(version_files)} version(s) in {ppo_versions_dir}")
    else:
        step("RL policy version store", False,
             f"{ppo_versions_dir} does not exist")

    # ── Step 3: RLAgent instantiation ────────────────────────────
    print("\n── Step 3: RLAgent Instantiation ──")
    try:
        from ml.rl_agent import get_rl_agent
        agent = get_rl_agent()
        step("RLAgent import + singleton", True)
    except Exception as e:
        step("RLAgent import + singleton", False, str(e))
        _summary()
        return

    try:
        status = agent.status()
        model_loaded = status.get("model_loaded", False)
        source = status.get("source", "unknown")
        step("RLAgent.status()", True,
             f"model_loaded={model_loaded}, source={source}")
    except Exception as e:
        step("RLAgent.status()", False, str(e))

    # ── Step 4: Observation space dimensions ─────────────────────
    print("\n── Step 4: Observation Space Dimensions ──")
    try:
        obs_size = agent.expected_observation_size()
        if obs_size is not None:
            step("expected_observation_size()", True, f"n_features={obs_size}")
        else:
            step("expected_observation_size()", False, "returned None (model not loaded)")
    except Exception as e:
        step("expected_observation_size()", False, str(e))

    # Check against ForexTradingEnv definition
    try:
        from ml.rl_environment import ForexTradingEnv, FEATURE_SCHEMA
        step("ForexTradingEnv import", True,
             f"base FEATURE_SCHEMA has {len(FEATURE_SCHEMA)} features")

        # ACTION constants
        from ml.rl_environment import (
            ACTION_HOLD, ACTION_BUY, ACTION_SELL, ACTION_CLOSE,
            ACTION_MAP,
        )
        step("Action space mapping", True,
             f"HOLD={ACTION_HOLD}, BUY={ACTION_BUY}, SELL={ACTION_SELL}, CLOSE={ACTION_CLOSE}")
    except Exception as e:
        step("ForexTradingEnv import / action mapping", False, str(e))

    # ── Step 5: RLAction dataclass shape ─────────────────────────
    print("\n── Step 5: RLAction Dataclass Shape ──")
    try:
        from ml.rl_agent import RLAction
        action_fields = [f.name for f in RLAction.__dataclass_fields__.values()]
        required_fields = ["action", "action_name", "confidence", "reason", "model_loaded", "source"]
        missing = [f for f in required_fields if f not in action_fields]
        if not missing:
            step("RLAction fields", True,
                 f"all {len(required_fields)} required fields: {action_fields}")
        else:
            step("RLAction fields", False, f"missing: {missing}")
    except Exception as e:
        step("RLAction dataclass", False, str(e))

    # ── Step 6: RLAgent.predict() ────────────────────────────────
    print("\n── Step 6: RLAgent.predict() ──")
    rl_result = None
    try:
        import numpy as np
        # Build a dummy observation if we know the size
        if obs_size is not None:
            dummy_state = np.zeros(obs_size, dtype=np.float32)
        else:
            # Use the FEATURE_SCHEMA length + 6 position/account features
            try:
                from ml.rl_environment import FEATURE_SCHEMA
                dummy_state = np.zeros(len(FEATURE_SCHEMA) + 6, dtype=np.float32)
            except Exception:
                dummy_state = np.zeros(22, dtype=np.float32)  # typical size

        rl_result = agent.predict(dummy_state, ensemble_signal="BUY", ensemble_confidence=0.65)

        if rl_result is not None:
            d = rl_result.to_dict() if hasattr(rl_result, 'to_dict') else dict(rl_result)
            step("RLAgent.predict()", True,
                 f"action_name={d.get('action_name')}, "
                 f"confidence={d.get('confidence', 0):.3f}, "
                 f"source={d.get('source')}, "
                 f"model_loaded={d.get('model_loaded')}")
        else:
            step("RLAgent.predict()", False, "returned None")
    except Exception as e:
        step("RLAgent.predict()", False, str(e))

    # ── Step 7: Shape check against DecisionAgent read path ──────
    print("\n── Step 7: Shape Check (DecisionAgent reads) ──")
    # DecisionAgent reads: analysis_out.get("rl_agent", {}).get("confidence")
    # where rl_agent confidence is 0-1 scale, then multiplied by 100
    if rl_result is not None:
        d = rl_result.to_dict() if hasattr(rl_result, 'to_dict') else dict(rl_result)
        rl_conf = d.get("confidence", 0)

        # Verify it's 0-1 range
        if isinstance(rl_conf, (int, float)):
            in_range = 0.0 <= rl_conf <= 1.0
            step("RL confidence in 0-1 range", in_range,
                 f"value={rl_conf:.3f}")
        else:
            step("RL confidence in 0-1 range", False,
                 f"got {type(rl_conf).__name__}")

        # Simulate the DecisionAgent read path
        simulated_analysis_out = {
            "rl_agent": {
                "action_name": d.get("action_name", "HOLD"),
                "confidence": rl_conf,
            }
        }
        readback_conf = simulated_analysis_out["rl_agent"]["confidence"]
        # DecisionAgent does: rl_conf = float(rl_ctx.get("confidence", 0)) * 100
        scaled_conf = float(readback_conf) * 100

        if isinstance(scaled_conf, (int, float)) and 0 <= scaled_conf <= 100:
            step("Round-trip analysis_out['rl_agent']['confidence']", True,
                 f"raw={readback_conf:.3f}, scaled={scaled_conf:.0f}% (DecisionAgent sees this)")
        else:
            step("Round-trip analysis_out['rl_agent']['confidence']", False,
                 f"scaled={scaled_conf}")
    else:
        step("Shape check", False, "RLAgent.predict() returned None — skip")

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