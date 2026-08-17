"""
ml/train_ppo_quick.py — Quick PPO bootstrap (compat wrapper) (Day 102+)
========================================================================
AUDIT FIX (2026-08-17, RL winrate/frequency project, round 8):

This script used to be a SEPARATE, independent training path from
ml/train_rl_v2.py — its own PPO setup, and critically its own
environment: `ml.rl_environment.ForexTradingEnv` (v1), which has the
exact same root-cause bugs that were found and fixed in v2 this round:
  - SL/TP only checked on action_name == "HOLD" steps (missed
    stop-outs and take-profits on every other action)
  - forced episode-end close returned a hardcoded reward of 0.0,
    hiding the real PnL of whatever position was open when the
    episode was cut off
  - "win_rate" computed as "fraction of episodes with positive
    cumulative reward" — not real trade-level win rate
  - no best-checkpoint tracking, no minimum-trade-count floor, so a
    degenerate never-trades policy could be saved as final
  - raw, non-normalized features
  - no cooldown / overtrade guard beyond whatever the reward engine
    penalized (which the agent could and did train straight through)

This script also wrote directly to ppo_forex_latest.zip with no
"best" checkpoint concept at all — so running it could silently
produce and deploy (as the fallback when ppo_forex_best.zip doesn't
exist yet) a broken model, undoing the v2 fixes entirely, without any
error or warning.

Rather than duplicate-fix the same bugs in a second environment
implementation (two copies of the same logic drifting apart is how
this happened in the first place), this script is now a thin
CLI-compatible wrapper around the fixed `ml.train_rl_v2` pipeline —
same command-line interface as before, but it trains through the one
verified-correct implementation (ForexTradingEnvV2 + RewardEngineV2 +
the round-1-through-7 fixes: per-step SL/TP checks, real episode-end
reward, real trade-based metrics, cooldown + daily cap, wide-vs-min SL,
26-feature normalized observation, best-checkpoint selection gated on
a minimum trade count, and ppo_forex_best.zip / ppo_forex_best_meta.json
output).

Usage (unchanged):
    python -m ml.train_ppo_quick
    python -m ml.train_ppo_quick --symbol EURUSD --timeframe M15 --bars 100000
    python -m ml.train_ppo_quick --timesteps 100000

`--debug-synthetic` is no longer supported — ForexTradingEnvV2's
training path always uses real historical data via
`load_historical_data_v2()`, consistent with the "production models
MUST use real data" rule this script always intended to enforce.
`--bars` is accepted for compatibility but currently unused: v2's data
loader pulls the full available history for the requested pair/
timeframe rather than a fixed bar count.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import get_logger

log = get_logger("train_ppo_quick")


def train_quick(
    symbol: str = "EURUSD",
    timeframe: str = "M15",
    bars: int = 100000,
    timesteps: int = 50000,
    use_synthetic: bool = False,
) -> dict:
    """Compat wrapper — delegates to ml.train_rl_v2.train_rl_agent_v2().

    Signature unchanged so any existing caller (scripts, cron jobs,
    other modules importing `train_quick` directly) keeps working.
    """
    if use_synthetic:
        log.error(
            "[TrainPPOQuick] --debug-synthetic is no longer supported here. "
            "Synthetic data isn't wired through the v2 pipeline (real data "
            "only, by design — see module docstring). Use real MT5/CSV data."
        )
        return {"error": "debug-synthetic not supported in the v2-backed wrapper"}

    if bars != 100000:
        log.info(
            f"[TrainPPOQuick] Note: --bars={bars} is accepted for CLI "
            f"compatibility but not used — the v2 pipeline loads the full "
            f"available history for {symbol}/{timeframe} rather than a "
            f"fixed bar count."
        )

    from ml.train_rl_v2 import train_rl_agent_v2

    result = train_rl_agent_v2(
        pair=symbol,
        timeframe=timeframe,
        total_timesteps=timesteps,
    )

    if "error" in result:
        return result

    # Reshape to the old return schema for backward compatibility with
    # any caller that reads specific keys off this function's result.
    return {
        "status": "success",
        "symbol": symbol,
        "timeframe": timeframe,
        "bars_requested": bars,
        "timesteps": timesteps,
        "model_path": result.get("model_path"),
        "episodes": result.get("episodes"),
        "win_rate": result.get("win_rate"),
        "avg_reward": result.get("avg_reward"),
        "best_eval_reward": result.get("best_eval_reward"),
        "meta_path": result.get("model_path", "").replace(".zip", "_meta.json")
        if result.get("model_path") else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quick PPO bootstrap (compat wrapper around ml.train_rl_v2)"
    )
    parser.add_argument("--symbol", type=str, default="EURUSD",
                        help="Trading symbol (default: EURUSD)")
    parser.add_argument("--timeframe", type=str, default="M15",
                        help="Timeframe: M15, H1, H4, D1 (default: M15)")
    parser.add_argument("--bars", type=int, default=100000,
                        help="Accepted for compatibility; not used by v2 (see docstring)")
    parser.add_argument("--timesteps", type=int, default=50000,
                        help="Training timesteps (default: 50000)")
    parser.add_argument("--debug-synthetic", action="store_true",
                        help="No longer supported — see module docstring")
    args = parser.parse_args()

    log.info(f"Symbol: {args.symbol} | Timeframe: {args.timeframe}")
    log.info(f"Training timesteps: {args.timesteps}")
    log.info("[TrainPPOQuick] Delegating to ml.train_rl_v2 (fixed pipeline)...")

    result = train_quick(
        symbol=args.symbol,
        timeframe=args.timeframe,
        bars=args.bars,
        timesteps=args.timesteps,
        use_synthetic=args.debug_synthetic,
    )

    if "error" in result:
        print(f"\n[ERROR] {result['error']}")
        sys.exit(1)
    else:
        print(f"\n{'='*60}")
        print("[PPO Quick] Training complete!")
        print(f"{'='*60}")
        for k, v in result.items():
            if k != "status":
                print(f"  {k}: {v}")
        print(f"{'='*60}")
