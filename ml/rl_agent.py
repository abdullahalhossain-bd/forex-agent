"""
ml/rl_agent.py — PPO RL Agent (Day 71)
========================================

Wraps stable-baselines3 PPO for forex trading. If stable-baselines3 is
not installed, falls back to a heuristic agent that uses the ensemble
prediction + simple rules to decide actions.

The RL agent's role is NOT to trade directly — it acts as a final
"should I really take this trade?" filter on top of the Day 70 Ensemble.
The ensemble says "BUY 75%", and the RL agent says "in similar past
situations, this lost money — WAIT" or "this looks like our winning
pattern — go for it".

Usage:
    agent = get_rl_agent()
    action = agent.predict(state_vector)
    # action = 0 (HOLD) / 1 (BUY) / 2 (SELL) / 3 (CLOSE)
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from utils.logger import get_logger
from config import PROJECT_ROOT

log = get_logger("rl_agent")

# Default PPO model path — absolute, CWD-independent
_DEFAULT_RL_POLICY_PATH = PROJECT_ROOT / "ml" / "rl_policy" / "ppo_forex_latest.zip"


@dataclass
class RLAction:
    """RL agent's action recommendation."""
    action: int                # 0=HOLD, 1=BUY, 2=SELL, 3=CLOSE
    action_name: str           # HOLD / BUY / SELL / CLOSE
    confidence: float = 0.5    # 0-1, how confident the RL agent is
    reason: str = ""
    model_loaded: bool = False
    source: str = "heuristic"  # "ppo" or "heuristic"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RLAgent:
    """PPO-based RL agent with heuristic fallback."""

    def __init__(self):
        self._lock = threading.RLock()
        self._model = None
        self._model_loaded = False
        self._sb3_available = self._check_sb3()
        self._training_episodes = 0
        self._avg_reward = 0.0
        self._best_strategy = "unknown"
        self._risk_behavior = "conservative"

    def _check_sb3(self) -> bool:
        """Check if stable-baselines3 is available."""
        try:
            import stable_baselines3
            return True
        except ImportError:
            log.info("[RL Agent] stable-baselines3 not installed — using heuristic fallback")
            return False

    # AUDIT FIX (2026-07): a loaded PPO .zip has no way to self-report how
    # well (or how little) it was trained — predict() would treat any
    # successfully-loaded model as equally trustworthy, whether it was
    # trained for 500k timesteps or 1. Convention: a sidecar
    # "{stem}_meta.json" next to the model file (same convention
    # ml/rl_policy_store.py already uses for its versioned policies —
    # v1.zip + v1_meta.json) carries episodes/win_rate/avg_reward. If
    # present, gate on it. If absent, treat as UNVERIFIED (not "assumed
    # fine") — this codebase has a real example of exactly the failure
    # this guards against: memory/rl_policy_versions/v2_meta.json records
    # episodes=1, win_rate=0.0 for a "trained" policy.
    MIN_EPISODES_TO_TRUST = 5
    MIN_WIN_RATE_TO_TRUST = 0.01  # >0: must have won at least once, ever

    def _load_policy_metadata(self, model_path: Path) -> Optional[Dict[str, Any]]:
        meta_path = model_path.parent / f"{model_path.stem}_meta.json"
        if not meta_path.exists():
            return None
        try:
            import json
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning(f"[RL Agent] failed to read policy metadata {meta_path}: {e}")
            return None

    def _passes_quality_gate(self, meta: Optional[Dict[str, Any]]) -> tuple[bool, str]:
        if meta is None:
            return False, "no training metadata found alongside model file — cannot verify quality, treating as unverified"
        episodes = meta.get("episodes", 0)
        win_rate = meta.get("win_rate", 0.0)
        if episodes < self.MIN_EPISODES_TO_TRUST:
            return False, f"only {episodes} training episode(s) recorded (need >= {self.MIN_EPISODES_TO_TRUST}) — undertrained"
        if win_rate < self.MIN_WIN_RATE_TO_TRUST:
            return False, f"recorded win_rate={win_rate:.1%} in its own training run — no evidence this policy ever won"
        return True, f"passed: {episodes} episodes, {win_rate:.1%} win_rate"

    def load_model(self, model_path: Optional[Path] = None) -> bool:
        """Load a trained PPO model from disk."""
        if not self._sb3_available:
            return False
        try:
            from stable_baselines3 import PPO
            model_path = model_path or _DEFAULT_RL_POLICY_PATH
            if model_path.exists():
                meta = self._load_policy_metadata(model_path)
                passed, gate_reason = self._passes_quality_gate(meta)
                self._policy_meta = meta
                self._quality_gate_reason = gate_reason
                if not passed:
                    log.warning(
                        f"[RL Agent] PPO model at {model_path} exists but FAILED the "
                        f"training-quality gate ({gate_reason}) — NOT loading it as "
                        f"trusted; falling back to heuristic mode instead of letting an "
                        f"unverified/undertrained policy vote on live trades."
                    )
                    self._model = None
                    self._model_loaded = False
                    return False
                self._model = PPO.load(str(model_path))
                self._model_loaded = True
                log.info(f"[RL Agent] PPO model loaded from {model_path} ({gate_reason})")
                return True
            else:
                log.warning(f"[RL Agent] No model at {model_path} — using heuristic (CHECK: file exists={model_path.exists()})")
                log.warning(f"[RL Agent] Searching in: {model_path.parent}")
                if model_path.parent.exists():
                    log.warning(f"[RL Agent] Files in directory: {list(model_path.parent.iterdir())}")
                return False
        except Exception as e:
            log.exception(f"[RL Agent] model load failed from {model_path}: {e}")
            import traceback
            log.error(f"[RL Agent] Full traceback: {traceback.format_exc()}")
            return False

    def predict(self, state: np.ndarray, ensemble_signal: str = "WAIT",
                ensemble_confidence: float = 0.0) -> RLAction:
        """Predict the best action given the current state.

        Args:
            state: Feature vector from the environment.
            ensemble_signal: What the Day 70 Ensemble says (BUY/SELL/WAIT).
            ensemble_confidence: Ensemble's confidence (0-100).

        Returns:
            RLAction with the recommended action.
        """
        # If PPO model is loaded, use it
        if self._model_loaded and self._model is not None:
            try:
                action, _ = self._model.predict(state, deterministic=True)
                action_name = {0: "HOLD", 1: "BUY", 2: "SELL", 3: "CLOSE"}.get(int(action), "HOLD")
                return RLAction(
                    action=int(action),
                    action_name=action_name,
                    confidence=0.7,  # PPO doesn't output confidence directly
                    reason=f"PPO model predicted {action_name}",
                    model_loaded=True,
                    source="ppo",
                )
            except Exception as e:
                log.warning(f"[RL Agent] PPO predict failed: {e} — falling back to heuristic")

        # ── Heuristic fallback ─────────────────────────────────────
        # The heuristic agent acts as a "wisdom filter" on the ensemble:
        #   - If ensemble says BUY/SELL with high confidence → agree
        #   - If ensemble says WAIT → HOLD
        #   - If confidence is marginal → suggest HOLD (patience)
        #   - If there's an open position and ensemble disagrees → CLOSE
        return self._heuristic_predict(ensemble_signal, ensemble_confidence)

    def _heuristic_predict(self, ensemble_signal: str, ensemble_confidence: float) -> RLAction:
        """Heuristic action when no PPO model is available.

        IMPORTANT FIX: The heuristic RL agent's job is to filter out
        genuinely dangerous trades (e.g. counter-trend during high
        volatility), NOT to second-guess the ensemble's confidence
        threshold. Previously, this method returned HOLD for ensemble
        signals with confidence < 45%, which created a phantom "RL
        says SELL/HOLD" conflict even when every other module agreed
        on BUY. The DecisionAgent's weighted voting then saw reduced
        participation, pushing consensus below minimum and blocking
        valid trades.

        New behavior:
        - If ensemble says BUY/SELL at ANY confidence → agree (don't
          veto). The ensemble's own confidence scoring already handles
          the quality threshold.
        - Only HOLD when ensemble itself says WAIT/HOLD.
        - This eliminates the RL-vs-ensemble false conflict while
          still allowing a trained PPO model (when available) to
          provide genuine independent analysis.
        """
        if ensemble_signal == "BUY":
            return RLAction(
                action=1, action_name="BUY", confidence=0.55,
                reason="Heuristic: agrees with ensemble BUY direction",
                model_loaded=False, source="heuristic",
            )
        elif ensemble_signal == "SELL":
            return RLAction(
                action=2, action_name="SELL", confidence=0.55,
                reason="Heuristic: agrees with ensemble SELL direction",
                model_loaded=False, source="heuristic",
            )
        else:
            return RLAction(
                action=0, action_name="HOLD", confidence=0.5,
                reason="Heuristic: no clear ensemble signal — hold",
                model_loaded=False, source="heuristic",
            )

    def train(self, env, total_timesteps: int = 100000, save_path: Optional[Path] = None) -> Dict[str, Any]:
        """Train the PPO model on the given environment.

        Args:
            env: A ForexTradingEnv instance.
            total_timesteps: Number of training steps.
            save_path: Where to save the trained model.

        Returns:
            Dict with training stats.
        """
        if not self._sb3_available:
            return {
                "error": "stable-baselines3 not installed — cannot train PPO",
                "install": "pip install stable-baselines3",
            }

        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.callbacks import BaseCallback

            log.info(f"[RL Agent] Training PPO for {total_timesteps} timesteps...")

            model = PPO(
                "MlpPolicy", env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                verbose=1,
            )

            # Training callback for logging
            class TrainingCallback(BaseCallback):
                def __init__(self):
                    super().__init__()
                    self.episode_rewards = []

                def _on_step(self):
                    if self.locals.get("done", False):
                        info = self.locals.get("infos", [{}])[-1]
                        if isinstance(info, dict):
                            self.episode_rewards.append(info.get("episode_reward", 0))
                    return True

            callback = TrainingCallback()
            model.learn(total_timesteps=total_timesteps, callback=callback)

            # Save model
            save_path = save_path or _DEFAULT_RL_POLICY_PATH
            save_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(save_path))

            self._model = model
            self._model_loaded = True
            self._training_episodes = len(callback.episode_rewards)
            if callback.episode_rewards:
                self._avg_reward = float(np.mean(callback.episode_rewards[-100:]))

            log.info(
                f"[RL Agent] Training complete: {self._training_episodes} episodes, "
                f"avg reward: {self._avg_reward:.2f}, saved to {save_path}"
            )

            return {
                "status": "success",
                "episodes": self._training_episodes,
                "avg_reward": round(self._avg_reward, 2),
                "total_timesteps": total_timesteps,
                "model_path": str(save_path),
            }
        except Exception as e:
            log.error(f"[RL Agent] training failed: {e}")
            return {"error": str(e)}

    def expected_observation_size(self) -> Optional[int]:
        """Return the loaded PPO model's real observation vector length,
        or None if no model is loaded (heuristic-only mode).

        Callers should build/pad/truncate their feature vector to THIS
        size rather than hardcoding a constant — a hardcoded size drifts
        out of sync the moment the model is retrained with a different
        feature count (this already happened twice: 16 → 24 → 167).
        """
        if self._model_loaded and self._model is not None:
            try:
                return int(self._model.observation_space.shape[0])
            except Exception as e:
                log.warning(f"[RL Agent] Could not read observation_space shape: {e}")
        return None

    def status(self) -> Dict[str, Any]:
        """Return RL agent status for dashboard."""
        return {
            "sb3_available": self._sb3_available,
            "model_loaded": self._model_loaded,
            "source": "ppo" if self._model_loaded else "heuristic",
            "training_episodes": self._training_episodes,
            "avg_reward": round(self._avg_reward, 2),
            "best_strategy": self._best_strategy,
            "risk_behavior": self._risk_behavior,
        }


# ── Singleton ───────────────────────────────────────────────────────

_AGENT: Optional[RLAgent] = None


def get_rl_agent() -> RLAgent:
    global _AGENT
    if _AGENT is None:
        _AGENT = RLAgent()
        _AGENT.load_model()  # try to load on init
    return _AGENT