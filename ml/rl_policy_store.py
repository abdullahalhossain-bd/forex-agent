"""
ml/rl_policy_store.py — RL policy versioning (Day 71)
=======================================================

Manages RL policy versions on disk with rollback support.
Each version has metadata (training episodes, avg reward, win rate).

Directory: memory/rl_policy_versions/
  ├── v1.zip + v1_meta.json
  ├── v2.zip + v2_meta.json
  ├── latest.zip (symlink or copy of current best)
  └── _registry.json
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

log = get_logger("rl_policy_store")

POLICY_DIR = Path("memory/rl_policy_versions")
REGISTRY_PATH = POLICY_DIR / "_registry.json"


class RLPolicyStore:
    """Versioned RL policy persistence."""

    def __init__(self):
        POLICY_DIR.mkdir(parents=True, exist_ok=True)
        self._registry = self._load_registry()

    def _load_registry(self) -> Dict[str, Any]:
        if REGISTRY_PATH.exists():
            try:
                return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"versions": [], "latest": None}

    def _save_registry(self) -> None:
        try:
            REGISTRY_PATH.write_text(json.dumps(self._registry, indent=2, default=str), encoding="utf-8")
        except Exception as e:
            log.warning(f"[RLPolicyStore] registry save failed: {e}")

    def save_policy(
        self,
        model_path: Path,
        episodes: int = 0,
        avg_reward: float = 0.0,
        win_rate: float = 0.0,
        notes: str = "",
    ) -> str:
        """Save a trained policy as a new version. Returns version label."""
        version_num = len(self._registry["versions"]) + 1
        version_label = f"v{version_num}"
        dest_path = POLICY_DIR / f"{version_label}.zip"

        try:
            shutil.copy2(str(model_path), str(dest_path))
        except Exception as e:
            log.error(f"[RLPolicyStore] copy failed: {e}")
            return ""

        meta = {
            "version": version_label,
            "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "episodes": episodes,
            "avg_reward": round(avg_reward, 2),
            "win_rate": round(win_rate, 2),
            "notes": notes,
            "path": str(dest_path),
        }

        # Save meta
        meta_path = POLICY_DIR / f"{version_label}_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        # Update registry
        self._registry["versions"].append(meta)
        self._registry["latest"] = version_label
        self._save_registry()

        # Also copy as "latest"
        latest_path = POLICY_DIR / "latest.zip"
        try:
            shutil.copy2(str(dest_path), str(latest_path))
        except Exception:
            pass

        log.info(f"[RLPolicyStore] saved {version_label} | episodes={episodes} avg_reward={avg_reward:.2f}")
        return version_label

    def load_policy(self, version: Optional[str] = None) -> Optional[Path]:
        """Load a policy path. If version=None, loads latest."""
        if version is None:
            version = self._registry.get("latest")
        if version is None:
            return None
        path = POLICY_DIR / f"{version}.zip"
        return path if path.exists() else None

    def rollback(self, to_version: str) -> bool:
        """Roll back to a previous policy version."""
        versions = [v["version"] for v in self._registry["versions"]]
        if to_version not in versions:
            log.warning(f"[RLPolicyStore] version {to_version} not found")
            return False
        self._registry["latest"] = to_version
        self._save_registry()
        log.info(f"[RLPolicyStore] rolled back to {to_version}")
        return True

    def list_versions(self) -> List[Dict[str, Any]]:
        """List all policy versions."""
        return self._registry.get("versions", [])

    def stats(self) -> Dict[str, Any]:
        """Return policy store stats."""
        versions = self._registry.get("versions", [])
        return {
            "total_versions": len(versions),
            "latest": self._registry.get("latest"),
            "versions": [
                {"version": v["version"], "episodes": v.get("episodes", 0),
                 "avg_reward": v.get("avg_reward", 0), "win_rate": v.get("win_rate", 0)}
                for v in versions
            ],
        }


# ── Singleton ───────────────────────────────────────────────────────

_STORE: Optional[RLPolicyStore] = None


def get_rl_policy_store() -> RLPolicyStore:
    global _STORE
    if _STORE is None:
        _STORE = RLPolicyStore()
    return _STORE
