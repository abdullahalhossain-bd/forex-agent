# learning/strategy_config.py  —  Day 55 | Strategy Configuration + Version Control ⭐⭐⭐⭐⭐
# ============================================================
# AI-এর "live configuration" — কোন pairs active, কোন session preferred,
# current risk %, এবং পুরনো configuration গুলোর version history।
#
# Day 55 spec অনুযায়ী AI সরাসরি পুরনো strategy overwrite করে না —
# প্রতিটা পরিবর্তনের আগে নতুন version snapshot হিসেবে save হয়, যাতে
# পরে compare এবং rollback করা যায়।
#
# strategy_versions/
#   v1.0.json, v1.1.json, v1.2.json ...
# ============================================================

import json
import os
from datetime import datetime, timezone
from typing import Optional

from utils.logger import get_logger

log = get_logger("learning.strategy_config")

STRATEGY_CONFIG_PATH = "memory/strategy_config.json"
VERSIONS_DIR         = "memory/strategy_versions"

DEFAULT_CONFIG = {
    "version":            "1.0",
    "active_pairs":       ["EURUSD", "GBPUSD", "USDJPY"],
    "disabled_pairs":     {},   # pair -> {reason, disabled_at}
    "session_preference": {},   # pair -> {preferred, avoid}
    "risk_percent":       1.0,
    "base_risk_percent":  1.0,
    "updated_at":         None,
}


class StrategyConfig:
    """
    Live trading configuration manager + version control।

    Usage:
        cfg = StrategyConfig()

        cfg.get_active_pairs()
        cfg.remove_pair("GBPUSD", reason="Negative expectancy")
        cfg.set_session_preference("EURUSD", preferred="London", avoid="Asian")
        cfg.set_risk(0.8, reason="High volatility")

        cfg.save_version(label="1.1", notes="Removed GBPUSD, lowered risk")
        cfg.list_versions()
        cfg.rollback_to("1.0")
    """

    def __init__(self):
        os.makedirs("memory", exist_ok=True)
        os.makedirs(VERSIONS_DIR, exist_ok=True)
        if not os.path.exists(STRATEGY_CONFIG_PATH):
            self._save(DEFAULT_CONFIG.copy())

    # ──────────────────────────────────────────────────────────
    # PAIRS
    # ──────────────────────────────────────────────────────────

    def get_active_pairs(self) -> list:
        return self._load().get("active_pairs", [])

    def remove_pair(self, pair: str, reason: str) -> dict:
        cfg = self._load()
        if pair in cfg["active_pairs"]:
            cfg["active_pairs"].remove(pair)
        cfg["disabled_pairs"][pair] = {
            "reason":      reason,
            "disabled_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save(cfg)
        log.warning(f"[StrategyConfig] ⛔ Pair REMOVED: {pair} | {reason}")
        return {"removed": pair, "reason": reason, "active_pairs": cfg["active_pairs"]}

    def add_pair(self, pair: str, reason: str = "Re-enabled") -> dict:
        cfg = self._load()
        if pair not in cfg["active_pairs"]:
            cfg["active_pairs"].append(pair)
        cfg["disabled_pairs"].pop(pair, None)
        self._save(cfg)
        log.info(f"[StrategyConfig] ✅ Pair ADDED: {pair} | {reason}")
        return {"added": pair, "active_pairs": cfg["active_pairs"]}

    def get_disabled_pairs(self) -> dict:
        return self._load().get("disabled_pairs", {})

    # ──────────────────────────────────────────────────────────
    # SESSION PREFERENCE
    # ──────────────────────────────────────────────────────────

    def set_session_preference(self, pair: str, preferred: str = None, avoid: str = None) -> dict:
        cfg = self._load()
        entry = cfg["session_preference"].get(pair, {})
        if preferred:
            entry["preferred_session"] = preferred
        if avoid:
            entry["avoid"] = avoid
        cfg["session_preference"][pair] = entry
        self._save(cfg)
        log.info(f"[StrategyConfig] Session preference set for {pair}: {entry}")
        return entry

    def get_session_preference(self, pair: str) -> Optional[dict]:
        return self._load().get("session_preference", {}).get(pair)

    # ──────────────────────────────────────────────────────────
    # RISK
    # ──────────────────────────────────────────────────────────

    def get_risk(self) -> float:
        return self._load().get("risk_percent", DEFAULT_CONFIG["risk_percent"])

    def set_risk(self, new_risk: float, reason: str) -> dict:
        cfg = self._load()
        old_risk = cfg.get("risk_percent", DEFAULT_CONFIG["risk_percent"])
        cfg["risk_percent"] = round(new_risk, 2)
        self._save(cfg)
        log.info(f"[StrategyConfig] Risk changed: {old_risk}% → {new_risk}% | {reason}")
        return {"old_risk": old_risk, "new_risk": new_risk, "reason": reason}

    # ──────────────────────────────────────────────────────────
    # VERSION CONTROL  ⭐⭐⭐⭐⭐
    # ──────────────────────────────────────────────────────────

    def save_version(self, label: str, notes: str = "", performance_snapshot: dict = None) -> dict:
        """
        বর্তমান configuration একটা নতুন immutable version হিসেবে save করো।
        পুরনো version কখনো overwrite হয় না।
        """
        cfg = self._load()
        cfg["version"] = label
        cfg["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._save(cfg)

        version_record = {
            "label":     label,
            "notes":     notes,
            "config":    cfg,
            "performance_snapshot": performance_snapshot or {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        path = os.path.join(VERSIONS_DIR, f"v{label}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(version_record, f, indent=2, default=str)

        log.info(f"[StrategyConfig] 💾 Version saved: v{label} — {notes}")
        return version_record

    def list_versions(self) -> list:
        if not os.path.isdir(VERSIONS_DIR):
            return []
        versions = []
        for fname in sorted(os.listdir(VERSIONS_DIR)):
            if fname.endswith(".json"):
                try:
                    with open(os.path.join(VERSIONS_DIR, fname), encoding="utf-8") as f:
                        versions.append(json.load(f))
                except Exception:
                    continue
        versions.sort(key=lambda v: v.get("created_at", ""))
        return versions

    def get_version(self, label: str) -> Optional[dict]:
        path = os.path.join(VERSIONS_DIR, f"v{label}.json")
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def compare_versions(self, label_a: str, label_b: str) -> dict:
        """
        দুইটা version-এর performance snapshot compare করো।
        Example output: "Version 1.2 better by +18%"
        """
        va = self.get_version(label_a)
        vb = self.get_version(label_b)
        if not va or not vb:
            return {"comparable": False, "reason": "One or both versions not found"}

        pa = va.get("performance_snapshot", {}).get("win_rate", 0)
        pb = vb.get("performance_snapshot", {}).get("win_rate", 0)
        diff = pb - pa

        return {
            "comparable":  True,
            "a":           {"label": label_a, "win_rate": pa},
            "b":           {"label": label_b, "win_rate": pb},
            "diff_pct":    round(diff, 1),
            "verdict": (
                f"Version {label_b} better by {diff:+.1f}%" if diff > 0 else
                f"Version {label_a} better by {abs(diff):.1f}%" if diff < 0 else
                "No significant difference"
            ),
        }

    def rollback_to(self, label: str) -> dict:
        """
        নির্দিষ্ট version-এ ফিরে যাও। বর্তমান config-কে ওই version দিয়ে replace
        করা হয়, কিন্তু rollback-ও নিজে একটা নতুন version হিসেবে save হয় —
        যাতে ইতিহাস হারিয়ে না যায়।
        """
        version = self.get_version(label)
        if not version:
            return {"success": False, "reason": f"Version {label} not found"}

        restored_cfg = version["config"]
        self._save(restored_cfg)

        rollback_label = f"{label}-rollback-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        self.save_version(
            label=rollback_label,
            notes=f"Rolled back to v{label}",
            performance_snapshot=version.get("performance_snapshot", {}),
        )

        log.warning(f"[StrategyConfig] ⏮️ ROLLED BACK to v{label}")
        return {"success": True, "restored_from": label, "config": restored_cfg}

    # ──────────────────────────────────────────────────────────
    # PRINT
    # ──────────────────────────────────────────────────────────

    def print_config(self) -> None:
        cfg = self._load()
        bar = "═" * 56
        print(f"\n{bar}")
        print("  ⚙️   STRATEGY CONFIGURATION  (Day 55)")
        print(bar)
        print(f"  Version       : {cfg.get('version')}")
        print(f"  Active pairs  : {', '.join(cfg.get('active_pairs', []))}")
        if cfg.get("disabled_pairs"):
            print(f"  Disabled pairs:")
            for p, d in cfg["disabled_pairs"].items():
                print(f"    ⛔ {p} — {d.get('reason')}")
        print(f"  Risk %        : {cfg.get('risk_percent')}%")
        if cfg.get("session_preference"):
            print(f"  Session prefs :")
            for p, s in cfg["session_preference"].items():
                print(f"    {p}: preferred={s.get('preferred_session')} avoid={s.get('avoid')}")
        print(f"  Updated at    : {cfg.get('updated_at')}")
        print(bar + "\n")

    # ──────────────────────────────────────────────────────────
    # STORAGE
    # ──────────────────────────────────────────────────────────

    def _load(self) -> dict:
        if not os.path.exists(STRATEGY_CONFIG_PATH):
            return DEFAULT_CONFIG.copy()
        try:
            with open(STRATEGY_CONFIG_PATH, encoding="utf-8") as f:
                cfg = json.load(f)
                # Backward-compat: ensure all default keys exist
                for k, v in DEFAULT_CONFIG.items():
                    cfg.setdefault(k, v)
                return cfg
        except Exception:
            return DEFAULT_CONFIG.copy()

    def _save(self, cfg: dict) -> None:
        with open(STRATEGY_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, default=str)