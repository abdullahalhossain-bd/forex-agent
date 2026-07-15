# learning/rule_updater.py  —  Day 52 | Dynamic Rule Update System ⭐⭐⭐⭐⭐
# ============================================================
# DeepMistakeAnalyzer থেকে approved rule changes এখানে apply হয়।
# DecisionAgent এবং MasterAnalyst এই rules দেখে confidence adjust করে।
#
# Key Design: Overfitting রোধে rule update হয় শুধুমাত্র:
#   1. Same mistake >= MISTAKE_THRESHOLD (5) বার হলে
#   2. Human approval পেলে
#   3. Statistical significance থাকলে
# ============================================================

import json
import os
from datetime import datetime, timezone

from utils.logger import get_logger

log = get_logger("learning.rule_updater")

RULE_STORE_PATH   = "memory/pattern_rules.json"
RULE_HISTORY_PATH = "memory/rule_history.json"


class RuleUpdater:
    """
    Pattern confidence rules manage করে।

    Usage:
        updater = RuleUpdater()

        # Rule দেখো:
        conf = updater.get_confidence("Bullish Engulfing", "RANGING")

        # Rule update করো (only after human approval via DeepAnalyzer):
        updater.apply_rule(pattern="Bullish Engulfing", condition="RANGING", new_confidence=55)

        # DecisionAgent-এ inject করার জন্য:
        adjustment = updater.get_confidence_adjustment("Bullish Engulfing", "RANGING")
    """

    DEFAULT_CONFIDENCE = 75   # কোনো rule না থাকলে default

    def __init__(self):
        os.makedirs("memory", exist_ok=True)

    # ──────────────────────────────────────────────────────────
    # READ
    # ──────────────────────────────────────────────────────────

    def get_confidence(self, pattern: str, condition: str) -> int:
        """
        Pattern + market condition-এর জন্য current confidence পাও।

        Example:
            get_confidence("Bullish Engulfing", "RANGING") → 55
            (After 7 losses in ranging market, reduced from 75 to 55)
        """
        rules = self._load_rules()
        key = self._key(pattern, condition)
        rule = rules.get(key)

        if rule:
            conf = rule.get("confidence", self.DEFAULT_CONFIDENCE)
            log.info(f"[RuleUpdater] Rule found: {key} → {conf}%")
            return conf

        return self.DEFAULT_CONFIDENCE

    def get_confidence_adjustment(self, pattern: str, condition: str) -> dict:
        """
        DecisionAgent-এ inject করার format।
        Returns delta from default (positive = boost, negative = reduce).
        """
        current = self.get_confidence(pattern, condition)
        delta = current - self.DEFAULT_CONFIDENCE
        rules = self._load_rules()
        rule = rules.get(self._key(pattern, condition))

        return {
            "pattern":           pattern,
            "condition":         condition,
            "current_confidence": current,
            "adjustment":        delta,
            "has_rule":          rule is not None,
            "lesson":            rule.get("lesson", "") if rule else "",
        }

    def get_all_rules(self) -> dict:
        return self._load_rules()

    # ──────────────────────────────────────────────────────────
    # WRITE (only called after human approval)
    # ──────────────────────────────────────────────────────────

    def apply_rule(
        self,
        pattern: str,
        condition: str,
        new_confidence: int,
        lesson: str = "",
        approved_by: str = "human",
    ) -> dict:
        """
        Approved rule change apply করো।
        Direct call করো না — DeepMistakeAnalyzer.approve_rule_change() দিয়ে।
        """
        rules = self._load_rules()
        key = self._key(pattern, condition)

        old_conf = rules.get(key, {}).get("confidence", self.DEFAULT_CONFIDENCE)
        new_conf = max(10, min(95, new_confidence))  # 10-95 range clamp

        rule = {
            "pattern":      pattern,
            "condition":    condition,
            "confidence":   new_conf,
            "lesson":       lesson,
            "approved_by":  approved_by,
            "updated_at":   datetime.now(timezone.utc).isoformat(),
            "version":      rules.get(key, {}).get("version", 0) + 1,
        }

        rules[key] = rule
        self._save_rules(rules)

        # History track
        self._add_to_history(key, old_conf, new_conf, lesson)

        log.info(f"[RuleUpdater] Rule updated: {key} | {old_conf}% → {new_conf}% (Δ{new_conf - old_conf:+d}%)")
        return {"success": True, "key": key, "old": old_conf, "new": new_conf, "rule": rule}

    def reset_rule(self, pattern: str, condition: str) -> dict:
        """Rule সরিয়ে default-এ ফিরে যাও।"""
        rules = self._load_rules()
        key = self._key(pattern, condition)

        if key in rules:
            old = rules.pop(key)
            self._save_rules(rules)
            log.info(f"[RuleUpdater] Rule RESET: {key}")
            return {"success": True, "removed": old}
        return {"success": False, "reason": "Rule not found"}

    # ──────────────────────────────────────────────────────────
    # PRINT
    # ──────────────────────────────────────────────────────────

    def print_all_rules(self) -> None:
        rules = self._load_rules()
        bar = "═" * 60
        print(f"\n{bar}")
        print("  📏  ACTIVE PATTERN RULES  (Day 52)")
        print(bar)
        print(f"  Default confidence (no rule): {self.DEFAULT_CONFIDENCE}%")
        print()

        if not rules:
            print("  No rules yet — all patterns using default confidence")
        else:
            for key, rule in rules.items():
                delta = rule["confidence"] - self.DEFAULT_CONFIDENCE
                arrow = "⬇️" if delta < 0 else "⬆️"
                print(f"  {arrow}  {key:<40} {rule['confidence']}% ({delta:+d}%)")
                print(f"      Lesson: {rule.get('lesson', 'N/A')[:70]}")
                print(f"      Updated: {rule.get('updated_at', '')[:19]}")
                print()
        print(bar + "\n")

    # ──────────────────────────────────────────────────────────
    # STORAGE
    # ──────────────────────────────────────────────────────────

    def _key(self, pattern: str, condition: str) -> str:
        return f"{pattern}_{condition}".replace(" ", "_")

    def _load_rules(self) -> dict:
        if not os.path.exists(RULE_STORE_PATH):
            return {}
        try:
            with open(RULE_STORE_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_rules(self, rules: dict) -> None:
        with open(RULE_STORE_PATH, "w", encoding="utf-8") as f:
            json.dump(rules, f, indent=2, default=str)

    def _add_to_history(self, key: str, old: int, new: int, lesson: str) -> None:
        history = []
        if os.path.exists(RULE_HISTORY_PATH):
            try:
                with open(RULE_HISTORY_PATH, encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                history = []

        history.append({
            "key":       key,
            "old":       old,
            "new":       new,
            "change":    new - old,
            "lesson":    lesson,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        with open(RULE_HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history[-500:], f, indent=2, default=str)