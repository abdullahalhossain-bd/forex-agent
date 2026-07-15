# memory/pattern_memory.py  —  Day 16 | Structured Pattern Memory

"""
Vector search similarity খুঁজবে,
কিন্তু trading rules-এর জন্য structured JSON বেশি reliable।

3 টি JSON file manage করে:
- winning_patterns.json
- losing_patterns.json
- lessons.json
"""

import json
from pathlib import Path
from datetime import datetime


MEMORY_DIR = Path("memory")


class PatternMemory:
    """
    Structured pattern memory।
    Fast, reliable, no ML dependency।
    """

    def __init__(self):
        MEMORY_DIR.mkdir(exist_ok=True)
        self._winning  = self._load("winning_patterns.json")
        self._losing   = self._load("losing_patterns.json")
        self._lessons  = self._load("lessons.json")

    def _load(self, filename: str) -> list:
        path = MEMORY_DIR / filename
        if path.exists():
            return json.loads(path.read_text())
        return []

    def _save(self, filename: str, data: list):
        path = MEMORY_DIR / filename
        path.write_text(json.dumps(data, indent=2))

    # ── Win / Loss Patterns ────────────────────────────────────

    def add_winning_pattern(self, pattern: dict):
        """
        Winning trade-এর pattern save করো।

        pattern = {
            "pair":      "EURUSD",
            "timeframe": "15m",
            "setup":     "Hammer at daily support + RSI oversold",
            "regime":    "TRENDING_BULLISH",
            "rsi":       35,
            "pnl":       50,
            "rr":        2.1,
        }
        """
        pattern["date"] = datetime.now().isoformat()
        self._winning.append(pattern)
        self._save("winning_patterns.json", self._winning)
        print(f"✅ Winning pattern saved: {pattern.get('setup', '')[:50]}")

    def add_losing_pattern(self, pattern: dict, lesson: str):
        """Losing trade-এর pattern + lesson save করো।"""
        pattern["date"]   = datetime.now().isoformat()
        pattern["lesson"] = lesson
        self._losing.append(pattern)
        self._save("losing_patterns.json", self._losing)
        print(f"📝 Losing pattern saved: {lesson[:50]}")

    def add_lesson(self, lesson: str, category: str = "general", pair: str = ""):
        """AI-এর শেখা lesson save করো।"""
        entry = {
            "lesson":   lesson,
            "category": category,
            "pair":     pair,
            "date":     datetime.now().isoformat(),
        }
        self._lessons.append(entry)
        self._save("lessons.json", self._lessons)

    # ── Query ─────────────────────────────────────────────────

    def find_similar_winning(self, pair: str, regime: str, pattern: str) -> list:
        """একই setup-এ আগে জেতা trades খুঁজো।"""
        results = []
        for p in self._winning:
            score = 0
            if p.get("pair")    == pair:    score += 2
            if p.get("regime")  == regime:  score += 2
            if p.get("pattern") == pattern: score += 3
            if score >= 2:
                results.append({**p, "match_score": score})
        return sorted(results, key=lambda x: x["match_score"], reverse=True)[:3]

    def find_similar_losing(self, pair: str, regime: str) -> list:
        """একই condition-এ আগে হারা trades খুঁজো।"""
        results = []
        for p in self._losing:
            if p.get("pair") == pair or p.get("regime") == regime:
                results.append(p)
        return results[:3]

    def get_lessons_for_pair(self, pair: str) -> list:
        """Specific pair-এর জন্য lessons।"""
        return [
            l for l in self._lessons
            if l.get("pair", "") in ("", pair)
        ]

    def get_win_rate_by_pattern(self) -> dict:
        """Pattern-wise win rate calculate করো।"""
        stats = {}

        for p in self._winning:
            pat = p.get("pattern", "unknown")
            stats.setdefault(pat, {"wins": 0, "losses": 0})
            stats[pat]["wins"] += 1

        for p in self._losing:
            pat = p.get("pattern", "unknown")
            stats.setdefault(pat, {"wins": 0, "losses": 0})
            stats[pat]["losses"] += 1

        result = {}
        for pat, data in stats.items():
            total = data["wins"] + data["losses"]
            result[pat] = {
                "wins":     data["wins"],
                "losses":   data["losses"],
                "total":    total,
                "win_rate": round(data["wins"] / total * 100, 1) if total else 0,
            }
        return result

    # ── Summary ───────────────────────────────────────────────

    def get_summary_for_decision(self, pair: str, regime: str, pattern: str) -> dict:
        """
        Decision agent-এ inject করার জন্য summary।
        """
        winning = self.find_similar_winning(pair, regime, pattern)
        losing  = self.find_similar_losing(pair, regime)
        lessons = self.get_lessons_for_pair(pair)[:3]
        wr_map  = self.get_win_rate_by_pattern()

        pattern_wr = wr_map.get(pattern, {}).get("win_rate", None)

        return {
            "similar_wins":    len(winning),
            "similar_losses":  len(losing),
            "pattern_win_rate": pattern_wr,
            "top_lessons":     [l["lesson"] for l in lessons],
            "warning":         len(losing) > len(winning),
        }

    def print_stats(self):
        bar = "─" * 48
        print(f"\n{bar}")
        print(f"  📂  PATTERN MEMORY STATS")
        print(bar)
        print(f"  Winning Patterns : {len(self._winning)}")
        print(f"  Losing Patterns  : {len(self._losing)}")
        print(f"  Lessons Learned  : {len(self._lessons)}")
        wr_map = self.get_win_rate_by_pattern()
        if wr_map:
            print(f"\n  Pattern Win Rates:")
            for pat, data in wr_map.items():
                print(f"  {pat:25s}: {data['win_rate']}% ({data['wins']}/{data['total']})")
        print(bar)