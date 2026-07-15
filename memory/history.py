# memory/history.py
# ============================================================
# Analysis History — AI Learning Memory
# প্রতিটা analysis save হবে, পরে AI নিজের ভুল থেকে শিখবে
# ============================================================

import json
import os
from datetime import datetime
from utils.logger import get_logger

log      = get_logger(__name__)
MEM_DIR  = "memory"
MEM_FILE = os.path.join(MEM_DIR, "analysis_history.json")

os.makedirs(MEM_DIR, exist_ok=True)


class AnalysisHistory:
    """
    প্রতিটা analysis run save করো।
    Structure:
    [
      {
        "time":           "2026-06-19T10:30:00",
        "pair":           "EUR/USDT",
        "timeframe":      "15m",
        "price":          1.14730,
        "bias":           "STRONG_SELL",
        "confidence":     62,
        "recommendation": "🔴 SELL BIAS ...",
        "has_conflict":   true,
        "result":         null    ← পরে trade result update হবে
      }
    ]
    """

    def save(self, symbol: str, timeframe: str, bias_ctx: dict, ind_ctx: dict):
        """এই run-এর analysis save করো"""
        record = {
            "time":           datetime.now().isoformat(timespec='seconds'),
            "pair":           symbol,
            "timeframe":      timeframe,
            "price":          ind_ctx.get('price'),
            "trend":          ind_ctx.get('trend'),
            "rsi":            ind_ctx.get('rsi'),
            "bias":           bias_ctx.get('bias'),
            "confidence":     bias_ctx.get('confidence_pct'),
            "recommendation": bias_ctx.get('recommendation'),
            "has_conflict":   bias_ctx.get('has_conflict', False),
            "result":         None,   # trade outcome — পরে update হবে
        }

        history = self._load()
        history.append(record)
        self._save(history)

        log.info(f"History saved: {symbol} {bias_ctx.get('bias')} "
                 f"{bias_ctx.get('confidence_pct')}%")
        return record

    def update_result(self, index: int, result: str, pnl: float = None):
        """
        Trade outcome update করো।
        result: 'win' | 'loss' | 'breakeven'
        """
        history = self._load()
        if 0 <= index < len(history):
            history[index]['result'] = result
            history[index]['pnl']    = pnl
            self._save(history)
            log.info(f"Result updated: index={index} result={result} pnl={pnl}")

    def get_recent(self, n: int = 10) -> list:
        """সর্বশেষ N analysis দেখো"""
        return self._load()[-n:]

    def get_stats(self) -> dict:
        """Win rate, bias accuracy stats"""
        history  = self._load()
        with_result = [h for h in history if h.get('result')]

        if not with_result:
            return {"message": "No completed trades yet"}

        wins   = sum(1 for h in with_result if h['result'] == 'win')
        losses = sum(1 for h in with_result if h['result'] == 'loss')
        total  = len(with_result)

        return {
            "total_analyses": len(history),
            "completed_trades": total,
            "wins":   wins,
            "losses": losses,
            "win_rate_%": round(wins / total * 100, 1) if total else 0,
        }

    def print_recent(self, n: int = 5):
        recent = self.get_recent(n)
        print("\n" + "═" * 52)
        print(f"  📚  ANALYSIS HISTORY  (last {n})")
        print("═" * 52)
        if not recent:
            print("  No history yet.")
        for r in recent:
            conflict = "⚠️" if r.get('has_conflict') else "  "
            result   = r.get('result') or '—'
            print(f"  {r['time'][11:19]}  {r['pair']:<12} "
                  f"{r['bias']:<14} {r['confidence']}%  "
                  f"{conflict}  result: {result}")
        print("═" * 52 + "\n")

    def _load(self) -> list:
        if not os.path.exists(MEM_FILE):
            return []
        try:
            with open(MEM_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []

    def _save(self, history: list):
        with open(MEM_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=True)