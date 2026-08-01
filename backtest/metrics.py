# backtest/metrics.py — Performance Analysis & Reporting
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime, timezone
from collections import defaultdict
import numpy as np, pandas as pd
log = logging.getLogger(__name__)
@dataclass
class BacktestMetrics:
    total_trades: int = 0; winning_trades: int = 0; losing_trades: int = 0
    win_rate: float = 0.0; profit_factor: float = 0.0; total_pnl_usd: float = 0.0
    total_pnl_pips: float = 0.0; avg_win_pips: float = 0.0; avg_loss_pips: float = 0.0
    avg_win_usd: float = 0.0; avg_loss_usd: float = 0.0; expectancy_r: float = 0.0
    avg_rr: float = 0.0; max_drawdown_pct: float = 0.0; max_drawdown_usd: float = 0.0
    sharpe_ratio: float = 0.0; sortino_ratio: float = 0.0; calmar_ratio: float = 0.0
    avg_hold_bars: int = 0; starting_balance: float = 0.0; ending_balance: float = 0.0
    total_return_pct: float = 0.0; pair_breakdown: Dict = None; session_breakdown: Dict = None
    strategy_breakdown: Dict = None; monthly_returns: Dict = None
    def to_dict(self): return {k: v for k, v in self.__dict__.items()}
    def to_table(self):
        lines = ["="*50, "  BACKTEST RESULTS", "="*50,
            f"  Total Trades     : {self.total_trades}",
            f"  Win Rate         : {self.win_rate:.1f}%",
            f"  Profit Factor    : {self.profit_factor:.2f}",
            f"  Total P&L        : ${self.total_pnl_usd:.2f} ({self.total_pnl_pips:.1f} pips)",
            f"  Avg Win          : {self.avg_win_pips:.1f}p (${self.avg_win_usd:.2f})",
            f"  Avg Loss         : {self.avg_loss_pips:.1f}p (${self.avg_loss_usd:.2f})",
            f"  Expectancy       : {self.expectancy_r:.2f}R",
            f"  Avg R:R          : 1:{self.avg_rr:.1f}",
            f"  Max Drawdown     : {self.max_drawdown_pct:.1f}% (${self.max_drawdown_usd:.2f})",
            f"  Sharpe Ratio     : {self.sharpe_ratio:.2f}",
            f"  Sortino Ratio    : {self.sortino_ratio:.2f}",
            f"  Calmar Ratio     : {self.calmar_ratio:.2f}",
            f"  Starting Balance : ${self.starting_balance:.2f}",
            f"  Ending Balance   : ${self.ending_balance:.2f}",
            f"  Total Return     : {self.total_return_pct:.1f}%", "="*50]
        if self.pair_breakdown:
            lines.append("\n  --- Pair Breakdown ---")
            for p, s in sorted(self.pair_breakdown.items(), key=lambda x: x[1].get("win_rate", 0), reverse=True):
                lines.append(f"  {p:10s} : {s['trades']:3d} trades | WR={s['win_rate']:.1f}% | P&L={s['pnl_pips']:.0f}p | PF={s['profit_factor']:.2f}")
        if self.session_breakdown:
            lines.append("\n  --- Session Breakdown ---")
            for s, st in sorted(self.session_breakdown.items(), key=lambda x: x[1].get("win_rate", 0), reverse=True):
                lines.append(f"  {s:10s} : {st['trades']:3d} trades | WR={st['win_rate']:.1f}% | P&L={st['pnl_pips']:.0f}p")
        if self.strategy_breakdown:
            lines.append("\n  --- Strategy Breakdown ---")
            for s, st in sorted(self.strategy_breakdown.items(), key=lambda x: x[1].get("win_rate", 0), reverse=True):
                lines.append(f"  {s:15s} : {st['trades']:3d} trades | WR={st['win_rate']:.1f}% | P&L={st['pnl_pips']:.0f}p")
        if self.monthly_returns:
            lines.append("\n  --- Monthly Returns ---")
            for m, r in sorted(self.monthly_returns.items()):
                lines.append(f"  {m} : {'🟢' if r>=0 else '🔴'} {r:+.2f}%")
        lines.append("="*50); return "\n".join(lines)

def calculate_metrics(trades, starting_balance=10000.0, ending_balance=None, risk_free_rate=0.02, timeframe=None):
    if not trades: return BacktestMetrics(starting_balance=starting_balance, ending_balance=starting_balance)
    m = BacktestMetrics(); m.total_trades = len(trades); m.starting_balance = starting_balance
    if ending_balance is None: ending_balance = starting_balance + sum(t.pnl_usd for t in trades)
    m.ending_balance = ending_balance
    wins = [t for t in trades if t.pnl_usd > 0]; losses = [t for t in trades if t.pnl_usd < 0]
    m.winning_trades = len(wins); m.losing_trades = len(losses)
    m.win_rate = len(wins) / len(trades) * 100 if trades else 0
    m.total_pnl_usd = round(sum(t.pnl_usd for t in trades), 2)
    m.total_pnl_pips = round(sum(t.pnl_pips for t in trades), 1)
    m.avg_win_pips = round(np.mean([t.pnl_pips for t in wins]), 1) if wins else 0
    m.avg_loss_pips = round(np.mean([t.pnl_pips for t in losses]), 1) if losses else 0
    m.avg_win_usd = round(np.mean([t.pnl_usd for t in wins]), 2) if wins else 0
    m.avg_loss_usd = round(np.mean([t.pnl_usd for t in losses]), 2) if losses else 0
    gp = sum(t.pnl_usd for t in wins); gl = abs(sum(t.pnl_usd for t in losses))
    m.profit_factor = round(gp / gl, 2) if gl > 0 else 0
    if wins and losses:
        aw = np.mean([abs(t.pnl_pips) for t in wins]); al = np.mean([abs(t.pnl_pips) for t in losses])
        if al > 0: m.expectancy_r = round((len(wins)/len(trades) * aw - len(losses)/len(trades) * al) / al, 2)
        m.avg_rr = round(abs(m.avg_win_pips / m.avg_loss_pips), 1) if m.avg_loss_pips != 0 else 0
    eq = [starting_balance]
    for t in trades: eq.append(eq[-1] + t.pnl_usd)
    eq = np.array(eq); rm = np.maximum.accumulate(eq); rm_safe = rm.copy(); rm_safe[rm_safe == 0] = np.nan; dd = (rm_safe - eq) / rm_safe * 100
    m.max_drawdown_pct = round(float(np.max(dd)), 2); m.max_drawdown_usd = round(float(np.max(rm - eq)), 2)
    if len(trades) > 1:
        BARS_PER_DAY = {"M1": 960, "M5": 288, "M15": 96, "H1": 24, "H4": 6, "D1": 1}
        bars_per_day = BARS_PER_DAY.get(timeframe or "M15", 96)
        tpy = 252 * bars_per_day
        rets = np.array([t.pnl_usd / starting_balance for t in trades])
        ar = np.mean(rets); sr = np.std(rets, ddof=1)
        if sr > 0: m.sharpe_ratio = round((ar * tpy - risk_free_rate) / sr * np.sqrt(tpy), 2)
        ds = rets[rets < 0]
        if len(ds) > 0:
            dstd = np.std(ds, ddof=1)
            if dstd > 0: m.sortino_ratio = round((ar * tpy - risk_free_rate) / dstd * np.sqrt(tpy), 2)
    if m.max_drawdown_pct > 0:
        tr = (ending_balance - starting_balance) / starting_balance * 100
        m.calmar_ratio = round(tr / m.max_drawdown_pct, 2)
    m.total_return_pct = round((ending_balance - starting_balance) / starting_balance * 100, 2)
    m.avg_hold_bars = int(np.mean([t.hold_bars for t in trades])) if trades else 0
    # Breakdowns
    pd_data = defaultdict(list)
    for t in trades: pd_data[t.symbol].append(t)
    m.pair_breakdown = {}
    for p, pt in pd_data.items():
        pw = [t for t in pt if t.pnl_usd > 0]; pl = [t for t in pt if t.pnl_usd <= 0]
        pgp = sum(t.pnl_usd for t in pw); pgl = abs(sum(t.pnl_usd for t in pl))
        m.pair_breakdown[p] = {"trades": len(pt), "win_rate": round(len(pw)/len(pt)*100, 1) if pt else 0, "pnl_pips": round(sum(t.pnl_pips for t in pt), 1), "pnl_usd": round(sum(t.pnl_usd for t in pt), 2), "profit_factor": round(pgp/pgl, 2) if pgl > 0 else 0}
    sd = defaultdict(list)
    for t in trades:
        try:
            et = datetime.fromisoformat(t.entry_time); h = et.hour
            s = "London_NY_Overlap" if 13 <= h < 16 else "London" if 7 <= h < 16 else "NewYork" if 13 <= h < 22 else "Asian"
            sd[s].append(t)
        except Exception as e: sd["Unknown"].append(t)
    m.session_breakdown = {}
    for s, st in sd.items():
        sw = [t for t in st if t.pnl_usd > 0]
        m.session_breakdown[s] = {"trades": len(st), "win_rate": round(len(sw)/len(st)*100, 1) if st else 0, "pnl_pips": round(sum(t.pnl_pips for t in st), 1), "pnl_usd": round(sum(t.pnl_usd for t in st), 2)}
    std = defaultdict(list)
    for t in trades: std[t.strategy or "unknown"].append(t)
    m.strategy_breakdown = {}
    for s, st in std.items():
        sw = [t for t in st if t.pnl_usd > 0]; sl = [t for t in st if t.pnl_usd <= 0]
        sgp = sum(t.pnl_usd for t in sw); sgl = abs(sum(t.pnl_usd for t in sl))
        m.strategy_breakdown[s] = {"trades": len(st), "win_rate": round(len(sw)/len(st)*100, 1) if st else 0, "pnl_pips": round(sum(t.pnl_pips for t in st), 1), "pnl_usd": round(sum(t.pnl_usd for t in st), 2), "profit_factor": round(sgp/sgl, 2) if sgl > 0 else 0}
    md = defaultdict(float)
    for t in trades:
        try:
            mk = datetime.fromisoformat(t.entry_time).strftime("%Y-%m"); md[mk] += t.pnl_usd
        except Exception as e: pass
    m.monthly_returns = {mk: round(p / starting_balance * 100, 2) for mk, p in md.items()}
    return m
