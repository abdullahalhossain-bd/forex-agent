from dataclasses import dataclass, asdict

from utils.logger import get_logger

# Day 99+ V5 FIX (Audit Issue #1): import the proper pip-value lookup
# from core/constants.py instead of hard-coding "9.0 if JPY else 10.0".
# The hard-coded heuristic was 10× too large for metals — XAUUSD's real
# pip value is $1.00/pip (pip=$0.01, lot=100oz → $1/pip), not $10.00 —
# causing the simulator to compute lot sizes 10× too large for gold,
# over-risking every XAUUSD/XAGUSD backtest by an order of magnitude.
# The shared get_pip_value_usd() in core/constants.py already has the
# correct per-symbol values (verified for FX majors, JPY crosses, and
# metals). Falling back to 10.0 if the import fails preserves the old
# behavior for environments without core/constants.py.
try:
    from core.constants import get_pip_value_usd as _get_pip_value_usd
    _PIP_VALUE_LOOKUP_AVAILABLE = True
except Exception:
    _PIP_VALUE_LOOKUP_AVAILABLE = False
    _get_pip_value_usd = None

log = get_logger("backtest_simulator")

PIP_SIZE = {
    # ── Majors ──
    "EURUSD": 0.0001,
    "GBPUSD": 0.0001,
    "USDJPY": 0.01,
    "USDCHF": 0.0001,
    "AUDUSD": 0.0001,
    "USDCAD": 0.0001,
    "NZDUSD": 0.0001,
    # ── JPY crosses (ALL must be 0.01, not 0.0001!) ──
    # Round-20 audit fix: previously only USDJPY was listed — ALL other
    # JPY crosses (EURJPY, GBPJPY, AUDJPY, etc.) fell through to DEFAULT
    # (0.0001), making their SL/TP 100× too tight. This silently corrupted
    # backtest results for any JPY cross pair.
    "EURJPY": 0.01,
    "GBPJPY": 0.01,
    "AUDJPY": 0.01,
    "NZDJPY": 0.01,
    "CADJPY": 0.01,
    "CHFJPY": 0.01,
    # ── Other crosses ──
    "EURGBP": 0.0001,
    "EURCHF": 0.0001,
    "EURAUD": 0.0001,
    "EURCAD": 0.0001,
    "EURNZD": 0.0001,
    "GBPCHF": 0.0001,
    "GBPAUD": 0.0001,
    "GBPCAD": 0.0001,
    "GBPNZD": 0.0001,
    "AUDCHF": 0.0001,
    "AUDCAD": 0.0001,
    "AUDNZD": 0.0001,
    "NZDCHF": 0.0001,
    "NZDCAD": 0.0001,
    "CADCHF": 0.0001,
    # ── Metals ──
    "XAUUSD": 0.01,    # Gold: 0.01 per oz (not 0.0001!)
    "XAGUSD": 0.001,   # Silver: 0.001 per oz
    "XPTUSD": 0.01,    # Platinum
    "XPDUSD": 0.01,    # Palladium
    # ── Scandinavian crosses ──
    "EURNOK": 0.0001,
    "EURSEK": 0.0001,
    "GBPSEK": 0.0001,
    "GBPNOK": 0.0001,
    # ── Asia Pacific ──
    "USDCNH": 0.0001,
    "USDHKD": 0.0001,
    "USDSGD": 0.0001,
    "USDMXN": 0.0001,
    "USDTHB": 0.0001,
    "USDSAR": 0.0001,
    "USDAED": 0.0001,
    "USDTRY": 0.0001,
    "USDZAR": 0.0001,
    "AUDSGD": 0.0001,
    "NZDSGD": 0.0001,
    "CADHKD": 0.0001,
    "SGDJPY": 0.01,    # JPY cross → 0.01
    "HKDJPY": 0.01,    # JPY cross → 0.01
    "MXNJPY": 0.01,    # JPY cross → 0.01
    # ── Default ──
    # Round-20: smart default — if symbol contains "JPY", use 0.01;
    # otherwise 0.0001. This catches any pair not explicitly listed
    # above (e.g. exotic JPY crosses) without falling back to the
    # wrong 0.0001.
    "DEFAULT": 0.0001,
}


def _get_pip_size(symbol: str) -> float:
    """Get pip size for a symbol, with smart JPY detection.

    Round-20 audit fix: the old PIP_SIZE.get(symbol, 0.0001) fallback
    was wrong for ANY JPY cross not explicitly listed in the dict.
    EURJPY, GBPJPY, AUDJPY etc. all got 0.0001 (100× too small),
    silently corrupting SL/TP distances in backtests.

    Now: if the symbol isn't in the dict, check if it contains "JPY"
    → use 0.01. Otherwise 0.0001. This is the same heuristic the pip
    VALUE calculation already uses (line 79: `9.0 if "JPY" in symbol`),
    so size and value are now consistent.
    """
    size = PIP_SIZE.get(symbol)
    if size is not None:
        return size
    # Smart default: JPY pairs use 0.01, everything else 0.0001
    if "JPY" in symbol.upper():
        return 0.01
    return PIP_SIZE["DEFAULT"]


def _get_pip_value(symbol: str) -> float:
    """Get per-standard-lot pip value in USD for a symbol.

    Day 99+ V5 FIX (Audit Issue #1 — Critical): the OLD code used a
    hard-coded `9.0 if "JPY" in symbol else 10.0` heuristic, which is
    WRONG for metals:
        XAUUSD: real pip value = $1.00 (pip=$0.01, lot=100oz → $1/pip)
                old heuristic returned $10.00 → 10× over-risk
        XAGUSD: real pip value = $5.00 (pip=$0.001, lot=5000oz → $5/pip)
                old heuristic returned $10.00 → 2× over-risk
    This caused lot sizes for XAUUSD/XAGUSD backtests to be 10× / 2×
    too large, silently corrupting every metals backtest.

    Now delegates to core.constants.get_pip_value_usd() which has the
    correct per-symbol values for FX majors, JPY crosses, metals, and
    indices. Falls back to the old heuristic only if the import failed
    (preserves backward compatibility for environments without
    core/constants.py — but logs a warning so the operator knows
    metals backtests will be over-risked).
    """
    if _PIP_VALUE_LOOKUP_AVAILABLE and _get_pip_value_usd is not None:
        try:
            return float(_get_pip_value_usd(symbol))
        except Exception:
            pass
    # Fallback: old heuristic. Log once so the operator knows metals
    # backtests will be over-risked.
    if not getattr(_get_pip_value, "_warned", False):
        log.warning(
            "[Simulator] core.constants.get_pip_value_usd unavailable — "
            "falling back to 9.0/10.0 heuristic. METALS BACKTESTS WILL "
            "BE OVER-RISKED (XAUUSD 10×, XAGUSD 2×). Install/fix "
            "core/constants.py to resolve."
        )
        _get_pip_value._warned = True
    return 9.0 if "JPY" in symbol.upper() else 10.0


SPREAD_PIPS = {
    "EURUSD": 1.2,
    "GBPUSD": 1.5,
    "USDJPY": 1.3,
    "USDCHF": 1.8,
    "AUDUSD": 1.4,
    "USDCAD": 1.7,
    "DEFAULT": 1.5,
}


@dataclass
class TradePosition:
    pair: str
    strategy: str
    strategy_version: str
    direction: str
    entry_time: str
    entry_index: int
    entry_requested: float
    entry_price: float
    sl: float
    tp: float
    lot: float
    confidence: int
    rr_ratio: float
    risk_usd: float
    stop_pips: float
    slippage_pips: float
    spread_pips: float
    commission_per_lot: float
    reason: str
    pattern: str
    regime: str
    session: str
    timeout_candles: int


class ForexSimulator:
    def __init__(
        self,
        commission_per_lot: float = 7.0,
        max_slippage_pips: float = 0.8,
        default_timeout_candles: int = 192,
    ):
        self.commission_per_lot = commission_per_lot
        self.max_slippage_pips = max_slippage_pips
        self.default_timeout_candles = default_timeout_candles

    def open_position(
        self,
        candle,
        signal: dict,
        pair: str,
        balance: float,
        risk_per_trade: float = 0.01,
        candle_index: int = 0,
    ) -> TradePosition:
        symbol = self._clean_pair(pair)
        direction = str(signal["signal"]).upper()
        pip = _get_pip_size(symbol)
        stop_pips = max(float(signal.get("stop_pips", 15)), 1.0)
        rr_ratio = max(float(signal.get("rr_ratio", 2.0)), 1.0)
        risk_usd = round(balance * risk_per_trade, 2)
        # Day 99+ V5 FIX (Audit Issue #1): use proper per-symbol pip
        # value lookup instead of the hard-coded JPY/10.0 heuristic.
        # This fixes XAUUSD/XAGUSD being 10×/2× over-risked in backtests.
        pip_value = _get_pip_value(symbol)
        lot = round(max(0.01, min(risk_usd / (stop_pips * pip_value), 100.0)), 2)

        requested = float(candle["open"])
        spread_pips = float(signal.get("spread_pips", SPREAD_PIPS.get(symbol, SPREAD_PIPS["DEFAULT"])))
        slippage_pips = min(self.max_slippage_pips, self._deterministic_slippage(candle, pip))

        fill_adjustment = (spread_pips / 2 + slippage_pips) * pip
        entry_price = requested + fill_adjustment if direction == "BUY" else requested - fill_adjustment

        stop_distance = stop_pips * pip
        tp_distance = stop_distance * rr_ratio
        sl = entry_price - stop_distance if direction == "BUY" else entry_price + stop_distance
        tp = entry_price + tp_distance if direction == "BUY" else entry_price - tp_distance

        position = TradePosition(
            pair=symbol,
            strategy=signal.get("strategy_name", "Unknown"),
            strategy_version=signal.get("strategy_version", "v1"),
            direction=direction,
            entry_time=str(getattr(candle, "name", candle_index)),
            entry_index=candle_index,
            entry_requested=round(requested, 5),
            entry_price=round(entry_price, 5),
            sl=round(sl, 5),
            tp=round(tp, 5),
            lot=lot,
            confidence=int(round(signal.get("confidence", 0))),
            rr_ratio=round(rr_ratio, 2),
            risk_usd=risk_usd,
            stop_pips=round(stop_pips, 1),
            slippage_pips=round(slippage_pips, 2),
            spread_pips=round(spread_pips, 2),
            commission_per_lot=self.commission_per_lot,
            reason=signal.get("reason", ""),
            pattern=signal.get("pattern", "none"),
            regime=signal.get("regime", "unknown"),
            session=signal.get("session", "unknown"),
            timeout_candles=int(signal.get("timeout_candles", self.default_timeout_candles)),
        )
        log.info(
            f"[Simulator] OPEN {position.strategy} | {position.direction} {position.pair} "
            f"@ {position.entry_price} | SL {position.sl} | TP {position.tp} | Lot {position.lot}"
        )
        return position

    def evaluate_exit(self, position: TradePosition, candle, candle_index: int) -> dict | None:
        high = float(candle["high"])
        low = float(candle["low"])

        if candle_index - position.entry_index >= position.timeout_candles:
            return self._close_trade(position, float(candle["close"]), "TIMEOUT", candle, candle_index)

        if position.direction == "BUY":
            sl_hit = low <= position.sl
            tp_hit = high >= position.tp
        else:
            sl_hit = high >= position.sl
            tp_hit = low <= position.tp

        if sl_hit:
            return self._close_trade(position, position.sl, "SL HIT", candle, candle_index)
        if tp_hit:
            return self._close_trade(position, position.tp, "TP HIT", candle, candle_index)
        return None

    def force_close(self, position: TradePosition, candle, candle_index: int, reason: str = "END OF DATA") -> dict:
        return self._close_trade(position, float(candle["close"]), reason, candle, candle_index)

    def _close_trade(self, position: TradePosition, raw_exit: float, reason: str, candle, candle_index: int) -> dict:
        pip = _get_pip_size(position.pair)
        spread_exit_adjustment = (position.spread_pips / 2) * pip
        exit_price = raw_exit - spread_exit_adjustment if position.direction == "BUY" else raw_exit + spread_exit_adjustment

        if position.direction == "BUY":
            pnl_pips = (exit_price - position.entry_price) / pip
        else:
            pnl_pips = (position.entry_price - exit_price) / pip

        # Day 99+ V5 FIX (Audit Issue #1): use proper per-symbol pip
        # value lookup. The old `9.0 if JPY else 10.0` heuristic was
        # wrong for metals (XAUUSD returned $10/pip instead of $1/pip),
        # inflating PnL by 10× for every gold trade in the backtest.
        pip_value = _get_pip_value(position.pair)
        gross_pnl = pnl_pips * pip_value * position.lot
        commission = round(position.commission_per_lot * position.lot, 2)
        net_pnl = round(gross_pnl - commission, 2)

        return {
            **asdict(position),
            "exit_price": round(exit_price, 5),
            "exit_time": str(getattr(candle, "name", candle_index)),
            "exit_index": candle_index,
            "close_reason": reason,
            "result": "WIN" if net_pnl > 0 else ("LOSS" if net_pnl < 0 else "BREAKEVEN"),
            "pnl": net_pnl,
            "pnl_pips": round(pnl_pips, 1),
            "commission": commission,
            "spread_cost": round(position.spread_pips * pip_value * position.lot, 2),
            "bars_held": candle_index - position.entry_index,
        }

    def _deterministic_slippage(self, candle, pip: float) -> float:
        candle_range_pips = abs(float(candle["high"]) - float(candle["low"])) / pip if pip else 0
        return round(min(self.max_slippage_pips, max(0.05, candle_range_pips * 0.02)), 2)

    def _clean_pair(self, pair: str) -> str:
        # Round-14 fix: str.replace("USDT","USD") matched "USDT" ANYWHERE
        # in the string, not just as a trailing Tether-quote suffix — it
        # silently corrupted real forex codes containing "USDT" as a
        # substring: USDTRY (USD/Turkish Lira) -> USDRY, USDTHB
        # (USD/Thai Baht) -> USDHB. Only strip the trailing "T" when
        # USDT is genuinely a Tether suffix (e.g. BTCUSDT -> BTCUSD).
        cleaned = str(pair).upper().replace("/", "").replace("=X", "").strip()
        if cleaned.endswith("USDT"):
            cleaned = cleaned[:-1]
        return cleaned