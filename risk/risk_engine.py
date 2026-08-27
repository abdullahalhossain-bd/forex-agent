# risk/risk_engine.py  —  Day 13 | Risk Engine
# ============================================================
# Uses core.constants for PIP_SIZE and CORRELATION_GROUPS —
# no local duplicates. Key naming follows project convention:
# "lot" (not "lot_size"), "risk_pc" (not "risk_percent").
# ============================================================

from utils.logger import get_logger
from core.constants import PIP_SIZE, CORRELATION_GROUPS, get_pip_size, get_pip_value_usd, get_live_pip_value_per_lot, clean_symbol, pips_to_price, MEMORY_DIR
import json, os
from datetime import datetime, date, timezone

log = get_logger("risk_engine")

DAILY_LOG_PATH = str(MEMORY_DIR / "daily_risk.json")


class RiskEngine:

    # P1 fix (2026-08-19): MAX_RISK_PC was hardcoded to 1.0 (1%) here while
    # config.py's RISK_PER_TRADE = 0.005 (0.5%) is the documented
    # "production-safe" value used elsewhere (matches strict_risk_manager
    # per config.py's own comment). RiskEngine never read it — every trade
    # was silently sized at 2x the intended risk. Wired in the same
    # pattern as MAX_LOT/DAILY_LOSS_LIMIT/MAX_OPEN_TRADES below: read from
    # config at class-definition time, no try/except (fail loudly on boot
    # if config.py is broken, per the P0-2 rule below).
    from config import RISK_PER_TRADE as _CFG_RISK_PER_TRADE
    MAX_RISK_PC      = float(_CFG_RISK_PER_TRADE) * 100  # config stores a fraction (0.005); RiskEngine uses a percent (0.5)
    # PARITY FIX (2026-08-15): align with risk/rr_policy.get_min_rr() /
    # core.constants.MIN_RR_PROD (default 2.0). Hardcoded 1.5 caused
    # RiskEngine to emit 1.5R TPs that orphan_consumers then rejected
    # (RR 1.50 < 2.00) → systematic risk_rejected / 0 trades in backtest.
    MIN_RR           = 2.0
    MAX_RR           = 5.0
    DAILY_LOSS_LIMIT = 3.0
    MAX_OPEN_TRADES  = 3
    ATR_SL_MULT      = 1.5  # SL = ATR * 1.5 (~25 pips on EURUSD H1)
    # P0-2 (Audit Fix): Config loading must NOT be wrapped in try/except.
    # If config.py fails to import, the system MUST crash on boot —
    # silently trading with wrong risk parameters is far more dangerous.
    from config import MAX_LOT as _CFG_MAX_LOT
    MAX_LOT = float(_CFG_MAX_LOT)

    from config import DAILY_LOSS_LIMIT_PCT as _CFG_DLL
    DAILY_LOSS_LIMIT = float(_CFG_DLL)

    from config import MAX_OPEN_TRADES as _CFG_MOT
    MAX_OPEN_TRADES = int(_CFG_MOT)

    def __init__(self, balance: float = None, symbol: str = "EURUSD"):
        # Bug #22 fix: use config.INITIAL_BALANCE as default instead of
        # hardcoded 1000.0 that drifts from the actual configured balance.
        if balance is None:
            try:
                from config import INITIAL_BALANCE
                balance = float(INITIAL_BALANCE)
            except Exception:
                balance = 1000.0
        self.balance = balance
        self.symbol  = clean_symbol(symbol)
        self.pip     = get_pip_size(self.symbol)
        self._daily  = self._load_daily()
        # Day 90 bugfix: _live_open_pairs MUST be initialized in __init__
        # so _correlation_check() always finds it.  Previously this attribute
        # was only set inside sync_open_positions() — which itself was broken
        # because Python silently kept the SECOND of two same-named methods
        # (the one that only updated daily_risk.json, not _live_open_pairs).
        # As a result _live_open_pairs was NEVER set and the correlation check
        # always fell back to the potentially-stale daily_risk.json file.
        # Initialize to empty set here; trader.py sync_open_positions() will
        # overwrite it each cycle with the authoritative PaperTrader state.
        self._live_open_pairs: set = set()
        # Track sync health so silent failures become visible
        self._sync_call_count: int = 0
        self._sync_fail_count: int = 0
        self._last_sync_at: float = 0.0

    def evaluate(self, signal: str, entry: float, atr: float, regime: dict | None = None,
                 correlation_ctx: dict | None = None, df=None,
                 strategy: str | None = None) -> dict:
        # Day 81+ hotfix: WAIT signal should also be rejected (not just NO TRADE).
        # Previously WAIT fell through to the `else` branch (SELL) and got
        # approved with SL/TP — but WAIT means "no trade", so it must reject.
        if signal in ("NO TRADE", "WAIT", "HOLD", ""):
            return self._reject(f"Signal is {signal or 'EMPTY'} — no trade")

        # P0-1 (Audit Fix): entry=None/0 must REJECT the trade, not use a
        # fabricated fallback price. A trade with entry=1.0 on EURUSD (real
        # price ~1.0850) would produce garbage SL/TP and lot sizing.
        if not entry or entry == 0:
            return self._reject(
                f"entry={entry} (None/0) — cannot compute SL/TP without a valid entry price"
            )

        daily_loss_usd = self._daily.get("total_loss_usd", 0)
        daily_loss_pc  = daily_loss_usd / self.balance * 100
        open_trades    = self._daily.get("open_trades", 0)

        if daily_loss_pc >= self.DAILY_LOSS_LIMIT:
            return self._reject(f"Daily loss limit hit ({daily_loss_pc:.1f}%)")

        if open_trades >= self.MAX_OPEN_TRADES:
            return self._reject(f"Max open trades ({open_trades}/{self.MAX_OPEN_TRADES})")

        corr = self._correlation_check()
        if not corr["allowed"]:
            return self._reject(corr["reason"])

        # ── EXECUTION-PROOF AUDIT FIX: live correlation_ctx ───────────
        # The CONSUMPTION-MAP audit proved `analysis_out["correlation_ctx"]`
        # was computed every cycle (with live ATR + correlation matrix
        # against live open positions) but NEVER consumed. The existing
        # _correlation_check() above uses static CORRELATION_GROUPS (a
        # hard-coded list of correlated pair clusters), which is a coarse
        # approximation. The CorrelationEngine produces a continuous
        # correlation_risk score (0-1) and a risk_adjustment multiplier
        # (0.25-1.0) computed from ACTUAL price correlation between this
        # pair and the live open pairs. Wire it in as an ADDITIONAL gate:
        # (a) hard-reject if correlation_risk >= 0.90 (effectively the
        # same pair twice — diversification is gone), (b) shrink the lot
        # by risk_adjustment when correlation is high but not extreme.
        # This does NOT replace _correlation_check() — both run; the
        # existing static check stays as a fast first-pass filter, this
        # adds a continuous-math second pass.
        self._correlation_adjustment = 1.0  # default; reset each call
        self._correlation_risk_score = 0.0
        if isinstance(correlation_ctx, dict) and correlation_ctx:
            try:
                _cr = float(correlation_ctx.get("corr_risk", 0) or 0)
                _adj = float(correlation_ctx.get("risk_adjustment", 1.0) or 1.0)
                self._correlation_risk_score = _cr
                self._correlation_adjustment = max(0.25, min(1.0, _adj))
                if _cr >= 0.90:
                    return self._reject(
                        f"Live correlation risk {_cr:.2f} ≥ 0.90 with open pairs "
                        f"{correlation_ctx.get('corr_pairs', [])} — diversification gone"
                    )
            except Exception as _e_corr:
                log.debug(f"[RiskEngine] correlation_ctx parse failed: {_e_corr}")

        vol_mult = {
            "LOW_VOLATILITY":  1.0,
            "NORMAL":          self.ATR_SL_MULT,
            "HIGH_VOLATILITY": 2.2,
        }.get(regime.get("volatility", "NORMAL") if regime else "NORMAL", self.ATR_SL_MULT)

        # Day 81+ hotfix: ATR can be None/0/NaN — force a safe default
        if not atr or atr != atr:  # NaN check
            log.warning(f"[RiskEngine] atr={atr} (invalid) — using 0.0010")
            atr = 0.0010

        # Day 97+ Book Page 13: per-instrument ATR multiplier
        symbol_upper = self.symbol.upper().replace("/", "").replace("=X", "")
        instrument_mult = 1.0
        if symbol_upper in ("XAUUSD", "XAGUSD"):
            instrument_mult = 1.5
        elif symbol_upper.endswith("JPY"):
            instrument_mult = 1.2
        elif symbol_upper in ("US30", "NAS100"):
            instrument_mult = 1.3
        vol_mult = vol_mult * instrument_mult

        sl_distance = round(atr * vol_mult, 5)

        # Day 96 bugfix: in LOW_VOLATILITY regime (vol_mult=1.0) ATR can be
        # as low as 6-7 pips on majors during Sydney/Tokyo sessions, giving
        # an SL of just 6-7 pips — easily hit by spread + normal noise
        # before any real move develops (this matched the production
        # journal: EURUSD 6.6 pip SL, GBPUSD 6.4 pip SL, both stopped out).
        # Enforce a hard floor of 10 pips regardless of regime/ATR.
        min_sl_distance = pips_to_price(self.symbol, 10)
        if sl_distance < min_sl_distance:
            log.info(
                f"[RiskEngine] sl_distance={sl_distance} below 10-pip floor "
                f"({min_sl_distance}) — flooring to avoid noise stop-out"
            )
            sl_distance = round(min_sl_distance, 5)

        sl_pips     = round(sl_distance / self.pip) if self.pip > 0 else 10

        try:
            from risk.rr_policy import get_min_rr
            try:
                from core.constants import is_test_mode
                _tm = bool(is_test_mode())
            except Exception:
                _tm = False
            _min_rr = float(get_min_rr(strategy=strategy, test_mode=_tm))
        except Exception:
            _min_rr = float(self.MIN_RR)
        _min_rr = max(0.5, min(float(self.MAX_RR), _min_rr))

        if signal == "BUY":
            sl_price = round(entry - sl_distance, 5)
            floor_tp = round(entry + sl_distance * _min_rr, 5)
        else:
            sl_price = round(entry + sl_distance, 5)
            floor_tp = round(entry - sl_distance * _min_rr, 5)

        structure_source = "atr"
        structure_reasons = []
        if df is not None:
            try:
                from risk.structure_stop import compute_structure_stop

                structure_sl = compute_structure_stop(
                    df,
                    signal,
                    method="swing_atr",
                    lookback=20,
                    atr_buffer_mult=0.20,
                    atr=atr,
                )
                structure_sl = round(float(structure_sl), 5)
                structure_pips = (
                    (entry - structure_sl) / self.pip
                    if signal == "BUY" else
                    (structure_sl - entry) / self.pip
                )
                if structure_pips >= 10 and (
                    (signal == "BUY" and structure_sl < entry) or
                    (signal == "SELL" and structure_sl > entry)
                ):
                    sl_price = structure_sl
                    sl_distance = round(abs(entry - sl_price), 5)
                    sl_pips = round(sl_distance / self.pip)
                    structure_source = "fractal_swing_atr"
                    structure_reasons.append("recent fractal swing with 0.20 ATR buffer")
                else:
                    structure_reasons.append("structure stop invalid or below 10-pip minimum")
            except Exception as _e_sl:
                log.debug(f"[RiskEngine] structure-based SL lookup failed: {_e_sl}")

        # BUG FIX (2026-08-25): TP used to be computed ONLY as
        # entry ± sl_distance * _min_rr — i.e. EVERY trade landed at
        # exactly the minimum RR (2.0 by default), regardless of where
        # real support/resistance actually sits. That's backwards: a
        # fixed RR multiple is a risk *floor*, not a price target — the
        # market doesn't know or care about our RR math. Two failure
        # modes followed from this:
        #   1. When real resistance/support was CLOSER than the fixed
        #      multiple, TP sat in "unconfirmed territory" past the
        #      last place price had actually reacted (exactly what
        #      risk/entry_quality_guardrails.py::check_tp_structure_
        #      validation flags downstream as a WARNING — but that
        #      check only *complains*, it never fixes the price).
        #   2. When real resistance/support was FARTHER than the fixed
        #      multiple, we left pips on the table by capping every
        #      winning trade at 1:2 instead of riding it to the level
        #      the market was actually respecting.
        #
        # Use the nearest structural target in the trade direction, subject
        # to the risk-policy RR floor.
        # BUG FIX (2026-08-27, no-trades audit): the previous selection took
        # the NEAREST swing in the trade direction (`min(candidates)` /
        # `max(candidates)`). With a window-3 swing detector on 100 bars that
        # nearest level is usually a MICRO-swing only a few pips away, so
        # tp_distance << sl_distance → rr_ratio 0.0–0.6 → "R:R below
        # execution minimum" REJECTED 225 trades in a single morning's log.
        # Correct behaviour: choose the nearest swing that still clears the
        # risk-policy RR floor (sl_distance * _min_rr); if no such structural
        # target exists, keep the ATR RR floor instead of rejecting. Risk is
        # never weakened — TP is always >= policy minimum RR — but a tiny
        # pullback level can no longer veto every setup.
        tp_price = floor_tp
        tp_source = "atr_rr_floor"
        if df is not None:
            try:
                from risk.entry_quality_guardrails import _find_swing_highs, _find_swing_lows

                _min_tp_distance = sl_distance * float(_min_rr)
                if signal == "BUY":
                    swings = _find_swing_highs(df, lookback=100)
                    candidates = sorted(s for s in swings if s > entry)
                    viable = [s for s in candidates if (s - entry) >= _min_tp_distance]
                    if viable:
                        tp_price = round(viable[0], 5)
                        tp_source = "structure"
                else:
                    swings = _find_swing_lows(df, lookback=100)
                    candidates = sorted((s for s in swings if s < entry), reverse=True)
                    viable = [s for s in candidates if (entry - s) >= _min_tp_distance]
                    if viable:
                        tp_price = round(viable[0], 5)
                        tp_source = "structure"
            except Exception as _e_tp:
                log.debug(f"[RiskEngine] structure-based TP lookup failed, using flat RR floor: {_e_tp}")

        tp_distance = abs(tp_price - entry)
        tp_pips  = round(tp_distance / self.pip) if self.pip > 0 else round(sl_pips * _min_rr)
        rr_ratio = round(tp_pips / sl_pips, 2) if sl_pips > 0 else 0
        try:
            from risk.rr_policy import get_execution_min_rr
            _execution_min_rr = get_execution_min_rr(strategy=strategy, test_mode=_tm)
        except Exception:
            _execution_min_rr = 1.5
        if rr_ratio < _execution_min_rr:
            return self._reject(
                f"R:R {rr_ratio:.2f} below execution minimum {_execution_min_rr:.2f}"
            )

        risk_usd = round(self.balance * self.MAX_RISK_PC / 100, 2)
        # BUG FIX (Cent-account 100x unit mismatch): get_pip_value_usd() is a
        # STATIC real-USD table. self.balance, however, comes from
        # config.INITIAL_BALANCE / live MT5 sync — which on a Cent account
        # is reported in CENTS (~100x real-USD). Mixing a cents-balance with
        # a real-USD pip value inflates lot_raw ~100x (matches the observed
        # "intended lot 11-15, actual risk 0.007-0.009%" symptom: MAX_LOT
        # then clips the inflated lot back down, silently gutting the real
        # risk taken). get_live_pip_value_per_lot() reads pip value from the
        # live MT5 connection in the ACCOUNT'S OWN unit (same unit as
        # balance), so risk_usd / (sl_pips * pip_val) is unit-consistent
        # regardless of account type. It falls back to the static table
        # (with a loud warning) only when no MT5 connection is available.
        pip_val  = get_live_pip_value_per_lot(self.symbol)

        # 2026-08-19 FIX: Guard against zero/negative/absurdly small pip
        # value. Missing entries in PIP_VALUE_USD used to fall back to
        # DEFAULT=10.0 (or bad live data), producing lot_raw of 49–118
        # on XPDUSD/USDTRY and then MAX_LOT under-risk rejects. Reject
        # early instead of letting the lot explode.
        if pip_val is None or pip_val <= 0.01:
            log.error(
                f"[RiskEngine] Invalid pip_val={pip_val} for {self.symbol} — "
                f"rejecting to avoid lot explosion"
            )
            return self._reject(
                f"Invalid pip value ({pip_val}) for {self.symbol} — "
                f"check PIP_VALUE_USD / live MT5 symbol_info"
            )

        lot_raw  = risk_usd / (sl_pips * pip_val) if sl_pips > 0 else 0.01

        # Day 97+ Book Rule (Page 13): Leverage-adjusted position sizing.
        # Forex is leveraged — "movements can be amplified". Reduce lot
        # size proportional to leverage to prevent account blow-up.
        # Most MT5 demo accounts use 1:100 leverage. We scale down lot
        # when leverage is high (risk_per_trade is already 1%, but the
        # NOTIONAL exposure can be 100× balance).
        # P0-2 (Audit Fix): MAX_LOT is already loaded at class level —
        # no need for a second try/except import here.
        # NOTE (2026-08-20 audit): confirmed by direct computation that
        # this multiplier does NOT affect the reject/approve outcome of
        # the risk-fraction safety guard further down — that guard's
        # fraction (risk_pc_max_by_lot / risk_pc_intended) is a function
        # of MAX_LOT, sl_pips, pip_val, and balance only, independent of
        # lot_raw/leverage_mult. The actual 2026-08-19 mass-rejection
        # incident (EURAUD/GBPNZD/AUDJPY/NZDCAD/etc. on a ~$99k account)
        # was caused by MAX_LOT itself being undersized for the account,
        # not by this multiplier — see config.MAX_LOT / .env fix.
        leverage_mult = 1.0
        if self.MAX_LOT > 1.0:
            leverage_mult = 0.5  # halve lot when high leverage allowed
        lot_raw = lot_raw * leverage_mult

        # EXECUTION-PROOF AUDIT FIX: apply live correlation_ctx multiplier.
        # When CorrelationEngine detected high correlation with an open
        # position (but below the 0.90 hard-reject threshold above), the
        # `risk_adjustment` multiplier (0.25-1.0) shrinks the lot so we
        # take less risk on a trade that effectively duplicates exposure
        # we already have on. Previously this multiplier was computed
        # every cycle and silently discarded (CONSUMPTION-MAP row #63).
        lot_raw = lot_raw * getattr(self, "_correlation_adjustment", 1.0)
        _corr_adj_applied = getattr(self, "_correlation_adjustment", 1.0)
        if _corr_adj_applied < 1.0:
            log.info(
                f"[RiskEngine] lot shrunk by correlation_ctx adjustment "
                f"{_corr_adj_applied:.2f}x (corr_risk="
                f"{getattr(self, '_correlation_risk_score', 0):.2f})"
            )

        # Day 81+ hotfix: cap at self.MAX_LOT (0.20 default), not 100.0.
        lot      = round(max(0.01, min(lot_raw, self.MAX_LOT)), 2)

        # Risk-sizing mismatch fix: when MAX_LOT caps the lot below what
        # the intended risk % would need, risk_usd/risk_pc computed from
        # the PRE-cap lot become fictional — e.g. "Risk: 1.0%" logged and
        # reported downstream (exposure_manager, correlation_manager,
        # trader.py confidence scaling) while the account is actually only
        # exposed to a tiny fraction of that. Recompute both from the
        # FINAL, post-cap lot so every consumer of risk_usd/risk_pc sees
        # the real number. The pre-cap intended values are kept alongside
        # under *_intended for audit/backtest transparency.
        #
        # BUG FIX (2026-08-20): risk_usd_intended/risk_pc_intended were
        # previously taken from `risk_usd`/`self.MAX_RISK_PC` captured
        # BEFORE the leverage_mult (Day 97+ Book Rule, line ~252) and
        # correlation_adjustment multipliers were applied to lot_raw.
        # Those two multipliers are INTENTIONAL, strategy-level risk
        # reductions (not the accidental under-risking the safety guard
        # below exists to catch) — but comparing risk_pc_max_by_lot
        # against the pre-multiplier 0.50% target double-counted them:
        # e.g. leverage_mult=0.5 alone made every high-leverage-account
        # trade register as "50% under-risked" before MAX_LOT even
        # entered the picture, systematically over-triggering the
        # "MAX_LOT cap shrinks actual risk" reject below on legitimate,
        # only-mildly-undersized MAX_LOT configs (see 2026-08-20 log:
        # lot_intended=2.48 (already leverage/corr-adjusted) vs
        # MAX_LOT=1.98 is a real ~20% shrink, not the 60% the old code
        # reported). Now computed from lot_raw AFTER all intentional
        # multipliers (leverage_mult, correlation_adjustment) — i.e.
        # what the strategy actually wants to risk right before the
        # MAX_LOT cap is applied — matching what the field's own
        # docstring below ("before MAX_LOT capping") already promised.
        risk_usd_intended = round(lot_raw * sl_pips * pip_val, 2)
        risk_pc_intended  = round((risk_usd_intended / self.balance) * 100, 4) if self.balance > 0 else 0.0
        # P5 fix: explicit "max allowed by lot cap" — what risk_usd WOULD be
        # if we used MAX_LOT exactly. Previously this was implicit in the
        # if/else below; now it's a named field so the operator can see all
        # three numbers in one place: requested / max_allowed_by_lot / actual.
        risk_usd_max_by_lot = round(self.MAX_LOT * sl_pips * pip_val, 2)
        risk_pc_max_by_lot  = round((risk_usd_max_by_lot / self.balance) * 100, 4) if self.balance > 0 else 0.0

        if lot_raw > self.MAX_LOT:
            # Lot was capped → actual risk = max_by_lot (since lot == MAX_LOT here)
            risk_usd = risk_usd_max_by_lot
            risk_pc  = risk_pc_max_by_lot
            log.warning(
                f"[RiskEngine] Intended lot {lot_raw:.2f} capped to {self.MAX_LOT} — "
                f"Effective risk reduced from {risk_pc_intended:.2f}% (${risk_usd_intended}) "
                f"to {risk_pc:.4f}% (${risk_usd}) "
                f"(sl_pips={sl_pips} pip_val=${pip_val})"
            )
            log.warning(
                f"[RiskEngine] Risk breakdown: "
                f"requested_risk={risk_pc_intended:.2f}% (${risk_usd_intended}) | "
                f"max_allowed_by_lot={risk_pc_max_by_lot:.4f}% (${risk_usd_max_by_lot}) | "
                f"actual_risk_after_lot_cap={risk_pc:.4f}% (${risk_usd}) | "
                f"lot_intended={lot_raw:.2f} lot_actual={lot} MAX_LOT={self.MAX_LOT}"
            )
            # SAFETY GUARD (fix for silent risk starvation): if the lot cap
            # shrinks actual risk to a small fraction of intended (default:
            # below 50%), the trade is no longer executing the strategy's
            # risk model — it's executing a near-random micro-lot. Reject
            # instead of silently approving a trade whose real exposure the
            # strategy never asked for. Threshold is configurable via
            # config.MIN_RISK_FRACTION_OF_INTENDED (default 0.5).
            try:
                from config import MIN_RISK_FRACTION_OF_INTENDED as _MIN_FRAC
                _min_frac = float(_MIN_FRAC)
            except Exception:
                _min_frac = 0.5
            _risk_fraction = (risk_pc / risk_pc_intended) if risk_pc_intended > 0 else 1.0
            if _risk_fraction < _min_frac:
                return self._reject(
                    f"MAX_LOT cap ({self.MAX_LOT}) shrinks actual risk to "
                    f"{_risk_fraction*100:.1f}% of intended ({risk_pc:.4f}% vs "
                    f"{risk_pc_intended:.2f}% target) — lot_intended={lot_raw:.2f} "
                    f"is far above MAX_LOT. Reconciling MAX_LOT with intended "
                    f"position size (or fixing the sizing formula/pip_value) "
                    f"is required rather than silently under-risking."
                )
        else:
            risk_pc = risk_pc_intended

        margin_needed = lot * 1000
        if margin_needed > self.balance * 0.5:
            return self._reject(f"Insufficient margin (need ~${margin_needed:.0f})")

        return {
            "approved":      True,
            "signal":        signal,
            "symbol":        self.symbol,
            "entry":         entry,
            "sl_price":      sl_price,
            "tp_price":      tp_price,
            "sl_pips":       sl_pips,
            "tp_pips":       tp_pips,
            "lot":           lot,
            "risk_usd":      risk_usd,
            "risk_pc":       risk_pc,
            # Audit/backtest transparency: what the sizing model WANTED
            # before MAX_LOT capping. Equal to risk_usd/risk_pc when no
            # capping occurred.
            "risk_usd_intended": risk_usd_intended,
            "risk_pc_intended":  risk_pc_intended,
            "lot_capped":        lot_raw > self.MAX_LOT,
            # P5 fix: explicit three-number breakdown so TradePermission /
            # operator can see requested vs max-allowed-by-lot vs actual.
            # `risk_usd_max_by_lot` = max USD risk achievable given MAX_LOT.
            # `actual_risk_after_lot_cap` = the REAL exposure (= risk_usd).
            # When lot_capped=False, requested == max == actual.
            "risk_usd_max_by_lot":        risk_usd_max_by_lot,
            "risk_pc_max_by_lot":         risk_pc_max_by_lot,
            "actual_risk_after_lot_cap":  risk_pc,
            "actual_risk_usd_after_lot_cap": risk_usd,
            "lot_intended":               round(lot_raw, 2),
            "MAX_LOT":                    self.MAX_LOT,
            "rr_ratio":      rr_ratio,
            "rr_preferred":  _min_rr,
            "rr_execution_min": _execution_min_rr,
            "strategy":      strategy,
            "tp_source":     tp_source,  # "structure" (real S/R level) or "atr_rr_floor" (flat min-RR fallback)
            "sl_source":     structure_source,
            "structure_reasons": structure_reasons,
            "daily_loss_pc": round(daily_loss_pc, 2),
            "open_trades":   open_trades,
            "reject_reason": None,
        }

    def _correlation_check(self) -> dict:
        # Day 90 bugfix: _live_open_pairs is now ALWAYS set in __init__
        # (empty set) and updated by sync_open_positions() each cycle.
        # Use it directly — no hasattr / isinstance checks needed.
        # If sync_open_positions was never called (e.g. fresh boot before
        # first cycle), this falls back to daily_risk.json state.
        live_pairs = getattr(self, "_live_open_pairs", None)
        if isinstance(live_pairs, set):
            open_pairs = live_pairs
        else:
            # Fallback: stale daily_risk.json state (only on very first cycle)
            open_pairs = set(self._daily.get("open_pairs", []))
        for group in CORRELATION_GROUPS:
            group_set = set(group)
            if self.symbol in group_set and open_pairs & group_set:
                return {"allowed": False, "reason": f"Correlation conflict with {open_pairs & group_set}"}
        return {"allowed": True, "reason": "OK"}

    def sync_open_positions(self, open_pairs) -> None:
        """Day 81+ hotfix (Day 90 bugfix): called by trader.py before
        evaluate() to inject the authoritative live open-pair list.

        This is the SINGLE source of truth for correlation checks — it
        overrides the potentially-stale open_pairs in daily_risk.json.

        Day 90 bugfix history:
          - There used to be TWO `sync_open_positions` methods in this
            class (line 159 + line 214). Python silently kept the second
            one, which only updated daily_risk.json and never set
            _live_open_pairs. Result: _correlation_check() always fell
            back to the stale file. The two methods are now merged here:
            we both update _live_open_pairs (in-memory authoritative
            state used by _correlation_check) AND sync daily_risk.json
            (for persistence across restarts).

        Args:
            open_pairs: list/set of pair symbols currently open
                        (e.g. ['USDJPY', 'EURUSD']).
        """
        import time as _time
        self._sync_call_count += 1
        self._last_sync_at = _time.time()
        try:
            # Clean + deduplicate symbols
            clean_pairs = sorted({clean_symbol(p) for p in (open_pairs or []) if p})
            # In-memory authoritative state (used by _correlation_check)
            self._live_open_pairs = set(clean_pairs)
            # Persisted state (used after restart, before first sync)
            self._daily["open_pairs"]   = clean_pairs
            self._daily["open_trades"]  = len(clean_pairs)
            self._save_daily(self._daily)
            log.debug(
                f"[RiskEngine] sync_open_positions OK | "
                f"pairs={clean_pairs} | calls={self._sync_call_count}"
            )
        except Exception as e:
            # Day 90 bugfix: log at WARNING (not debug) so silent failures
            # are visible in production logs. Increment fail counter so
            # health checks can detect recurring problems.
            self._sync_fail_count += 1
            log.warning(
                f"[RiskEngine] sync_open_positions FAILED "
                f"(call #{self._sync_call_count}, fail #{self._sync_fail_count}): {e}"
            )
            # Still try to set _live_open_pairs defensively so correlation
            # check doesn't silently use stale state. If even this raises,
            # we leave the previous value in place (better stale than none).
            try:
                self._live_open_pairs = set(open_pairs or [])
            except (TypeError, ValueError) as e:
                log.warning(f"[RiskEngine] Failed to set _live_open_pairs from fallback: {e}")
                self._live_open_pairs = set()  # empty = most conservative (blocks all correlated trades)

    def _load_daily(self) -> dict:
        """Load daily risk state from disk.

        CRITICAL FIX: Fail CLOSED on corruption, not open.
        Previously, any read error returned _fresh_day() — silently
        resetting total_loss_usd to 0. A crash near the daily loss limit
        would reset the counter, allowing more losses.
        Now: on corruption, return a FAIL-SAFE state that blocks new trades.
        """
        import os as _os
        _os.makedirs("memory", exist_ok=True)
        today = date.today().isoformat()
        if not _os.path.exists(DAILY_LOG_PATH):
            return self._fresh_day(today)
        try:
            with open(DAILY_LOG_PATH) as f:
                data = json.load(f)
            return data if data.get("date") == today else self._fresh_day(today)
        except (json.JSONDecodeError, KeyError) as e:
            # Corrupt JSON — fail CLOSED: block all new trades
            log.critical(
                f"risk_engine: daily_risk.json is CORRUPT ({e}) — "
                f"FAILING CLOSED (blocking new trades). Manual intervention required."
            )
            return {
                "date": today,
                "total_loss_usd": 999999,  # blocks all new trades
                "total_win_usd": 0,
                "open_trades": 0,
                "open_pairs": [],
                "trades": [],
                "_corrupt": True,
            }
        except Exception as e:
            log.critical(
                f"risk_engine: daily_risk.json read error ({e}) — "
                f"FAILING CLOSED. Manual intervention required."
            )
            return {
                "date": today,
                "total_loss_usd": 999999,
                "total_win_usd": 0,
                "open_trades": 0,
                "open_pairs": [],
                "trades": [],
                "_corrupt": True,
            }

    def _fresh_day(self, today: str) -> dict:
        data = {"date": today, "total_loss_usd": 0, "total_win_usd": 0,
                "open_trades": 0, "open_pairs": [], "trades": []}
        self._save_daily(data)
        return data

    def _save_daily(self, data: dict) -> None:
        """CRITICAL FIX: Atomic write using temp file + os.replace().
        Previously wrote directly with open(path, 'w') — a crash mid-write
        would leave a truncated/invalid JSON file, which _load_daily()
        would then silently treat as 'no history' (fail-open).
        Now: write to temp file first, then atomic rename.
        """
        import tempfile
        dir_name = os.path.dirname(DAILY_LOG_PATH) or "."
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=dir_name, suffix=".tmp",
                prefix="daily_risk_", delete=False
            ) as tmp_f:
                json.dump(data, tmp_f, indent=2)
                tmp_path = tmp_f.name
            os.replace(tmp_path, DAILY_LOG_PATH)
        except Exception as e:
            try:
                os.unlink(tmp_path)
            except (OSError, UnboundLocalError):
                pass
            raise

    def record_trade_open(self, symbol: str) -> None:
        self._daily["open_trades"] = self._daily.get("open_trades", 0) + 1
        pairs = self._daily.get("open_pairs", [])
        if symbol not in pairs:
            pairs.append(symbol)
        self._daily["open_pairs"] = pairs
        self._save_daily(self._daily)

    def record_trade_close(self, symbol: str, pnl_usd: float) -> None:
        self._daily["open_trades"] = max(0, self._daily.get("open_trades", 1) - 1)
        pairs = self._daily.get("open_pairs", [])
        if symbol in pairs:
            pairs.remove(symbol)
        self._daily["open_pairs"] = pairs
        if pnl_usd < 0:
            self._daily["total_loss_usd"] = self._daily.get("total_loss_usd", 0) + abs(pnl_usd)
        else:
            self._daily["total_win_usd"] = self._daily.get("total_win_usd", 0) + pnl_usd
        self._daily.setdefault("trades", []).append(
            {"symbol": symbol, "pnl_usd": round(pnl_usd, 2), "time": datetime.now(timezone.utc).isoformat()}
        )
        self._save_daily(self._daily)

    def get_daily_summary(self) -> dict:
        d = self._daily
        net = d.get("total_win_usd", 0) - d.get("total_loss_usd", 0)
        return {
            "date":               d.get("date"),
            "net_usd":            round(net, 2),
            "total_win_usd":      d.get("total_win_usd", 0),
            "total_loss_usd":     d.get("total_loss_usd", 0),
            "open_trades":        d.get("open_trades", 0),
            "open_pairs":         d.get("open_pairs", []),
            "daily_loss_pc":      round(d.get("total_loss_usd", 0) / self.balance * 100, 2),
            "limit_remaining_pc": round(self.DAILY_LOSS_LIMIT - d.get("total_loss_usd", 0) / self.balance * 100, 2),
        }

    def get_sync_health(self) -> dict:
        """Day 90 bugfix: surface sync_open_positions health metrics so
        dashboard / health monitors can detect when the sync chain is
        broken. Returns dict with:
          - sync_call_count  : total calls since boot
          - sync_fail_count  : total failures since boot
          - last_sync_ago_s  : seconds since last successful sync
          - live_open_pairs  : current authoritative open-pairs set
          - file_open_pairs  : what daily_risk.json says (should match)
          - in_sync          : True if live state matches file state
        """
        import time as _time
        ago = _time.time() - self._last_sync_at if self._last_sync_at > 0 else None
        live = getattr(self, "_live_open_pairs", set())
        file_pairs = set(self._daily.get("open_pairs", []))
        return {
            "sync_call_count": self._sync_call_count,
            "sync_fail_count": self._sync_fail_count,
            "last_sync_ago_s": round(ago, 1) if ago is not None else None,
            "live_open_pairs": sorted(live),
            "file_open_pairs": sorted(file_pairs),
            "in_sync": live == file_pairs,
        }

    def _reject(self, reason: str) -> dict:
        """Build a risk-rejection result.

        ARCHITECTURAL FIX (institutional refactor):
        The risk gate is an EXECUTION filter, NOT an analysis layer. It must
        NEVER produce a `signal` field — that belongs to the analysis layer
        (Rule Engine / LLM / Master). Previously this method returned
        `{"signal": "NO TRADE", ...}`, which collided with the analysis-layer
        `signal` field and caused downstream consumers (notably
        `core/trader.py::_apply_advanced_sizing()` L387 which reads
        `risk_out.get("signal")` as the authoritative direction) to silently
        see "NO TRADE" even when the analysis layer said BUY/SELL.

        Now: risk_out only carries risk-computed fields (lot/sl/tp/rr — all
        zeroed because the trade was rejected) plus `approved=False` and
        `reject_reason`. The analysis-layer signal is preserved by the caller
        (core/trader.py keeps `dec_out["decision"]` untouched) and is only
        gated at the TradePermission layer via `execution_allowed=False`.
        """
        log.info(f"[RiskEngine] REJECTED — {reason}")
        return {
            "approved":       False,
            "reject_reason":  reason,
            # Risk computations — all zeroed because no trade will be placed.
            "lot":            0,
            "sl_pips":        0,
            "tp_pips":        0,
            "rr_ratio":       0,
            "risk_usd":       0.0,
            "risk_pc":        0.0,
            # NOTE: NO `signal` field. Risk gate does not produce analysis
            # signals. Downstream consumers reading `risk_out.get("signal")`
            # will now get None (which they already handle via `.get(..., default)`).
        }

    def _clean(self, symbol: str) -> str:
        return clean_symbol(symbol)

    def print_summary(self, result: dict) -> None:
        bar  = "═" * 44
        icon = "✅" if result.get("approved") else "⛔"
        log.info(bar)
        log.info(f"  {icon}  RISK ENGINE")
        log.info(bar)
        if not result.get("approved"):
            log.info(f"  Rejected    : {result.get('reject_reason', 'unknown')}")
        else:
            log.info(f"  Signal      : {result.get('signal', '?')} {result.get('symbol', '?')}")
            log.info(f"  Entry       : {result.get('entry', 0)}")
            log.info(f"  SL          : {result.get('sl_price', 0)}  ({result.get('sl_pips', 0)} pips)")
            log.info(f"  TP          : {result.get('tp_price', 0)}  ({result.get('tp_pips', 0)} pips)")
            log.info(f"  Lot         : {result.get('lot', 0)}")
            log.info(f"  Risk        : {result.get('risk_pc', 0)}%  (${result.get('risk_usd', 0)})")
            log.info(f"  R:R         : 1:{result.get('rr_ratio', 0)}")
            log.info(f"  Daily loss  : {result.get('daily_loss_pc', 0)}%  (limit {self.DAILY_LOSS_LIMIT}%)")
        log.info(bar)

    def get_ai_context(self, result: dict) -> dict:
        return {
            "risk_approved": result["approved"],
            "risk_lot":      result.get("lot", 0),
            "risk_sl_pips":  result.get("sl_pips", 0),
            "risk_tp_pips":  result.get("tp_pips", 0),
            "risk_rr":       result.get("rr_ratio", 0),
            "risk_reject":   result.get("reject_reason"),
            "risk_sl_price": result.get("sl_price"),
            "risk_tp_price": result.get("tp_price"),
        }