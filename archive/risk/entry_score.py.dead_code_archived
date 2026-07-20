"""risk/entry_score.py — 100-point entry scoring system"""
from __future__ import annotations
from dataclasses import dataclass, field
from utils.logger import get_logger
log = get_logger("entry_score")

MIN_SCORE_TO_TRADE = 70
STRONG_TRADE_THRESHOLD = 85
MIN_RR = 2.0
MAX_SPREAD_PIPS = 10
MIN_ATR_PCT = 0.0003
MAX_ATR_PCT = 0.008

@dataclass
class EntryScoreResult:
    score: int = 0
    should_trade: bool = False
    is_strong: bool = False
    hard_block: str = ""
    components: dict = field(default_factory=dict)
    recommendation: str = "NO TRADE"

def compute_entry_score(df, ind_ctx, sr_ctx, regime, mtf_bias, direction, entry, sl, tp, atr, spread_pips=1.0, news_ctx=None, structure_ctx=None):
    result = EntryScoreResult()
    ind_ctx = ind_ctx or {}; sr_ctx = sr_ctx or {}; regime = regime or {}
    news_ctx = news_ctx or {}; structure_ctx = structure_ctx or {}
    components = {}
    if not news_ctx.get("news_trade_allowed", True):
        result.hard_block = "News block"; return result
    if spread_pips > MAX_SPREAD_PIPS:
        result.hard_block = f"Spread {spread_pips} > {MAX_SPREAD_PIPS}"; return result
    if entry > 0 and sl > 0 and tp > 0:
        risk = abs(entry - sl); reward = abs(tp - entry)
        if risk > 0:
            rr = reward / risk
            if rr < MIN_RR - 0.01:
                result.hard_block = f"R/R {rr:.2f} < {MIN_RR}"; return result
    price_loc = str(sr_ctx.get("price_location", "mid_range")).lower()
    if "mid" in price_loc:
        result.hard_block = "Mid-range"; return result
    if entry > 0 and atr > 0:
        atr_pct = atr / entry
        if atr_pct < MIN_ATR_PCT or atr_pct > MAX_ATR_PCT:
            result.hard_block = f"ATR {atr_pct:.5f} extreme"; return result
    # Scoring
    r_dir = str(regime.get("direction","")).upper(); r_name = str(regime.get("regime","")).upper()
    mtf_u = str(mtf_bias or "").upper()
    ts = 15
    if r_name == "TRENDING" and (("BULL" in r_dir and direction=="BUY") or ("BEAR" in r_dir and direction=="SELL")): ts = 20
    if direction=="BUY" and "BULL" in mtf_u: ts = min(ts+5,20)
    elif direction=="SELL" and "BEAR" in mtf_u: ts = min(ts+5,20)
    components["trend"] = ts
    sr_s = 5
    if "support" in price_loc and direction=="BUY": sr_s = 20
    elif "resistance" in price_loc and direction=="SELL": sr_s = 20
    elif "golden" in price_loc: sr_s = 15
    components["sr"] = sr_s
    liq_s = 5
    if structure_ctx.get("liquidity_sweep"): liq_s = 20
    components["liq"] = liq_s
    vr = float(ind_ctx.get("volume_ratio",1.0) or 1.0)
    vs = 15 if vr >= 1.5 else 10 if vr >= 1.2 else 5 if vr >= 1.0 else 2
    components["vol"] = vs
    ms = 0; rsi = float(ind_ctx.get("rsi",50) or 50); mc = str(ind_ctx.get("macd_cross","")).lower(); adx = float(ind_ctx.get("adx",0) or 0)
    if direction=="BUY":
        if "bull" in mc: ms += 8
        if 40 <= rsi <= 65: ms += 4
        if adx >= 25: ms += 3
    else:
        if "bear" in mc: ms += 8
        if 35 <= rsi <= 60: ms += 4
        if adx >= 25: ms += 3
    components["mom"] = min(ms, 15)
    rrs = 0
    if entry > 0 and sl > 0 and tp > 0:
        r = abs(entry-sl); w = abs(tp-entry)
        if r > 0:
            rr = w/r
            rrs = 10 if rr >= 3.0 else 8 if rr >= 2.5 else 6 if rr >= 2.0 else 2
    components["rr"] = rrs
    total = sum(components.values())
    result.score = int(total); result.components = components
    if total >= STRONG_TRADE_THRESHOLD:
        result.should_trade = True; result.is_strong = True; result.recommendation = f"STRONG TRADE ({total}/100)"
    elif total >= MIN_SCORE_TO_TRADE:
        result.should_trade = True; result.recommendation = f"MARGINAL ({total}/100)"
    else:
        result.recommendation = f"NO TRADE ({total}/100 < {MIN_SCORE_TO_TRADE})"
    log.info(f"[EntryScore] {direction} score={total}/100 → {result.recommendation}")
    return result
