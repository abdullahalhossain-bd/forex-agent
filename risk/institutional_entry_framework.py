"""risk/institutional_entry_framework.py — 200-point institutional framework"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from utils.logger import get_logger
log = get_logger("institutional_entry")
MIN_SCORE_TO_TRADE = 130; GOOD_TRADE_THRESHOLD = 160; APLUS_THRESHOLD = 180
MIN_RR = 2.0; MAX_SPREAD_PIPS = 10; MIN_ATR_PCT = 0.0003; MAX_ATR_PCT = 0.008
W = {"market_structure":25,"smc":25,"liquidity":20,"mtf_alignment":20,"session_timing":15,"volume_vwap":15,"momentum":15,"risk_reward":10,"news_macro":10,"candlestick":10,"spread_execution":10,"trend_direction":10,"volatility_regime":10,"psychology":5}
TOTAL_W = sum(W.values())

@dataclass
class InstitutionalEntryResult:
    score: int = 0; max_score: int = TOTAL_W; should_trade: bool = False
    trade_quality: str = "NO TRADE"; hard_block: str = ""
    components: dict = field(default_factory=dict); position_size_mult: float = 1.0
    recommendations: list = field(default_factory=list)

def evaluate_institutional_entry(direction, entry, sl, tp, df=None, ind_ctx=None, sr_ctx=None, regime=None, mtf_bias=None, mtf_data=None, structure_ctx=None, smc_ctx=None, session_ctx=None, news_ctx=None, liquidity_ctx=None, volume_ctx=None, spread_pips=1.0, revenge_ctx=None):
    result = InstitutionalEntryResult()
    ind_ctx = ind_ctx or {}; sr_ctx = sr_ctx or {}; regime = regime or {}
    structure_ctx = structure_ctx or {}; smc_ctx = smc_ctx or {}; session_ctx = session_ctx or {}
    news_ctx = news_ctx or {}; liquidity_ctx = liquidity_ctx or {}; mtf_data = mtf_data or {}; revenge_ctx = revenge_ctx or {}
    direction = direction.upper()
    # Hard blocks
    if not news_ctx.get("news_trade_allowed", True): result.hard_block = "News"; return result
    if spread_pips > MAX_SPREAD_PIPS: result.hard_block = "Spread"; return result
    if entry > 0 and sl > 0 and tp > 0:
        r = abs(entry-sl) if direction=="BUY" else abs(sl-entry)
        w = abs(tp-entry) if direction=="BUY" else abs(entry-tp)
        if r > 0 and w/r < MIN_RR - 0.01: result.hard_block = "R/R too low"; return result
    if "mid" in str(sr_ctx.get("price_location","mid_range")).lower(): result.hard_block = "Mid-range"; return result
    atr = float(ind_ctx.get("atr",0.001) or 0.001)
    if entry > 0:
        ap = atr/entry
        if ap < MIN_ATR_PCT or ap > MAX_ATR_PCT: result.hard_block = "ATR extreme"; return result
    now = datetime.now(timezone.utc)
    if now.weekday() == 4 and now.hour >= 20: result.hard_block = "Friday"; return result
    if revenge_ctx.get("is_revenge"): result.hard_block = "Revenge"; return result
    # Scoring (simplified — uses same logic as entry_score but with more dimensions)
    c = {}
    bos = str(structure_ctx.get("structure_bos","NONE")).upper()
    c["market_structure"] = 25 if ("BULL" in bos and direction=="BUY") or ("BEAR" in bos and direction=="SELL") else 12 if structure_ctx.get("structure_bias") else 5
    smc_sig = str(smc_ctx.get("smc_signal","WAIT")).upper()
    c["smc"] = min(15 if smc_sig == direction else 0 + (5 if smc_ctx.get("order_block") else 0) + (3 if smc_ctx.get("fvg") else 0), 25)
    c["liquidity"] = min((15 if structure_ctx.get("liquidity_sweep") else 0) + (3 if liquidity_ctx.get("equal_lows") and direction=="BUY" else 0) + (3 if liquidity_ctx.get("equal_highs") and direction=="SELL" else 0), 20) or 3
    mtf_u = str(mtf_bias or "").upper()
    c["mtf_alignment"] = 15 if (direction=="BUY" and "BULL" in mtf_u) or (direction=="SELL" and "BEAR" in mtf_u) else 5
    c["session_timing"] = 12 if session_ctx.get("quality") == "HIGH" else 8 if session_ctx.get("quality") == "MEDIUM" else 5
    vr = float(ind_ctx.get("volume_ratio",1.0) or 1.0); vwap = float(ind_ctx.get("vwap",0) or 0)
    vv = (8 if vr>=1.5 else 5 if vr>=1.2 else 3) + (7 if vwap>0 and ((direction=="BUY" and entry<=vwap) or (direction=="SELL" and entry>=vwap)) else 2)
    c["volume_vwap"] = min(vv, 15)
    rsi = float(ind_ctx.get("rsi",50) or 50); mc = str(ind_ctx.get("macd_cross","")).lower(); adx = float(ind_ctx.get("adx",0) or 0)
    ms = 0
    if direction=="BUY":
        if "bull" in mc: ms+=5
        if 40<=rsi<=65: ms+=4
        if adx>=25: ms+=3
    else:
        if "bear" in mc: ms+=5
        if 35<=rsi<=60: ms+=4
        if adx>=25: ms+=3
    c["momentum"] = min(ms,15)
    rrs = 0
    if entry>0 and sl>0 and tp>0:
        r=abs(entry-sl) if direction=="BUY" else abs(sl-entry); w=abs(tp-entry) if direction=="BUY" else abs(entry-tp)
        if r>0: rr=w/r; rrs=10 if rr>=3 else 8 if rr>=2.5 else 6 if rr>=2 else 2
    c["risk_reward"] = rrs
    c["news_macro"] = 10 if news_ctx.get("risk_level","LOW")=="LOW" else 6
    pat = str(ind_ctx.get("pattern","none")).lower()
    cs = 10 if any(x in pat for x in ["hammer","engulf","star"]) else 5 if "doji" in pat else 2
    c["candlestick"] = cs
    c["spread_execution"] = 10 if spread_pips<=1 else 8 if spread_pips<=2 else 5 if spread_pips<=4 else 3
    r_dir = str(regime.get("direction","")).upper(); r_name = str(regime.get("regime","")).upper()
    ts = 10 if r_name=="TRENDING" and (("BULL" in r_dir and direction=="BUY") or ("BEAR" in r_dir and direction=="SELL")) else 5 if r_name=="RANGING" else 3
    c["trend_direction"] = ts
    c["volatility_regime"] = 10 if regime.get("volatility")=="NORMAL" else 6 if regime.get("volatility")=="HIGH" else 4
    c["psychology"] = 5
    total = sum(c.values()); result.score = int(total); result.components = c
    if total >= APLUS_THRESHOLD: result.should_trade=True; result.trade_quality="A+ SETUP"; result.position_size_mult=1.1
    elif total >= GOOD_TRADE_THRESHOLD: result.should_trade=True; result.trade_quality="GOOD TRADE"; result.position_size_mult=1.0
    elif total >= MIN_SCORE_TO_TRADE: result.should_trade=True; result.trade_quality="MARGINAL"; result.position_size_mult=0.5
    log.info(f"[InstitutionalEntry] {direction} score={total}/{TOTAL_W} → {result.trade_quality}")
    return result
