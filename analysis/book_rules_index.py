# analysis/book_rules_index.py
# ============================================================
# Machine-Readable Book Rules Registry
# ============================================================
# Maps every extractable rule from "The Only Technical Analysis
# Book You Will Ever Need" (Pages 1-151) to its implementation.
#
# This registry can be queried programmatically to:
#   - List all rules by chapter
#   - List all rules by category (pattern, indicator, risk, etc.)
#   - Find implementation file/function for a given rule
#   - Verify completeness (all rules have implementations)
#   - Generate documentation
#
# Usage:
#   from analysis.book_rules_index import BOOK_RULES, get_rules_by_chapter
#   chapter_5_rules = get_rules_by_chapter(5)
# ============================================================

from dataclasses import dataclass, field
from typing import Optional, List, Dict


@dataclass(frozen=True)
class BookRule:
    """Single rule extracted from the book."""
    rule_id: str                    # e.g., "P106-HAMMER"
    page: int                       # book page number
    chapter: int                    # chapter number
    category: str                   # "pattern" | "indicator" | "risk" | "trend" | "strategy"
    name: str                       # human-readable name
    rule_type: str                  # "deterministic" | "needs_confirmation" | "design_principle"
    implementation_file: str        # path to implementation
    implementation_function: str    # function/class name
    description: str                # short description
    no_trade_condition: bool = False  # True if this rule defines a NO_TRADE state


# ═══════════════════════════════════════════════════════════════
# COMPLETE RULE REGISTRY (Pages 1-151)
# ═══════════════════════════════════════════════════════════════

BOOK_RULES: List[BookRule] = [


    # ── CHAPTER 1: TA FUNDAMENTALS (Pages 9-21) ──────────────
    BookRule("P9-TA_FOUNDATION", 9, 1, "fundamental", "TA Foundation", "design_principle",
             "data/indicators.py", "Indicators.add_all",
             "OHLC data is the foundation of TA"),
    BookRule("P15-LIMITATIONS", 15, 1, "fundamental", "TA Limitations", "design_principle",
             "agents/analysis_agent.py", "AnalysisAgent.run",
             "TA should be combined with fundamental/sentiment analysis"),

    # ── CHAPTER 2: S/R, VOLUME, CHARTS (Pages 22-36) ─────────
    BookRule("P22-SR_ZONE", 22, 2, "pattern", "S/R Zone Detection", "deterministic",
             "analysis/support_resistance.py", "SupportResistance.analyze",
             "Swing high/low cluster → S/R zone"),
    BookRule("P25-ZONE_STRENGTH", 25, 2, "pattern", "Zone Strength (2/3/4+)", "deterministic",
             "analysis/support_resistance.py", "_classify_strength",
             "2=Weak, 3=Medium, 4+=Strong"),
    BookRule("P25-ROLE_REVERSAL", 25, 2, "pattern", "Role Reversal", "deterministic",
             "analysis/support_resistance.py", "_detect_role_reversal",
             "Broken support→resistance, broken resistance→support"),
    BookRule("P28-OBV", 28, 2, "indicator", "On-Balance Volume", "deterministic",
             "data/indicators.py", "_add_obv",
             "Volume confirms price trend"),
    BookRule("P32-VOLUME_RSI", 32, 2, "indicator", "Volume RSI", "deterministic",
             "data/indicators_ext.py", "_volume_rsi",
             "RSI applied to volume"),

    # ── CHAPTER 3: INDICATORS (Pages 37-55) ──────────────────
    BookRule("P38-MA", 38, 3, "indicator", "Moving Average", "deterministic",
             "data/indicators.py", "_add_moving_averages",
             "SMA + EMA, trend identification"),
    BookRule("P42-RSI", 42, 3, "indicator", "RSI (14)", "deterministic",
             "data/indicators.py", "_add_rsi",
             "RSI = 100 - 100/(1+RS), overbought>70, oversold<30"),
    BookRule("P44-STOCHASTIC", 44, 3, "indicator", "Stochastic", "deterministic",
             "data/indicators.py", "_add_stochastic",
             "%K + %D oscillator"),
    BookRule("P46-FIBONACCI", 46, 3, "indicator", "Fibonacci Retracement", "deterministic",
             "analysis/fibonacci.py", "FibonacciEngine.analyze",
             "23.6%, 38.2%, 50%, 61.8%, 78.6%"),
    BookRule("P50-BOLLINGER", 50, 3, "indicator", "Bollinger Bands", "deterministic",
             "data/indicators.py", "_add_bollinger",
             "SMA ± 2×StdDev"),
    BookRule("P53-ATR", 53, 3, "indicator", "ATR", "deterministic",
             "analysis/_engine_utils.py", "atr_series",
             "Volatility measure for SL sizing"),

    # ── CHAPTER 4: TREND + MTF (Pages 56-71) ─────────────────
    BookRule("P59-HH_HL_UPTREND", 59, 4, "trend", "HH/HL Uptrend", "deterministic",
             "analysis/structure.py", "MarketStructureEngine",
             "Higher Highs + Higher Lows = uptrend"),
    BookRule("P59-LH_LL_DOWNTREND", 59, 4, "trend", "LH/LL Downtrend", "deterministic",
             "analysis/structure.py", "MarketStructureEngine",
             "Lower Highs + Lower Lows = downtrend"),
    BookRule("P59-SIDEWAYS", 59, 4, "trend", "Sideways (no-trade)", "deterministic",
             "analysis/structure.py", "MarketStructureEngine",
             "No clear HH/HL or LH/LL → WAIT",
             no_trade_condition=True),
    BookRule("P62-TRENDLINE", 62, 4, "pattern", "Wick-based Trendline", "deterministic",
             "analysis/trendline_engine.py", "TrendlineEngine.analyze",
             "Connect swing wick extremes"),
    BookRule("P63-BOS", 63, 4, "trend", "Break of Structure", "deterministic",
             "analysis/structure.py", "_detect_bos",
             "Trend continuation signal"),
    BookRule("P63-CHOCH", 63, 4, "trend", "Change of Character", "deterministic",
             "analysis/structure.py", "_detect_choch",
             "Trend reversal signal (needs confirmation)"),
    BookRule("P69-MTF_3TIER", 69, 4, "trend", "3-Tier MTF System", "deterministic",
             "analysis/structure_mtf.py", "MTFStructureEngine",
             "Trend→Signal→Entry timeframe hierarchy"),

    # ── CHAPTER 5: CANDLESTICK PATTERNS (Pages 72-99) ────────
    BookRule("P79-HAMMER", 79, 5, "pattern", "Hammer", "deterministic",
             "analysis/high_reliability_patterns.py", "_detect_hammer",
             "Long lower wick ≥2× body, bullish reversal"),
    BookRule("P80-SHOOTING_STAR", 80, 5, "pattern", "Shooting Star", "deterministic",
             "analysis/high_reliability_patterns.py", "_detect_shooting_star",
             "Long upper wick ≥2× body, bearish reversal"),
    BookRule("P82-INVERTED_HAMMER", 82, 5, "pattern", "Inverted Hammer", "deterministic",
             "analysis/high_reliability_patterns.py", "_detect_inverted_hammer",
             "Hammer shape in downtrend (needs confirmation)"),
    BookRule("P83-HANGING_MAN", 83, 5, "pattern", "Hanging Man", "deterministic",
             "analysis/high_reliability_patterns.py", "_detect_hanging_man",
             "Hammer shape in uptrend, bearish warning"),
    BookRule("P85-DOJI", 85, 5, "pattern", "Doji", "deterministic",
             "analysis/high_reliability_patterns.py", "_detect_doji",
             "Open≈Close, indecision",
             no_trade_condition=True),  # Multi-Doji → WAIT
    BookRule("P88-BULL_MARUBOZU", 88, 5, "pattern", "Bullish Marubozu", "deterministic",
             "analysis/high_reliability_patterns.py", "_detect_bullish_marubozu",
             "Body ≥90% range, strong buyer momentum"),
    BookRule("P88-BEAR_MARUBOZU", 88, 5, "pattern", "Bearish Marubozu", "deterministic",
             "analysis/high_reliability_patterns.py", "_detect_bearish_marubozu",
             "Body ≥90% range, strong seller momentum"),
    BookRule("P90-BULL_ENGULFING", 90, 5, "pattern", "Bullish Engulfing", "deterministic",
             "analysis/high_reliability_patterns.py", "_detect_bullish_engulfing",
             "2nd candle engulfs 1st body, bullish reversal"),
    BookRule("P90-BEAR_ENGULFING", 90, 5, "pattern", "Bearish Engulfing", "deterministic",
             "analysis/high_reliability_patterns.py", "_detect_bearish_engulfing",
             "2nd candle engulfs 1st body, bearish reversal"),
    BookRule("P92-TWEEZER_TOP", 92, 5, "pattern", "Tweezer Top", "deterministic",
             "analysis/high_reliability_patterns.py", "_detect_tweezer_top",
             "Equal high rejected twice, bearish"),
    BookRule("P92-TWEEZER_BOTTOM", 92, 5, "pattern", "Tweezer Bottom", "deterministic",
             "analysis/high_reliability_patterns.py", "_detect_tweezer_bottom",
             "Equal low rejected twice, bullish"),
    BookRule("P94-PIERCING_LINE", 94, 5, "pattern", "Piercing Line", "deterministic",
             "analysis/high_reliability_patterns.py", "_detect_piercing_line",
             "Bullish candle closes ≥50% into prior bearish body"),
    BookRule("P94-DARK_CLOUD", 94, 5, "pattern", "Dark Cloud Cover", "deterministic",
             "analysis/high_reliability_patterns.py", "_detect_dark_cloud_cover",
             "Bearish candle closes ≥50% into prior bullish body"),
    BookRule("P96-HARAMI", 96, 5, "pattern", "Harami (Bull/Bear)", "deterministic",
             "analysis/high_reliability_patterns.py", "_detect_harami",
             "Small candle inside large candle, momentum weakening"),
    BookRule("P97-MORNING_STAR", 97, 5, "pattern", "Morning Star", "deterministic",
             "analysis/high_reliability_patterns.py", "_detect_morning_star",
             "Bearish→indecision→bullish, confirmed reversal"),
    BookRule("P97-EVENING_STAR", 97, 5, "pattern", "Evening Star", "deterministic",
             "analysis/high_reliability_patterns.py", "_detect_evening_star",
             "Bullish→indecision→bearish, confirmed reversal"),
    BookRule("P98-THREE_SOLDIERS", 98, 5, "pattern", "Three White Soldiers", "deterministic",
             "analysis/high_reliability_patterns.py", "_detect_three_white_soldiers",
             "3 consecutive large bullish candles, continuation"),
    BookRule("P98-THREE_CROWS", 98, 5, "pattern", "Three Black Crows", "deterministic",
             "analysis/high_reliability_patterns.py", "_detect_three_black_crows",
             "3 consecutive large bearish candles, continuation"),
    BookRule("P98-THREE_INSIDE_UP", 98, 5, "pattern", "Three Inside Up", "deterministic",
             "analysis/high_reliability_patterns.py", "_detect_three_inside_up",
             "Bullish Harami + confirmation candle"),
    BookRule("P98-THREE_INSIDE_DOWN", 98, 5, "pattern", "Three Inside Down", "deterministic",
             "analysis/high_reliability_patterns.py", "_detect_three_inside_down",
             "Bearish Harami + confirmation candle"),

    # ── CHAPTER 6: CHART PATTERNS (Pages 100-118) ────────────
    BookRule("P103-DOUBLE_TOP", 103, 6, "pattern", "Double Top", "deterministic",
             "analysis/advanced_patterns.py", "detect_double_top_bottom",
             "Two peaks at same level, bearish reversal"),
    BookRule("P103-DOUBLE_BOTTOM", 103, 6, "pattern", "Double Bottom", "deterministic",
             "analysis/advanced_patterns.py", "detect_double_top_bottom",
             "Two troughs at same level, bullish reversal"),
    BookRule("P106-HEAD_SHOULDERS", 106, 6, "pattern", "Head & Shoulders", "deterministic",
             "analysis/advanced_patterns.py", "detect_head_and_shoulders",
             "3-peak reversal with neckline break"),
    BookRule("P107-RISING_WEDGE", 107, 6, "pattern", "Rising Wedge", "deterministic",
             "analysis/advanced_patterns.py", "detect_wedge",
             "Converging up trendlines, bearish (counter-intuitive)"),
    BookRule("P108-FALLING_WEDGE", 108, 6, "pattern", "Falling Wedge", "deterministic",
             "analysis/advanced_patterns.py", "detect_wedge",
             "Converging down trendlines, bullish"),
    BookRule("P110-BULL_FLAG", 110, 6, "pattern", "Bullish Flag", "deterministic",
             "analysis/advanced_patterns.py", "detect_flag",
             "Strong up→sideways consolidation→breakout up"),
    BookRule("P110-BEAR_FLAG", 110, 6, "pattern", "Bearish Flag", "deterministic",
             "analysis/advanced_patterns.py", "detect_flag",
             "Strong down→sideways consolidation→breakout down"),
    BookRule("P111-CUP_HANDLE", 111, 6, "pattern", "Cup with Handle", "needs_confirmation",
             "analysis/advanced_patterns.py", "detect_cup_and_handle",
             "U-shape cup + shallow handle, bullish continuation"),
    BookRule("P112-RECTANGLE", 112, 6, "pattern", "Rectangle", "deterministic",
             "analysis/advanced_patterns.py", "detect_rectangle",
             "Price between parallel horizontal lines",
             no_trade_condition=True),  # No breakout → NO_TRADE
    BookRule("P113-ASCENDING_TRIANGLE", 113, 6, "pattern", "Ascending Triangle", "deterministic",
             "analysis/advanced_patterns.py", "detect_triangle",
             "Flat resistance + rising support, bullish"),
    BookRule("P114-DESCENDING_TRIANGLE", 114, 6, "pattern", "Descending Triangle", "deterministic",
             "analysis/advanced_patterns.py", "detect_triangle",
             "Flat support + falling resistance, bearish"),
    BookRule("P115-SYMMETRICAL_TRIANGLE", 115, 6, "pattern", "Symmetrical Triangle", "deterministic",
             "analysis/advanced_patterns.py", "detect_triangle",
             "Converging trendlines, direction-neutral until breakout",
             no_trade_condition=True),  # No breakout → NO_TRADE

    # ── CHAPTER 7: TRADING STRATEGIES (Pages 119-130) ────────
    BookRule("P120-MOMENTUM_SCREEN", 120, 7, "strategy", "52-Week High Momentum", "deterministic",
             "analysis/advanced_patterns.py", "detect_momentum_screen",
             "Price within 10% of high = momentum candidate"),
    BookRule("P124-POSITION_SIZING", 124, 7, "risk", "1-2% Position Sizing", "deterministic",
             "risk/position_sizer.py", "PositionSizer.calculate",
             "Max 1-2% account risk per trade"),
    BookRule("P126-RISK_REWARD", 126, 7, "risk", "Risk-Reward Gate", "deterministic",
             "risk/risk_engine.py", "RiskEngine",
             "Reject trades <1:1, prefer ≥1:2"),

    # ── CHAPTER 8: RISK MANAGEMENT (Pages 131-141) ───────────
    BookRule("P134-STOP_LOSS", 134, 8, "risk", "Stop-Loss Discipline", "deterministic",
             "risk/risk_engine.py", "RiskEngine.calculate",
             "Always use stop-loss"),
    BookRule("P134-TRAILING_STOP", 134, 8, "risk", "Trailing Stop", "deterministic",
             "risk/risk_engine.py", "RiskEngine",
             "Adjust SL to lock in profits"),
    BookRule("P136-DIVERSIFICATION", 136, 8, "risk", "Correlation-Based Diversification", "deterministic",
             "risk/book_guardrails.py", "check_correlation_exposure",
             "Avoid stacking correlated FX pairs"),
    BookRule("P138-ANTI_REVENGE", 138, 8, "risk", "Anti-Revenge-Trading", "deterministic",
             "risk/book_guardrails.py", "check_anti_revenge_trading",
             "Block oversized trades after loss streak",
             no_trade_condition=True),
    BookRule("P138-COST_AWARE_EV", 138, 8, "risk", "Cost-Aware EV Gate", "deterministic",
             "risk/book_guardrails.py", "check_cost_aware_ev",
             "Net EV after costs must be > 0",
             no_trade_condition=True),

    # ── CONCLUSION (Pages 142-151) ───────────────────────────
    BookRule("P142-HYBRID_SYSTEM", 142, 9, "fundamental", "Hybrid TA+Fundamental", "design_principle",
             "agents/analysis_agent.py", "AnalysisAgent.run",
             "Combine TA with fundamental + sentiment analysis"),


    # ════════════════════════════════════════════════════════════
    # BOOK 5 (Frank Miller — Supply & Demand)
    # Chapter numbers 51-64 reserved for this book (chapters 1-14).
    # Rule IDs prefixed with "B5-" to avoid collision with
    # Candlestick Bible rules above.
    # ════════════════════════════════════════════════════════════

    # ── BOOK 5 CHAPTER 6: ODD ENHANCERS SCORING (Pages 61-75) ─
    # The book's central quantitative framework: 4 compulsory + 2
    # optional "odd enhancers" producing a zone tradability score.

    BookRule("B5-P62-SCORING_SYSTEM", 62, 56, "strategy", "Odd Enhancers Scoring Framework",
             "design_principle",
             "analysis/odd_enhancers.py", "OddEnhancerScorer.score_zone",
             "zone_score = sum(4 compulsory enhancer scores, each 0-3 normalized, max=12)"),

    BookRule("B5-P63-STRENGTH_OF_MOVE", 63, 56, "pattern", "Enhancer 1: Strength of Move",
             "deterministic",
             "analysis/odd_enhancers.py", "OddEnhancerScorer._score_strength_of_move",
             "Strong/fast departure (≥2 ERCs) → 3.0; 1 ERC → 1.5; 0 ERCs → 0 (skip)"),

    BookRule("B5-P65-WEAK_DEPARTURE", 65, 56, "risk", "Weak Departure = Skip",
             "deterministic",
             "analysis/odd_enhancers.py", "OddEnhancerScorer._score_strength_of_move",
             "Indecision-candle-dominated departure → score 0 → SKIP",
             no_trade_condition=True),

    BookRule("B5-P65-BASE_CANDLE_COUNT", 65, 56, "pattern", "Enhancer 2: Time at the Zone",
             "deterministic",
             "analysis/odd_enhancers.py", "OddEnhancerScorer._score_time_at_zone",
             "Base ≤3 candles → 3.0; 4-5 → 1.5; ≥6 → 0 (skip)"),

    BookRule("B5-P66-BASE_TOO_LONG", 66, 56, "risk", "Base ≥6 Candles = Skip",
             "deterministic",
             "analysis/odd_enhancers.py", "OddEnhancerScorer._score_time_at_zone",
             "≥6 candles in base = orders likely used up → SKIP",
             no_trade_condition=True),

    BookRule("B5-P68-FRESHNESS_SCORING", 68, 56, "pattern", "Enhancer 3: Fresh Zone",
             "deterministic",
             "analysis/odd_enhancers.py", "OddEnhancerScorer._score_freshness",
             "0 retests → 3.0; 1 retest → 1.5; ≥2 retests → 0 (skip)"),

    BookRule("B5-P68-STALE_ZONE", 68, 56, "risk", "Stale Zone (≥2 retests) = Skip",
             "deterministic",
             "analysis/odd_enhancers.py", "OddEnhancerScorer._score_freshness",
             "≥2 retests = hardly any pending orders remain → SKIP",
             no_trade_condition=True),

    BookRule("B5-P69-RR_SCORING", 69, 56, "risk", "Enhancer 4: Risk/Reward Ratio",
             "deterministic",
             "analysis/odd_enhancers.py", "OddEnhancerScorer._score_risk_reward",
             "R:R ≥1:3 → 3.0; 1:1.5-1:2 → 1.5; <1:1.5 → 0 (skip)"),

    BookRule("B5-P69-RR_TOO_LOW", 69, 56, "risk", "R:R <1:1.5 = Skip",
             "deterministic",
             "analysis/odd_enhancers.py", "OddEnhancerScorer._score_risk_reward",
             "R:R <1:1.5 → ignore zone entirely",
             no_trade_condition=True),

    BookRule("B5-P72-ORIGINAL_ZONE", 72, 56, "pattern", "Optional Enhancer 5: Original Zone",
             "needs_confirmation",
             "analysis/odd_enhancers.py", "OddEnhancerScorer._check_original_zone",
             "Zone formed independently (not as reaction to prior zone) → confidence boost"),

    BookRule("B5-P73-OVERLAPPING_ZONES", 73, 56, "strategy", "Optional Enhancer 6: Overlapping Zones",
             "needs_confirmation",
             "analysis/odd_enhancers.py", "OddEnhancerScorer._check_overlapping",
             "Multi-TF zone confluence; HTF zones inherently stronger than LTF zones"),

    BookRule("B5-P74-TIER_A_DECISION", 74, 56, "strategy", "Tier A Decision (Score ≥10)",
             "deterministic",
             "analysis/odd_enhancers.py", "OddEnhancerScorer.score_zone",
             "Total ≥10/12 → full-conviction trade, limit order at proximal line"),

    BookRule("B5-P74-TIER_B_DECISION", 74, 56, "strategy", "Tier B Decision (Score 7-9, no zero)",
             "deterministic",
             "analysis/odd_enhancers.py", "OddEnhancerScorer.score_zone",
             "Score 7-9 with no individual zero → conditional trade, use Tier-B entry tactic"),

    BookRule("B5-P74-MARKET_ORDER_ENTRY", 74, 56, "strategy", "Tier-B Market-Order Entry",
             "deterministic",
             "analysis/odd_enhancers.py", "TierBEntryStateMachine.check_market_order_entry",
             "Wait for candle close inside zone, enter at piercing candle's close"),

    BookRule("B5-P75-CONFIRMATION_ENTRY", 75, 56, "strategy", "Tier-B Confirmation-Order Entry",
             "deterministic",
             "analysis/odd_enhancers.py", "TierBEntryStateMachine.check_confirmation_entry",
             "Pierce + reversal momentum candle closing back past proximal → enter at close"),


    # ── BOOK 5 CHAPTER 7: PRICE-ACTION CONFLUENCE (Pages 80-88) ─
    # The book layers 5 PA patterns on top of zones as an unofficial
    # 7th enhancer (confluence). Most pattern detectors already exist
    # in analysis/high_reliability_patterns.py and analysis/patterns.py;
    # the confluence CHECK is wired into odd_enhancers._check_pa_confluence.

    BookRule("B5-P80-PA_CONFLUENCE", 80, 57, "strategy", "PA Confluence Layer (Unofficial 7th Enhancer)",
             "needs_confirmation",
             "analysis/odd_enhancers.py", "OddEnhancerScorer._check_pa_confluence",
             "5 PA patterns (pin bar/inside bar/H&S/double top-bottom/engulfing) at zone = confluence boost"),

    BookRule("B5-P80-PIN_BAR", 80, 57, "pattern", "Pin Bar at Zone",
             "needs_confirmation",
             "analysis/pin_bar_strategy.py", "PinBarStrategy",
             "Small body + long tail at zone proximal = rejection signal"),

    BookRule("B5-P81-PIN_BAR_WARNING", 81, 57, "risk", "Pin Bar Not Standalone",
             "design_principle",
             "analysis/pin_bar_strategy.py", "PinBarStrategy",
             "Pin bars are not reliable standalone — combine with zone + structure"),

    BookRule("B5-P82-INSIDE_BAR", 82, 57, "pattern", "Inside Bar at Zone",
             "needs_confirmation",
             "analysis/high_reliability_patterns.py", "HighReliabilityPatternDetector",
             "Candle fully within prior candle's range; reversal or continuation context-dependent"),

    BookRule("B5-P83-INSIDE_BAR_CONTEXT", 83, 57, "strategy", "Inside Bar Zone-Context Filter",
             "deterministic",
             "analysis/odd_enhancers.py", "OddEnhancerScorer._check_pa_confluence",
             "Same inside bar tradable near demand (reversal) / RBR (continuation), NOT near supply"),

    BookRule("B5-P83-HEAD_SHOULDERS", 83, 57, "pattern", "Head & Shoulders at Zone",
             "needs_confirmation",
             "analysis/advanced_patterns.py", "detect_head_and_shoulders",
             "3-peak reversal pattern; stronger when combined with S/D zone"),

    BookRule("B5-P84-HS_ENTRY_METHODS", 84, 57, "strategy", "H&S Two Entry Tactics",
             "deterministic",
             "analysis/advanced_patterns.py", "detect_head_and_shoulders",
             "Conservative=neckline break; Aggressive=limit at left-shoulder price with distal SL"),

    BookRule("B5-P85-DOUBLE_TOP_BOTTOM", 85, 57, "pattern", "Double Top/Bottom at Zone",
             "needs_confirmation",
             "analysis/advanced_patterns.py", "detect_double_top_bottom",
             "Two peaks/troughs at similar level; slight weakening on 2nd attempt = stronger signal"),

    BookRule("B5-P86-DOUBLE_TB_MTF", 86, 57, "strategy", "Double Top/Bottom MTF Confirmation",
             "needs_confirmation",
             "analysis/mtf_analyzer.py", "MTFAnalyzer",
             "Confirm double top/bottom on smaller TF before entry"),

    BookRule("B5-P87-ENGULFING", 87, 57, "pattern", "Engulfing Pattern at Zone",
             "needs_confirmation",
             "analysis/high_reliability_patterns.py", "HighReliabilityPatternDetector",
             "2-candle reversal; 2nd candle fully engulfs 1st; most frequent of the 5 PA patterns"),

    BookRule("B5-P88-CONFLUENCE_THESIS", 88, 57, "design_principle", "Zone + PA Confluence Thesis",
             "design_principle",
             "analysis/odd_enhancers.py", "OddEnhancerScorer.score_zone",
             "Zones mark imbalance; PA patterns act as confirmation layered on top"),


    # ── BOOK 5 CHAPTER 8: FLIP ZONES (Pages 89-90) ─────────────
    # State-transition rule: zone.type flips on confirmed break (close
    # beyond distal line). Implemented in analysis/flip_zones.py.

    BookRule("B5-P89-FLIP_ZONE_DEMAND", 89, 58, "strategy", "Flip Zone: Demand→Supply",
             "deterministic",
             "analysis/flip_zones.py", "FlipZoneDetector.update",
             "Demand zone broken down (close < distal) → reclassified as supply"),

    BookRule("B5-P89-FLIP_ZONE_SUPPLY", 89, 58, "strategy", "Flip Zone: Supply→Demand",
             "deterministic",
             "analysis/flip_zones.py", "FlipZoneDetector.update",
             "Supply zone broken up (close > distal) → reclassified as demand"),

    BookRule("B5-P89-CONFIRMED_BREAK", 89, 58, "risk", "Confirmed Break = Close Beyond Distal",
             "deterministic",
             "analysis/flip_zones.py", "FlipZoneDetector.update",
             "Wick pierce does NOT flip a zone — only a candle CLOSE beyond distal counts"),

    BookRule("B5-P90-FLIP_EXAMPLE", 90, 58, "pattern", "Flip Zone Examples (S/R Role Reversal)",
             "deterministic",
             "analysis/flip_zones.py", "FlipZoneDetector",
             "Support→resistance and resistance→support — ported to S/D framework"),


    # ── BOOK 5 CHAPTER 11: CCI CONFLUENCE (Pages 120-125) ──────
    # Complete CCI entry/add/exit state machine layered on zones.

    BookRule("B5-P120-CCI_ENTRY", 120, 61, "indicator", "CCI Entry at Zone",
             "deterministic",
             "analysis/cci_state_machine.py", "CCIStateMachine.evaluate",
             "Long at demand zone if CCI < -100; short at supply zone if CCI > +100"),

    BookRule("B5-P122-ZONE_FAILURE_DIAGNOSIS", 122, 61, "risk", "Zone Failure Diagnostic",
             "deterministic",
             "analysis/cci_state_machine.py", "CCIStateMachine.diagnose_zone_failure",
             "Failed zones diagnosed via same enhancers: freshness, trend-line, departure, CCI"),

    BookRule("B5-P123-CCI_OVERBOUGHT_EXIT", 123, 61, "indicator", "CCI +200 Short Confirmation",
             "deterministic",
             "analysis/cci_state_machine.py", "CCIStateMachine.evaluate",
             "CCI ≈ +200 at supply zone retest = strong short confirmation"),

    BookRule("B5-P124-FULL_CONFLUENCE_STACK", 124, 61, "strategy", "Full Confluence Stack",
             "deterministic",
             "analysis/cci_state_machine.py", "CCIStateMachine._score_confluence",
             "Trend line + S/D zone + CCI extreme = fully optimized (3/3 confluence)"),

    BookRule("B5-P125-CCI_EXIT_RULE", 125, 61, "risk", "CCI Exit Rule",
             "deterministic",
             "analysis/cci_state_machine.py", "CCIStateMachine._check_exit_long",
             "Exit long when CCI < +100; exit short when CCI > -100 (momentum fading)",
             no_trade_condition=True),

    BookRule("B5-P125-CCI_ADD_RULE", 125, 61, "strategy", "CCI Add-to-Position Rule",
             "deterministic",
             "analysis/cci_state_machine.py", "CCIStateMachine.evaluate",
             "Add long only if CCI > 0; add short only if CCI < 0"),

    BookRule("B5-P125-CCI_AMBIGUOUS", 125, 61, "risk", "CCI Near Zero = Ambiguous",
             "deterministic",
             "analysis/cci_state_machine.py", "CCIStateMachine.evaluate",
             "|CCI| < 20 = uncertain state (correction vs reversal unclear) → HOLD",
             no_trade_condition=True),

    BookRule("B5-P125-CCI_NOT_STANDALONE", 125, 61, "design_principle", "CCI Not Standalone",
             "design_principle",
             "analysis/cci_state_machine.py", "CCIStateMachine",
             "CCI is a confluence layer, NOT a standalone signal — confirmation only"),


    # ── BOOK 5 CHAPTER 12: MULTI-FRAME CURVE (Pages 126-135) ───
    # The book's most quantitatively rich methodology: divide price
    # range between nearest zones into High/Equilibrium/Low thirds.

    BookRule("B5-P126-MTF_PRINCIPLE", 126, 62, "strategy", "MTF Alignment Principle",
             "design_principle",
             "analysis/curve_mtf.py", "CurveMTF.resolve_conflict",
             "Higher timeframe takes priority; trade only when timeframes align"),

    BookRule("B5-P127-TRADING_STYLES", 127, 62, "strategy", "Four Trading Styles",
             "design_principle",
             "analysis/curve_mtf.py", "TradingStyle",
             "Scalper/Day/Swing/Position — by holding period and trade frequency"),

    BookRule("B5-P129-TF_TRIPLET_LOOKUP", 129, 62, "strategy", "Timeframe Triplet per Style",
             "deterministic",
             "analysis/curve_mtf.py", "TIMEFRAME_TRIPLET",
             "Scalper=15m/5m/1m, Day=1d/4h/1h, Swing=1w/1d/4h, Position=1M/1w/1d"),

    BookRule("B5-P130-CURVE_DEFINITION", 130, 62, "strategy", "Curve Definition",
             "deterministic",
             "analysis/curve_mtf.py", "CurveMTF.from_zones",
             "Curve = price range between proximal lines of nearest demand and supply zones"),

    BookRule("B5-P131-CURVE_SPLIT_THIRDS", 131, 62, "strategy", "Curve Split into Thirds",
             "deterministic",
             "analysis/curve_mtf.py", "CurveMTF.from_zones",
             "subzone_width = (upper_proximal - lower_proximal) / 3; boundaries at +1w and +2w"),

    BookRule("B5-P132-FIB_CURVE_ALTERNATIVE", 132, 62, "strategy", "Fibonacci Curve Alternative",
             "deterministic",
             "analysis/curve_mtf.py", "CurveMTF.fib_levels_for_curve",
             "Fib tool set to 33% and 66% (NOT standard 23.6/38.2/61.8/78.6) = same as thirds"),

    BookRule("B5-P133-CURVE_BIAS_RULE", 133, 62, "strategy", "Curve Bias Decision Rule",
             "deterministic",
             "analysis/curve_mtf.py", "CurveMTF.get_bias",
             "High/Very High=SELL_ONLY; Low/Very Low=BUY_ONLY; Equilibrium=TREND_FOLLOW_OR_NO_TRADE"),

    BookRule("B5-P135-HTF_OVERRIDE", 135, 62, "strategy", "HTF Override Hierarchy",
             "deterministic",
             "analysis/curve_mtf.py", "CurveMTF.resolve_conflict",
             "'The longer frame always wins' — LTF signals only actionable if they agree with HTF bias",
             no_trade_condition=True),


    # ── BOOK 5 CHAPTER 13: TRADE MANAGEMENT (Pages 151-152) ────
    # Extended walkthrough conclusion + generalized curve-position rule.

    BookRule("B5-P151-TRADE_WALKTHROUGH", 151, 63, "strategy", "MTF Trade Walkthrough Conclusion",
             "design_principle",
             "analysis/curve_mtf.py", "CurveMTF",
             "Extended MTF entry walkthrough validates top-down zone + curve methodology"),

    BookRule("B5-P152-CURVE_PERSISTENCE", 152, 63, "strategy", "Curve Bias Persistence Rule",
             "deterministic",
             "analysis/curve_mtf.py", "Curve.bias_for",
             "Bias holds until price crosses into OPPOSITE extreme zone — persistence/invalidation"),


    # ── BOOK 5 CHAPTER 14: RISK MANAGEMENT (Pages 153-157) ─────
    # Complete risk system: position sizing, margin calls, drawdown throttling.

    BookRule("B5-P154-RISK_PER_TRADE", 154, 64, "risk", "Risk Per Trade (2% / 1%)",
             "deterministic",
             "analysis/risk_management.py", "RiskManager.base_risk_pct",
             "Experienced=2% per trade; Beginner=1% until 3x account growth"),

    BookRule("B5-P154-MARGIN_CALL", 154, 64, "risk", "Margin Call Detection",
             "deterministic",
             "analysis/risk_management.py", "MarginCallDetector.is_margin_call",
             "Margin call when account_loss% × leverage ≥ 100%",
             no_trade_condition=True),

    BookRule("B5-P154-BEGINNER_GRADUATION", 154, 64, "risk", "Beginner Graduation Rule",
             "deterministic",
             "analysis/risk_management.py", "RiskManager.graduate_to_experienced",
             "Beginner (1% risk) graduates to experienced (2%) when account triples (3x)"),

    BookRule("B5-P155-POSITION_SIZING", 155, 64, "risk", "Position Sizing Master Formula",
             "deterministic",
             "analysis/risk_management.py", "PositionSizer.size_for_stock",
             "position_size = risk_amount / |entry_price - stop_price|"),

    BookRule("B5-P155-FOREX_SIZING_SAME_CCY", 155, 64, "risk", "Forex Sizing (Quote=Account Ccy)",
             "deterministic",
             "analysis/risk_management.py", "PositionSizer.size_for_forex",
             "When quote ccy = account ccy: position = risk_amount / pip_distance"),

    BookRule("B5-P156-FOREX_SIZING_DIFF_CCY", 156, 64, "risk", "Forex Sizing (Quote≠Account Ccy)",
             "deterministic",
             "analysis/risk_management.py", "PositionSizer.size_for_forex",
             "Convert risk to quote ccy first (risk × exchange_rate), then divide by pip distance"),

    BookRule("B5-P155-DRAWDOWN_SIMULATOR", 155, 64, "risk", "Compounding Drawdown Math",
             "deterministic",
             "analysis/risk_management.py", "DrawdownSimulator.simulate_losing_streak",
             "remaining_equity = initial_equity × (1 - risk%)^n_losses"),

    BookRule("B5-P157-DRAWDOWN_CIRCUIT_BREAKER", 157, 64, "risk", "Drawdown Circuit Breaker (20% halt)",
             "deterministic",
             "analysis/risk_management.py", "RiskManager.update",
             "≥20% drawdown from peak → halt trading for rest of month",
             no_trade_condition=True),

    BookRule("B5-P157-DRAWDOWN_RISK_REDUCTION", 157, 64, "risk", "Drawdown Risk Reduction (25%)",
             "deterministic",
             "analysis/risk_management.py", "RiskManager.current_risk_pct",
             "During any drawdown → reduce risk_per_trade by 25% (2% → 1.5%)"),

    BookRule("B5-P157-RISK_RESTORE_ON_NEW_HIGH", 157, 64, "risk", "Risk Restore on New Equity High",
             "deterministic",
             "analysis/risk_management.py", "RiskManager.update",
             "Restore original risk_pct only when equity makes a NEW HIGH"),
]


# ═══════════════════════════════════════════════════════════════
# QUERY FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def get_all_rules() -> List[BookRule]:
    """Return all book rules."""
    return BOOK_RULES


def get_rules_by_chapter(chapter: int) -> List[BookRule]:
    """Return all rules from a specific chapter."""
    return [r for r in BOOK_RULES if r.chapter == chapter]


def get_rules_by_category(category: str) -> List[BookRule]:
    """Return all rules of a specific category.
    Categories: 'pattern', 'indicator', 'risk', 'trend', 'strategy', 'fundamental'
    """
    return [r for r in BOOK_RULES if r.category == category]


def get_no_trade_conditions() -> List[BookRule]:
    """Return all rules that define NO_TRADE states."""
    return [r for r in BOOK_RULES if r.no_trade_condition]


def get_deterministic_rules() -> List[BookRule]:
    """Return all deterministic (directly codeable) rules."""
    return [r for r in BOOK_RULES if r.rule_type == "deterministic"]


def get_rules_needing_confirmation() -> List[BookRule]:
    """Return rules that need additional confirmation logic."""
    return [r for r in BOOK_RULES if r.rule_type == "needs_confirmation"]


def get_design_principles() -> List[BookRule]:
    """Return architectural design principles."""
    return [r for r in BOOK_RULES if r.rule_type == "design_principle"]


def find_rule(rule_id: str) -> Optional[BookRule]:
    """Find a specific rule by ID."""
    for r in BOOK_RULES:
        if r.rule_id == rule_id:
            return r
    return None


def get_implementation_map() -> Dict[str, List[str]]:
    """Return {file_path: [function_names]} mapping."""
    impl_map: Dict[str, List[str]] = {}
    for r in BOOK_RULES:
        if r.implementation_file not in impl_map:
            impl_map[r.implementation_file] = []
        if r.implementation_function not in impl_map[r.implementation_file]:
            impl_map[r.implementation_file].append(r.implementation_function)
    return impl_map


def get_stats() -> dict:
    """Return summary statistics about the rule registry."""
    return {
        "total_rules":         len(BOOK_RULES),
        "deterministic":       len(get_deterministic_rules()),
        "needs_confirmation":  len(get_rules_needing_confirmation()),
        "design_principles":   len(get_design_principles()),
        "no_trade_conditions": len(get_no_trade_conditions()),
        "by_chapter":          {ch: len(get_rules_by_chapter(ch)) for ch in range(1, 65)},
        "by_category":         {cat: len(get_rules_by_category(cat))
                                for cat in ["pattern", "indicator", "risk", "trend",
                                            "strategy", "fundamental", "design_principle"]},
        "implementation_files": len(get_implementation_map()),
    }


# ═══════════════════════════════════════════════════════════════
# CLI ENTRY
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  BOOK RULES REGISTRY — Master Index")
    print("=" * 60)

    stats = get_stats()
    print(f"\nTotal Rules:           {stats['total_rules']}")
    print(f"Deterministic:         {stats['deterministic']}")
    print(f"Needs Confirmation:    {stats['needs_confirmation']}")
    print(f"Design Principles:     {stats['design_principles']}")
    print(f"No-Trade Conditions:   {stats['no_trade_conditions']}")
    print(f"Implementation Files:  {stats['implementation_files']}")

    print(f"\nBy Chapter:")
    for ch, count in stats["by_chapter"].items():
        if count > 0:
            label = (f"Book5-Chapter{ch-50}" if ch >= 51
                     else f"Chapter {ch}")
            print(f"  {label}: {count} rules")

    print(f"\nBy Category:")
    for cat, count in stats["by_category"].items():
        if count > 0:
            print(f"  {cat}: {count} rules")

    print(f"\nNo-Trade Conditions ({len(get_no_trade_conditions())}):")
    for r in get_no_trade_conditions():
        print(f"  [{r.rule_id}] {r.name} (P{r.page})")

    print("\n" + "=" * 60)
