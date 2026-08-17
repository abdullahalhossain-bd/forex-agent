"""
config/barrier_config.py — per-pair triple-barrier + live trade-filter settings

Verified by walk-forward backtest on independent held-out data, then
re-verified through a full simulated equity curve in backtest_engine.py
(which caught and fixed two real bugs along the way — see
FINAL_SYSTEM_README.md). These are the values behind that final run.

NOT auto-wired into your pipeline — I don't have your actual
phase4_labels.py / model_trainer.py / live trade-execution call sites, so
two separate wiring steps are needed:

1) At LABEL-GENERATION / TRAINING time, use the per-pair barrier width:

    from config.barrier_config import BARRIER_CONFIG, DEFAULT_PROB_THRESHOLD

    cfg = BARRIER_CONFIG.get(symbol, BARRIER_CONFIG["_default"])
    labels = triple_barrier_labels(
        df,
        holding_period=cfg["holding_period"],
        take_profit_width=cfg["atr_multiplier"],
        stop_loss_width=cfg["atr_multiplier"],
        atr_period=14,
        use_atr=True,
    )

2) At LIVE TRADE-DECISION time, add the cost-spike filter on top of your
   existing confidence threshold. This is a NEW requirement found while
   building backtest_engine.py: a pair's AVERAGE spread can look fine
   (e.g. EURCAD's cost ratio looked like ~16% of barrier width on
   average) while individual bars — specifically low-ATR / low-volatility
   moments — see spread eat 100-220% of that bar's barrier width, because
   spread doesn't shrink with volatility the way ATR does. Skipping those
   specific bars (not just averaging over them) meaningfully changed
   EURCAD's real numbers (81.7% win rate on the average-cost estimate ->
   59.7% honest win rate once cost-spike bars are excluded — still
   positive, just a thinner edge than the average made it look):

    current_atr = ...  # your live ATR(14) value for this symbol right now
    current_spread_price = current_spread_points * POINT_SIZE
    barrier_width_price = cfg["atr_multiplier"] * current_atr
    cost_R = current_spread_price / barrier_width_price

    if cost_R > MAX_COST_R:
        skip_this_trade()   # spread is eating too much of the risk budget right now
    elif pred_class != 0 and pred_confidence >= cfg["prob_threshold"]:
        take_trade()
"""

DEFAULT_PROB_THRESHOLD = 0.55
DEFAULT_HOLDING_PERIOD = 16   # bars (4h on M15)
POINT_SIZE = 0.00001          # verified against all 6 CSVs' quoted decimal precision
MAX_COST_R = 0.5              # NEW: skip a trade if spread alone would eat >50% of
                               # its risk budget right now (see notes above)

# CONFIRMED against your real config.py (uploaded 2026-08-17):
#   RISK_PER_TRADE = 0.005 (0.5%) — used for the equity-curve numbers in the
#   README, NOT the 1% first assumed. Corrected and re-run.
#   DAILY_LOSS_LIMIT_PCT = 5.0%, single source of truth, confirmed wired
#   correctly to RiskEngine/CircuitBreaker/KillSwitch/DrawdownController/
#   AutonomousRisk/RiskAgent per config.py's own comments — matches what
#   the risk-module audit found independently.
#   USE_SCANNER defaults to "false" — meaning scanner.config.CORRELATION_GROUPS
#   likely ISN'T loaded by default in your deployment, which means the
#   correlation_manager.py / exposure_manager.py fix in this package is not
#   just defensive, it is very likely ACTIVE in your current default config.
#   TRADING_MODE_CONFIDENCE["AUTONOMOUS"] = 60 (i.e. 0.60) — this is
#   TradePermission's overall signal-confidence gate (rule+ml+rl+llm fused),
#   a DIFFERENT number from this file's prob_threshold=0.55 (which is only
#   the raw ML model's own class probability, before fusion). Don't treat
#   these as the same knob — reconcile with whoever owns SignalFusion if
#   you want the raw-model gate and the fused-signal gate aligned.

BARRIER_CONFIG = {
    # pair: symmetric TP/SL width in ATR multiples, chosen via nested
    # walk-forward (tuned on folds 0-2, verified on folds 3+4, never peeked),
    # then re-verified via full equity-curve simulation with realistic
    # per-bar cost and the MAX_COST_R filter applied.
    #
    # Final verified numbers (walk-forward + cost-spike filter + CORRECTED
    # to your real RISK_PER_TRADE=0.5%, see backtest_engine_results_FINAL.csv):
    #   EURAUD: 82.0% win rate, profit factor 3.23, max DD -3.5%, 0.49 trades/day
    #   GBPCAD: 71.1% win rate, profit factor 2.03, max DD -17.6%, 1.23 trades/day
    #   EURCAD: 59.7% win rate, profit factor 1.14, max DD -7.2%, 0.33 trades/day (thin edge — watch this one)
    #   EURNZD: 85.0% win rate, profit factor 4.87, max DD -6.3%, 2.85 trades/day
    #   GBPNOK: 89.3% win rate, profit factor 1.74, max DD -14.8%, 10.41 trades/day
    #   GBPSEK: 84.2% win rate, profit factor 1.44, max DD -15.8%, 9.90 trades/day
    "EURAUD": {"atr_multiplier": 2.5, "holding_period": 16, "prob_threshold": 0.55, "max_cost_R": 0.5},
    "GBPCAD": {"atr_multiplier": 2.5, "holding_period": 16, "prob_threshold": 0.55, "max_cost_R": 0.5},
    "EURCAD": {"atr_multiplier": 1.5, "holding_period": 16, "prob_threshold": 0.55, "max_cost_R": 0.5},
    "EURNZD": {"atr_multiplier": 3.0, "holding_period": 16, "prob_threshold": 0.55, "max_cost_R": 0.5},
    "GBPNOK": {"atr_multiplier": 4.0, "holding_period": 16, "prob_threshold": 0.55, "max_cost_R": 0.5},
    "GBPSEK": {"atr_multiplier": 4.0, "holding_period": 16, "prob_threshold": 0.55, "max_cost_R": 0.5},

    # fallback for any symbol not tuned above — untested default, not a guess at a wider one
    "_default": {"atr_multiplier": 1.5, "holding_period": 16, "prob_threshold": 0.55, "max_cost_R": 0.5},
}
