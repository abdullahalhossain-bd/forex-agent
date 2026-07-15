# broker/__init__.py
# ============================================================
# MetaTrader5 broker integration package.
# Submodules:
#   - mt5_connection  : MT5 terminal initialize / login / disconnect
#   - mt5_data         : Tick + multi-timeframe candle data feed
#   - account_manager  : Symbol resolution, market status, trade permission
#   - order_manager    : Market / limit / modify / close order execution
#   - journal_bridge   : Mirror MT5 trades into the local DB (trades table)
#   - health_monitor   : Auto-reconnect watcher for the MT5 connection
#   - safety_guard     : Spread / news / off-hours broker-side protection
#   - spread_monitor   : Real-time spread surveillance + alert
#   - economic_calendar: High-impact news event schedule
#   - position_manager : Aggregated open-position view + breakeven/trailing
#   - symbol_manager   : Symbol availability + digits/contract-size lookup
#   - market_data_manager: Unified MT5 + cache data access layer
#   - data_validator   : Tick/candle sanity checks before strategy use
#   - mt5_data         : Legacy candle/tick helper (Day 32)
# ============================================================
