# Day 74 fix: lazy imports to avoid ta dependency at package level.
# Old eager imports caused ModuleNotFoundError when ta not installed.
# Now each module is imported on demand.
__all__ = ["BacktestEngine", "HistoricalDataLoader", "ForexSimulator", "BacktestReport", "load_data"]
