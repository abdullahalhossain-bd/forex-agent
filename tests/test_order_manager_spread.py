from broker.order_manager import _get_spread_limit_pips


def test_spread_limit_uses_symbol_specific_thresholds():
    assert _get_spread_limit_pips("EURUSD") == 3.0
    assert _get_spread_limit_pips("XAUUSD") == 50.0
    assert _get_spread_limit_pips("UNKNOWN") == 10.0


def test_spread_limit_falls_back_to_default_when_account_manager_is_unavailable():
    import broker.order_manager as order_manager

    original_module = order_manager._get_spread_limit_pips
    try:
        order_manager._get_spread_limit_pips = lambda symbol: 99.0
        assert order_manager._get_spread_limit_pips("EURUSD") == 99.0
    finally:
        order_manager._get_spread_limit_pips = original_module
