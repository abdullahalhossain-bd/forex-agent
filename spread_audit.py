from __future__ import annotations

import csv
import math
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import MetaTrader5 as mt5


# ============================================================
# CONFIG
# ============================================================

BROKER_SUFFIX = "m"

# Number of historical ticks requested per symbol.
TICK_COUNT = 5000

# How far back to search.
LOOKBACK_DAYS = 7

# Output
OUTPUT_DIR = Path("spread_audit_output")
CSV_FILE = OUTPUT_DIR / "spread_audit_results.csv"


# ============================================================
# 48 SYMBOLS
# ============================================================

_MAJOR_PAIRS = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "USDCHF",
    "USDCAD",
    "AUDUSD",
    "NZDUSD",
]

_DISABLED_SYMBOLS_REFERENCE = [

    # Minors / Crosses
    "EURGBP",
    "EURJPY",
    "EURCHF",
    "EURAUD",
    "EURCAD",
    "EURNZD",

    "GBPJPY",
    "GBPCHF",
    "GBPAUD",
    "GBPCAD",
    "GBPNZD",

    "AUDJPY",
    "AUDCHF",
    "AUDCAD",
    "AUDNZD",

    "NZDJPY",
    "NZDCHF",
    "NZDCAD",

    "CADJPY",
    "CADCHF",
    "CHFJPY",

    # Metals
    "XAUUSD",
    "XAGUSD",
    "XPTUSD",
    "XPDUSD",

    # Exotic
    "USDTRY",
    "USDZAR",

    # Additional Crosses
    "EURNOK",
    "EURSEK",
    "GBPSEK",
    "GBPNOK",

    "AUDSGD",
    "NZDSGD",
    "SGDJPY",
    "HKDJPY",
    "MXNJPY",

    # Asia Pacific
    "USDCNH",
    "USDHKD",
    "USDSGD",
    "USDMXN",
    "USDTHB",
]

BASE_SYMBOLS = list(
    dict.fromkeys(
        _MAJOR_PAIRS +
        _DISABLED_SYMBOLS_REFERENCE
    )
)


# ============================================================
# MT5 CONNECTION
# ============================================================

def connect_mt5():

    if not mt5.initialize():

        print(
            "[ERROR] MT5 initialization failed:",
            mt5.last_error()
        )

        return False

    terminal = mt5.terminal_info()
    account = mt5.account_info()

    print("=" * 78)
    print("MT5 CONNECTION")
    print("=" * 78)

    if terminal:
        print(
            f"Terminal : {terminal.name}"
        )

    if account:
        print(
            f"Account  : {account.login}"
        )

        print(
            f"Server   : {account.server}"
        )

    print()

    return True


# ============================================================
# SYMBOL RESOLUTION
# ============================================================

def resolve_symbol(base_symbol: str) -> Optional[str]:

    candidates = []

    if BROKER_SUFFIX:
        candidates.append(
            base_symbol + BROKER_SUFFIX
        )

    candidates.append(base_symbol)

    # Exact lookup
    for candidate in candidates:

        info = mt5.symbol_info(candidate)

        if info is not None:

            mt5.symbol_select(
                candidate,
                True
            )

            return candidate

    # Fallback search
    symbols = mt5.symbols_get()

    if not symbols:
        return None

    base_upper = base_symbol.upper()

    matches = [
        item.name
        for item in symbols
        if item.name.upper().startswith(
            base_upper
        )
    ]

    if not matches:
        return None

    # Prefer suffix
    suffix_upper = BROKER_SUFFIX.upper()

    for name in matches:

        if (
            suffix_upper
            and name.upper()
            == base_upper + suffix_upper
        ):
            mt5.symbol_select(
                name,
                True
            )

            return name

    name = matches[0]

    mt5.symbol_select(
        name,
        True
    )

    return name


# ============================================================
# SYMBOL SPEC
# ============================================================

def get_symbol_spec(symbol: str):

    info = mt5.symbol_info(symbol)

    if info is None:
        return None

    return {
        "symbol": info.name,
        "digits": info.digits,
        "point": info.point,
        "trade_tick_size":
            getattr(
                info,
                "trade_tick_size",
                None
            ),
        "trade_tick_value":
            getattr(
                info,
                "trade_tick_value",
                None
            ),
        "contract_size":
            getattr(
                info,
                "trade_contract_size",
                None
            ),
        "volume_min":
            getattr(
                info,
                "volume_min",
                None
            ),
        "volume_max":
            getattr(
                info,
                "volume_max",
                None
            ),
        "volume_step":
            getattr(
                info,
                "volume_step",
                None
            ),
        "currency_base":
            getattr(
                info,
                "currency_base",
                None
            ),
        "currency_profit":
            getattr(
                info,
                "currency_profit",
                None
            ),
        "currency_margin":
            getattr(
                info,
                "currency_margin",
                None
            ),
    }


# ============================================================
# HISTORICAL TICKS
# ============================================================

def get_historical_ticks(
    symbol: str,
    count: int = TICK_COUNT,
):

    utc_to = datetime.now(
        timezone.utc
    )

    utc_from = utc_to - timedelta(
        days=LOOKBACK_DAYS
    )

    # COPY_TICKS_ALL includes:
    # bid
    # ask
    # last
    # volume
    # time
    # flags

    ticks = mt5.copy_ticks_range(
        symbol,
        utc_from,
        utc_to,
        mt5.COPY_TICKS_ALL
    )

    if ticks is None:

        print(
            f"    [ERROR] "
            f"copy_ticks_range failed: "
            f"{mt5.last_error()}"
        )

        return None

    if len(ticks) == 0:

        return []

    # We only need the latest N ticks.
    if len(ticks) > count:

        ticks = ticks[-count:]

    return ticks


# ============================================================
# SPREAD EXTRACTION
# ============================================================

def extract_spreads(
    symbol: str,
    ticks,
):

    info = mt5.symbol_info(symbol)

    if info is None:
        return []

    point = info.point

    if point <= 0:
        return []

    results = []

    for tick in ticks:

        bid = float(tick["bid"])
        ask = float(tick["ask"])

        if bid <= 0 or ask <= 0:
            continue

        if ask < bid:
            continue

        raw_spread = ask - bid

        spread_points = (
            raw_spread / point
        )

        if (
            not math.isfinite(
                spread_points
            )
            or spread_points < 0
        ):
            continue

        results.append(
            {
                "time": int(
                    tick["time"]
                ),
                "bid": bid,
                "ask": ask,
                "spread_price":
                    raw_spread,
                "spread_points":
                    spread_points,
            }
        )

    return results


# ============================================================
# PERCENTILE
# ============================================================

def percentile(
    values,
    p
):

    if not values:
        return math.nan

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    index = (
        len(values) - 1
    ) * p

    lower = math.floor(index)
    upper = math.ceil(index)

    if lower == upper:
        return values[lower]

    return (
        values[lower]
        + (
            values[upper]
            - values[lower]
        )
        * (
            index - lower
        )
    )


# ============================================================
# STATISTICS
# ============================================================

def calculate_statistics(
    samples
):

    if not samples:
        return {}

    values = [
        x["spread_points"]
        for x in samples
    ]

    return {
        "samples": len(values),

        "min":
            min(values),

        "median":
            statistics.median(values),

        "mean":
            statistics.mean(values),

        "p75":
            percentile(
                values,
                0.75
            ),

        "p90":
            percentile(
                values,
                0.90
            ),

        "p95":
            percentile(
                values,
                0.95
            ),

        "p99":
            percentile(
                values,
                0.99
            ),

        "max":
            max(values),
    }


# ============================================================
# LIVE SPREAD
# ============================================================

def get_live_spread(symbol):

    tick = mt5.symbol_info_tick(
        symbol
    )

    if tick is None:
        return None

    info = mt5.symbol_info(
        symbol
    )

    if info is None:
        return None

    if (
        tick.bid <= 0
        or tick.ask <= 0
        or info.point <= 0
    ):
        return None

    raw = (
        tick.ask - tick.bid
    )

    points = (
        raw / info.point
    )

    return {
        "bid": tick.bid,
        "ask": tick.ask,
        "spread_price": raw,
        "spread_points": points,
    }


# ============================================================
# REPORT
# ============================================================

def print_report(
    base_symbol,
    actual_symbol,
    spec,
    stats,
    live,
):

    print()
    print("-" * 78)

    print(
        f"{base_symbol} -> {actual_symbol}"
    )

    print("-" * 78)

    print(
        f"Point              : "
        f"{spec['point']}"
    )

    print(
        f"Digits             : "
        f"{spec['digits']}"
    )

    print(
        f"Tick size          : "
        f"{spec['trade_tick_size']}"
    )

    print(
        f"Contract size      : "
        f"{spec['contract_size']}"
    )

    print()

    print(
        "[HISTORICAL SPREAD]"
    )

    for key in [
        "samples",
        "min",
        "median",
        "mean",
        "p75",
        "p90",
        "p95",
        "p99",
        "max",
    ]:

        value = stats.get(key)

        if isinstance(
            value,
            float
        ):

            print(
                f"{key:18} : "
                f"{value:.4f} points"
            )

        else:

            print(
                f"{key:18} : "
                f"{value}"
            )

    print()

    print(
        "[LIVE]"
    )

    if live:

        print(
            f"Bid                : "
            f"{live['bid']}"
        )

        print(
            f"Ask                : "
            f"{live['ask']}"
        )

        print(
            f"Spread price       : "
            f"{live['spread_price']}"
        )

        print(
            f"Spread points      : "
            f"{live['spread_points']:.4f}"
        )

        if stats.get("p95", 0) > 0:

            ratio = (
                live["spread_points"]
                / stats["p95"]
            )

            print(
                f"Live / P95         : "
                f"{ratio:.3f}x"
            )

    else:

        print(
            "No live tick."
        )


# ============================================================
# CSV
# ============================================================

def save_csv(results):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    fields = [
        "base_symbol",
        "actual_symbol",

        "digits",
        "point",
        "trade_tick_size",
        "trade_tick_value",
        "contract_size",

        "volume_min",
        "volume_max",
        "volume_step",

        "currency_base",
        "currency_profit",
        "currency_margin",

        "samples",

        "spread_min_points",
        "spread_median_points",
        "spread_mean_points",
        "spread_p75_points",
        "spread_p90_points",
        "spread_p95_points",
        "spread_p99_points",
        "spread_max_points",

        "live_bid",
        "live_ask",
        "live_spread_price",
        "live_spread_points",
        "live_vs_p95",

        "audit_time_utc",
    ]

    with CSV_FILE.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()

        for row in results:

            writer.writerow(row)

    print()
    print("=" * 78)
    print(
        "CSV SAVED:"
    )
    print(
        CSV_FILE.resolve()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print("MT5 HISTORICAL SPREAD AUDIT")
    print("=" * 78)

    print(
        f"Pairs        : "
        f"{len(BASE_SYMBOLS)}"
    )

    print(
        f"Tick target  : "
        f"{TICK_COUNT:,}"
    )

    print(
        f"Lookback     : "
        f"{LOOKBACK_DAYS} days"
    )

    print()

    if len(BASE_SYMBOLS) != 48:

        print(
            "[WARNING] "
            f"Expected 48 pairs, "
            f"found {len(BASE_SYMBOLS)}"
        )

    if not connect_mt5():
        return

    # --------------------------------------------------------
    # Resolve
    # --------------------------------------------------------

    print(
        "=" * 78
    )

    print(
        "SYMBOL RESOLUTION"
    )

    print(
        "=" * 78
    )

    resolved = []

    for base in BASE_SYMBOLS:

        actual = resolve_symbol(
            base
        )

        if actual:

            print(
                f"  OK   "
                f"{base:10} -> "
                f"{actual}"
            )

            resolved.append(
                (
                    base,
                    actual
                )
            )

        else:

            print(
                f"  FAIL "
                f"{base:10} -> "
                f"NOT FOUND"
            )

    print()

    print(
        f"Resolved: "
        f"{len(resolved)}/"
        f"{len(BASE_SYMBOLS)}"
    )

    print()

    results = []

    # --------------------------------------------------------
    # Audit
    # --------------------------------------------------------

    total = len(resolved)

    for index, (
        base,
        actual,
    ) in enumerate(
        resolved,
        start=1
    ):

        print()
        print(
            "=" * 78
        )

        print(
            f"AUDIT "
            f"{index}/{total} "
            f"| {base} -> {actual}"
        )

        print(
            "=" * 78
        )

        spec = get_symbol_spec(
            actual
        )

        if spec is None:

            print(
                "[ERROR] "
                "No symbol specification."
            )

            continue

        # Historical ticks
        print(
            "Loading historical ticks..."
        )

        ticks = get_historical_ticks(
            actual,
            TICK_COUNT
        )

        if ticks is None:

            print(
                "[ERROR] "
                "Historical tick request failed."
            )

            continue

        print(
            f"Historical ticks received: "
            f"{len(ticks):,}"
        )

        # Spread
        samples = extract_spreads(
            actual,
            ticks
        )

        if not samples:

            print(
                "[WARNING] "
                "No valid bid/ask spread data."
            )

            continue

        print(
            f"Valid spread samples: "
            f"{len(samples):,}"
        )

        # Stats
        stats = calculate_statistics(
            samples
        )

        # Live
        live = get_live_spread(
            actual
        )

        # Report
        print_report(
            base,
            actual,
            spec,
            stats,
            live
        )

        # Live / P95
        live_vs_p95 = None

        if (
            live
            and stats.get("p95", 0) > 0
        ):

            live_vs_p95 = (
                live["spread_points"]
                / stats["p95"]
            )

        # CSV
        results.append(
            {
                "base_symbol": base,
                "actual_symbol": actual,

                "digits":
                    spec["digits"],

                "point":
                    spec["point"],

                "trade_tick_size":
                    spec["trade_tick_size"],

                "trade_tick_value":
                    spec["trade_tick_value"],

                "contract_size":
                    spec["contract_size"],

                "volume_min":
                    spec["volume_min"],

                "volume_max":
                    spec["volume_max"],

                "volume_step":
                    spec["volume_step"],

                "currency_base":
                    spec["currency_base"],

                "currency_profit":
                    spec["currency_profit"],

                "currency_margin":
                    spec["currency_margin"],

                "samples":
                    stats.get("samples"),

                "spread_min_points":
                    stats.get("min"),

                "spread_median_points":
                    stats.get("median"),

                "spread_mean_points":
                    stats.get("mean"),

                "spread_p75_points":
                    stats.get("p75"),

                "spread_p90_points":
                    stats.get("p90"),

                "spread_p95_points":
                    stats.get("p95"),

                "spread_p99_points":
                    stats.get("p99"),

                "spread_max_points":
                    stats.get("max"),

                "live_bid":
                    live["bid"]
                    if live else None,

                "live_ask":
                    live["ask"]
                    if live else None,

                "live_spread_price":
                    live["spread_price"]
                    if live else None,

                "live_spread_points":
                    live["spread_points"]
                    if live else None,

                "live_vs_p95":
                    live_vs_p95,

                "audit_time_utc":
                    datetime.now(
                        timezone.utc
                    ).isoformat(),
            }
        )

        # Overall progress
        percent = (
            index / total
        ) * 100

        print()
        print(
            f"OVERALL PROGRESS: "
            f"{index}/{total} "
            f"({percent:.1f}%)"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    if results:

        save_csv(
            results
        )

    print()
    print("=" * 78)
    print("AUDIT COMPLETE")
    print("=" * 78)

    print(
        f"Successful: "
        f"{len(results)}/{total}"
    )

    print(
        f"CSV: "
        f"{CSV_FILE.resolve()}"
    )

    mt5.shutdown()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()