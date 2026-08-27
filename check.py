import os, json, math
from datetime import datetime, timezone
import MetaTrader5 as mt5
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ---------- AGENT WHITELIST ----------
WHITELIST = [
    # majors
    "EURUSDm","GBPUSDm","USDJPYm","AUDUSDm","NZDUSDm","USDCADm","USDCHFm",
    # crosses
    "EURGBPm","EURJPYm","GBPJPYm","AUDJPYm","CADJPYm","NZDJPYm","CHFJPYm",
    "EURAUDm","EURCADm","GBPCADm","AUDCADm","NZDCADm","GBPAUDm",
    "EURNZDm","GBPNZDm","AUDNZDm",
    # metal
    "XAUUSDm",
]

def pip_size(info):
    return info.point * 10 if info.digits in (3, 5) else info.point

def tick_fresh(tick, max_age=300):
    """Market খোলা আছে কিনা — tick ৫ মিনিটের নতুন হতে হবে"""
    if tick is None or tick.time == 0 or tick.bid == 0:
        return False
    return (datetime.now(timezone.utc).timestamp() - tick.time) <= max_age

def collect():
    cfg, rows = {}, []
    for name in WHITELIST:
        if not mt5.symbol_select(name, True):
            continue
        info = mt5.symbol_info(name)
        if info is None:
            continue
        tick = mt5.symbol_info_tick(name)
        fresh = tick_fresh(tick)
        ps = pip_size(info)

        if fresh:
            spread_pts = (tick.ask - tick.bid) / info.point
        else:
            spread_pts = float(info.spread)
        spread_pips = spread_pts * info.point / ps

        broker_min_pips = info.trade_stops_level * info.point / ps
        safe_sl = max(broker_min_pips * 2, spread_pips * 15, 1.0)

        rows.append({
            "Symbol": name, "Digits": info.digits,
            "Fresh": "LIVE" if fresh else "STALE/CLOSED",
            "Spread(pips)": round(spread_pips, 2),
            "BrokerMinSL(pips)": round(broker_min_pips, 2),
            "SafeMinSLTP(pips)": math.ceil(safe_sl * 10) / 10,
            "MinLot": info.volume_min,
        })

        # ---- Agent config ----
        cfg[name] = {
            "digits": info.digits,
            "pip_size": ps,
            "live": fresh,
            "max_spread_at_entry": round(max(spread_pips * 1.5, spread_pips + 0.5), 1),
            "min_sl": math.ceil(safe_sl * 10) / 10,
            "min_tp": math.ceil(safe_sl * 10) / 10,
            "rr_min": 1.5,
            "max_open_positions_per_symbol": 1,
        }

    return pd.DataFrame(rows), cfg


def main():
    if not mt5.initialize(path=os.getenv("MT5_PATH"),
                          login=int(os.getenv("MT5_LOGIN")),
                          password=os.getenv("MT5_PASSWORD"),
                          server=os.getenv("MT5_SERVER")):
        print("❌", mt5.last_error()); return

    df, cfg = collect()
    print(df.to_string(index=False))

    with open("agent_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print("\n📁 agent_config.json saved — agent startup-এ এটা load করবে")

    mt5.shutdown()


if __name__ == "__main__":
    main()