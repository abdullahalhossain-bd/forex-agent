from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent.parent / "data"
files = sorted([p for p in root.glob("*.csv") if p.is_file()])
print(f"Found {len(files)} CSV files")
issues = []
for p in files:
    try:
        df = pd.read_csv(p, encoding="utf-8-sig")
        cols = [str(c).strip().lower() for c in df.columns]
        if "time" not in cols and not any("date" in c or "time" in c for c in cols):
            issues.append((p.name, "missing time column", cols))
            continue
        if "time" not in cols:
            if "datetime" in cols:
                df = df.rename(columns={df.columns[cols.index("datetime")]: "time"})
            elif "datetime_utc" in cols:
                df = df.rename(columns={df.columns[cols.index("datetime_utc")]: "time"})
            elif "date" in cols:
                df = df.rename(columns={df.columns[cols.index("date")]: "time"})
            else:
                candidates = [c for c in cols if "date" in c or "time" in c]
                if len(candidates) == 1:
                    df = df.rename(columns={df.columns[cols.index(candidates[0])]: "time"})
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        missing_t = int(df["time"].isna().sum())
        dupes = int(df["time"].duplicated().sum())
        if missing_t or dupes:
            issues.append((p.name, "timestamp issues", missing_t, dupes))
        for col in ["open", "high", "low", "close"]:
            if col not in df.columns:
                issues.append((p.name, "missing column", col))
                continue
            n_m = int(df[col].isna().sum())
            if n_m:
                issues.append((p.name, f"missing values {col}", n_m))
            n_nonpos = int((df[col] <= 0).sum())
            if n_nonpos:
                issues.append((p.name, f"nonpositive {col}", n_nonpos))
        if all(c in df.columns for c in ["open", "high", "low", "close"]):
            bad = int(((df["high"] < df["low"]) | (df["high"] < df["open"]) | (df["high"] < df["close"]) | (df["low"] > df["open"]) | (df["low"] > df["close"])).sum())
            if bad:
                issues.append((p.name, "ohlc logic violations", bad))
    except Exception as e:
        issues.append((p.name, "load failed", repr(e)))

print(f"Issues found: {len(issues)}")
for issue in issues[:200]:
    print(issue)
