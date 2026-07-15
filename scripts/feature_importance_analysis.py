#!/usr/bin/env python3
"""
scripts/feature_importance_analysis.py — Day 97+ Feature Importance Analyzer
==============================================================================
Analyzes which of the 100+ features are actually predictive of trade outcomes.

Uses 3 methods:
  1. Correlation with outcome (quick, linear)
  2. Mutual information (non-linear relationships)
  3. Permutation importance via Random Forest (most reliable)

Output:
  - Ranked list of top 20 features by importance
  - Recommendation: which features to drop (importance < threshold)
  - Multicollinearity check (features that are redundant)

Usage:
    python scripts/feature_importance_analysis.py

Reads from: database/trader.db (analysis table with context_json)
            memory/decision_history.jsonl (trade outcomes)
"""

import sys
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learn not installed — install with: pip install scikit-learn")
    sys.exit(1)


def load_trade_features() -> pd.DataFrame:
    """Load trade features + outcomes from DB + decision history.

    Returns DataFrame with:
      - feature columns (from context_json in analysis table)
      - 'outcome' column: 1 = WIN, 0 = LOSS
    """
    db_path = ROOT / "database" / "trader.db"
    if not db_path.exists():
        print(f"❌ DB not found: {db_path}")
        sys.exit(1)

    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        # Get trades with outcomes
        rows = conn.execute("""
            SELECT t.pair, t.type, t.result, t.pnl, t.confidence,
                   t.pattern, t.regime, t.trend, t.rsi, t.session,
                   t.context_json, t.open_time
            FROM trades t
            WHERE t.result IN ('WIN', 'LOSS')
            ORDER BY t.open_time DESC
            LIMIT 500
        """).fetchall()

    if not rows:
        print("❌ No trades with outcomes (WIN/LOSS) found in DB")
        print("   Need at least 50 closed trades to run feature importance analysis")
        sys.exit(1)

    records = []
    for row in rows:
        ctx = {}
        if row["context_json"]:
            try:
                ctx = json.loads(row["context_json"])
            except Exception:
                pass

        record = {
            "pair": row["pair"],
            "type": row["type"],
            "result": row["result"],
            "pnl": row["pnl"],
            "confidence": row["confidence"],
            "pattern": row["pattern"],
            "regime": row["regime"],
            "trend": row["trend"],
            "rsi": row["rsi"],
            "session": row["session"],
        }
        # Flatten context_json features
        for k, v in ctx.items():
            if isinstance(v, (int, float)):
                record[f"ctx_{k}"] = v
        records.append(record)

    df = pd.DataFrame(records)
    df["outcome"] = (df["result"] == "WIN").astype(int)

    # Encode categorical features
    for col in ("pair", "type", "pattern", "regime", "trend", "session"):
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Drop non-numeric / target columns
    feature_cols = [c for c in df.columns if c not in ("result", "outcome", "context_json", "open_time")]
    X = df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
    y = df["outcome"]

    print(f"✅ Loaded {len(df)} trades ({y.sum()} wins, {len(y) - y.sum()} losses)")
    print(f"   Features: {len(X.columns)}")
    return X, y


def analyze_correlation(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Method 1: Pearson correlation with outcome."""
    correlations = {}
    for col in X.columns:
        try:
            corr = abs(np.corrcoef(X[col], y)[0, 1])
            if not np.isnan(corr):
                correlations[col] = corr
        except Exception:
            pass
    return pd.Series(correlations).sort_values(ascending=False)


def analyze_mutual_info(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Method 2: Mutual information (captures non-linear relationships)."""
    try:
        mi = mutual_info_classif(X, y, random_state=42, discrete_features=False)
        return pd.Series(dict(zip(X.columns, mi))).sort_values(ascending=False)
    except Exception as e:
        print(f"⚠️ Mutual info failed: {e}")
        return pd.Series()


def analyze_permutation_importance(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Method 3: Random Forest permutation importance."""
    try:
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        importances = pd.Series(rf.feature_importances_, index=X.columns)
        return importances.sort_values(ascending=False)
    except Exception as e:
        print(f"⚠️ RF importance failed: {e}")
        return pd.Series()


def check_multicollinearity(X: pd.DataFrame, threshold: float = 0.85) -> list:
    """Find pairs of features that are highly correlated (redundant)."""
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    redundant = []
    for col in upper.columns:
        high_corr = upper[col][upper[col] > threshold].index.tolist()
        for h in high_corr:
            redundant.append((col, h, upper[col][h]))
    return redundant


def main():
    print("═" * 60)
    print("  🔬  FEATURE IMPORTANCE ANALYSIS  (Day 97+)")
    print("═" * 60)
    print()

    X, y = load_trade_features()

    if len(X) < 30:
        print(f"⚠️ Only {len(X)} trades — results will be noisy. Need 50+ for reliability.")
    if len(X) < 10:
        print("❌ Not enough data. Run more trades first.")
        return

    print()
    print("── Method 1: Correlation with Outcome ──")
    corr = analyze_correlation(X, y)
    print("Top 15 features by |correlation|:")
    for feat, val in corr.head(15).items():
        bar = "█" * int(val * 50)
        print(f"  {feat:<30} {val:.4f} {bar}")
    print()

    print("── Method 2: Mutual Information ──")
    mi = analyze_mutual_info(X, y)
    if len(mi) > 0:
        print("Top 15 features by mutual information:")
        for feat, val in mi.head(15).items():
            bar = "█" * int(val * 50)
            print(f"  {feat:<30} {val:.4f} {bar}")
    print()

    print("── Method 3: Random Forest Importance ──")
    rf_imp = analyze_permutation_importance(X, y)
    if len(rf_imp) > 0:
        print("Top 15 features by RF importance:")
        for feat, val in rf_imp.head(15).items():
            bar = "█" * int(val * 100)
            print(f"  {feat:<30} {val:.4f} {bar}")
    print()

    print("── Multicollinearity Check (redundant features) ──")
    redundant = check_multicollinearity(X, threshold=0.85)
    if redundant:
        print(f"Found {len(redundant)} highly-correlated pairs (>|0.85|):")
        for f1, f2, corr in redundant[:10]:
            print(f"  {f1} ↔ {f2}  (r={corr:.2f}) — drop one")
    else:
        print("No highly-correlated pairs found (good).")
    print()

    # Recommendation
    print("── Recommendation ──")
    if len(rf_imp) > 0:
        top_features = rf_imp.head(15).index.tolist()
        drop_features = rf_imp[rf_imp < 0.01].index.tolist()
        print(f"  ✅ KEEP (top 15 by RF importance): {', '.join(top_features[:10])}...")
        print(f"  ❌ DROP (importance < 0.01): {len(drop_features)} features")
        if drop_features:
            print(f"     {', '.join(drop_features[:5])}...")
        print()
        print(f"  Current: {len(X.columns)} features")
        print(f"  Recommended: {len(top_features)} features (simplify by {len(X.columns) - len(top_features)})")
    print()

    print("═" * 60)
    print("  💡 This analysis should be re-run after every 50 new trades")
    print("  to detect feature drift and adapt the model.")
    print("═" * 60)


if __name__ == "__main__":
    main()
