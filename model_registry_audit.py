"""
model_registry_audit.py — model registry health audit + safe prune.

AUDIT FIX (2026-08-26 overfitting/portability review):

  memory/ml_models/_registry.json had:
    - 171/197 version entries whose model_path points at the ORIGINAL
      author's Windows filesystem (D:\\Projects\\forex\\...) even though
      the repo was cloned elsewhere;
    - most shockingly, ZERO .pkl binaries are committed under
      memory/ml_models/ (only .pkl.meta stubs) — every XGBoost /
      RandomForest / CatBoost "champion" registered here has no actual
      binary in this repo. Only the 43 LSTM *.keras files really exist.
      ml/model_store.ModelStore._resolve_path() already degrades
      gracefully (returns None → loader skips), but nothing ever cleaned
      the registry, so dashboards/predictors keep consulting hundreds of
      phantom champions and humans trust metrics that were produced on
      the author's machine (and 119 entries carry EMPTY metrics anyway).

Usage:
    python model_registry_audit.py            # audit only (read-only)
    python model_registry_audit.py --prune    # backup + drop unresolvable versions

This tool is standalone (stdlib-only) so it never pulls heavy ML deps.
Prune ALWAYS writes a timestamped backup next to the registry first.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path, PurePosixPath, PureWindowsPath

BASE_DIR = Path(__file__).resolve().parent / "memory" / "ml_models"
REGISTRY_PATH = BASE_DIR / "_registry.json"


def resolve_model_path(raw_path: str) -> bool:
    """Mirror ModelStore._resolve_path() existence logic (stdlib-only)."""
    if not raw_path:
        return False
    # 1) new-style relative path
    if (BASE_DIR / raw_path).exists():
        return True
    # 2) legacy absolute path valid on THIS machine
    if Path(raw_path).exists():
        return True
    # 3) legacy absolute path from ANOTHER machine: recover last two parts
    parts: list[str] = []
    for flavor in (PureWindowsPath, PurePosixPath):
        try:
            p = flavor(raw_path)
            if len(p.parts) > 1:
                parts = list(p.parts)
                break
        except Exception:
            continue
    if len(parts) >= 2:
        return (BASE_DIR / parts[-2] / parts[-1]).exists()
    return False


def audit() -> dict:
    reg = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    models = reg.get("models", {})
    stats = {
        "keys": len(models),
        "versions_total": 0,
        "resolvable": 0,
        "missing_binary": 0,
        "empty_metrics": 0,
        "windows_style_paths": 0,
        "by_model_type": {},
    }
    detail = {}
    for key, entry in models.items():
        for v in entry.get("versions", []):
            stats["versions_total"] += 1
            mp = str(v.get("model_path", ""))
            ok = resolve_model_path(mp)
            empty_m = not v.get("metrics")
            win_p = ("D:\\" in mp or "C:\\" in mp or "\\" in mp)
            stats["resolvable" if ok else "missing_binary"] += 1
            if empty_m:
                stats["empty_metrics"] += 1
            if win_p:
                stats["windows_style_paths"] += 1
            mtype = key.rsplit("_", 1)[-1]
            stats["by_model_type"].setdefault(mtype, {"ok": 0, "missing": 0})
            stats["by_model_type"][mtype]["ok" if ok else "missing"] += 1
            detail.setdefault(key, []).append(
                {"version": v.get("version"), "ok": ok, "empty_metrics": empty_m}
            )
    return {"stats": stats, "detail": detail}


def prune(backup_first: bool = True) -> None:
    reg = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    models = reg.get("models", {})
    if backup_first:
        bak = REGISTRY_PATH.with_suffix(f".json.bak.{time.strftime('%Y%m%d_%H%M%S')}")
        shutil.copy2(REGISTRY_PATH, bak)
        print(f"[backup] {bak}")

    removed_versions = 0
    dropped_keys = []
    for key in list(models.keys()):
        kept = []
        for v in models[key].get("versions", []):
            if resolve_model_path(str(v.get("model_path", ""))):
                kept.append(v)
            else:
                removed_versions += 1
        if kept:
            models[key]["versions"] = kept
        else:
            dropped_keys.append(key)
            del models[key]

    tmp = REGISTRY_PATH.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(reg, f, indent=2, default=str)
    tmp.replace(REGISTRY_PATH)
    print(f"[prune] removed {removed_versions} phantom versions, "
          f"dropped {len(dropped_keys)} fully-empty keys")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prune", action="store_true",
                    help="backup registry, then drop versions whose binaries cannot be resolved")
    args = ap.parse_args()

    if not REGISTRY_PATH.exists():
        print(f"registry not found: {REGISTRY_PATH}")
        return 1

    report = audit()
    s = report["stats"]
    print(json.dumps(s, indent=2))

    healthy_types = {k: v for k, v in s["by_model_type"].items() if v["ok"] > 0}
    print("\nsummary:")
    print(f"  REAL binaries present for model types : {sorted(healthy_types)} or none"
          if healthy_types else "  NO model type has any resolvable binary")
    print(f"  phantom (binary-missing) versions     : {s['missing_binary']}/{s['versions_total']}")

    if args.prune:
        prune()
    else:
        print("\n(read-only audit — rerun with --prune to clean the registry)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
