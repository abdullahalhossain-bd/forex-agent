"""
audit_duplicate_modules.py — Audit tool for duplicate module triplets

Identifies which variant of each duplicate module set is actually imported
by live runtime code, so you can safely delete the orphans.

Run:
    python scripts/audit_duplicate_modules.py

Safe to run anytime — read-only.
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Duplicate module groups (canonical listed FIRST)
DUPLICATE_GROUPS = {
    "confidence_calibrator": [
        "intelligence/confidence_calibrator.py",   # canonical (used by confluence_engine.py)
        "hybrid/confidence_calibrator.py",         # DEAD per obsolete.py
        "memory/confidence_calibrator.py",         # SUPERSEDED per obsolete.py
    ],
    "execution_router": [
        "execution/execution_router.py",           # canonical (used by core/trader.py)
        "hybrid/execution_router.py",              # DUPLICATE per obsolete.py
    ],
    "decision_validator": [
        "core/decision_validator.py",              # canonical (used by master_decision.py)
        "hybrid/decision_validator.py",            # DEAD per obsolete.py
    ],
}


def find_importers(target_module: str) -> list[str]:
    """Find all .py files in the repo that import the given module."""
    importers = []
    target = target_module.replace("/", ".").replace(".py", "")
    for py_file in ROOT.rglob("*.py"):
        rel = py_file.relative_to(ROOT)
        if str(rel).startswith("scripts/"):
            continue
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(target):
                importers.append(str(rel))
                break
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(target):
                        importers.append(str(rel))
                        break
    return sorted(set(importers))


def main():
    print("=" * 70)
    print("  DUPLICATE MODULE AUDIT")
    print("=" * 70)
    print()
    for group_name, modules in DUPLICATE_GROUPS.items():
        print(f"── {group_name} ({len(modules)} variants) ──")
        for i, mod_path in enumerate(modules):
            tag = "CANONICAL" if i == 0 else "ORPHAN    "
            module_dot = mod_path.replace("/", ".").replace(".py", "")
            full_path = ROOT / mod_path
            if not full_path.exists():
                print(f"  [{tag}] {mod_path}  (file not found — already deleted)")
                continue
            importers = find_importers(module_dot)
            print(f"  [{tag}] {mod_path}  ({len(full_path.read_text().splitlines())} lines)")
            if importers:
                for imp in importers:
                    print(f"           ← imported by {imp}")
            else:
                print(f"           ← zero importers")
        print()
    print("=" * 70)
    print("  SAFE-TO-DELETE candidates (orphans with zero importers):")
    print("=" * 70)
    for group_name, modules in DUPLICATE_GROUPS.items():
        for i, mod_path in enumerate(modules):
            if i == 0:
                continue  # skip canonical
            module_dot = mod_path.replace("/", ".").replace(".py", "")
            if not (ROOT / mod_path).exists():
                continue
            importers = find_importers(module_dot)
            if not importers:
                print(f"  - {mod_path}")
    print()


if __name__ == "__main__":
    main()
