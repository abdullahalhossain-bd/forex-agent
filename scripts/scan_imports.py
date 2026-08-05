import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import pkgutil
import importlib

root = Path(PROJECT_ROOT)
packages = []
for p in root.iterdir():
    if p.is_dir() and (p / "__init__.py").exists():
        packages.append(p.name)

print('Found packages:', packages)
errors = []
for pkg in packages:
    try:
        importlib.import_module(pkg)
        print(f'OK: {pkg}')
    except Exception as e:
        print(f'ERROR importing {pkg}: {e.__class__.__name__}: {e}')
        errors.append((pkg, str(e)))

print('\nSummary:')
if not errors:
    print('All top-level packages imported successfully')
    sys.exit(0)
else:
    print(f'{len(errors)} package(s) failed to import:')
    for pkg, err in errors:
        print(f' - {pkg}: {err}')
    sys.exit(2)
