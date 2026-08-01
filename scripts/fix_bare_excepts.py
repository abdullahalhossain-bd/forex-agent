#!/usr/bin/env python3
"""
scripts/fix_bare_excepts.py
===========================
Fix all bare `except:` clauses by replacing with `except Exception:`.
Bare except catches KeyboardInterrupt + SystemExit, which prevents
Ctrl+C from working and can leave the system in an inconsistent state.
"""
import re
from pathlib import Path

fixes = []

# Files to scan
scan_dirs = ['analysis', 'backtest', 'risk', 'agents', 'core', 'data']
project_root = Path('/home/z/my-project/forex_ai')

for d in scan_dirs:
    p = project_root / d
    if not p.exists():
        continue
    for f in p.rglob('*.py'):
        if '__pycache__' in str(f):
            continue
        try:
            content = f.read_text()
            original = content
            # Pattern: bare `except:` (not `except Exception:` or `except SomeError:`)
            # Match `except:` followed by newline or whitespace
            new_content = re.sub(
                r'\bexcept\s*:',
                'except Exception:',
                content
            )
            if new_content != original:
                f.write_text(new_content)
                count = original.count('except:') - new_content.count('except:')
                # Actually count replacements
                old_count = len(re.findall(r'\bexcept\s*:', original))
                new_count = len(re.findall(r'\bexcept\s*:', new_content))
                replaced = old_count - new_count
                fixes.append((str(f).replace(str(project_root) + '/', ''), replaced))
        except Exception as e:
            print(f"  Error processing {f}: {e}")

print(f"Fixed {len(fixes)} files:")
for f, n in fixes:
    print(f"  {f}: {n} bare except → except Exception")

total = sum(n for _, n in fixes)
print(f"\nTotal bare except clauses fixed: {total}")
