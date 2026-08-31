import sys
import os
import runpy
import traceback

ROOT = os.path.abspath(os.getcwd())

def norm(p):
    try:
        return os.path.normcase(os.path.abspath(p))
    except:
        return ""

def inside_root(p):
    p = norm(p)
    return p.startswith(norm(ROOT))

class TraceFinder:
    def __init__(self):
        self.seen = set()

    def trace(self, frame, event, arg):
        if event == "call":
            filename = frame.f_code.co_filename

            if filename and filename.endswith(".py") and inside_root(filename):
                self.seen.add(norm(filename))

        return self.trace

finder = TraceFinder()

sys.settrace(finder.trace)

try:
    runpy.run_path(os.path.join(ROOT, "main.py"), run_name="__main__")
except SystemExit:
    pass
except KeyboardInterrupt:
    pass
except Exception:
    traceback.print_exc()
finally:
    sys.settrace(None)

print("\n===== RUNTIME FILES =====")

for f in sorted(finder.seen):
    print(f)
