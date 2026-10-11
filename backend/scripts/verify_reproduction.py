"""Compare a freshly generated results directory with the committed ``results/`` files.

    QVK_RESULTS_DIR=/tmp/fresh python -m scripts.run_experiments forecast --seeds 5   # etc.
    python -m scripts.verify_reproduction /tmp/fresh

Wall-clock quantities (latency, generation time) are ignored; every other number must agree to 1e-9.
Exit status is non-zero if anything differs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from app import config as C

IGNORE = ("latency_ms", "generation_seconds")


def diff(a, b, path="", out=None):
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if any(t in k for t in IGNORE):
                continue
            if k not in a or k not in b:
                out.append(f"missing key {path}/{k}")
            else:
                diff(a[k], b[k], f"{path}/{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"length {path}: {len(a)} vs {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                diff(x, y, f"{path}[{i}]", out)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if abs(a - b) > 1e-9 * max(1.0, abs(a)):
            out.append(f"{path}: {a} vs {b}")
    elif a != b:
        out.append(f"{path}: {a!r} vs {b!r}")
    return out


def main() -> int:
    fresh = Path(sys.argv[1])
    bad = 0
    for name in ("simulator", "forecast", "kernel", "noise"):
        f = fresh / f"{name}.json"
        if not f.exists():
            print(f"[skip] {name}.json not in {fresh}")
            continue
        d = diff(json.loads(f.read_text()), json.loads((C.RESULTS_DIR / f"{name}.json").read_text()))
        print(f"[{'OK' if not d else 'DIFF'}] {name}.json" + ("" if not d else f": {len(d)} differences, first: {d[0]}"))
        bad += bool(d)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
