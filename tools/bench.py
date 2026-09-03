#!/usr/bin/env python3
"""Benchmark every module's bench cases: the kernel against the naive implementation.

    tools/bench.py                       modules whose bench.json is missing or stale
    tools/bench.py --module orbits       one module (still skipped if current; add --force)
    tools/bench.py --force               rerun regardless of fingerprints
    tools/bench.py --status              list which modules are current or why they are not
    tools/bench.py --show                print every committed bench row without running anything

Each module's results live in `modules/NAME/bench.json` with a fingerprint of the module tree,
the module trees it includes, and the runtime. A module is rerun when any of those changed.
After running, `tools/manifest.py generate` rewrites BENCHMARKS.md from the records.

Naive time is measured on the full family when that finishes inside --naive-limit seconds;
otherwise it is measured on a prefix of members and scaled to the family size, and the row says
so. Kernel time is measured on the full family, single-threaded and with every core, and the
answers of both runs (and of naive, when it ran in full) must agree byte for byte.
"""
from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))
import lemmakernel as lk  # noqa: E402
from tools import harness as H  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", action="append")
    ap.add_argument("--case", action="append", help="run only these cases; the record is not written")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--naive-limit", type=float, default=10.0, help="seconds; above this the naive time is extrapolated")
    ap.add_argument("--backend", default=None)
    a = ap.parse_args()
    threads = os.cpu_count() or 1
    selected = [m for m in H.modules() if not a.module or m.name in a.module]

    if a.status or a.show:
        for mod in selected:
            why = H.bench_staleness(mod)
            rec = H.bench_record(mod)
            when = f" ({rec['measured_at']}, {rec['machine']['host']})" if rec else ""
            print(f"{mod.name:28s} {'current' if why is None else 'stale: ' + why}{when}")
            if a.show and rec:
                for r in rec["rows"]:
                    print("  " + H.format_bench_row(r).replace("\n", "\n  "))
        return 0

    ran = 0
    for mod in selected:
        why = None if a.force or a.case else H.bench_staleness(mod)
        if why is None and not a.force and not a.case:
            print(f"{mod.name}: current, skipped")
            continue
        print(f"== {mod.name}" + (f" ({why})" if why else ""))
        ctx = lk.Context(a.backend)
        naive = mod.naive()
        rows = []
        for case in mod.cases(ctx, random.Random(1)):
            if case.bench is None or (a.case and case.name not in a.case):
                continue
            r = H.bench_case(mod, ctx, case, naive, a.naive_limit, threads)
            rows.append(r)
            print(H.format_bench_row(r))
        if a.case:
            continue
        path = H.write_bench_record(mod, rows, a.naive_limit, threads)
        ran += 1
        print(f"wrote {path.relative_to(ROOT)} ({len(rows)} row(s))")
    if ran:
        subprocess.run([sys.executable, str(ROOT / "tools" / "manifest.py"), "generate"], check=True)
    if any(r["naive_extrapolated"] for m in selected for r in (H.bench_record(m) or {"rows": []})["rows"]):
        print("~ naive time extrapolated from a prefix of members")
    return 0


if __name__ == "__main__":
    sys.exit(main())
