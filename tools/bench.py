#!/usr/bin/env python3
"""Benchmark every module's bench cases: the kernel against the naive implementation.

    tools/bench.py                       every module, every bench case
    tools/bench.py --module orbits       one module
    tools/bench.py --case NAME           one case
    tools/bench.py --json                machine-readable

Naive time is measured on the full family when that finishes inside --naive-limit seconds;
otherwise it is measured on a prefix of members and scaled to the family size, and the row says
so. Kernel time is measured on the full family, single-threaded and with every core, and the
answers of both runs (and of naive, when it ran in full) must agree byte for byte.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))
import lemmakernel as lk  # noqa: E402
from tools import harness as H  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", action="append")
    ap.add_argument("--case", action="append")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--naive-limit", type=float, default=10.0, help="seconds; above this the naive time is extrapolated")
    ap.add_argument("--backend", default=None)
    a = ap.parse_args()
    threads = os.cpu_count() or 1
    rows = []
    for mod in H.modules():
        if a.module and mod.name not in a.module:
            continue
        ctx = lk.Context(a.backend)
        naive = mod.naive()
        for case in mod.cases(ctx, random.Random(1)):
            if case.bench is None or (a.case and case.name not in a.case):
                continue
            r = H.bench_case(mod, ctx, case, naive, a.naive_limit, threads)
            r["module"] = mod.name
            rows.append(r)
            if not a.json:
                print(H.format_bench_row(r, threads))
    if a.json:
        print(json.dumps(rows, indent=2))
    elif any(r["naive_extrapolated"] for r in rows):
        print("~ naive time extrapolated from a prefix of members")


if __name__ == "__main__":
    main()
