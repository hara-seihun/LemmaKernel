#!/usr/bin/env python3
"""gfp benchmark: the kernel against the naive implementation on the same requests.

    modules/gfp/bench/bench.py                 every case
    modules/gfp/bench/bench.py --case NAME     one case
    modules/gfp/bench/bench.py --json          machine-readable

Naive time is measured on the full family when that finishes inside --naive-limit seconds;
otherwise it is measured on a prefix and scaled to the family size, and the row says so.
Kernel time is measured on the full family, single-threaded and with every core.
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
import lemmakernel as lk  # noqa: E402

_spec = importlib.util.spec_from_file_location("gfp_naive", ROOT / "modules" / "gfp" / "naive" / "naive.py")
naive = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(naive)


def rand_batch(rng, p, count, rows, cols):
    return lk.matrix(p, [[[rng.randrange(p) for _ in range(cols)] for _ in range(rows)] for _ in range(count)])


def cases(ctx):
    """name -> (family handle, op, reduction, args, what it answers)"""
    rng = random.Random(7)
    out = {}
    out["subsets_independent"] = (
        ctx.subsets(rand_batch(rng, 2, 20, 1, 10), 6), "gfp.full_row_rank", "count", {},
        "how many 6-subsets of 20 vectors in F_2^10 are independent")
    out["grassmannian_image_rank"] = (
        ctx.transform(ctx.grassmannian(2, 8, 4), rand_batch(rng, 2, 1, 8, 6)), "gfp.rank", "histogram", {},
        "rank distribution of the image of every 4-subspace of F_2^8 under a fixed 8x6 map")
    out["explicit_rref"] = (
        ctx.explicit(rand_batch(rng, 7, 100_000, 6, 8)), "gfp.rref", "all", {},
        "rref of 100k random 6x8 matrices over F_7")
    out["explicit_inverse"] = (
        ctx.explicit(rand_batch(rng, 251, 50_000, 8, 8)), "gfp.inverse", "all", {},
        "inverse of 50k random 8x8 matrices over F_251")
    out["all_matrices_invertible"] = (
        ctx.all_matrices(3, 4, 4), "gfp.full_col_rank", "count", {},
        "|GL(4, F_3)| by counting every 4x4 matrix over F_3 (43 million members)")
    out["grassmannian_span_hits"] = (
        ctx.stack(ctx.grassmannian(2, 10, 4), rand_batch(rng, 2, 1, 2, 10)), "gfp.in_span", "hits",
        {"target": rand_batch(rng, 2, 1, 1, 10), "limit": 4},
        "which 4-subspaces of F_2^10, extended by two fixed rows, span a target vector (53 million members)")
    return out


def time_kernel(ctx, fam, op, red, args, threads):
    ctx.threads = threads
    t = time.perf_counter()
    h = ctx.run(op, fam, red, **args)
    dt = time.perf_counter() - t
    return dt, h


def time_naive(desc, op, red, args, size, limit):
    """Full naive run when affordable; otherwise a prefix scaled up."""
    prefix = 2000
    t = time.perf_counter()
    for m in itertools.islice(naive.iter_members(desc), prefix):
        _naive_one(m, op, args, naive.prime(desc))
    per = (time.perf_counter() - t) / min(prefix, size)
    if per * size <= limit:
        t = time.perf_counter()
        result = naive.run(op, desc, red, **args)
        return time.perf_counter() - t, False, result
    return per * size, True, None


def _naive_one(m, op, args, p):
    op = op.removeprefix("gfp.")
    if op in ("rank", "nullity", "full_row_rank", "full_col_rank"):
        return naive.rank(m, p)
    if op == "in_span":
        return naive.in_span(m, args["target"].member(0)[0], p)
    if op == "rref":
        return naive.rref(m, p)
    if op == "inverse":
        return naive.inverse(m, p)
    if op == "nullspace":
        return naive.nullspace(m, p)
    raise ValueError(op)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", action="append")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--naive-limit", type=float, default=20.0, help="seconds; above this the naive time is extrapolated")
    ap.add_argument("--backend", default=None)
    a = ap.parse_args()
    ctx = lk.Context(a.backend)
    ncpu = os.cpu_count() or 1
    rows = []
    for name, (fam, op, red, args, what) in cases(ctx).items():
        if a.case and name not in a.case:
            continue
        size = ctx.size(fam)
        desc = fam.value()
        naive_s, extrapolated, naive_result = time_naive(desc, op, red, args, size, a.naive_limit)
        one_s, h1 = time_kernel(ctx, fam, op, red, args, 1)
        all_s, hn = time_kernel(ctx, fam, op, red, args, ncpu)
        assert h1.export() == hn.export(), f"{name}: thread count changed the answer"
        if naive_result is not None:
            assert naive_result.encode() == h1.export(), f"{name}: kernel disagrees with naive"
        rows.append({
            "case": name, "what": what, "op": op, "reduction": red, "members": size,
            "naive_s": naive_s, "naive_extrapolated": extrapolated,
            "kernel_1_thread_s": one_s, f"kernel_{ncpu}_threads_s": all_s,
            "speedup_1_thread": naive_s / one_s, "speedup_all_threads": naive_s / all_s,
            "answer": repr(hn.value())[:120],
        })
        if not a.json:
            r = rows[-1]
            print(f"{name:28s} {size:>12,d} members  naive {naive_s:9.2f}s{'~' if extrapolated else ' '}  "
                  f"kernel x1 {one_s:8.3f}s  x{ncpu} {all_s:8.3f}s  speedup {r['speedup_1_thread']:9.0f}x / {r['speedup_all_threads']:9.0f}x")
            print(f"    {what}")
            print(f"    {r['answer']}")
    if a.json:
        print(json.dumps(rows, indent=2))
    elif any(r["naive_extrapolated"] for r in rows):
        print("~ naive time extrapolated from a 2000-member prefix")


if __name__ == "__main__":
    main()
