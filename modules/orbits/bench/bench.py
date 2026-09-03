#!/usr/bin/env python3
"""orbits benchmark: the kernel against the naive implementation on the same requests.

    modules/orbits/bench/bench.py                 every case
    modules/orbits/bench/bench.py --case NAME     one case
    modules/orbits/bench/bench.py --json          machine-readable

Naive time is measured on the full family when that finishes inside --naive-limit seconds;
otherwise it is measured on a prefix of members (one orbit search each) and scaled to the family
size, and the row says so. Kernel time is measured on the full family, single-threaded and with
every core.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
import lemmakernel as lk  # noqa: E402

_spec = importlib.util.spec_from_file_location("orbits_naive", ROOT / "modules" / "orbits" / "naive" / "naive.py")
naive = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(naive)


def unit_vectors(p, n):
    return lk.matrix(p, [[int(i == j) for j in range(n)] for i in range(n)])


def dihedral(n):
    return [[(i + 1) % n for i in range(n)], [(-i) % n for i in range(n)]]


def hypercube(n):
    """Generators of the symmetry group of the n-cube on its 2^n vertices (bit strings)."""
    flip = [v ^ 1 for v in range(1 << n)]
    swap = [(v & ~3) | ((v & 1) << 1) | ((v >> 1) & 1) for v in range(1 << n)]
    cycle = [((v << 1) | (v >> (n - 1))) & ((1 << n) - 1) for v in range(1 << n)]
    return [flip, swap, cycle]


def companion(coeffs, p):
    """Companion matrix of the monic polynomial x^n + c_{n-1} x^{n-1} + ... + c_0, acting on rows:
    multiplication by x in the basis 1, x, ..., x^{n-1} of F_p[x]/(f)."""
    n = len(coeffs)
    m = [[0] * n for _ in range(n)]
    for i in range(n - 1):
        m[i][i + 1] = 1
    m[n - 1] = [(-c) % p for c in coeffs]
    return m


def frobenius(coeffs, p):
    """The map v -> v^p on F_p[x]/(f) in the same basis; with `companion` it generates the
    normaliser of the Singer cycle, of order n (p^n - 1)."""
    n = len(coeffs)
    x = companion(coeffs, p)
    rows, power = [], [1] + [0] * (n - 1)
    for _ in range(n):
        rows.append(power)
        for _ in range(p):
            power = [sum(power[i] * x[i][j] for i in range(n)) % p for j in range(n)]
    return rows


def cases(ctx):
    """name -> (family handle, op, reduction, args, what it answers)"""
    out = {}
    D24 = ctx.perms(24, dihedral(24))
    beads = ctx.subsets(unit_vectors(2, 24), 8)
    out["bracelets_24_8"] = (
        beads, "orbits.is_canonical", "count", {"group": D24},
        "bracelets with 8 black and 16 white beads: orbits of D_24 on 8-subsets of 24 positions")
    out["bracelets_stabilizers"] = (
        beads, "orbits.stabilizer_order", "histogram", {"group": D24},
        "how symmetric each such bracelet is: stabiliser orders in D_24")
    out["bracelets_burnside"] = (
        ctx.group_elements(D24), "orbits.fixed_points", "all", {"on": beads},
        "fixed 8-subsets of every element of D_24 (Burnside count of the bracelets)")
    out["cube5_triples"] = (
        ctx.subsets(unit_vectors(2, 32), 3), "orbits.is_canonical", "hits", {"group": ctx.perms(32, hypercube(5)), "limit": 8},
        "3-vertex configurations of the 5-cube up to symmetry (group of order 3840)")
    f7 = [1, 1, 0, 0, 0, 0, 0]  # x^7 + x + 1, primitive over F_2
    singer7 = lk.matrix(2, [companion(f7, 2), frobenius(f7, 2)])  # order 7 * 127 = 889
    out["pg62_planes_singer"] = (
        ctx.grassmannian(2, 7, 3), "orbits.orbit_size", "histogram", {"group": singer7},
        "orbits of the Singer normaliser of GL(7,2) (order 889) on the 11811 planes of PG(6,2)")
    f4 = [1, 1, 0, 0]  # x^4 + x + 1, primitive over F_2
    singer4 = lk.matrix(2, [companion(f4, 2), frobenius(f4, 2)])  # order 60
    out["all_4x4_f2_singer_normaliser"] = (
        ctx.all_matrices(2, 4, 4), "orbits.canonical_index", "histogram", {"group": singer4},
        "every 4x4 matrix over F_2 under right multiplication by a group of order 60: orbit representatives by index")
    return out


def time_kernel(ctx, fam, op, red, args, threads):
    ctx.threads = threads
    t = time.perf_counter()
    h = ctx.run(op, fam, red, **args)
    dt = time.perf_counter() - t
    return dt, h


def naive_args(args):
    return {k: (v.value() if isinstance(v, lk.Handle) else v) for k, v in args.items()}


def time_naive(desc, op, red, args, size, limit):
    """Full naive run when affordable; otherwise a prefix of orbit searches scaled up."""
    args = naive_args(args)
    prefix = 200
    t = time.perf_counter()
    if op == "orbits.fixed_points":
        on = args["on"]
        members, p = naive.gfp.members(on)
        ks = naive.keys(on, members)
        (gens,) = desc.children
        elements = naive.perm_closure(gens.tolist())
        for g in elements[:prefix]:
            sum(naive.act(on, g, m, p) == m for m in ks)
    else:
        members, p = naive.gfp.members(desc)
        ks = naive.keys(desc, members)
        rank = naive.ranking(ks)
        gens = naive.generators_of(args["group"])
        for i in range(min(prefix, size)):
            naive.orbit(desc, gens, ks, rank, i, p)
    per = (time.perf_counter() - t) / min(prefix, size)
    if per * size <= limit:
        t = time.perf_counter()
        result = naive.run(op, desc, red, **args)
        return time.perf_counter() - t, False, result
    return per * size, True, None


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
            print(f"{name:32s} {size:>12,d} members  naive {naive_s:9.2f}s{'~' if extrapolated else ' '}  "
                  f"kernel x1 {one_s:8.3f}s  x{ncpu} {all_s:8.3f}s  speedup {r['speedup_1_thread']:9.0f}x / {r['speedup_all_threads']:9.0f}x")
            print(f"    {what}")
            print(f"    {r['answer']}")
    if a.json:
        print(json.dumps(rows, indent=2))
    elif any(r["naive_extrapolated"] for r in rows):
        print("~ naive time extrapolated from a prefix of members")


if __name__ == "__main__":
    main()
