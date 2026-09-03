"""The obvious implementation of the gfp module, in plain Python.

Every family is materialised member by member in its canonical order; every operation is
Gauss-Jordan elimination from scratch on that member. Nothing is shared, nothing is pruned.

This is what the fast backends are tested against, byte for byte on the interchange encoding,
and what the benchmark reports the speed-up against. Keep it readable before keeping it quick.

    naive.run(op, family, reduction, **args) -> interchange object

`family` is a lemmakernel.interchange.Family (what a family handle exports).
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))
from lemmakernel import naive as rt  # noqa: E402  (families and reductions, shared by every module)
from lemmakernel.interchange import Basis, Family, Inverses, Matrix, Solutions, Witness  # noqa: E402


# ---- arithmetic ---------------------------------------------------------------------------------

def rref_with_transform(rows: list[list[int]], p: int):
    """Gauss-Jordan with first-nonzero-row pivoting. Returns (R, T, pivots) with T*A = R."""
    m = len(rows)
    n = len(rows[0]) if m else 0
    R = [list(r) for r in rows]
    T = [[int(i == j) for j in range(m)] for i in range(m)]
    pivots = []
    r = 0
    for c in range(n):
        if r == m:
            break
        pr = next((i for i in range(r, m) if R[i][c]), None)
        if pr is None:
            continue
        R[r], R[pr] = R[pr], R[r]
        T[r], T[pr] = T[pr], T[r]
        inv = pow(R[r][c], p - 2, p)
        R[r] = [(x * inv) % p for x in R[r]]
        T[r] = [(x * inv) % p for x in T[r]]
        for i in range(m):
            if i != r and R[i][c]:
                f = R[i][c]
                R[i] = [(a - f * b) % p for a, b in zip(R[i], R[r])]
                T[i] = [(a - f * b) % p for a, b in zip(T[i], T[r])]
        pivots.append(c)
        r += 1
    return R, T, pivots


def rref(rows, p):
    R, _, piv = rref_with_transform(rows, p)
    return R, piv


def rank(rows, p):
    return len(rref(rows, p)[1])


def nullspace(rows, p):
    """Canonical basis: one vector per free column f, x_f = 1, x_pivot_i = -R[i][f]."""
    R, piv = rref(rows, p)
    n = len(rows[0])
    out = []
    for f in range(n):
        if f in piv:
            continue
        v = [0] * n
        v[f] = 1
        for i, pc in enumerate(piv):
            v[pc] = (-R[i][f]) % p
        out.append(v)
    return out


def in_span(rows, target, p):
    return rank(rows, p) == rank(rows + [target], p)


def solve(rows, rhs, p):
    """x with A x^T = rhs^T, free coordinates zero; None when inconsistent."""
    aug = [r + [b] for r, b in zip(rows, rhs)]
    n = len(rows[0])
    R, piv = rref(aug, p)
    if n in piv:
        return None
    x = [0] * n
    for i, pc in enumerate(piv):
        x[pc] = R[i][n]
    return x


def inverse(rows, p):
    n = len(rows)
    aug = [r + [int(i == j) for j in range(n)] for i, r in enumerate(rows)]
    R, piv = rref(aug, p)
    if piv != list(range(n)):
        return None
    return [r[n:] for r in R]


matmul = rt.matmul
members = rt.members  # kept for callers that materialise a family through this module


# ---- operations and reductions -------------------------------------------------------------------

def _flat(mats):
    return [x for m in mats for r in m for x in r]


def run(op: str, family: Family, reduction: str = "all", prefix: int | None = None, **args):
    """`prefix`: answer for the first `prefix` members only (the benchmark's timing sample)."""
    op = op.removeprefix("gfp.")
    ms, p = list(itertools.islice(rt.iter_members(family), prefix)), rt.prime(family)
    size = len(ms)
    rows = len(ms[0]) if ms else family.params.get("rows", 0)
    cols = len(ms[0][0]) if ms else family.params.get("cols", 0)

    if op in ("solve", "inverse", "rref_witness"):
        if family.kind != "explicit":
            raise ValueError(f"{op} is defined on explicit families only")
        if reduction != "all":
            raise ValueError(f"{op} values only reduce with `all`")
        if op == "rref_witness":
            Rs, Ts = [], []
            for m in ms:
                R, T, _ = rref_with_transform(m, p)
                Rs.append(R)
                Ts.append(T)
            return Witness(p, size, rows, cols, _flat(Rs), _flat(Ts))
        if op == "inverse":
            invs = [inverse(m, p) for m in ms]
            zero = [[0] * rows for _ in range(rows)]
            return Inverses(p, size, rows, [int(i is not None) for i in invs], _flat([i or zero for i in invs]))
        rhs: Matrix = args["rhs"]
        sols = [solve(m, rhs.member(i)[0], p) for i, m in enumerate(ms)]
        return Solutions(p, size, cols, [int(s is not None) for s in sols], [x for s in sols for x in (s or [0] * cols)])

    if op == "rref":
        return Matrix(p, size, rows, cols, _flat([rref(m, p)[0] for m in ms]))
    if op == "nullspace":
        vecs = [nullspace(m, p) for m in ms]
        offsets = [0]
        for v in vecs:
            offsets.append(offsets[-1] + len(v))
        return Basis(p, size, cols, offsets, _flat(vecs))

    if op in ("rank", "nullity"):
        values = [rank(m, p) if op == "rank" else cols - rank(m, p) for m in ms]
        return rt.reduce_int(reduction, values, ms, p)

    if op == "full_row_rank":
        flags = [rank(m, p) == rows for m in ms]
    elif op == "full_col_rank":
        flags = [rank(m, p) == cols for m in ms]
    elif op == "in_span":
        target = args["target"].member(0)[0]
        flags = [in_span(m, target, p) for m in ms]
    else:
        raise ValueError(f"unknown operation {op}")

    return rt.reduce_bool(reduction, flags, ms, p, **args)
