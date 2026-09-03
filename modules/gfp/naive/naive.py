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
from lemmakernel.interchange import (Basis, Count, Family, Histogram, Hits, Integers, Inverses, Matrix,  # noqa: E402
                                     Solutions, Witness)


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


def matmul(a, b, p):
    return [[sum(x * y for x, y in zip(row, col)) % p for col in zip(*b)] for row in a]


# ---- families -----------------------------------------------------------------------------------

def _batch_members(m: Matrix):
    return [m.member(i) for i in range(m.count)]


def grassmannian_members(p, n, h):
    """Every h-dim subspace of F_p^n as its rref basis; pivot sets lexicographic, then free
    entries row-major lexicographic. Yields lists of rows."""
    for piv in itertools.combinations(range(n), h):
        free = [(i, c) for i in range(h) for c in range(piv[i] + 1, n) if c not in piv]
        for digits in itertools.product(range(p), repeat=len(free)):
            rows = [[0] * n for _ in range(h)]
            for i, pc in enumerate(piv):
                rows[i][pc] = 1
            for (i, c), d in zip(free, digits):
                rows[i][c] = d
            yield rows


def iter_members(f: Family):
    """Yield the members of a family in canonical order as lists of rows."""
    if f.kind == "explicit":
        (batch,) = f.children
        yield from _batch_members(batch)
    elif f.kind == "subsets":
        (dictionary,) = f.children
        vecs = [m[0] for m in _batch_members(dictionary)] if dictionary.rows == 1 else dictionary.member(0)
        for idx in itertools.combinations(range(len(vecs)), f.params["k"]):
            yield [vecs[i] for i in idx]
    elif f.kind == "grassmannian":
        yield from grassmannian_members(f.params["p"], f.params["n"], f.params["h"])
    elif f.kind == "all_matrices":
        p, rows, cols = f.params["p"], f.params["rows"], f.params["cols"]
        for digits in itertools.product(range(p), repeat=rows * cols):
            yield [list(digits[r * cols:(r + 1) * cols]) for r in range(rows)]
    elif f.kind == "transform":
        inner, C = f.children
        c, p = C.member(0), C.p
        for m in iter_members(inner):
            yield matmul(m, c, p)
    elif f.kind == "stack":
        inner, rows = f.children
        extra = rows.member(0)
        for m in iter_members(inner):
            yield m + extra
    else:
        raise ValueError(f"unknown family {f.kind}")


def prime(f: Family) -> int:
    return f.params["p"] if "p" in f.params else prime(f.children[0]) if isinstance(f.children[0], Family) else f.children[0].p


def members(f: Family):
    """Materialise a family in canonical order as a list of matrices (lists of rows), plus p."""
    return list(iter_members(f)), prime(f)


# ---- operations and reductions -------------------------------------------------------------------

def _flat(mats):
    return [x for m in mats for r in m for x in r]


def run(op: str, family: Family, reduction: str = "all", prefix: int | None = None, **args):
    """`prefix`: answer for the first `prefix` members only (the benchmark's timing sample)."""
    op = op.removeprefix("gfp.")
    ms, p = list(itertools.islice(iter_members(family), prefix)), prime(family)
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
        if reduction == "all":
            return Integers(values)
        if reduction == "histogram":
            bins = [0] * (max(values) + 1 if values else 0)
            for v in values:
                bins[v] += 1
            return Histogram(size, size, bins)
        raise ValueError(f"reduction {reduction} does not accept integer values")

    if op == "full_row_rank":
        flags = [rank(m, p) == rows for m in ms]
    elif op == "full_col_rank":
        flags = [rank(m, p) == cols for m in ms]
    elif op == "in_span":
        target = args["target"].member(0)[0]
        flags = [in_span(m, target, p) for m in ms]
    else:
        raise ValueError(f"unknown operation {op}")

    if reduction == "all":
        return Integers([int(f) for f in flags])
    if reduction == "count":
        return Count(sum(flags), size, size)
    if reduction == "hits":
        idx = [i for i, f in enumerate(flags) if f]
        limit = min(args.get("limit", 0), len(idx))
        mem = Matrix(p, limit, rows, cols, _flat([ms[i] for i in idx[:limit]]))
        return Hits(p, rows, cols, len(idx), size, size, idx, mem)
    raise ValueError(f"reduction {reduction} does not accept boolean values")
