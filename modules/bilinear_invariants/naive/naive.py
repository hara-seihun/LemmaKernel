"""Direct per-matrix implementation of bilinear_invariants."""
from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
from lemmakernel import naive as rt  # noqa: E402
from lemmakernel.interchange import Family, Matrix  # noqa: E402

SUPPORTED = {"explicit", "symmetric_matrices", "alternating_matrices"}


def is_prime(p):
    return p >= 2 and all(p % d for d in range(2, math.isqrt(p) + 1))


def is_symmetric(a):
    return all(a[i][j] == a[j][i] for i in range(len(a)) for j in range(len(a)))


def is_alternating(a, p):
    return all(a[i][i] == 0 for i in range(len(a))) and all(
        (a[i][j] + a[j][i]) % p == 0 for i in range(len(a)) for j in range(len(a)))


def rank(a, p):
    m = [list(row) for row in a]
    rows, cols = len(m), len(m[0])
    r = 0
    for col in range(cols):
        pivot = next((i for i in range(r, rows) if m[i][col]), None)
        if pivot is None:
            continue
        m[r], m[pivot] = m[pivot], m[r]
        inverse = pow(m[r][col], p - 2, p)
        m[r] = [x * inverse % p for x in m[r]]
        for i in range(r + 1, rows):
            factor = m[i][col]
            if factor:
                m[i] = [(x - factor * y) % p for x, y in zip(m[i], m[r])]
        r += 1
        if r == rows:
            break
    return r


def determinant(a, p):
    m = [list(row) for row in a]
    value = 1
    for col in range(len(m)):
        pivot = next((i for i in range(col, len(m)) if m[i][col]), None)
        if pivot is None:
            return 0
        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]
            value = -value
        d = m[col][col]
        value = value * d % p
        inverse = pow(d, p - 2, p)
        for i in range(col + 1, len(m)):
            factor = m[i][col] * inverse % p
            if factor:
                m[i] = [(x - factor * y) % p for x, y in zip(m[i], m[col])]
    return value % p


def square_class(x, p):
    if x % p == 0:
        return 0
    return 1 if pow(x, (p - 1) // 2, p) == 1 else 2


def discriminant(a, p):
    r = rank(a, p)
    if r == 0:
        return 0
    for indices in itertools.combinations(range(len(a)), r):
        minor = [[a[i][j] for j in indices] for i in indices]
        d = determinant(minor, p)
        if d:
            return d
    raise AssertionError("a symmetric or alternating matrix has no full-rank principal minor")


def discriminant_class(a, p):
    return square_class(discriminant(a, p), p)


def least_nonsquare(p):
    return next(x for x in range(2, p) if square_class(x, p) == 2)


def alternating_label(p, n, r):
    out = [[0] * n for _ in range(n)]
    for i in range(0, r, 2):
        out[i][i + 1] = 1
        out[i + 1][i] = (-1) % p
    return out


def diagonal_label(p, n, r, last):
    out = [[0] * n for _ in range(n)]
    for i in range(r):
        out[i][i] = last if i + 1 == r else 1
    return out


def congruence_label(a, p):
    n = len(a)
    r = rank(a, p)
    if is_alternating(a, p):
        return alternating_label(p, n, r)
    if p == 2:
        return diagonal_label(p, n, r, 1)
    last = least_nonsquare(p) if discriminant_class(a, p) == 2 else 1
    return diagonal_label(p, n, r, last)


def run(op: str, family: Family, reduction: str = "all", prefix: int | None = None, **args):
    op = op.removeprefix("bilinear_invariants.")
    if family.kind not in SUPPORTED:
        raise ValueError("bilinear_invariants is defined on explicit, symmetric_matrices, and alternating_matrices families")
    matrices = list(itertools.islice(rt.iter_members(family), prefix))
    p = rt.prime(family)
    if not is_prime(p):
        raise ValueError("bilinear_invariants needs a prime field")
    if not matrices or not matrices[0] or any(len(a) != len(a[0]) for a in matrices):
        raise ValueError("bilinear_invariants needs nonempty square matrices")
    if any(not (is_symmetric(a) or is_alternating(a, p)) for a in matrices):
        raise ValueError("each matrix must be symmetric or alternating")
    n = len(matrices[0])

    if op == "congruence_label":
        labels = [congruence_label(a, p) for a in matrices]
        flat = [x for a in labels for row in a for x in row]
        return rt.reduce_values(reduction, Matrix(p, len(labels), n, n, flat))

    ranks = [rank(a, p) for a in matrices]
    if op == "rank":
        return rt.reduce_int(reduction, ranks, matrices, p)
    if op == "radical_dimension":
        return rt.reduce_int(reduction, [n - r for r in ranks], matrices, p)
    if op == "determinant":
        return rt.reduce_int(reduction, [determinant(a, p) for a in matrices], matrices, p)
    if op == "determinant_class":
        return rt.reduce_int(reduction, [square_class(determinant(a, p), p) for a in matrices], matrices, p)
    if op == "discriminant_class":
        return rt.reduce_int(reduction, [discriminant_class(a, p) for a in matrices], matrices, p)
    if op == "is_nondegenerate":
        return rt.reduce_bool(reduction, [r == n for r in ranks], matrices, p, **args)
    if op == "is_alternating":
        return rt.reduce_bool(reduction, [is_alternating(a, p) for a in matrices], matrices, p, **args)
    raise ValueError(f"unknown operation {op}")
