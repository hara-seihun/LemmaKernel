"""Straightforward integral-lattice calculations for the benchmark baseline.

Families are materialised by ``lemmakernel.naive``. Short vectors are found by scanning an exact
cofactor-derived coordinate box, rather than by sharing work or using Fincke-Pohst.
"""
from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
from lemmakernel import naive as rt  # noqa: E402
from lemmakernel.interchange import GRAMS, Family, ShortVectors, ThetaSeries  # noqa: E402


def determinant(a):
    """Exact Bareiss determinant with row pivoting."""
    n = len(a)
    if n == 0:
        return 1
    m = [list(row) for row in a]
    sign = 1
    previous = 1
    for k in range(n - 1):
        pivot = next((i for i in range(k, n) if m[i][k]), None)
        if pivot is None:
            return 0
        if pivot != k:
            m[k], m[pivot] = m[pivot], m[k]
            sign = -sign
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                m[i][j] = (m[i][j] * m[k][k] - m[i][k] * m[k][j]) // previous
        previous = m[k][k]
    return sign * m[-1][-1]


def valid_gram(g):
    n = len(g)
    return (n > 0 and all(len(row) == n for row in g)
            and all(g[i][j] == g[j][i] for i in range(n) for j in range(n))
            and all(determinant([row[:k] for row in g[:k]]) > 0 for k in range(1, n + 1)))


def norm(g, x):
    return sum(x[i] * g[i][j] * x[j] for i in range(len(g)) for j in range(len(g)))


def coordinate_bounds(g, bound):
    n = len(g)
    det = determinant(g)
    out = []
    for i in range(n):
        cofactor = determinant([[g[r][c] for c in range(n) if c != i] for r in range(n) if r != i])
        out.append(math.isqrt(bound * cofactor // det))
    return out


def short_vectors(g, bound, include_zero=False):
    bounds = coordinate_bounds(g, bound)
    vectors = []
    for x in itertools.product(*(range(-b, b + 1) for b in bounds)):
        q = norm(g, x)
        if q <= bound and (include_zero or any(x)):
            vectors.append(list(x))
    return vectors


def minimum(g):
    upper = min(g[i][i] for i in range(len(g)))
    return min(norm(g, x) for x in short_vectors(g, upper))


def kissing_number(g):
    value = minimum(g)
    return sum(norm(g, x) == value for x in short_vectors(g, value))


def theta_series(g, bound):
    coefficients = [0] * (bound + 1)
    for x in short_vectors(g, bound, include_zero=True):
        coefficients[norm(g, x)] += 1
    return coefficients


def run(op: str, family: Family, reduction: str = "all", prefix: int | None = None, **args):
    op = op.removeprefix("lattices_small.")
    if rt.prime(family) != GRAMS:
        raise ValueError("lattices_small operations need lattices.gram members")
    grams = list(itertools.islice(rt.iter_members(family), prefix))
    if not all(valid_gram(g) for g in grams):
        raise ValueError("Gram matrix must be symmetric positive definite")

    if op == "theta_series":
        bound = args["bound"]
        values = [theta_series(g, bound) for g in grams]
        return rt.reduce_values(reduction, ThetaSeries(len(grams), bound, [x for value in values for x in value]))
    if op == "short_vectors":
        bound = args["bound"]
        values = [short_vectors(g, bound) for g in grams]
        offsets = [0]
        for value in values:
            offsets.append(offsets[-1] + len(value))
        entries = [x for value in values for vector in value for x in vector]
        return rt.reduce_values(reduction, ShortVectors(len(grams), len(grams[0]) if grams else family.params.get("n", 0),
                                                        bound, offsets, entries))

    if op == "minimum":
        values = [minimum(g) for g in grams]
        return rt.reduce_int(reduction, values, grams, GRAMS)
    if op == "kissing_number":
        values = [kissing_number(g) for g in grams]
        return rt.reduce_int(reduction, values, grams, GRAMS)
    if op == "is_unimodular":
        flags = [determinant(g) == 1 for g in grams]
    elif op == "is_even":
        flags = [all(g[i][i] % 2 == 0 for i in range(len(g))) for g in grams]
    else:
        raise ValueError(f"unknown operation {op}")
    return rt.reduce_bool(reduction, flags, grams, GRAMS, **args)
