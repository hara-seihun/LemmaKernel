"""Plain Python baseline for counts and incidence in the subspace lattice.

The implementation materialises each family member, reduces its rows from scratch, and applies
the shared runtime reductions. It intentionally does not share work between members.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
from lemmakernel import naive as rt  # noqa: E402
from lemmakernel.interchange import Family  # noqa: E402

MATRIX_FAMILIES = {
    "explicit", "subsets", "grassmannian", "all_matrices", "symmetric_matrices",
    "transform", "stack", "subsets_of",
}


def gaussian_binomial(q: int, n: int, k: int) -> int:
    """[n choose k]_q by the Gaussian Pascal recurrence."""
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    row = [1] + [0] * k
    powers = [1]
    for _ in range(k):
        powers.append(powers[-1] * q)
    for i in range(1, n + 1):
        for j in range(min(i, k), 0, -1):
            row[j] = row[j - 1] + powers[j] * row[j]
    return row[k]


def flag_count(q: int, n: int, dims: list[int]) -> int:
    if any(d > n for d in dims) or any(a >= b for a, b in zip(dims, dims[1:])):
        return 0
    value = 1
    previous = 0
    for d in dims:
        value *= gaussian_binomial(q, n - previous, d - previous)
        previous = d
    return value


def rank(rows: list[list[int]], p: int) -> int:
    if not rows:
        return 0
    a = [list(row) for row in rows]
    r = 0
    for c in range(len(a[0])):
        pivot = next((i for i in range(r, len(a)) if a[i][c]), None)
        if pivot is None:
            continue
        a[r], a[pivot] = a[pivot], a[r]
        inverse = pow(a[r][c], p - 2, p)
        a[r] = [(x * inverse) % p for x in a[r]]
        for i in range(r + 1, len(a)):
            if a[i][c]:
                factor = a[i][c]
                a[i] = [(x - factor * y) % p for x, y in zip(a[i], a[r])]
        r += 1
        if r == len(a):
            break
    return r


def _vectors(matrix):
    return rt.vectors_of(matrix)


def run(op: str, family: Family, reduction: str = "all", prefix: int | None = None, **args):
    op = op.removeprefix("lattice_of_subspaces.")
    members = list(itertools.islice(rt.iter_members(family), prefix))
    member_prime = rt.prime(family)

    if op in ("gaussian_binomial", "flag_count"):
        q, n = int(args["q"]), int(args["n"])
        if q < 2:
            raise ValueError("q must be at least 2")
        if op == "gaussian_binomial":
            if family.kind != "range":
                raise ValueError("gaussian_binomial is defined on range families only")
            values = [gaussian_binomial(q, n, member[0][0]) for member in members]
        else:
            if family.kind != "words":
                raise ValueError("flag_count is defined on words families only")
            values = [flag_count(q, n, member[0]) for member in members]
        return rt.reduce_int(reduction, values, members, member_prime)

    if family.kind not in MATRIX_FAMILIES:
        raise ValueError(f"{op} is defined on matrix families only")
    p = member_prime

    if op in ("contained_subspace_count", "containing_subspace_count"):
        h = int(args["h"])
        values = []
        for member in members:
            r = rank(member, p)
            if op == "contained_subspace_count":
                values.append(gaussian_binomial(p, r, h))
            else:
                cols = len(member[0]) if member else family.params.get("cols", 0)
                values.append(gaussian_binomial(p, cols - r, h - r) if r <= h else 0)
        return rt.reduce_int(reduction, values, members, p)

    if op not in ("contains", "is_contained_in"):
        raise ValueError(f"unknown operation {op}")
    fixed = args["subspace"]
    subspace = _vectors(fixed)
    cols = len(members[0][0]) if members and members[0] else family.params.get("cols", 0)
    if fixed.p != p or any(len(row) != cols for row in subspace):
        raise ValueError("subspace must be over the same prime with the same number of columns")
    fixed_rank = rank(subspace, p)
    if op == "contains":
        flags = [rank(member + subspace, p) == rank(member, p) for member in members]
    else:
        flags = [rank(subspace + member, p) == fixed_rank for member in members]
    return rt.reduce_bool(reduction, flags, members, p, **args)
