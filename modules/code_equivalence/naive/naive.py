"""Plain Python baseline for linear codes up to monomial equivalence.

Every member is materialised and reduced to its rref basis. Every monomial map of F_p^n is then
applied to that basis, from scratch, and the image is reduced again and ranked in the runtime's
Grassmannian order. The class representative is the image of least rank; the orbit is the set of
distinct images. Nothing is shared between members and nothing is pruned, which is the point: it
is the obvious implementation, and the benchmark baseline.
"""
from __future__ import annotations

import importlib.util
import itertools
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
from lemmakernel import naive as rt  # noqa: E402
from lemmakernel.interchange import Basis, Family  # noqa: E402

_spec = importlib.util.spec_from_file_location("gfp_naive", ROOT / "modules" / "gfp" / "naive" / "naive.py")
gfp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gfp)

OPS = ("is_canonical", "canonical_index", "canonical_form", "orbit_size", "aut_order")


def code_basis(rows, p):
    """The unique rref basis of the row span, without zero rows."""
    return [row for row in gfp.rref(rows, p)[0] if any(row)]


def grassmannian_index(basis, p, n):
    """Index of an rref basis in grassmannian(p, n, k): pivot sets in lexicographic order, then
    the free entries as base-p digits in row-major order."""
    k = len(basis)
    if k == 0:
        return 0
    pivots = tuple(next(j for j, x in enumerate(row) if x) for row in basis)
    index = 0
    for earlier in itertools.combinations(range(n), k):
        if earlier == pivots:
            break
        index += p ** sum(1 for i in range(k) for c in range(earlier[i] + 1, n) if c not in earlier)
    for i in range(k):
        for c in range(pivots[i] + 1, n):
            if c not in pivots:
                index = index * p + basis[i][c]
    return index


def monomial_maps(p, n, scalars):
    """Every monomial map as (permutation, scale). With scalars, the group has order
    (p-1)^n n!; without them it is the n! coordinate permutations."""
    scales = list(itertools.product(range(1, p), repeat=n)) if scalars else [(1,) * n]
    return [(perm, scale) for perm in itertools.permutations(range(n)) for scale in scales]


def image(basis, perm, scale, p):
    return [[scale[j] * row[perm[j]] % p for j in range(len(perm))] for row in basis]


def classify(rows, p, n, maps, scalars):
    """(own index, least index, representative basis, orbit size) for one member."""
    basis = code_basis(rows, p)
    own = grassmannian_index(basis, p, n)
    best, best_basis, orbit = None, None, set()
    for perm, scale in maps:
        candidate = code_basis(image(basis, perm, scale, p), p)
        index = grassmannian_index(candidate, p, n)
        orbit.add(index)
        if best is None or index < best:
            best, best_basis = index, candidate
    return own, best, best_basis, len(orbit)


def run(op: str, family: Family, reduction: str = "all", prefix: int | None = None, **args):
    op = op.removeprefix("code_equivalence.")
    if op not in OPS:
        raise ValueError(f"unknown operation {op}")
    scalars = args["scalars"]
    if scalars not in (0, 1):
        raise ValueError("scalars must be 1 for the monomial group or 0 for coordinate permutations")
    members = list(itertools.islice(rt.iter_members(family), prefix))
    p = rt.prime(family)
    n = len(members[0][0]) if members else family.params.get("cols", 0)
    maps = monomial_maps(p, n, scalars)
    classes = [classify(m, p, n, maps, scalars) for m in members]

    if op == "is_canonical":
        return rt.reduce_bool(reduction, [own == least for own, least, _, _ in classes], members, p, **args)
    if op == "canonical_form":
        bases = [basis for _, _, basis, _ in classes]
        offsets = [0]
        for basis in bases:
            offsets.append(offsets[-1] + len(basis))
        return rt.reduce_values(reduction, Basis(p, len(members), n, offsets,
                                                 [x for basis in bases for row in basis for x in row]))
    if op == "canonical_index":
        values = [least for _, least, _, _ in classes]
    elif op == "orbit_size":
        values = [size for _, _, _, size in classes]
    else:
        order = (p - 1) ** n * math.factorial(n) if scalars else math.factorial(n)
        values = [order // size for _, _, _, size in classes]
    return rt.reduce_int(reduction, values, members, p)
