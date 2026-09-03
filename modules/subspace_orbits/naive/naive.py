"""Plain Python baseline for matrix-group orbits on row spaces.

Every member is materialised. Every orbit performs a fresh breadth-first search, reducing each
image from scratch. The implementation follows the canonical Grassmannian order directly.
"""
from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
from lemmakernel import naive as rt  # noqa: E402
from lemmakernel.interchange import Family  # noqa: E402

_spec = importlib.util.spec_from_file_location("gfp_naive", ROOT / "modules" / "gfp" / "naive" / "naive.py")
gfp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gfp)


def grassmannian_derived(family: Family) -> bool:
    if family.kind == "grassmannian":
        return True
    if family.kind in ("transform", "stack"):
        return grassmannian_derived(family.children[0])
    return False


def subspace_rref(rows, p):
    return [row for row in gfp.rref(rows, p)[0] if any(row)]


def subspace_index(rows, p, n):
    """Index of an rref basis in grassmannian(p, n, rank)."""
    R = subspace_rref(rows, p)
    h = len(R)
    if h == 0:
        return 0
    pivots = tuple(next(i for i, x in enumerate(row) if x) for row in R)
    index = 0
    for earlier in itertools.combinations(range(n), h):
        if earlier == pivots:
            break
        free_count = sum(1 for i in range(h) for c in range(earlier[i] + 1, n) if c not in earlier)
        index += p ** free_count
    free = [(i, c) for i in range(h) for c in range(pivots[i] + 1, n) if c not in pivots]
    within = 0
    for i, c in free:
        within = within * p + R[i][c]
    return index + within


def orbit(rows, generators, p):
    start = subspace_rref(rows, p)
    queue = [start]
    seen = {tuple(map(tuple, start))}
    for current in queue:
        for generator in generators:
            image = subspace_rref(gfp.matmul(current, generator, p), p)
            key = tuple(map(tuple, image))
            if key not in seen:
                seen.add(key)
                queue.append(image)
    return queue


def normalise_matrix(a, p):
    first = next(x for row in a for x in row if x)
    inv = pow(first, p - 2, p)
    return [[x * inv % p for x in row] for row in a]


def matrix_closure(generators, p, projective):
    n = len(generators[0])
    identity = [[int(i == j) for j in range(n)] for i in range(n)]
    queue = [identity]
    seen = {tuple(map(tuple, identity))}
    for current in queue:
        for generator in generators:
            image = gfp.matmul(current, generator, p)
            if projective:
                image = normalise_matrix(image, p)
            key = tuple(map(tuple, image))
            if key not in seen:
                seen.add(key)
                queue.append(image)
    return queue


def family_cols(family):
    if family.kind == "grassmannian":
        return family.params["n"]
    if family.kind == "transform":
        return family.children[1].cols
    if family.kind == "stack":
        return family.children[1].cols
    raise ValueError(f"{family.kind} is not a subspace family")


def validate(family, group, projective):
    if not grassmannian_derived(family):
        raise ValueError("subspace_orbits accepts a Grassmannian and its transform/stack derivatives only")
    if projective not in (0, 1):
        raise ValueError("projective must be 0 or 1")
    p = rt.prime(family)
    n = family_cols(family)
    if getattr(group, "p", 0) != p:
        raise ValueError("group must be a gfp.matrix batch over the family's prime")
    generators = group.tolist()
    if not generators or any(len(a) != n or any(len(row) != n for row in a) or gfp.rank(a, p) != n for a in generators):
        raise ValueError("group generators must be invertible n x n matrices")
    return generators, p, n


def run(op: str, family: Family, reduction: str = "all", prefix: int | None = None, **args):
    op = op.removeprefix("subspace_orbits.")
    if op not in ("is_canonical", "canonical_index", "orbit_size", "stabilizer_order"):
        raise ValueError(f"unknown operation {op}")
    projective = args["projective"]
    generators, p, n = validate(family, args["group"], projective)
    members = list(itertools.islice(rt.iter_members(family), prefix))
    member_orbits = [orbit(member, generators, p) for member in members]
    least = [min(subspace_index(image, p, n) for image in images) for images in member_orbits]

    if op == "is_canonical":
        flags = [subspace_index(member, p, n) == canonical for member, canonical in zip(members, least)]
        return rt.reduce_bool(reduction, flags, members, p, **args)
    if op == "canonical_index":
        values = least
    elif op == "orbit_size":
        values = [len(images) for images in member_orbits]
    else:
        order = len(matrix_closure(generators, p, projective))
        values = [order // len(images) for images in member_orbits]
    return rt.reduce_int(reduction, values, members, p)
