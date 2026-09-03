"""The obvious implementation of the orbits module, in plain Python.

Members of the acted-on family are materialised in canonical order and ranked with a dict, the
group is closed under composition when its elements are needed, and every orbit is a
breadth-first search over the generators. Nothing is shared between members.

    naive.run(op, family, reduction, **args) -> interchange object

`family` is a lemmakernel.interchange.Family; `group` is a Perms or a Matrix batch.
"""
from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
from lemmakernel.interchange import Count, Family, Histogram, Hits, Integers, Matrix, Perms  # noqa: E402

_spec = importlib.util.spec_from_file_location("gfp_naive", ROOT / "modules" / "gfp" / "naive" / "naive.py")
gfp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gfp)


# ---- groups -------------------------------------------------------------------------------------

def compose(g, h):
    """Permutation x -> h(g(x))."""
    return [h[g[x]] for x in range(len(g))]


def perm_closure(gens: list[list[int]]) -> list[list[int]]:
    """Every element of the generated group, sorted lexicographically."""
    n = len(gens[0])
    identity = list(range(n))
    seen = {tuple(identity)}
    queue = [identity]
    for g in queue:
        for gen in gens:
            h = compose(g, gen)
            if tuple(h) not in seen:
                seen.add(tuple(h))
                queue.append(h)
    return sorted(queue)


def matrix_closure(gens: list[list[list[int]]], p: int) -> list[list[list[int]]]:
    n = len(gens[0])
    identity = [[int(i == j) for j in range(n)] for i in range(n)]
    seen = {tuple(map(tuple, identity))}
    queue = [identity]
    for a in queue:
        for gen in gens:
            b = gfp.matmul(a, gen, p)
            key = tuple(map(tuple, b))
            if key not in seen:
                seen.add(key)
                queue.append(b)
    return queue


# ---- actions on members -------------------------------------------------------------------------

def keys(family: Family, members):
    """What the group permutes: a subset as its sorted dictionary positions (duplicate dictionary
    rows stay distinct members), a matrix as itself."""
    if family.kind == "subsets":
        (dictionary,) = family.children
        return [list(c) for c in itertools.combinations(range(dictionary.count if dictionary.rows == 1 else dictionary.rows), family.params["k"])]
    return members


def act(family: Family, generator, key, p):
    """Image of a member's key under one generator, canonicalised."""
    if family.kind == "subsets":
        return sorted(generator[i] for i in key)
    member = key
    if family.kind == "grassmannian":
        return gfp.rref(gfp.matmul(member, generator, p), p)[0]
    if family.kind == "all_matrices":
        return gfp.matmul(member, generator, p)
    raise ValueError(f"no action on {family.kind} families")


def generators_of(group):
    if isinstance(group, Perms):
        return group.tolist()
    return group.tolist()  # Matrix batch: list of n x n matrices


def ranking(ks):
    """key -> member index."""
    return {tuple(map(tuple, k)) if isinstance(k[0], list) else tuple(k): i for i, k in enumerate(ks)}


def orbit(family, gens, ks, rank, index, p):
    """Set of member indices in the orbit of member `index`; `ks` are the members' keys and
    `rank` is `ranking(ks)`."""
    seen = {index}
    queue = [index]
    for i in queue:
        for g in gens:
            image = act(family, g, ks[i], p)
            j = rank[tuple(map(tuple, image)) if isinstance(image[0], list) else tuple(image)]
            if j not in seen:
                seen.add(j)
                queue.append(j)
    return seen


# ---- operations -----------------------------------------------------------------------------------

def _reduce_int(values, reduction, size):
    if reduction == "all":
        return Integers(values)
    if reduction == "histogram":
        bins = [0] * (max(values) + 1 if values else 0)
        for v in values:
            bins[v] += 1
        return Histogram(size, size, bins)
    raise ValueError(f"reduction {reduction} does not accept integer values")


def _reduce_bool(flags, reduction, members, p, rows, cols, **args):
    size = len(flags)
    if reduction == "all":
        return Integers([int(f) for f in flags])
    if reduction == "count":
        return Count(sum(flags), size, size)
    if reduction == "hits":
        idx = [i for i, f in enumerate(flags) if f]
        limit = min(args.get("limit", 0), len(idx))
        mem = Matrix(p, limit, rows, cols, [x for i in idx[:limit] for r in members[i] for x in r])
        return Hits(p, rows, cols, len(idx), size, size, idx, mem)
    raise ValueError(f"reduction {reduction} does not accept boolean values")


def run(op: str, family: Family, reduction: str = "all", **args):
    op = op.removeprefix("orbits.")
    if op == "projective_action":
        if family.kind != "explicit":
            raise ValueError("projective_action is defined on explicit families only")
        (batch,) = family.children
        p = batch.p
        pts = args["points"]
        points = [m[0] for m in pts.tolist()] if pts.rows == 1 else pts.member(0)

        def normalise(v):
            lead = next(x for x in v if x)
            inv = pow(lead, p - 2, p)
            return [(x * inv) % p for x in v]
        out = []
        for a in batch.tolist():
            out.append([points.index(normalise(gfp.matmul([v], a, p)[0])) for v in points])
        return Perms(len(points), len(out), [x for g in out for x in g])

    if op == "fixed_points":
        if family.kind != "group_elements":
            raise ValueError("fixed_points is defined on group_elements families only")
        (gens,) = family.children
        elements = perm_closure(gens.tolist())
        on: Family = args["on"]
        members, p = gfp.members(on)
        ks = keys(on, members)
        values = [sum(act(on, g, m, p) == m for m in ks) for g in elements]
        return _reduce_int(values, reduction, len(elements))

    if op not in ("is_canonical", "canonical_index", "orbit_size", "stabilizer_order"):
        raise ValueError(f"unknown operation {op}")
    if family.kind not in ("subsets", "grassmannian", "all_matrices"):
        raise ValueError(f"{op} is defined on subsets, grassmannian, all_matrices families only")
    group = args["group"]
    gens = generators_of(group)
    members, p = gfp.members(family)
    rows = len(members[0]) if members else 0
    cols = len(members[0][0]) if members else 0
    ks = keys(family, members)
    rank = ranking(ks)
    orbits = [orbit(family, gens, ks, rank, i, p) for i in range(len(members))]
    if op == "is_canonical":
        flags = [min(o) == i for i, o in enumerate(orbits)]
        return _reduce_bool(flags, reduction, members, p, rows, cols, **args)
    if op == "canonical_index":
        values = [min(o) for o in orbits]
    elif op == "orbit_size":
        values = [len(o) for o in orbits]
    else:
        order = len(perm_closure(gens)) if isinstance(group, Perms) else len(matrix_closure(gens, p))
        values = [order // len(o) for o in orbits]
    return _reduce_int(values, reduction, len(members))
