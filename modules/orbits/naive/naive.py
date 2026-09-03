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
from lemmakernel import naive as rt  # noqa: E402  (families and reductions, shared by every module)
from lemmakernel.interchange import Family, Perms  # noqa: E402

_spec = importlib.util.spec_from_file_location("gfp_naive", ROOT / "modules" / "gfp" / "naive" / "naive.py")
gfp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gfp)

perm_closure = rt.perm_closure


# ---- actions on members -------------------------------------------------------------------------

def keys(family: Family, members):
    """What the group permutes: a subset as its sorted dictionary positions (duplicate dictionary
    rows stay distinct members), a matrix as itself."""
    if family.kind in ("subsets", "subsets_of"):
        return [list(c) for c in itertools.combinations(range(len(rt.dictionary(family))), family.params["k"])]
    return members


def act(family: Family, generator, key, p):
    """Image of a member's key under one generator, canonicalised."""
    if family.kind in ("subsets", "subsets_of"):
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

def run(op: str, family: Family, reduction: str = "all", prefix: int | None = None, **args):
    """`prefix`: answer for the first `prefix` members only (the benchmark's timing sample)."""
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
        for a in batch.tolist()[:prefix]:
            out.append([points.index(normalise(gfp.matmul([v], a, p)[0])) for v in points])
        return Perms(len(points), len(out), [x for g in out for x in g])

    if op == "fixed_points":
        if family.kind != "group_elements":
            raise ValueError("fixed_points is defined on group_elements families only")
        (gens,) = family.children
        elements = perm_closure(gens.tolist())[:prefix]
        on: Family = args["on"]
        members, p = rt.members(on)
        ks = keys(on, members)
        values = [sum(act(on, g, m, p) == m for m in ks) for g in elements]
        return rt.reduce_int(reduction, values, [[g] for g in elements], 0)

    if op not in ("is_canonical", "canonical_index", "orbit_size", "stabilizer_order"):
        raise ValueError(f"unknown operation {op}")
    if family.kind not in ("subsets", "subsets_of", "grassmannian", "all_matrices"):
        raise ValueError(f"{op} is defined on subsets, subsets_of, grassmannian, all_matrices families only")
    group = args["group"]
    gens = generators_of(group)
    members, p = rt.members(family)
    ks = keys(family, members)
    rank = ranking(ks)
    if prefix is not None:
        members = members[:prefix]
    orbits = [orbit(family, gens, ks, rank, i, p) for i in range(len(members))]
    if op == "is_canonical":
        flags = [min(o) == i for i, o in enumerate(orbits)]
        return rt.reduce_bool(reduction, flags, members, p, **args)
    if op == "canonical_index":
        values = [min(o) for o in orbits]
    elif op == "orbit_size":
        values = [len(o) for o in orbits]
    else:
        order = len(perm_closure(gens)) if isinstance(group, Perms) else len(rt.matrix_closure(gens, p))
        values = [order // len(o) for o in orbits]
    return rt.reduce_int(reduction, values, members, p)
