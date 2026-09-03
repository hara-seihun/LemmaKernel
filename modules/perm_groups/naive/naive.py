"""Plain Python baseline for permutation groups.

A family member is materialised as a matrix whose rows are permutation generators. Group order,
membership, point orbits and block tests use full group closure. Only the materialised BSGS value
runs the deterministic Schreier construction, since its exact transversals are part of the output.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
from lemmakernel import naive as rt  # noqa: E402
from lemmakernel.interchange import Bsgs, Family, Partitions, Perms  # noqa: E402


def compose(g, h):
    return tuple(h[x] for x in g)


def inverse(g):
    out = [0] * len(g)
    for i, x in enumerate(g):
        out[x] = i
    return tuple(out)


def canonical_generators(gens):
    return sorted({tuple(g) for g in gens if tuple(g) != tuple(range(len(g)))})


def valid_group(gens, n):
    return bool(gens) and all(len(g) == n and sorted(g) == list(range(n)) for g in gens)


def closure(gens):
    return [tuple(g) for g in rt.perm_closure([list(g) for g in gens])]


def orbit_partition(elements, n):
    return [min(g[i] for g in elements) for i in range(n)]


def transitive(elements, n):
    return len({g[0] for g in elements}) == n


def primitive(elements, n):
    if not transitive(elements, n):
        return False
    points = range(n)
    for k in range(2, n):
        for rest in itertools.combinations(range(1, n), k - 1):
            block = (0,) + rest
            block_set = set(block)
            is_block = True
            for g in elements:
                image = {g[i] for i in block}
                if image != block_set and image & block_set:
                    is_block = False
                    break
            if is_block:
                return False
    return True


def schreier_chain(input_gens, n):
    identity = tuple(range(n))
    gens = canonical_generators(input_gens)
    levels = []
    while gens:
        base = next((i for i in range(n) if any(g[i] != i for g in gens)), None)
        if base is None:
            break
        reps = {base: identity}
        queue = [base]
        for x in queue:
            ux = reps[x]
            for s in gens:
                y = s[x]
                if y not in reps:
                    reps[y] = compose(ux, s)
                    queue.append(y)
        next_gens = []
        for x in queue:
            ux = reps[x]
            for s in gens:
                y = s[x]
                t = compose(compose(ux, s), inverse(reps[y]))
                if t != identity:
                    next_gens.append(t)
        levels.append((base, gens, reps, queue))
        gens = sorted(set(next_gens))
    return levels


def base_and_strong_generators(gens, n):
    levels = schreier_chain(gens, n)
    base = [level[0] for level in levels]
    strong = sorted({g for _, level_gens, _, _ in levels for g in level_gens})
    return base, strong


def run(op: str, family: Family, reduction: str = "all", prefix: int | None = None, **args):
    op = op.removeprefix("perm_groups.")
    if family.kind not in ("subsets", "explicit"):
        raise ValueError(f"{op} is defined on subsets and explicit families only")
    groups = list(itertools.islice(rt.iter_members(family), prefix))
    n = len(groups[0][0]) if groups and groups[0] else 0
    if not groups or any(not valid_group(gens, n) for gens in groups):
        raise ValueError("family members must be nonempty lists of same-degree permutation rows")

    target = None
    if op == "contains":
        targets: Perms = args["target"]
        if targets.count != 1:
            raise ValueError("contains target must contain exactly one permutation")
        if targets.n != n:
            raise ValueError("contains target must have the same degree as the groups")
        target = tuple(targets.member(0))

    if op == "base_and_strong_generators":
        bases, strongs = [], []
        for gens in groups:
            base, strong = base_and_strong_generators(gens, n)
            bases.append(base)
            strongs.append(strong)
        base_offsets = [0]
        strong_offsets = [0]
        for base, strong in zip(bases, strongs):
            base_offsets.append(base_offsets[-1] + len(base))
            strong_offsets.append(strong_offsets[-1] + len(strong))
        value = Bsgs(len(groups), n, base_offsets, strong_offsets,
                     [x for base in bases for x in base],
                     [x for strong in strongs for g in strong for x in g])
        return rt.reduce_values(reduction, value)

    elements = [closure(gens) for gens in groups]
    if op == "orbit_partition":
        value = Partitions(len(groups), n, [x for elems in elements for x in orbit_partition(elems, n)])
        return rt.reduce_values(reduction, value)
    if op == "order":
        return rt.reduce_int(reduction, [len(elems) for elems in elements], groups, 0)
    if op == "contains":
        return rt.reduce_bool(reduction, [target in elems for elems in elements], groups, 0, **args)
    if op == "is_transitive":
        return rt.reduce_bool(reduction, [transitive(elems, n) for elems in elements], groups, 0, **args)
    if op == "is_primitive":
        return rt.reduce_bool(reduction, [primitive(elems, n) for elems in elements], groups, 0, **args)
    raise ValueError(f"unknown operation {op}")
