"""Naive orbit counting by materialising the acted-on family.

The generic backend uses cycle formulas. This implementation instead visits each subset or word
and applies every group element, which makes it the benchmark baseline and a readable second
opinion.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
from lemmakernel import naive as rt  # noqa: E402
from lemmakernel.interchange import BurnsideCounts, CycleIndex, Family, Perms  # noqa: E402


def cycle_type(g: list[int]) -> tuple[int, ...]:
    counts = [0] * len(g)
    seen = [False] * len(g)
    for i in range(len(g)):
        if seen[i]:
            continue
        length = 0
        j = i
        while not seen[j]:
            seen[j] = True
            length += 1
            j = g[j]
        counts[length - 1] += 1
    return tuple(counts)


def _degree(family: Family) -> int:
    if family.kind in ("subsets", "subsets_of"):
        return len(rt.dictionary(family))
    if family.kind == "words":
        return family.params["length"]
    raise ValueError("burnside operations are defined on subsets, subsets_of, and words families only")


def _members(family: Family):
    if family.kind in ("subsets", "subsets_of"):
        yield from itertools.combinations(range(_degree(family)), family.params["k"])
    elif family.kind == "words":
        yield from itertools.product(range(family.params["alphabet"]), repeat=family.params["length"])
    else:
        raise ValueError("burnside operations are defined on subsets, subsets_of, and words families only")


def _act(family: Family, g: list[int], member: tuple[int, ...]) -> tuple[int, ...]:
    if family.kind in ("subsets", "subsets_of"):
        return tuple(sorted(g[i] for i in member))
    image = [0] * len(member)
    for i, value in enumerate(member):
        image[g[i]] = value
    return tuple(image)


def _one_perm(obj, degree: int) -> list[int]:
    if not isinstance(obj, Perms) or obj.count != 1:
        raise ValueError("g must be a single permutation")
    if obj.n != degree:
        raise ValueError(f"permutation has {obj.n} points but the family has {degree} positions")
    return obj.member(0)


def _group(obj, degree: int) -> list[list[int]]:
    if not isinstance(obj, Perms):
        raise ValueError("group must be a permutation group")
    if obj.count == 0:
        raise ValueError("group needs at least one generator")
    if obj.n != degree:
        raise ValueError(f"permutations have {obj.n} points but the family has {degree} positions")
    return rt.perm_closure(obj.tolist())


def run(op: str, family: Family, reduction: str = "all", prefix: int | None = None, **args):
    op = op.removeprefix("burnside.")
    if reduction != "all":
        raise ValueError(f"{op} values only reduce with `all`")
    degree = _degree(family)

    if op == "fixed_count":
        g = _one_perm(args["g"], degree)
        members = itertools.islice(_members(family), prefix)
        return BurnsideCounts([sum(_act(family, g, member) == member for member in members)])

    if op not in ("orbit_count", "cycle_index"):
        raise ValueError(f"unknown operation {op}")
    elements = _group(args["group"], degree)

    if op == "cycle_index":
        multiplicities: dict[tuple[int, ...], int] = {}
        for g in elements:
            typ = cycle_type(g)
            multiplicities[typ] = multiplicities.get(typ, 0) + 1
        return CycleIndex(degree, len(elements), [(multiplicities[typ], list(typ)) for typ in sorted(multiplicities)])

    members = itertools.islice(_members(family), prefix)
    count = sum(all(member <= _act(family, g, member) for g in elements) for member in members)
    return BurnsideCounts([count])
