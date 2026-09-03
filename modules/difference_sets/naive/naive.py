"""Direct Python implementation of difference sets in finite permutation groups.

The family is materialised, then every ordered pair in every candidate subset is recomputed.
The ambient element order comes from ``group_elements`` and is also the column order of
``difference_multiset``.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
from lemmakernel import naive as rt  # noqa: E402
from lemmakernel.interchange import Family, Matrix, NATURALS, Perms  # noqa: E402


def inverse(g: list[int]) -> list[int]:
    out = [0] * len(g)
    for i, image in enumerate(g):
        out[image] = i
    return out


def quotient(x: list[int], y: list[int]) -> list[int]:
    yi = inverse(y)
    return [x[preimage] for preimage in yi]


def difference_multiset(elements: list[list[int]], subset: list[list[int]]) -> list[int]:
    index = {tuple(g): i for i, g in enumerate(elements)}
    counts = [0] * len(elements)
    for x in subset:
        for y in subset:
            counts[index[tuple(quotient(x, y))]] += 1
    return counts


def _ambient(family: Family) -> tuple[list[list[int]], list[list[list[int]]]]:
    if family.kind != "subsets_of":
        raise ValueError("difference_sets needs subsets_of(group_elements(G), k)")
    (inner,) = family.children
    if inner.kind != "group_elements":
        raise ValueError("difference_sets needs subsets_of(group_elements(G), k)")
    (generators,) = inner.children
    elements = rt.perm_closure(generators.tolist())
    return elements, list(rt.iter_members(family))


def is_difference_set(elements: list[list[int]], subset: list[list[int]], counts: list[int]) -> bool:
    v, k = len(elements), len(subset)
    if v <= 1 or k * (k - 1) % (v - 1):
        return False
    target = k * (k - 1) // (v - 1)
    identity = list(range(len(elements[0])))
    return all(c == (k if g == identity else target) for g, c in zip(elements, counts))


def is_pds(elements: list[list[int]], subset: list[list[int]], counts: list[int], lam: int, mu: int) -> bool:
    v, k = len(elements), len(subset)
    identity = list(range(len(elements[0])))
    chosen = {tuple(g) for g in subset}
    if tuple(identity) in chosen or any(tuple(inverse(g)) not in chosen for g in subset):
        return False
    if k * (k - 1) != k * lam + (v - 1 - k) * mu:
        return False
    return all(c == (k if g == identity else lam if tuple(g) in chosen else mu)
               for g, c in zip(elements, counts))


def forbidden_subgroup(elements: list[list[int]], forbidden: Perms) -> list[list[int]]:
    if forbidden.count == 0 or forbidden.n != len(elements[0]):
        raise ValueError("forbidden must generate a subgroup of the ambient group")
    subgroup = rt.perm_closure(forbidden.tolist())
    ambient = {tuple(g) for g in elements}
    if any(tuple(g) not in ambient for g in subgroup):
        raise ValueError("forbidden must generate a subgroup of the ambient group")
    return subgroup


def is_relative_difference_set(elements: list[list[int]], subset: list[list[int]], counts: list[int],
                               subgroup: list[list[int]]) -> bool:
    outside, k = len(elements) - len(subgroup), len(subset)
    if outside == 0 or k * (k - 1) % outside:
        return False
    target = k * (k - 1) // outside
    identity = list(range(len(elements[0])))
    forbidden = {tuple(g) for g in subgroup}
    return all(c == (k if g == identity else 0 if tuple(g) in forbidden else target)
               for g, c in zip(elements, counts))


def run(op: str, family: Family, reduction: str = "all", prefix: int | None = None, **args):
    op = op.removeprefix("difference_sets.")
    elements, members = _ambient(family)
    members = list(itertools.islice(members, prefix))
    multisets = [difference_multiset(elements, subset) for subset in members]

    if op == "difference_multiset":
        out = Matrix(NATURALS, len(members), 1, len(elements), [x for counts in multisets for x in counts])
        return rt.reduce_values(reduction, out)
    if op == "is_difference_set":
        flags = [is_difference_set(elements, subset, counts) for subset, counts in zip(members, multisets)]
    elif op == "is_pds":
        flags = [is_pds(elements, subset, counts, args["lambda"], args["mu"])
                 for subset, counts in zip(members, multisets)]
    elif op == "is_relative_difference_set":
        subgroup = forbidden_subgroup(elements, args["forbidden"])
        flags = [is_relative_difference_set(elements, subset, counts, subgroup)
                 for subset, counts in zip(members, multisets)]
    else:
        raise ValueError(f"unknown operation {op}")
    return rt.reduce_bool(reduction, flags, members, 0, **args)
