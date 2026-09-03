"""Plain Python baseline for extremal predicates on finite set systems.

Each member of ``subsets_of(words(2, n), m)`` is materialised as a list of binary incidence
words. Predicates are recomputed from scratch for every member. The generic backend shares the
prefix work that this implementation deliberately repeats.
"""
from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))
from lemmakernel import naive as rt  # noqa: E402
from lemmakernel.interchange import Family  # noqa: E402


def intersects(a: list[int], b: list[int]) -> bool:
    return any(x == y == 1 for x, y in zip(a, b))


def is_subset(a: list[int], b: list[int]) -> bool:
    return all(x != 1 or y == 1 for x, y in zip(a, b))


def is_intersecting(system: list[list[int]]) -> bool:
    return all(intersects(a, b) for a, b in itertools.combinations(system, 2))


def is_antichain(system: list[list[int]]) -> bool:
    return all(not is_subset(a, b) and not is_subset(b, a) for a, b in itertools.combinations(system, 2))


def is_sunflower(petals: tuple[list[int], ...]) -> bool:
    core = tuple(x & y for x, y in zip(petals[0], petals[1]))
    return all(tuple(x & y for x, y in zip(a, b)) == core for a, b in itertools.combinations(petals, 2))


def is_sunflower_free(k: int, system: list[list[int]]) -> bool:
    return not any(is_sunflower(petals) for petals in itertools.combinations(system, k))


def max_degree(system: list[list[int]], n: int) -> int:
    return max((sum(s[x] == 1 for s in system) for x in range(n)), default=0)


def shadow_size(system: list[list[int]]) -> int:
    shadow = set()
    for s in system:
        for i, bit in enumerate(s):
            if bit == 1:
                shadow.add(tuple(0 if i == j else x for j, x in enumerate(s)))
    return len(shadow)


def is_ekr_extremal(system: list[list[int]], n: int) -> bool:
    r = sum(system[0])
    return (r > 0 and 2 * r <= n and all(sum(s) == r for s in system)
            and is_intersecting(system) and len(system) == math.comb(n - 1, r - 1))


def is_sperner_extremal(system: list[list[int]], n: int) -> bool:
    return is_antichain(system) and len(system) == math.comb(n, n // 2)


def _shape(family: Family) -> tuple[int, int]:
    if family.kind != "subsets_of":
        raise ValueError("set_systems needs subsets_of(words(2,n),m)")
    (inner,) = family.children
    if inner.kind != "words" or inner.params["alphabet"] != 2:
        raise ValueError("set_systems needs subsets_of(words(2,n),m)")
    return inner.params["length"], family.params["k"]


def run(op: str, family: Family, reduction: str = "all", prefix: int | None = None, **args):
    """Evaluate the first ``prefix`` members when the benchmark asks for a sample."""
    op = op.removeprefix("set_systems.")
    n, _ = _shape(family)
    if op == "is_sunflower_free" and args["k"] < 2:
        raise ValueError("is_sunflower_free: k must be at least 2")

    systems = list(itertools.islice(rt.iter_members(family), prefix))
    p = rt.prime(family)
    if op == "max_degree":
        return rt.reduce_int(reduction, [max_degree(s, n) for s in systems], systems, p)
    if op == "shadow_size":
        return rt.reduce_int(reduction, [shadow_size(s) for s in systems], systems, p)

    if op == "is_intersecting":
        flags = [is_intersecting(s) for s in systems]
    elif op == "is_antichain":
        flags = [is_antichain(s) for s in systems]
    elif op == "is_sunflower_free":
        flags = [is_sunflower_free(args["k"], s) for s in systems]
    elif op == "is_ekr_extremal":
        flags = [is_ekr_extremal(s, n) for s in systems]
    elif op == "is_sperner_extremal":
        flags = [is_sperner_extremal(s, n) for s in systems]
    else:
        raise ValueError(f"unknown operation {op}")
    return rt.reduce_bool(reduction, flags, systems, p, **args)
