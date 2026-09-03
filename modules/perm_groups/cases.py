"""Cases for permutation-group operations.

Lean closes each oracle group explicitly, so these stay at degree at most five and order at most
120. The benchmark and invariants use groups whose closures are already too large for that oracle.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lemmakernel as lk  # noqa: E402
from tools.harness import Case, cyclic, dihedral, symmetric  # noqa: E402


def singleton_group(ctx, n, generators):
    batch = ctx.perms(n, generators)
    return ctx.subsets(batch, len(generators))


def cases(ctx, rng):
    del rng
    id4 = list(range(4))
    cycle4 = [1, 2, 3, 0]
    swap01 = [1, 0, 2, 3]
    swap_pairs = [1, 0, 3, 2]
    reverse4 = [0, 3, 2, 1]
    dictionary = ctx.perms(4, [id4, cycle4, swap01, swap_pairs, reverse4])
    subgroups = ctx.subsets(dictionary, 2)
    target = ctx.perms(4, [swap01])

    out = [
        Case("two-generator subgroups of S4", subgroups, "order"),
        Case("two-generator subgroups of S4", subgroups, "contains", {"target": target, "limit": 4}),
        Case("two-generator subgroups of S4", subgroups, "is_transitive", {"limit": 4}),
        Case("two-generator subgroups of S4", subgroups, "is_primitive", {"limit": 4}),
        Case("two-generator subgroups of S4", subgroups, "orbit_partition"),
        Case("two-generator subgroups of S4", subgroups, "base_and_strong_generators"),
    ]

    explicit_cyclic = ctx.explicit(ctx.perms(5, [cyclic(5)[0], dihedral(5)[1], list(range(5))]))
    out += [
        Case("explicit cyclic groups", explicit_cyclic, "order", reductions=["all", "histogram"]),
        Case("explicit cyclic groups", explicit_cyclic, "is_transitive", {"limit": 2}, reductions=["all", "count"]),
        Case("explicit cyclic groups", explicit_cyclic, "is_primitive", {"limit": 2}, reductions=["all", "first"]),
        Case("explicit cyclic groups", explicit_cyclic, "orbit_partition"),
        Case("explicit cyclic groups", explicit_cyclic, "base_and_strong_generators"),
    ]

    out += [
        Case("matrix rows are not permutations", ctx.explicit(lk.matrix(5, [[[1, 0], [0, 1]]])),
             "order", reductions=["all"], oracle=False),
        Case("two membership targets", singleton_group(ctx, 4, symmetric(4)), "contains",
             {"target": ctx.perms(4, [cycle4, swap01])}, reductions=["all"], oracle=False),
        Case("wrong-degree membership target", singleton_group(ctx, 4, symmetric(4)), "contains",
             {"target": ctx.perms(5, cyclic(5))}, reductions=["all"], oracle=False),
    ]

    out.append(Case(
        "S9 Schreier-Sims order",
        lambda: singleton_group(ctx, 9, symmetric(9)),
        "order",
        reductions=["all"],
        bench="all",
        oracle=False,
        what="order of S_9 from a 9-cycle and one transposition, without enumerating 362880 elements",
    ))
    return out


def invariants(ctx):
    s16 = singleton_group(ctx, 16, symmetric(16))
    assert ctx.value("perm_groups.order", s16).values == [math.factorial(16)]
    assert ctx.value("perm_groups.is_transitive", s16).values == [1]
    assert ctx.value("perm_groups.is_primitive", s16).values == [1]
    assert ctx.value("perm_groups.contains", s16, target=ctx.perms(16, [[1, 0] + list(range(2, 16))])).values == [1]

    base, strong = ctx.value("perm_groups.base_and_strong_generators", s16).member(0)
    assert len(base) == 15 and len(set(base)) == len(base)
    strong_group = singleton_group(ctx, 16, strong)
    assert ctx.value("perm_groups.order", strong_group).values == [math.factorial(16)]

    d8 = singleton_group(ctx, 8, dihedral(8))
    assert ctx.value("perm_groups.is_transitive", d8).values == [1]
    assert ctx.value("perm_groups.is_primitive", d8).values == [0]

    disjoint = singleton_group(ctx, 6, [[1, 2, 0, 4, 5, 3]])
    assert ctx.value("perm_groups.orbit_partition", disjoint).member(0) == [0, 0, 0, 3, 3, 3]
