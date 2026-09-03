"""Cases for subgroup enumeration.

The Lean reference constructs every subgroup from complete element lists. Oracle groups therefore
stay at order six or below. The benchmark uses S5, of order 120 with 156 subgroups.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lemmakernel as lk  # noqa: E402
from tools.harness import Case, symmetric  # noqa: E402


def singleton_group(ctx, n, generators):
    batch = ctx.perms(n, generators)
    return ctx.subsets(batch, len(generators))


def cases(ctx, rng):
    del rng
    identity4 = list(range(4))
    swap01 = [1, 0, 2, 3]
    cycle4 = [1, 2, 3, 0]
    cyclic_groups = ctx.explicit(ctx.perms(4, [identity4, swap01, cycle4]))

    s3_generators = symmetric(3)
    s3 = singleton_group(ctx, 3, s3_generators)
    parent_s3 = ctx.perms(3, s3_generators)
    identity3 = list(range(3))
    swap12 = [0, 2, 1]
    cycle3 = [1, 2, 0]
    candidates = ctx.subsets(ctx.perms(3, [identity3, s3_generators[1], swap12, cycle3]), 1)

    out = [
        Case("cyclic groups on four points", cyclic_groups, "subgroup_count"),
        Case("cyclic groups on four points", cyclic_groups, "conjugacy_classes"),
        Case("cyclic groups on four points", cyclic_groups, "maximal_subgroups"),
        Case("S3 subgroup classes", s3, "conjugacy_classes"),
        Case("S3 maximal subgroups", s3, "maximal_subgroups"),
        Case("candidate subgroups of S3", candidates, "is_normal", {"group": parent_s3, "limit": 4}),
        Case("transposition outside C3", ctx.explicit(ctx.perms(3, [[1, 0, 2]])), "is_normal",
             {"group": ctx.perms(3, [[1, 2, 0]])}, reductions=["all"]),
        Case("field matrices are not permutations", ctx.explicit(lk.matrix(5, [[1, 0], [0, 1]])),
             "subgroup_count", reductions=["all"], oracle=False),
        Case("matrix parent for normality", candidates, "is_normal",
             {"group": lk.matrix(2, [[[1, 0], [0, 1]]])}, reductions=["all"], oracle=False),
        Case("wrong-degree parent", candidates, "is_normal",
             {"group": ctx.perms(4, [[1, 2, 3, 0]])}, reductions=["all"], oracle=False),
    ]

    out.append(Case(
        "S5 subgroup census",
        lambda: singleton_group(ctx, 5, symmetric(5)),
        "subgroup_count",
        reductions=["all"],
        bench="all",
        oracle=False,
        what="all 156 subgroups of S5, a parent group of order 120",
    ))
    return out


def invariants(ctx):
    s4 = singleton_group(ctx, 4, symmetric(4))
    assert ctx.value("subgroups.subgroup_count", s4).values == [30]
    assert len(ctx.value("subgroups.conjugacy_classes", s4).member(0)) == 11
    assert len(ctx.value("subgroups.maximal_subgroups", s4).member(0)) == 3
