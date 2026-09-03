"""Cases for difference sets in small cyclic groups, plus one pruned search benchmark."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lemmakernel as lk  # noqa: E402
from tools.harness import Case, cyclic, symmetric, unit_vectors  # noqa: E402


def cyclic_subsets(ctx, n, k):
    group = ctx.perms(n, cyclic(n))
    return ctx.subsets_of(ctx.group_elements(group), k)


def cases(ctx, rng):
    del rng
    c4_pairs = cyclic_subsets(ctx, 4, 2)
    c7_triples = cyclic_subsets(ctx, 7, 3)
    c5_pairs = cyclic_subsets(ctx, 5, 2)
    n4 = ctx.perms(4, [[(i + 2) % 4 for i in range(4)]])
    s3_elements = ctx.group_elements(ctx.perms(3, symmetric(3)))

    out = [
        Case("C7 (7,3,1) difference sets", c7_triples, "is_difference_set", {"limit": 4}),
        Case("C4 differences", c4_pairs, "difference_multiset"),
        Case("C5 (5,2,0,1) PDS", c5_pairs, "is_pds", {"lambda": 0, "mu": 1, "limit": 4}),
        Case("C4 relative to C2", c4_pairs, "is_relative_difference_set", {"forbidden": n4, "limit": 4}),
        Case("S3 nonabelian", ctx.subsets_of(s3_elements, 2), "difference_multiset"),
        Case("S3 nonabelian", ctx.subsets_of(s3_elements, 3), "is_pds",
             {"lambda": 0, "mu": 3}, reductions=["count"]),
        Case("S3 nonabelian", ctx.subsets_of(s3_elements, 5), "is_difference_set", reductions=["count"]),
    ]

    plain = ctx.subsets(unit_vectors(2, 4), 2)
    out += [
        Case("plain subsets", plain, "is_difference_set", reductions=["count"], oracle=False),
        Case("subsets of a range", ctx.subsets_of(ctx.range(0, 4), 2), "is_difference_set",
             reductions=["count"], oracle=False),
        Case("foreign forbidden subgroup", c4_pairs, "is_relative_difference_set",
             {"forbidden": ctx.perms(4, [[1, 0, 2, 3]])}, reductions=["count"], oracle=False),
    ]

    out.append(Case(
        "C31 (31,6,1) search",
        lambda: cyclic_subsets(ctx, 31, 6),
        "is_difference_set",
        what="search all 6-subsets of C31 for cyclic projective-plane difference sets",
        bench="count",
        oracle=False,
    ))
    return out


def invariants(ctx):
    family = cyclic_subsets(ctx, 7, 3)
    multisets = ctx.value("difference_sets.difference_multiset", family)
    flags = ctx.value("difference_sets.is_difference_set", family).values
    identity_column = 0
    for i, counts in enumerate(multisets.tolist()):
        row = counts[0]
        assert sum(row) == 9
        assert row[identity_column] == 3
        assert bool(flags[i]) == all(x == 1 for x in row[1:])
