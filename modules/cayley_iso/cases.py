"""Cases for fixed-size Cayley CI classification."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lemmakernel as lk  # noqa: E402
from tools.harness import Case  # noqa: E402

FIXED_OPS = ["aut_class_count", "iso_class_count", "is_ci", "is_non_ci"]
GROUP_OPS = ["is_ci_group", "is_non_ci_group"]


def cyclic_table(n):
    return [[(a + b) % n for b in range(n)] for a in range(n)]


def product_table(a, b):
    return [[((x // b + y // b) % a) * b + (x % b + y % b) % b
             for y in range(a * b)] for x in range(a * b)]


def elementary2_table(rank):
    n = 1 << rank
    return [[a ^ b for b in range(n)] for a in range(n)]


def relabel(table, old_to_new):
    n = len(table)
    out = [[0] * n for _ in range(n)]
    for a in range(n):
        for b in range(n):
            out[old_to_new[a]][old_to_new[b]] = old_to_new[table[a][b]]
    return out


def table_family(ctx, tables):
    return ctx.group_tables(lk.naturals(tables))


def cases(ctx, rng):
    del rng
    c4 = cyclic_table(4)
    catalogue = table_family(ctx, [c4, product_table(2, 2), relabel(c4, [2, 0, 3, 1])])
    out = [Case("order-four catalogue at k=1", catalogue, op, args={"k": 1, "limit": 2})
           for op in FIXED_OPS]
    out += [Case("order-four whole-group census", catalogue, op, args={"limit": 2})
            for op in GROUP_OPS]

    out.append(Case("C4xC2 fixed-size census", lambda: table_family(ctx, [product_table(4, 2)]),
                    "is_non_ci", args={"k": 1}, reductions=["count"], bench="count", oracle=False,
                    what="the three inverse-closed singleton connections of C4xC2"))

    out.append(Case("matrix is not a group family", ctx.explicit(lk.matrix(2, [[[1]]])),
                    "is_ci", args={"k": 1}, reductions=["all"], oracle=False))
    return out


def invariants(ctx):
    family = table_family(ctx, [cyclic_table(8), product_table(4, 2), elementary2_table(3)])
    args = {"k": 1}
    aut_counts = ctx.value("cayley_iso.aut_class_count", family, **args).values
    iso_counts = ctx.value("cayley_iso.iso_class_count", family, **args).values
    ci = ctx.value("cayley_iso.is_ci", family, **args).values
    non_ci = ctx.value("cayley_iso.is_non_ci", family, **args).values
    hits = ctx.value("cayley_iso.is_non_ci", family, "hits", limit=3, **args)
    group_ci = ctx.value("cayley_iso.is_ci_group", family).values
    group_non_ci = ctx.value("cayley_iso.is_non_ci_group", family).values

    expected_non_ci = [int(a != i) for a, i in zip(aut_counts, iso_counts)]
    assert list(non_ci) == expected_non_ci
    assert list(ci) == [1 - value for value in expected_non_ci]
    assert list(hits.indices) == [i for i, value in enumerate(expected_non_ci) if value]
    assert list(group_non_ci) == [1 - value for value in group_ci]
    for index, table in enumerate([cyclic_table(8), product_table(4, 2), elementary2_table(3)]):
        fixed = [ctx.value("cayley_iso.is_ci", table_family(ctx, [table]), k=k).values[0]
                 for k in range(len(table))]
        assert group_ci[index] == int(all(fixed))
