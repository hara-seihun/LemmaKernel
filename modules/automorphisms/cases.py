"""Automorphism cases. The Lean oracle checks groups of order at most four because its reference
tries every permutation. The generic backend gets larger groups in invariants and the benchmark.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lemmakernel as lk  # noqa: E402
from tools.harness import Case, cyclic  # noqa: E402

OPS = ["aut_order", "aut_generators", "holomorph_order", "inner_aut_index"]


def cyclic_table(n):
    return [[(a + b) % n for b in range(n)] for a in range(n)]


def product_table(a, b):
    return [[((x // b + y // b) % a) * b + (x % b + y % b) % b
             for y in range(a * b)] for x in range(a * b)]


def dihedral_table(rotations):
    """D_(2*rotations), with labels (rotation, reflection bit)."""
    n = 2 * rotations
    return [[((x // 2 + (-1 if x % 2 else 1) * (y // 2)) % rotations) * 2 + (x + y) % 2
             for y in range(n)] for x in range(n)]


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


def preserves(table, permutation):
    n = len(table)
    return all(permutation[table[a][b]] == table[permutation[a]][permutation[b]]
               for a in range(n) for b in range(n))


def compose(a, b):
    return [b[a[x]] for x in range(len(a))]


def closure(n, generators):
    seen = {tuple(range(n))}
    queue = [list(range(n))]
    for a in queue:
        for g in generators:
            b = compose(a, g)
            if tuple(b) not in seen:
                seen.add(tuple(b))
                queue.append(b)
    return seen


def cases(ctx, rng):
    del rng
    c4 = cyclic_table(4)
    catalog4 = table_family(ctx, [c4, product_table(2, 2), relabel(c4, [2, 0, 3, 1])])
    out = [Case("order-four table catalogue", catalog4, op) for op in OPS]

    generated_c3 = ctx.generated_group(ctx.perms(3, cyclic(3)))
    out += [Case("C3 from permutation generators", generated_c3, op, reductions=["all"]) for op in OPS]
    out.append(Case("trivial group generators", table_family(ctx, [[[0]]]), "aut_generators"))

    out.append(Case("matrix is not a group family", ctx.explicit(lk.matrix(2, [[[1]]])),
                    "aut_order", reductions=["all"], oracle=False))

    order8 = [cyclic_table(8), product_table(4, 2), elementary2_table(3), dihedral_table(4)]
    out.append(Case("order8_catalogue", lambda: table_family(ctx, order8), "aut_order",
                    bench="histogram", oracle=False,
                    what="automorphism orders for four stored groups of order 8: C8, C4xC2, C2^3, and D8"))
    return out


def invariants(ctx):
    tables = [cyclic_table(8), product_table(4, 2), elementary2_table(3), dihedral_table(4)]
    family = table_family(ctx, tables)
    orders = ctx.value("automorphisms.aut_order", family).values
    holomorphs = ctx.value("automorphisms.holomorph_order", family).values
    outer = ctx.value("automorphisms.inner_aut_index", family).values
    generators = ctx.value("automorphisms.aut_generators", family)

    for i, table in enumerate(tables):
        gens = generators.member(i)
        assert all(preserves(table, g) for g in gens)
        assert len(closure(len(table), gens)) == orders[i]
        assert holomorphs[i] == len(table) * orders[i]
        center = sum(all(table[a][b] == table[b][a] for b in range(len(table))) for a in range(len(table)))
        assert outer[i] == orders[i] * center // len(table)
