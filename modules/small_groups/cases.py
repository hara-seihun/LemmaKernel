"""small_groups cases.

The oracle cases are the stored catalogues of small order plus explicit tables under an awkward
labelling. Walking a subgroup lattice costs the Lean kernel about a second per group of order 8,
so the two lattice counts are claimed on the catalogues of order 4 and 6 and on the quaternion
group alone; order 16 and 24 are left to the invariants, and the benchmark takes eighteen groups
of order 64 built here from cyclic, dihedral and dicyclic factors.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lemmakernel as lk  # noqa: E402
from tools.harness import Case  # noqa: E402

LATTICE_OPS = ["subgroup_count", "normal_subgroup_count"]
SCALAR_OPS = ["order", "exponent", "centre_order", "class_count", "derived_length"] + LATTICE_OPS
FLAG_OPS = ["is_nilpotent", "is_solvable"]
VALUE_OPS = ["centre", "derived_series"]
OPS = SCALAR_OPS + FLAG_OPS + VALUE_OPS


def cyclic_table(n):
    return [[(a + b) % n for b in range(n)] for a in range(n)]


def dihedral_table(rotations):
    """The dihedral group of order 2*rotations, labelled by (rotation, reflection bit)."""
    n = 2 * rotations
    return [[((x // 2 + (-1 if x % 2 else 1) * (y // 2)) % rotations) * 2 + (x + y) % 2
             for y in range(n)] for x in range(n)]


def dicyclic_table(n):
    """The dicyclic group of order 4n: a of order 2n and b with b² = aⁿ and b⁻¹ab = a⁻¹.

    Label a^i by i and a^i b by 2n + i, so n = 2 is the quaternion group and n a power of two
    gives the generalised quaternion group of order 4n.
    """
    m = 2 * n

    def product(x, y):
        i, j = x % m, y % m
        if x < m:
            return (i + j) % m if y < m else m + (i + j) % m
        return m + (i - j) % m if y < m else (i - j + n) % m

    return [[product(x, y) for y in range(2 * m)] for x in range(2 * m)]


def direct_product(a, b):
    """A × B, with the pair (x, y) labelled x*|B| + y."""
    n = len(b)
    return [[a[x // n][y // n] * n + b[x % n][y % n] for y in range(len(a) * n)]
            for x in range(len(a) * n)]


def product_of(*tables):
    out = tables[0]
    for table in tables[1:]:
        out = direct_product(out, table)
    return out


def order_64_groups():
    """Eighteen groups of order 64: the eleven abelian ones, and seven with a dihedral or
    dicyclic factor. The elementary abelian group has 2,825 subgroups, the most of the eighteen.
    """
    c, d, q = cyclic_table, dihedral_table, dicyclic_table
    abelian = [[2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 4], [2, 2, 4, 4], [4, 4, 4], [2, 2, 2, 8],
               [2, 4, 8], [8, 8], [2, 2, 16], [4, 16], [2, 32], [64]]
    return [product_of(*[c(k) for k in orders]) for orders in abelian] + [
        d(32), q(16), product_of(c(2), d(16)), product_of(c(2), q(8)),
        product_of(c(4), d(8)), product_of(c(2), c(2), d(8)), product_of(c(2), c(4), d(4))]


def relabel(table, old_to_new):
    """The same group with element `x` renamed `old_to_new[x]`, so label 0 need not be the unit."""
    n = len(table)
    out = [[0] * n for _ in range(n)]
    for a in range(n):
        for b in range(n):
            out[old_to_new[a]][old_to_new[b]] = old_to_new[table[a][b]]
    return out


def cases(ctx, rng):
    del rng
    out = []
    for n in (4, 6):
        family = ctx.group_catalogue(n)
        out += [Case(f"catalogue of order {n}", family, op, {"limit": 3}) for op in OPS]

    order8 = ctx.group_catalogue(8)
    out += [Case("catalogue of order 8", order8, op, {"limit": 3})
            for op in OPS if op not in LATTICE_OPS]

    quaternion = ctx.group_tables(lk.naturals([dicyclic_table(2)]))
    out += [Case("quaternion table", quaternion, op, {"limit": 1}) for op in OPS]

    shifted = ctx.group_tables(lk.naturals([relabel(cyclic_table(6), [3, 4, 5, 0, 1, 2]),
                                            relabel(dihedral_table(3), [5, 4, 3, 2, 1, 0])]))
    out += [Case("relabelled tables", shifted, op, {"limit": 2}) for op in OPS]

    out += [Case("explicit naturals are not a group family",
                 ctx.explicit(lk.naturals([[[0, 1], [1, 0]]])), "order", reductions=["all"],
                 oracle=False)]
    out += [Case("field matrix is not a group family", ctx.explicit(lk.matrix(2, [[[1]]])),
                 "class_count", reductions=["all"], oracle=False)]
    out += [Case("range is not a group family", ctx.range(0, 4), "is_solvable",
                 reductions=["all"], oracle=False)]

    out.append(Case("order64_subgroup_lattices",
                    lambda: ctx.group_tables(lk.naturals(order_64_groups())), "subgroup_count",
                    bench="histogram", oracle=False,
                    what="how many subgroups each of eighteen groups of order 64 has, from the 7"
                         " of the cyclic group to the 2,825 of the elementary abelian one"))
    return out


def divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def invariants(ctx):
    """Facts that hold for every group of order 16 and 24, beyond what the oracle cases check."""
    for n in (16, 24):
        family = ctx.group_catalogue(n)
        orders = ctx.value("small_groups.order", family).values
        exponents = ctx.value("small_groups.exponent", family).values
        centres = ctx.value("small_groups.centre", family)
        centre_orders = ctx.value("small_groups.centre_order", family).values
        classes = ctx.value("small_groups.class_count", family).values
        subgroup_counts = ctx.value("small_groups.subgroup_count", family).values
        normal_counts = ctx.value("small_groups.normal_subgroup_count", family).values
        nilpotent = ctx.value("small_groups.is_nilpotent", family).values
        solvable = ctx.value("small_groups.is_solvable", family).values
        series = ctx.value("small_groups.derived_series", family)
        lengths = ctx.value("small_groups.derived_length", family).values

        for i in range(len(orders)):
            assert orders[i] == n
            # Lagrange, and the class equation for an abelian group.
            assert n % exponents[i] == 0
            assert len(centres.member(i)) == centre_orders[i]
            assert n % centre_orders[i] == 0
            assert centre_orders[i] != n or classes[i] == n
            assert classes[i] >= centre_orders[i]
            # A group of prime-power order has a nontrivial centre and is nilpotent.
            assert (n != 16) or (centre_orders[i] > 1 and nilpotent[i] == 1)
            # Every group of order below 60 is solvable, so the derived series reaches 1.
            assert solvable[i] == 1
            terms = series.member(i)
            assert len(terms) == lengths[i] + 1
            assert terms[0] == list(range(n)) and len(terms[-1]) == 1
            for a, b in zip(terms, terms[1:]):
                assert set(b) < set(a) and len(a) % len(b) == 0
            # Nilpotent means the subgroups of each order are normal in the right numbers:
            # at the least, both counts include 1 and G, and normal subgroups are subgroups.
            assert 2 <= normal_counts[i] <= subgroup_counts[i]
            assert subgroup_counts[i] >= len(divisors(n))
