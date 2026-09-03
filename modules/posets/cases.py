"""Finite-poset cases for relation matrices, subset inclusion, and divisor posets."""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lemmakernel as lk  # noqa: E402
from tools.harness import Case, module, rotate  # noqa: E402

OPS = ["mobius_function", "linear_extension_count", "is_lattice", "is_distributive", "width", "height", "order_polynomial"]


def relation_from_le(n, le):
    return [[int(le(i, j)) for j in range(n)] for i in range(n)]


def labelled_posets():
    chain = relation_from_le(5, lambda i, j: i <= j)
    antichain = relation_from_le(5, lambda i, j: i == j)
    m3 = relation_from_le(5, lambda i, j: i == j or i == 0 or j == 4)
    n5 = relation_from_le(5, lambda i, j: i == j or i == 0 or j == 4 or (i, j) == (1, 2))
    vee_plus_point = relation_from_le(5, lambda i, j: i == j or (i in (0, 1) and j == 2))
    permutation = [3, 1, 4, 0, 2]
    relabelled_chain = [[chain[permutation[i]][permutation[j]] for j in range(5)] for i in range(5)]
    return [chain, antichain, m3, n5, vee_plus_point, relabelled_chain]


def cases(ctx, rng):
    del rng
    mod = module("posets")
    relations = ctx.explicit(lk.naturals(labelled_posets()))
    out = []
    for op in OPS:
        args = {"limit": 3}
        if op == "order_polynomial":
            args["t"] = 3
        out.append(Case("six labelled relations", relations, op, args))

    dictionary = lk.naturals([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1]])
    subset_family = ctx.subsets(dictionary, 4)
    divisor_family = ctx.range(1, 13)
    for i, op in enumerate(OPS):
        args = {"t": 4} if op == "order_polynomial" else {"limit": 3}
        out.append(Case("four chosen subsets", subset_family, op, args, reductions=rotate(mod, op, i)))
        out.append(Case("divisors 1 through 12", divisor_family, op, args, reductions=rotate(mod, op, i + 1)))

    out += [
        Case("non-poset relation", ctx.explicit(lk.naturals([[[1, 1, 0], [0, 1, 1], [0, 0, 1]]])), "width", oracle=False),
        Case("duplicate subsets", ctx.subsets(lk.naturals([[1, 0], [2, 0]]), 2), "height", oracle=False),
        Case("divisors of zero", ctx.range(0, 1), "height", oracle=False),
        Case("words are not posets", ctx.words(2, 3), "width", oracle=False),
    ]
    antichain25 = ctx.explicit(lk.naturals([relation_from_le(25, lambda i, j: i == j)]))
    out += [
        Case("25-element antichain", antichain25, "linear_extension_count", oracle=False),
        Case("25-element antichain", antichain25, "order_polynomial", {"t": 2}, oracle=False),
    ]

    all_four_bit_sets = lk.naturals([[int(mask & (1 << bit) != 0) for bit in range(4)] for mask in range(16)])
    out += [
        Case("eight_of_sixteen_subsets", lambda: ctx.subsets(all_four_bit_sets, 8), "linear_extension_count",
             bench="histogram", oracle=False,
             what="linear-extension distribution for all 8-element induced subposets of the Boolean lattice B_4"),
        Case("divisor_heights_100000", lambda: ctx.range(1, 100001), "height", bench="histogram", oracle=False,
             what="height distribution of the divisor posets of the first 100000 positive integers"),
    ]
    return out


def invariants(ctx):
    b3_elements = [[int(mask & (1 << bit) != 0) for bit in range(3)] for mask in range(8)]
    b3_relation = relation_from_le(8, lambda i, j: i & j == i)
    b3 = ctx.explicit(lk.naturals([b3_relation]))
    assert ctx.value("posets.linear_extension_count", b3).values == [48]
    assert ctx.value("posets.width", b3).values == [3]
    assert ctx.value("posets.height", b3).values == [4]
    assert ctx.value("posets.is_distributive", b3).values == [1]
    mu = ctx.value("posets.mobius_function", b3).member(0)
    assert mu[0][7] == -1

    as_subsets = ctx.subsets(lk.naturals(b3_elements), 8)
    for op in ("linear_extension_count", "width", "height", "is_lattice", "is_distributive"):
        assert ctx.run(f"posets.{op}", b3).export() == ctx.run(f"posets.{op}", as_subsets).export()

    d12 = ctx.range(12, 13)
    assert ctx.value("posets.height", d12).values == [4]
    assert ctx.value("posets.width", d12).values == [2]
    assert ctx.value("posets.is_distributive", d12).values == [1]
    assert ctx.value("posets.mobius_function", d12).member(0)[0][-1] == 0
