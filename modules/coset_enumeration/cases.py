"""Oracle and benchmark cases for bounded Todd-Coxeter enumeration."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lemmakernel as lk  # noqa: E402
from tools.harness import Case  # noqa: E402


def equations(ctx, generators, pairs):
    """Use the equation encoding shared with words_and_growth."""
    pairs = pairs or [([], [])]
    width = max(2 + len(left) + len(right) for left, right in pairs)
    alphabet = 2 * generators
    rows = []
    for left, right in pairs:
        row = [len(left), len(right), *left, *right]
        rows.append(row + [alphabet] * (width - len(row)))
    return ctx.naturals(rows)


def cases(ctx, rng):
    del rng
    c3_family = ctx.words(2, 2)
    c3_args = {
        "generators": 1,
        "relations": equations(ctx, 1, [([0, 0, 0], [])]),
        "subgroup": lk.naturals([[0, 1]]),
        "max_cosets": 8,
    }
    finite_args = {k: v for k, v in c3_args.items() if k != "subgroup"}

    mixed_family = ctx.words(2, 2)
    mixed_args = {
        "generators": 1,
        "relations": equations(ctx, 1, [([0, 1], [])]),
        "max_cosets": 6,
        "limit": 2,
    }
    c6_subgroup_args = {
        "generators": 1,
        "relations": equations(ctx, 1, [([0] * 6, [])]),
        "subgroup": lk.naturals([[0, 0]]),
        "max_cosets": 12,
    }

    out = [
        Case("C3 presentations", c3_family, "index", c3_args),
        Case("C3 presentations", c3_family, "permutation_representation", c3_args),
        Case("C3 below its table bound", c3_family, "index", {**c3_args, "max_cosets": 2},
             reductions=["all"]),
        Case("C3 below its table bound", c3_family, "permutation_representation",
             {**c3_args, "max_cosets": 2}),
        Case("C6 modulo its square subgroup", c3_family, "index", c6_subgroup_args,
             reductions=["all"]),
        Case("C6 modulo its square subgroup", c3_family, "permutation_representation", c6_subgroup_args),
        Case("finite cyclic presentations", c3_family, "is_finite", {**finite_args, "limit": 2}),
        Case("finite and unbounded cyclic presentations", mixed_family, "is_finite", mixed_args),
        Case("padded fixed relations", ctx.words(4, 1), "is_finite",
             {"generators": 2, "relations": equations(ctx, 2, [([0], [1]), ([2, 2, 2], [])]),
              "max_cosets": 12}, reductions=["all"]),
    ]

    out += [
        Case("wrong alphabet", ctx.words(3, 2), "index", c3_args, reductions=["all"], oracle=False),
        Case("relation has an invalid symbol", ctx.words(2, 2), "index",
             {**c3_args, "relations": lk.naturals([[1, 0, 4]])}, reductions=["all"], oracle=False),
        Case("zero bound", ctx.words(2, 2), "index", {**c3_args, "max_cosets": 0},
             reductions=["all"], oracle=False),
        Case("range is not presentations", ctx.range(0, 4), "index", c3_args,
             reductions=["all"], oracle=False),
    ]

    out.append(Case(
        "two involutions with a varying relator",
        lambda: ctx.words(4, 6),
        "is_finite",
        {"generators": 2, "relations": equations(ctx, 2, [([0, 0], []), ([2, 2], [])]),
         "max_cosets": 64},
        bench="count",
        oracle=False,
        what="which length-6 relators make <x,y | x^2,y^2,w> close within 64 Todd-Coxeter rows",
    ))
    return out


def invariants(ctx):
    trivial_subgroup = lk.naturals([[0, 1]])
    s3_relations = equations(ctx, 2, [([0], [1]), ([2, 2, 2], [])])
    s3_indices = ctx.value(
        "coset_enumeration.index", ctx.words(4, 4), generators=2, relations=s3_relations,
        subgroup=trivial_subgroup, max_cosets=32,
    ).values
    assert s3_indices[34] == 6

    a4_relations = equations(ctx, 2, [([0], [1]), ([2, 2, 2], []), ([0, 2, 0, 2, 0, 2], [])])
    a4_indices = ctx.value(
        "coset_enumeration.index", ctx.words(4, 2), generators=2, relations=a4_relations,
        subgroup=trivial_subgroup, max_cosets=64,
    ).values
    assert a4_indices[1] == 12
