"""Cases for linear codes up to monomial equivalence."""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lemmakernel as lk  # noqa: E402
from tools.harness import Case, module, rotate  # noqa: E402

OPS = ["is_canonical", "canonical_index", "canonical_form", "orbit_size", "aut_order"]


def binary_batch():
    """Zero, repetition, and three [4,2] codes: two of them equivalent, one not."""
    return lk.matrix(2, [
        [[0, 0, 0, 0], [0, 0, 0, 0]],
        [[1, 1, 1, 1], [0, 0, 0, 0]],
        [[1, 1, 0, 0], [0, 0, 1, 1]],
        [[1, 0, 1, 0], [0, 1, 0, 1]],
        [[1, 1, 1, 0], [0, 1, 1, 1]],
    ])


def cases(ctx, rng):
    del rng
    mod = module("code_equivalence")
    explicit = ctx.explicit(binary_batch())
    out = [
        Case(f"explicit binary {op.replace('_', ' ')}", explicit, op, {"scalars": 1, "limit": 3})
        for op in OPS
    ]

    families = [
        ("binary lines", ctx.grassmannian(2, 4, 1), 1),
        ("ternary lines", ctx.grassmannian(3, 3, 1), 1),
        ("ternary lines up to permutation", ctx.grassmannian(3, 3, 1), 0),
        ("parity extension", ctx.transform(ctx.grassmannian(2, 3, 1),
                                           lk.matrix(2, [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])), 1),
        ("ternary stacked generators", ctx.stack(ctx.grassmannian(3, 3, 1), lk.matrix(3, [[1, 1, 1]])), 1),
    ]
    for i, (name, family, scalars) in enumerate(families):
        for j, op in enumerate(OPS):
            out.append(Case(name, family, op, {"scalars": scalars, "limit": 2},
                            reductions=rotate(mod, op, i + j)))

    out += [
        # The classes of the whole Grassmannian, which is what a caller asks this module for.
        Case("binary [4,2] classes", ctx.grassmannian(2, 4, 2), "is_canonical", {"scalars": 1},
             reductions=["count"]),
        Case("words are not generator matrices", ctx.words(2, 3), "canonical_index",
             {"scalars": 1}, oracle=False),
        Case("invalid scalar flag", explicit, "canonical_index", {"scalars": 2}, oracle=False),
        Case(
            "binary [6,3] classes",
            lambda: ctx.grassmannian(2, 6, 3),
            "is_canonical",
            {"scalars": 1},
            reductions=["count"],
            bench="count",
            oracle=False,
            what="how many inequivalent binary [6,3] codes there are, out of 1395 subspaces",
        ),
    ]
    return out


def monomial_generators(p, n):
    """The monomial group of F_p^n as matrices: two permutations generate S_n, and one diagonal
    adds the scalars. subspace_orbits acts with these on the right, so they permute and scale
    coordinates exactly as this module does."""
    swap = [1, 0] + list(range(2, n))
    cycle = [(i + 1) % n for i in range(n)]
    generators = [[[int(perm[i] == j) for j in range(n)] for i in range(n)] for perm in (swap, cycle)]
    if p > 2:
        generators.append([[(2 if i == 0 else 1) if i == j else 0 for j in range(n)] for i in range(n)])
    return lk.matrix(p, generators)


def invariants(ctx):
    # The monomial group is a matrix group, so subspace_orbits answers the same questions from a
    # breadth-first orbit search instead of a search over information sets.
    for p, n, k in [(2, 5, 2), (3, 3, 1), (3, 4, 2)]:
        family = ctx.grassmannian(p, n, k)
        group = monomial_generators(p, n)
        for op in ("canonical_index", "orbit_size"):
            assert (ctx.value(f"code_equivalence.{op}", family, scalars=1).values ==
                    ctx.value(f"subspace_orbits.{op}", family, group=group, projective=0).values)

    p, n, k = 2, 5, 2
    family = ctx.grassmannian(p, n, k)
    size = ctx.size(family)
    canonical = ctx.value("code_equivalence.is_canonical", family, scalars=1).values
    indices = ctx.value("code_equivalence.canonical_index", family, scalars=1).values
    orbits = ctx.value("code_equivalence.orbit_size", family, scalars=1).values
    auts = ctx.value("code_equivalence.aut_order", family, scalars=1).values
    order = (p - 1) ** n * math.factorial(n)

    # A Grassmannian member's own index is its position, so a member is canonical exactly when it
    # is its own class representative, and each class is counted once with its full orbit.
    assert [i for i in range(size) if canonical[i]] == sorted(set(indices))
    assert all(indices[i] <= i for i in range(size))
    assert all(orbits[i] * auts[i] == order for i in range(size))
    assert sum(orbits[i] for i in range(size) if canonical[i]) == size
    assert all(orbits[i] == orbits[indices[i]] for i in range(size))

    forms = ctx.value("code_equivalence.canonical_form", family, scalars=1)
    representatives = ctx.value("gfp.rref", family).tolist()
    assert all(forms.member(i) == representatives[indices[i]] for i in range(size))

    # Without scalars the group is S_n, and its stabiliser is the permutation automorphism group
    # linear_codes reports.
    permuted = ctx.value("code_equivalence.aut_order", family, scalars=0).values
    assert permuted == ctx.value("linear_codes.aut_order", family).values
    # Over F_2 the monomial group is S_n, so both settings agree everywhere.
    assert permuted == auts
