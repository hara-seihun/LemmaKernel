"""Oracle and benchmark cases for lattice_of_subspaces.

Oracle families stay below a few dozen members because the incidence predicates ask Lean to run
row reduction for each member. The benchmark uses a larger Grassmannian.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lemmakernel as lk  # noqa: E402
from tools.harness import Case  # noqa: E402


def cases(ctx, rng):
    del rng
    mixed = lk.matrix(3, [
        [[1, 0, 0, 0], [2, 0, 0, 0]],
        [[1, 0, 1, 0], [0, 1, 0, 1]],
        [[1, 2, 0, 1], [0, 1, 1, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0]],
    ])
    line_f2 = lk.matrix(2, [[1, 0, 0]])
    plane_f3 = lk.matrix(3, [[1, 0, 1], [0, 1, 2]])

    out = [
        Case("Gaussian row F_2", ctx.range(0, 7), "gaussian_binomial", {"q": 2, "n": 6}),
        Case("three-step flag profiles", ctx.words(4, 3), "flag_count", {"q": 2, "n": 3}),
        Case("subspaces below mixed row spaces", ctx.explicit(mixed), "contained_subspace_count", {"h": 1}),
        Case("subspaces above mixed row spaces", ctx.explicit(mixed), "containing_subspace_count", {"h": 3}),
        Case("planes containing a fixed line", ctx.grassmannian(2, 3, 2), "contains",
             {"subspace": line_f2, "limit": 3}),
        Case("points in a fixed plane", ctx.grassmannian(3, 3, 1), "is_contained_in",
             {"subspace": plane_f3, "limit": 4}),

        Case("q below two", ctx.range(0, 4), "gaussian_binomial", {"q": 1, "n": 3},
             reductions=["all"], oracle=False),
        Case("flag count on a range", ctx.range(0, 4), "flag_count", {"q": 2, "n": 3},
             reductions=["all"], oracle=False),
        Case("wrong subspace width", ctx.grassmannian(2, 3, 1), "contains",
             {"subspace": lk.matrix(2, [[1, 0]]), "limit": 2}, reductions=["all"], oracle=False),

        Case("g84_contains_line", lambda: ctx.grassmannian(2, 8, 4), "contains",
             {"subspace": lk.matrix(2, [[1, 0, 0, 0, 0, 0, 0, 0]])},
             bench="count", oracle=False,
             what="4-subspaces of F_2^8 containing a fixed line, among all 200787 members of G_2(8,4)"),
    ]
    return out


def invariants(ctx):
    """Incidence counts on Grassmannians agree with the corresponding Gaussian binomials."""
    line = lk.matrix(2, [[1, 0, 0, 0, 0, 0]])
    spaces = ctx.grassmannian(2, 6, 3)
    above = ctx.value("lattice_of_subspaces.contains", spaces, "count", subspace=line).value
    expected_above = ctx.value("lattice_of_subspaces.gaussian_binomial", ctx.range(2, 3),
                               q=2, n=5).values[0]
    assert above == expected_above

    plane = lk.matrix(2, [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]])
    points = ctx.grassmannian(2, 6, 1)
    below = ctx.value("lattice_of_subspaces.is_contained_in", points, "count", subspace=plane).value
    expected_below = ctx.value("lattice_of_subspaces.gaussian_binomial", ctx.range(1, 2),
                               q=2, n=3).values[0]
    assert below == expected_below
