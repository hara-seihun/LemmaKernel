"""Oracle, rejection, invariant, and benchmark cases for quadratic_forms."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lemmakernel as lk  # noqa: E402
from tools.harness import Case  # noqa: E402


def cases(ctx, rng):
    del rng
    ternary_binary = ctx.symmetric_matrices(3, 2)
    hyperbolic_plane = lk.matrix(3, [[0, 1], [1, 0]])
    out = [
        Case("all ternary binary forms", ternary_binary, "form_type"),
        Case("all ternary binary forms", ternary_binary, "rank"),
        Case("all ternary binary forms", ternary_binary, "radical"),
        Case("all ternary binary forms", ternary_binary, "witt_index"),
        Case("all ternary binary forms", ternary_binary, "is_isometric",
             {"other": hyperbolic_plane, "limit": 3}),
        Case("all ternary binary forms", ternary_binary, "isotropic_point_count"),
    ]

    examples = lk.matrix(3, [
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 2]],
    ])
    examples_family = ctx.explicit(examples)
    other = lk.matrix(3, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    out += [
        Case("degenerate and parabolic examples", examples_family, "form_type", reductions=["all"]),
        Case("degenerate and parabolic examples", examples_family, "rank", reductions=["all"]),
        Case("degenerate and parabolic examples", examples_family, "radical"),
        Case("degenerate and parabolic examples", examples_family, "witt_index", reductions=["all"]),
        Case("degenerate and parabolic examples", examples_family, "is_isometric",
             {"other": other, "limit": 2}, reductions=["all", "count"]),
        Case("degenerate and parabolic examples", examples_family, "isotropic_point_count", reductions=["all"]),
    ]

    out += [
        Case("characteristic two", ctx.symmetric_matrices(2, 2), "rank",
             reductions=["all"], oracle=False),
        Case("nonsymmetric explicit form", ctx.explicit(lk.matrix(3, [[[1, 1], [0, 1]]])), "rank",
             reductions=["all"], oracle=False),
        Case("nonsquare explicit form", ctx.explicit(lk.matrix(3, [[[1, 0, 0], [0, 1, 0]]])), "rank",
             reductions=["all"], oracle=False),
        Case("wrong-sized comparison form", ternary_binary, "is_isometric",
             {"other": lk.matrix(3, [[1]]), "limit": 1}, reductions=["all"], oracle=False),
        Case("Grassmannian restrictions not shipped", ctx.grassmannian(3, 3, 2), "rank",
             reductions=["all"], oracle=False),
        Case("symmetric_4x4_F3", lambda: ctx.symmetric_matrices(3, 4), "form_type",
             what="type distribution of all 59,049 symmetric 4x4 matrices over F_3",
             bench="histogram", oracle=False),
    ]
    return out


def invariants(ctx):
    family = ctx.symmetric_matrices(3, 3)
    assert ctx.value("quadratic_forms.rank", family, "histogram").bins == \
        ctx.value("gfp.rank", family, "histogram").bins

    forms = lk.matrix(3, [
        [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 2]],
    ])
    form_family = ctx.explicit(forms)
    radicals = ctx.value("quadratic_forms.radical", form_family)
    for i in range(forms.count):
        matrix = forms.member(i)
        for vector in radicals.member(i):
            assert [sum(row[j] * vector[j] for j in range(3)) % 3 for row in matrix] == [0, 0, 0]

    assert ctx.value("quadratic_forms.form_type", form_family).values == [0, 2, 2]
    assert ctx.value("quadratic_forms.witt_index", form_family).values == [1, 1, 1]
    assert ctx.value("quadratic_forms.isotropic_point_count", form_family).values == [7, 4, 4]

    square_discriminant = lk.matrix(3, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    flags = ctx.value("quadratic_forms.is_isometric", form_family, other=square_discriminant).values
    assert flags == [0, 1, 0]
