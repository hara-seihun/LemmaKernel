"""Oracle and benchmark cases for binary matrices read as signs."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lemmakernel as lk  # noqa: E402
from tools.harness import Case, module, rotate  # noqa: E402

PREDICATES = ["is_hadamard", "is_skew", "is_regular", "is_conference"]

SYLVESTER4 = [
    [0, 0, 0, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 1],
    [0, 1, 1, 0],
]

SKEW4 = [
    [0, 0, 0, 0],
    [1, 0, 0, 1],
    [1, 1, 0, 0],
    [1, 0, 1, 0],
]

REGULAR4 = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
]


def cases(ctx, rng):
    mod = module("hadamard")
    all2 = ctx.all_matrices(2, 2, 2)
    out = [Case("all 2x2 signs", all2, op, {"limit": 3}) for op in PREDICATES]
    out.append(Case("all 2x2 signs", all2, "canonical_form"))

    named = ctx.explicit(lk.matrix(2, [
        SYLVESTER4,
        SKEW4,
        REGULAR4,
        [[0] * 4 for _ in range(4)],
        [[rng.randrange(2) for _ in range(4)] for _ in range(4)],
    ]))
    for i, op in enumerate(PREDICATES):
        out.append(Case("named 4x4 signs", named, op, {"limit": 2}, reductions=rotate(mod, op, i)))

    dictionary = lk.matrix(2, [
        [0, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 1],
        [0, 1, 1, 0], [1, 0, 0, 1], [1, 1, 0, 0],
    ])
    subsets = ctx.subsets(dictionary, 4)
    for i, op in enumerate(PREDICATES):
        out.append(Case("4-subsets of sign rows", subsets, op, {"limit": 2}, reductions=rotate(mod, op, i + 1)))

    transformed = ctx.transform(ctx.all_matrices(2, 2, 2), lk.matrix(2, [[1, 1], [0, 1]]))
    for i, op in enumerate(PREDICATES):
        out.append(Case("transformed 2x2 signs", transformed, op, {"limit": 2}, reductions=rotate(mod, op, i + 2)))

    ternary = ctx.explicit(lk.matrix(3, [[[0, 1], [2, 0]]]))
    for op in PREDICATES + ["canonical_form"]:
        out.append(Case("ternary sign input", ternary, op, reductions=["all"], oracle=False))
    out.append(Case("canonical form with count", all2, "canonical_form", oracle=False))

    out.append(Case(
        "all_4x4_hadamard",
        lambda: ctx.all_matrices(2, 4, 4),
        "is_hadamard",
        bench="count",
        oracle=False,
        what="count all 4x4 Hadamard sign matrices among the 65,536 binary matrices",
    ))
    return out


def _equivalent(m, row_order, col_order, row_flips, col_flips):
    return [[m[row_order[i]][col_order[j]] ^ row_flips[i] ^ col_flips[j]
             for j in range(len(col_order))] for i in range(len(row_order))]


def invariants(ctx):
    variants = [
        SYLVESTER4,
        _equivalent(SYLVESTER4, [2, 0, 3, 1], [3, 1, 0, 2], [1, 0, 1, 1], [0, 1, 0, 1]),
        _equivalent(SYLVESTER4, [3, 2, 1, 0], [1, 3, 2, 0], [0, 1, 1, 0], [1, 1, 0, 0]),
    ]
    forms = ctx.value("hadamard.canonical_form", ctx.explicit(lk.matrix(2, variants)))
    assert forms.member(0) == forms.member(1) == forms.member(2)
    again = ctx.value("hadamard.canonical_form", ctx.explicit(forms))
    assert again.tolist() == forms.tolist()

    known = ctx.explicit(lk.matrix(2, [SYLVESTER4, SKEW4, REGULAR4]))
    assert ctx.value("hadamard.is_hadamard", known).values == [1, 1, 1]
    assert ctx.value("hadamard.is_skew", known).values == [0, 1, 0]
    assert ctx.value("hadamard.is_regular", known).values == [0, 0, 1]
