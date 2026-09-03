"""Oracle and benchmark inputs for point sets in finite projective spaces."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lemmakernel as lk  # noqa: E402
from tools.harness import Case, projective_points  # noqa: E402

OPS = [
    "is_arc", "is_cap", "is_blocking_set", "is_hyperoval", "is_ovoid",
    "max_collinear", "spanned_rank", "secant_count", "tangent_count", "passant_count",
]


def cases(ctx, rng):
    # Three of these four points are a Fano-plane line. Its four triples include one line and
    # three triangles, which gives variation in rank, arc, cap and every line statistic.
    mixed = lk.matrix(2, [[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1]])
    triples = ctx.subsets(mixed, 3)
    out = [Case("Fano line and triangles", triples, op, {"limit": 2}) for op in OPS]

    pg23 = lk.matrix(3, [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
    ternary_triples = ctx.subsets(pg23, 3)
    for op in OPS:
        reduction = "count" if op.startswith("is_") else "histogram"
        out.append(Case("four ternary points", ternary_triples, op, reductions=[reduction]))

    wide_prime = lk.matrix(257, [[1, 0], [0, 1], [1, 256]])
    out.append(Case("F_257 projective rank", ctx.subsets(wide_prime, 2), "spanned_rank", reductions=["all"]))

    hyperoval = lk.matrix(2, [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
    out.append(Case("Fano hyperoval", ctx.subsets(hyperoval, 4), "is_hyperoval", {"limit": 1}))

    # e1,e2,e3,e4 and e1+e2+e3+e4 form the five-point ovoid in PG(3,2).
    ovoid = lk.matrix(2, [
        [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 1, 1, 1],
    ])
    ovoid_family = ctx.subsets(ovoid, 5)
    out += [
        Case("PG(3,2) ovoid", ovoid_family, "is_ovoid", {"limit": 1}),
        Case("PG(3,2) ovoid line counts", ovoid_family, "secant_count"),
        Case("PG(3,2) ovoid line counts", ovoid_family, "tangent_count"),
        Case("PG(3,2) ovoid line counts", ovoid_family, "passant_count"),
    ]

    out += [
        Case(
            "explicit point set",
            ctx.explicit(lk.matrix(2, [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])),
            "is_arc",
            reductions=["count"],
            oracle=False,
        ),
        Case("zero projective point", ctx.subsets(lk.matrix(2, [[0, 0], [1, 0]]), 1), "is_arc", reductions=["count"], oracle=False),
        Case("unnormalised projective point", ctx.subsets(lk.matrix(3, [[2, 0], [0, 1]]), 1), "is_arc", reductions=["count"], oracle=False),
        Case("duplicate projective point", ctx.subsets(lk.matrix(2, [[1, 0], [1, 0]]), 1), "is_arc", reductions=["count"], oracle=False),
    ]

    out.append(Case(
        "pg32_six_sets",
        lambda: ctx.subsets(projective_points(ctx, 2, 4), 6),
        "max_collinear",
        what="maximum line intersection for every 6-point subset of the 15 points of PG(3,2)",
        bench="histogram",
        oracle=False,
    ))
    return out


def invariants(ctx):
    hyperoval = lk.matrix(2, [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
    h = ctx.subsets(hyperoval, 4)
    assert ctx.value("projective_sets.is_hyperoval", h).values == [1]
    assert ctx.value("projective_sets.secant_count", h).values == [6]
    assert ctx.value("projective_sets.tangent_count", h).values == [0]
    assert ctx.value("projective_sets.passant_count", h).values == [1]

    ovoid = lk.matrix(2, [
        [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 1, 1, 1],
    ])
    o = ctx.subsets(ovoid, 5)
    assert ctx.value("projective_sets.is_ovoid", o).values == [1]
    assert ctx.value("projective_sets.secant_count", o).values == [10]
    assert ctx.value("projective_sets.tangent_count", o).values == [15]
    assert ctx.value("projective_sets.passant_count", o).values == [10]
