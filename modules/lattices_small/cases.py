"""Oracle and benchmark cases for small integral lattices."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lemmakernel as lk  # noqa: E402
from tools.harness import Case  # noqa: E402


def cases(ctx, rng):
    del rng
    rank_two = ctx.explicit(lk.gram([
        [[1, 0], [0, 1]],
        [[2, -1], [-1, 2]],
        [[2, 0], [0, 2]],
        [[2, 3], [3, 5]],
        [[4, 1], [1, 2]],
        [[2, 1], [1, 2]],
    ]))
    rank_two_series = ctx.explicit(lk.gram([
        [[1, 0], [0, 1]],
        [[2, -1], [-1, 2]],
        [[2, 0], [0, 2]],
    ]))
    out = [
        Case("rank-two integral lattices", rank_two, "minimum"),
        Case("rank-two integral lattices", rank_two, "kissing_number"),
        Case("rank-two integral lattices", rank_two, "is_unimodular", {"limit": 3}),
        Case("rank-two integral lattices", rank_two, "is_even", {"limit": 3}),
        Case("rank-two theta prefixes", rank_two_series, "theta_series", {"bound": 5}),
        Case("rank-two short vectors", rank_two_series, "short_vectors", {"bound": 4}),
    ]

    z2_index2 = ctx.sublattices(lk.gram([[1, 0], [0, 1]]), 2)
    out += [
        Case("index-two sublattices of Z2", z2_index2, "minimum", reductions=["histogram"]),
        Case("index-two sublattices of Z2", z2_index2, "kissing_number", reductions=["all"]),
        Case("index-two sublattices of Z2", z2_index2, "is_unimodular", reductions=["count"]),
        Case("index-two sublattices of Z2", z2_index2, "is_even", {"limit": 2}, reductions=["hits"]),
        Case("index-two sublattices of Z2", z2_index2, "theta_series", {"bound": 4}),
        Case("index-two sublattices of Z2", z2_index2, "short_vectors", {"bound": 3}),
    ]

    out += [
        Case("indefinite Gram", ctx.explicit(lk.gram([[1, 2], [2, 1]])), "minimum", oracle=False),
        Case("nonsymmetric Gram", ctx.explicit(lk.gram([[2, 1], [0, 2]])), "minimum", oracle=False),
        Case("field matrix is not a Gram", ctx.explicit(lk.matrix(2, [[1, 0], [0, 1]])), "minimum", oracle=False),
    ]

    z4 = ctx.explicit(lk.gram([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
    a4 = ctx.explicit(lk.gram([[2, -1, 0, 0], [-1, 2, -1, 0], [0, -1, 2, -1], [0, 0, -1, 2]]))
    out += [
        Case("Z4 theta through 12", z4, "theta_series", {"bound": 12}, bench="all", oracle=False,
             what="theta coefficients through q^12 for the standard rank-four lattice"),
        Case("A4 vectors through norm 12", a4, "short_vectors", {"bound": 12}, bench="all", oracle=False,
             what="all nonzero A4 coordinate vectors of squared norm at most 12"),
    ]
    return out


def invariants(ctx):
    grams = ctx.explicit(lk.gram([
        [[1, 0], [0, 1]],
        [[2, -1], [-1, 2]],
        [[2, 3], [3, 5]],
    ]))
    minima = ctx.value("lattices_small.minimum", grams).values
    kissing = ctx.value("lattices_small.kissing_number", grams).values
    theta = ctx.value("lattices_small.theta_series", grams, bound=8)
    vectors = ctx.value("lattices_small.short_vectors", grams, bound=8)
    for i, (minimum, number) in enumerate(zip(minima, kissing)):
        coefficients = theta.member(i)
        short = vectors.member(i)
        assert coefficients[0] == 1
        assert coefficients[minimum] == number
        assert len(short) == sum(coefficients) - 1
        assert short == sorted(short)
    family = ctx.sublattices(lk.gram([[1, 0], [0, 1]]), 4)
    assert ctx.size(family) == 7
    assert family.value().kind == "sublattices"
