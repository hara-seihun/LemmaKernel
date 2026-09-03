"""Oracle and benchmark cases for Latin squares."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lemmakernel as lk  # noqa: E402
from tools.harness import Case  # noqa: E402


def cyclic_square(n):
    return [[(i + j) % n for j in range(n)] for i in range(n)]


def cases(ctx, rng):
    del rng
    order3 = ctx.latin_squares(3)
    out = [
        Case("all order 3 squares", order3, "is_latin", {"limit": 3}),
        Case("all order 3 squares", order3, "has_orthogonal_mate", {"limit": 3}),
        Case("all order 3 squares", order3, "transversal_count"),
        Case("all order 3 squares", order3, "is_group_table", {"limit": 3}),
        Case("all order 3 squares", order3, "isotopy_canonical_form"),
    ]

    examples2 = ctx.explicit(lk.naturals([
        cyclic_square(2),
        [[1, 0], [0, 1]],
        [[0, 0], [1, 1]],
    ]))
    out += [
        Case("order 2 examples", examples2, "is_latin", reductions=["all"]),
        Case("order 2 examples", examples2, "has_orthogonal_mate", reductions=["all"]),
        Case("order 2 examples", examples2, "transversal_count", reductions=["all"]),
        Case("order 2 examples", examples2, "is_group_table", reductions=["all"]),
        Case("order 2 examples", examples2, "isotopy_canonical_form"),
    ]

    order4_groups = ctx.explicit(lk.naturals([
        cyclic_square(4),
        [[i ^ j for j in range(4)] for i in range(4)],
    ]))
    out += [
        Case("order 4 group tables", order4_groups, "has_orthogonal_mate", reductions=["all"]),
        Case("order 4 group tables", order4_groups, "transversal_count", reductions=["all"]),
        Case("order 4 group tables", order4_groups, "is_group_table", reductions=["all"]),
        Case("order 4 group tables", order4_groups, "isotopy_canonical_form"),
        Case("non-square array", ctx.explicit(lk.naturals([[0, 1, 2], [1, 2, 0]])),
             "is_latin", reductions=["all"], oracle=False),
        Case("order5_transversals", lambda: ctx.latin_squares(5), "transversal_count",
             reductions=["histogram"], bench="histogram", oracle=False,
             what="transversal-count distribution over all 161,280 Latin squares of order 5"),
    ]
    return out


def invariants(ctx):
    expected = [1, 2, 12, 576, 161280]
    assert [ctx.size(ctx.latin_squares(n)) for n in range(1, 6)] == expected

    order4 = ctx.latin_squares(4)
    latin = ctx.value("latin_squares.is_latin", order4, "count")
    assert latin.value == latin.family_size == 576

    samples = [ctx.member(order4, i).value().member(0) for i in (0, 17, 575)]
    first = ctx.value("latin_squares.isotopy_canonical_form", ctx.explicit(lk.naturals(samples)))
    second = ctx.value("latin_squares.isotopy_canonical_form", ctx.explicit(first))
    assert first.encode() == second.encode()

    z2 = ctx.explicit(lk.naturals(cyclic_square(2)))
    z3 = ctx.explicit(lk.naturals(cyclic_square(3)))
    assert ctx.value("latin_squares.has_orthogonal_mate", z2).values == [0]
    assert ctx.value("latin_squares.has_orthogonal_mate", z3).values == [1]
