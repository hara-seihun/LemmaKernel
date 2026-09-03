"""Oracle, rejection, benchmark, and invariant cases for group growth."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lemmakernel as lk  # noqa: E402
from tools.harness import Case  # noqa: E402


def equations(ctx, generators, pairs):
    """Encode variable-length word equations as fixed-width natural-number rows."""
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
    free_1 = equations(ctx, 1, [])
    free_2 = equations(ctx, 2, [])
    cyclic_2 = equations(ctx, 1, [([1], [0]), ([0, 0], [])])

    out = [
        Case("free group F1 balls", ctx.range(0, 4), "ball_size",
             {"generators": 1, "relations": free_1}),
        Case("free group F2 spheres", ctx.range(0, 4), "sphere_size",
             {"generators": 2, "relations": free_2}),
        Case("cyclic group C2 geodesic words", ctx.range(0, 4), "geodesic_count",
             {"generators": 1, "relations": cyclic_2}),
        Case("free group F1 length-three words", ctx.words(2, 3), "is_geodesic",
             {"generators": 1, "relations": free_1, "limit": 3}),
        Case("cyclic group C2 spheres", ctx.range(0, 4), "sphere_size",
             {"generators": 1, "relations": cyclic_2}, reductions=["all"]),
        Case("cyclic group C2 letters", ctx.words(2, 1), "is_geodesic",
             {"generators": 1, "relations": cyclic_2, "limit": 2}, reductions=["all", "count"]),
    ]

    invalid_symbol = ctx.naturals([[1, 0, 2]])
    raw_c3 = equations(ctx, 1, [([0, 0, 0], [])])
    out += [
        Case("ball size on words", ctx.words(2, 2), "ball_size",
             {"generators": 1, "relations": free_1}, reductions=["all"], oracle=False),
        Case("geodesic predicate on radii", ctx.range(0, 3), "is_geodesic",
             {"generators": 1, "relations": free_1}, reductions=["all"], oracle=False),
        Case("wrong word alphabet", ctx.words(3, 2), "is_geodesic",
             {"generators": 1, "relations": free_1}, reductions=["all"], oracle=False),
        Case("relation has an invalid symbol", ctx.range(0, 3), "sphere_size",
             {"generators": 1, "relations": invalid_symbol}, reductions=["all"], oracle=False),
        Case("nonconfluent C3 relator", ctx.range(0, 3), "sphere_size",
             {"generators": 1, "relations": raw_c3}, reductions=["all"], oracle=False),
    ]

    out.append(Case(
        "free group F2 geodesic prefix",
        lambda: ctx.range(0, 10),
        "geodesic_count",
        {"generators": 2, "relations": free_2},
        bench="all",
        oracle=False,
        what="geodesic-word counts through length 9 in the rank-two free group",
    ))
    return out


def invariants(ctx):
    free_2 = equations(ctx, 2, [])
    radii = ctx.range(0, 13)
    spheres = ctx.value("words_and_growth.sphere_size", radii,
                        generators=2, relations=free_2).values
    balls = ctx.value("words_and_growth.ball_size", radii,
                      generators=2, relations=free_2).values
    geodesics = ctx.value("words_and_growth.geodesic_count", radii,
                          generators=2, relations=free_2).values
    expected_spheres = [1] + [4 * 3 ** (n - 1) for n in range(1, 13)]
    assert spheres == expected_spheres == geodesics
    assert balls == [sum(expected_spheres[:n + 1]) for n in range(13)]

    cyclic_2 = equations(ctx, 1, [([1], [0]), ([0, 0], [])])
    short = ctx.range(0, 5)
    assert ctx.value("words_and_growth.sphere_size", short,
                     generators=1, relations=cyclic_2).values == [1, 1, 0, 0, 0]
    assert ctx.value("words_and_growth.ball_size", short,
                     generators=1, relations=cyclic_2).values == [1, 2, 2, 2, 2]
    assert ctx.value("words_and_growth.geodesic_count", short,
                     generators=1, relations=cyclic_2).values == [1, 2, 0, 0, 0]
