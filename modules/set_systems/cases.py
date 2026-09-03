"""Oracle and benchmark cases for set_systems.

The oracle inputs have at most 28 members. Benchmark inputs put millions of set systems behind a
small ``subsets_of(words(...))`` description.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.harness import Case  # noqa: E402


def systems(ctx, n, m):
    return ctx.subsets_of(ctx.words(2, n), m)


def cases(ctx, rng):
    del rng
    out = [
        Case("intersecting pairs on 3 points", systems(ctx, 3, 2), "is_intersecting", {"limit": 3}),
        Case("antichain pairs on 3 points", systems(ctx, 3, 2), "is_antichain", {"limit": 3}),
        Case("3-sunflowers on 2 points", systems(ctx, 2, 3), "is_sunflower_free", {"k": 3, "limit": 3}),
        Case("degrees of pairs on 3 points", systems(ctx, 3, 2), "max_degree"),
        Case("shadows of pairs on 3 points", systems(ctx, 3, 2), "shadow_size"),
        Case("EKR extremals on 2 points", systems(ctx, 2, 1), "is_ekr_extremal", {"limit": 3}),
        Case("Sperner extremals on 2 points", systems(ctx, 2, 2), "is_sperner_extremal", {"limit": 3}),
        Case("2-sunflowers on 3 points", systems(ctx, 3, 2), "is_sunflower_free", {"k": 2}, reductions=["count"]),
        Case("sunflower larger than system", systems(ctx, 2, 2), "is_sunflower_free", {"k": 4}, reductions=["all"]),
    ]

    out += [
        Case("intersecting_32_choose_5", lambda: systems(ctx, 5, 5), "is_intersecting", bench="count", oracle=False,
             what="count the pairwise-intersecting 5-set systems on a 5-point ground set"),
        Case("antichains_64_choose_5", lambda: systems(ctx, 6, 5), "is_antichain", bench="count", oracle=False,
             what="count 5-element antichains in the Boolean lattice on 6 points"),
        Case("shadows_32_choose_4", lambda: systems(ctx, 5, 4), "shadow_size", bench="histogram", oracle=False,
             what="distribution of lower-shadow sizes for every 4-set system on 5 points"),
    ]

    out += [
        Case("subsets of integers", ctx.subsets_of(ctx.range(0, 4), 2), "is_intersecting", reductions=["all"], oracle=False),
        Case("ternary words", ctx.subsets_of(ctx.words(3, 2), 2), "is_antichain", reductions=["all"], oracle=False),
        Case("sunflower k=1", systems(ctx, 2, 2), "is_sunflower_free", {"k": 1}, reductions=["all"], oracle=False),
    ]
    return out


def invariants(ctx):
    ekr = systems(ctx, 4, 3)
    assert ctx.value("set_systems.is_ekr_extremal", ekr, "count").value == 8

    sperner = systems(ctx, 3, 3)
    assert ctx.value("set_systems.is_sperner_extremal", sperner, "count").value == 2

    pairs = systems(ctx, 3, 2)
    degrees = ctx.value("set_systems.max_degree", pairs).values
    shadows = ctx.value("set_systems.shadow_size", pairs).values
    assert min(degrees) == 1 and max(degrees) == 2
    assert min(shadows) == 1 and max(shadows) == 5
