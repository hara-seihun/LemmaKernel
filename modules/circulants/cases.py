"""Circulant connection-set, spectrum, and CI-classification cases."""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lemmakernel as lk  # noqa: E402
from tools.harness import Case  # noqa: E402


def connections(ctx, n, k):
    return ctx.subsets_of(ctx.range(1, n), k)


def target(values):
    return lk.naturals([values])


def cases(ctx, rng):
    del rng
    directed5 = connections(ctx, 5, 2)
    directed6 = connections(ctx, 6, 2)
    undirected7 = connections(ctx, 7, 2)
    undirected8 = connections(ctx, 8, 2)

    out = [
        Case("directed spectra on Z5", directed5, "spectrum", {"n": 5, "directed": 1}),
        Case("explicit undirected spectra on Z5",
             ctx.explicit(ctx.naturals([[[1], [4]], [[2], [3]]])), "spectrum", {"n": 5, "directed": 0}),
        Case("undirected spectra on Z7", undirected7, "spectrum", {"n": 7, "directed": 0}),
        Case("directed isomorphism on Z6", directed6, "isomorphic",
             {"n": 6, "directed": 1, "target": target([1, 2]), "limit": 3}),
        Case("undirected isomorphism on Z8", undirected8, "isomorphic",
             {"n": 8, "directed": 0, "target": target([1]), "limit": 3}),
        Case("directed canonical sets on Z5", directed5, "is_canonical",
             {"n": 5, "directed": 1, "limit": 3}),
        Case("undirected canonical sets on Z8", undirected8, "is_canonical",
             {"n": 8, "directed": 0, "limit": 3}),
        Case("cyclic DCI orders", ctx.range(1, 25), "is_ci", {"directed": 1, "limit": 4}),
        Case("cyclic CI orders", ctx.range(1, 25), "is_ci", {"directed": 0, "limit": 4}),
    ]

    out += [
        Case("directed order 8 is outside Adam", connections(ctx, 8, 2), "is_canonical",
             {"n": 8, "directed": 1}, reductions=["count"], oracle=False),
        Case("invalid directed flag", ctx.range(1, 5), "is_ci", {"directed": 2},
             reductions=["count"], oracle=False),
        Case("connection set containing zero", ctx.subsets_of(ctx.range(0, 4), 1), "spectrum",
             {"n": 4, "directed": 1}, oracle=False),
        Case("zero order spectrum", connections(ctx, 4, 1), "spectrum",
             {"n": 0, "directed": 1}, oracle=False),
        Case("is_ci on connection sets", connections(ctx, 5, 2), "is_ci",
             {"directed": 1}, reductions=["count"], oracle=False),
    ]

    out += [
        Case("canonical 5-subsets on Z31", lambda: connections(ctx, 31, 5), "is_canonical",
             {"n": 31, "directed": 1}, bench="count", oracle=False,
             what="multiplier-isomorphism classes of valency-five circulant digraphs on 31 vertices"),
        Case("spectra of pairs on Z101", lambda: connections(ctx, 101, 2), "spectrum",
             {"n": 101, "directed": 1}, bench="all", oracle=False,
             what="exact character spectra of every valency-two circulant digraph on 101 vertices"),
    ]
    return out


def invariants(ctx):
    n, k = 11, 3
    family = connections(ctx, n, k)
    unit_perms = []
    for a in range(1, n):
        if math.gcd(a, n) == 1:
            unit_perms.append([(a * residue) % n - 1 for residue in range(1, n)])
    group = ctx.perms(n - 1, unit_perms)
    canonical = ctx.value("circulants.is_canonical", family, "count", n=n, directed=1).value
    orbit_count = ctx.value("burnside.orbit_count", family, group=group).values[0]
    assert canonical == orbit_count

    spectra = ctx.value("circulants.spectrum", connections(ctx, 7, 2), n=7, directed=1)
    for i in range(spectra.count):
        rows = spectra.member(i)
        assert all(len(row) == 2 for row in rows)
        assert rows[0] == [0, 0]
