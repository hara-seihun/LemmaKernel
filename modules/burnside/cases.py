"""Burnside cases. Oracle inputs keep both the group and the family small. Bench inputs have up
to 10^18 family members because the generic backend never visits them.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lemmakernel as lk  # noqa: E402
from tools.harness import Case, cyclic, dihedral, symmetric, unit_vectors  # noqa: E402


def cases(ctx, rng):
    del rng
    I3 = unit_vectors(2, 3)
    I4 = unit_vectors(2, 4)
    C3 = ctx.perms(3, cyclic(3))
    C4 = ctx.perms(4, cyclic(4))
    D4 = ctx.perms(4, dihedral(4))
    S4 = ctx.perms(4, symmetric(4))

    out = [
        Case("C4 on pairs", ctx.subsets(I4, 2), "orbit_count", {"group": C4}),
        Case("D4 on pairs", ctx.subsets(I4, 2), "orbit_count", {"group": D4}),
        Case("S4 on pairs", ctx.subsets(I4, 2), "orbit_count", {"group": S4}),
        Case("C3 on ternary words", ctx.words(3, 3), "orbit_count", {"group": C3}),
        Case("D4 on binary words", ctx.words(2, 4), "orbit_count", {"group": D4}),
        Case("C4 on subsets_of", ctx.subsets_of(ctx.range(0, 4), 2), "orbit_count", {"group": C4}),
        Case("C4 cycle index", ctx.words(2, 4), "cycle_index", {"group": C4}),
        Case("D4 cycle index", ctx.subsets(I4, 2), "cycle_index", {"group": D4}),
        Case("S4 cycle index", ctx.words(2, 4), "cycle_index", {"group": S4}),
        Case("quarter-turn fixes pairs", ctx.subsets(I4, 2), "fixed_count", {"g": ctx.perms(4, cyclic(4))}),
        Case("reflection fixes pairs", ctx.subsets(I4, 2), "fixed_count", {"g": ctx.perms(4, [dihedral(4)[1]])}),
        Case("swap fixes binary words", ctx.words(2, 4), "fixed_count", {"g": ctx.perms(4, [[1, 0, 2, 3]])}),
        Case("cycle fixes ternary words", ctx.words(3, 3), "fixed_count", {"g": C3}),
        Case("cycle fixes subsets_of", ctx.subsets_of(ctx.range(0, 3), 2), "fixed_count", {"g": C3}),
    ]

    out += [
        Case("permutations on a range", ctx.range(0, 4), "orbit_count", {"group": C4}, oracle=False),
        Case("C4 on length-five words", ctx.words(2, 5), "orbit_count", {"group": C4}, oracle=False),
        Case("matrix group on words", ctx.words(2, 2), "orbit_count",
             {"group": lk.matrix(2, [[[1, 0], [0, 1]]])}, oracle=False),
        Case("two permutations passed as g", ctx.words(2, 4), "fixed_count",
             {"g": ctx.perms(4, [list(range(4)), cyclic(4)[0]])}, oracle=False),
    ]

    I60 = unit_vectors(2, 60)
    out += [
        Case("binary_necklaces_60", lambda: ctx.words(2, 60), "orbit_count",
             {"group": ctx.perms(60, cyclic(60))}, bench="all", oracle=False,
             what="binary necklaces of length 60, counted from 60 cycle types instead of 2^60 words"),
        Case("balanced_bracelets_60", lambda: ctx.subsets(I60, 30), "orbit_count",
             {"group": ctx.perms(60, dihedral(60))}, bench="all", oracle=False,
             what="bracelets with 30 black beads among 60, counted without visiting the 60-choose-30 subsets"),
    ]
    return out


def invariants(ctx):
    I8 = unit_vectors(2, 8)
    D8 = ctx.perms(8, dihedral(8))
    subsets = ctx.subsets(I8, 4)
    burnside = ctx.value("burnside.orbit_count", subsets, group=D8).values[0]
    enumerated = ctx.value("orbits.is_canonical", subsets, "count", group=D8).value
    assert burnside == enumerated

    words = ctx.words(3, 8)
    index = ctx.value("burnside.cycle_index", words, group=D8)
    count = ctx.value("burnside.orbit_count", words, group=D8).values[0]
    assert sum(multiplicity for multiplicity, _ in index.terms) == index.denominator
    assert sum(multiplicity * 3 ** sum(cycles) for multiplicity, cycles in index.terms) == index.denominator * count
    assert index.terms == sorted(index.terms, key=lambda term: term[1])
