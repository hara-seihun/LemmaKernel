"""spreads_and_partitions cases: the inputs the harness runs against the backend, the naive
implementation and the Lean reference.

The kernel evaluates the reference by elimination, about one per pair of components per member,
so the oracle families here are a few dozen members of a few components each; the sizes that show
what the module is for are in the bench cases at the end. Every configuration named here (the
regular spread of PG(3,2), the packing of its 35 lines) is put to the reference like everything
else; none of it is trusted because it was computed elsewhere.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lemmakernel as lk  # noqa: E402
from tools.harness import Case, module, rotate  # noqa: E402

SET_OPS = ["is_partial_spread", "is_spread", "is_vector_space_partition", "intersecting_pairs"]

# Indices into the canonical order of grassmannian(2, 4, 2), the 35 lines of PG(3,2). SPREAD is
# the regular spread {y = cx} ∪ {x = 0} of F_4^2 read as F_2^4; PACKING is the first of the 240
# packings of PG(3,2): seven spreads which together are every line exactly once. MEETING lines
# are outside SPREAD, so each of them meets some line of it.
SPREAD = [0, 7, 9, 14, 34]
MEETING = [1, 2, 3]
PACKING = [[0, 6, 11, 13, 34], [1, 7, 21, 25, 28], [2, 9, 18, 26, 29], [3, 14, 19, 20, 32],
           [4, 10, 17, 27, 30], [5, 12, 16, 23, 33], [8, 15, 22, 24, 31]]


def flat(ctx, p, n, h, indices):
    """Members of the Grassmannian of F_p^n, each flattened to one row of h*n entries: the shape
    in which every operation here reads a component."""
    G = ctx.grassmannian(p, n, h)
    return [[x for row in ctx.member(G, i).value().member(0) for x in row] for i in indices]


def cases(ctx, rng):
    mod = module("spreads_and_partitions")
    lines = flat(ctx, 2, 4, 2, range(35))     # the 35 lines of PG(3,2), 8 entries each
    points = flat(ctx, 2, 3, 1, range(7))     # the 7 points of PG(2,2), 3 entries each
    spread = [lines[i] for i in SPREAD]

    # Five lines of PG(3,2): the regular spread, then one line of it swapped for a line it meets,
    # a repeated line, a rank-1 component and a zero component. Only the first is a spread, and
    # none of the others is a partial spread.
    fives = [spread,
             spread[:4] + [lines[MEETING[0]]],
             spread[:4] + [spread[0]],
             spread[:4] + [[1, 0, 0, 0] + [0] * 4],
             spread[:4] + [[0] * 8]]
    # Four lines: partial spreads too small to cover, and two sets that are not even that.
    fours = [spread[:4], spread[1:5], spread[:3] + [lines[MEETING[1]]], spread[:2] + spread[:2]]
    # Components of two dimensions in F_2^3: the plane <e0, e1> and the four points outside it
    # partition the seven nonzero vectors (3 + 4), which no set of equal dimensions can do here.
    # Points 0, 2, 4 of PG(2,2) are the ones inside that plane.
    plane = [1, 0, 0, 0, 1, 0]
    pad = [v + [0, 0, 0] for v in points]     # a point as a rank-1 2x3 component
    mixed = [[plane, pad[1], pad[3], pad[5], pad[6]],
             [plane, pad[0], pad[3], pad[5], pad[6]],
             [pad[1], pad[3], plane, pad[5], pad[6]],
             [plane, pad[1], pad[3], pad[5], pad[1]],
             [[0] * 6, pad[1], pad[3], pad[5], pad[6]]]

    families = [
        # (name, family, ambient dimension, every reduction on every operation)
        ("PG(1,2) triples", ctx.subsets_of(ctx.grassmannian(2, 2, 1), 3), 2, True),
        ("PG(2,2) points", ctx.subsets_of(ctx.grassmannian(2, 3, 1), 7), 3, True),
        ("PG(3,2) candidates", ctx.explicit(lk.matrix(2, fives)), 4, True),
        ("PG(3,2) four lines", ctx.explicit(lk.matrix(2, fours)), 4, True),
        ("F_2^3 mixed partitions", ctx.explicit(lk.matrix(2, mixed)), 3, True),
        ("PG(2,2) point pairs", ctx.subsets_of(ctx.grassmannian(2, 3, 1), 2), 3, False),
        ("pairs of vectors in F_2^2", ctx.all_matrices(2, 2, 2), 2, False),
        ("line pairs of PG(3,2)", ctx.subsets(lk.matrix(2, [lines[i] for i in SPREAD + MEETING]), 2), 4, False),
        ("line triples of PG(3,2)", ctx.subsets(lk.matrix(2, [lines[i] for i in SPREAD + MEETING[:1]]), 3), 4, False),
    ]
    out = []
    for i, (name, fam, n, full) in enumerate(families):
        for j, op in enumerate(SET_OPS):
            out.append(Case(name, fam, op, {"n": n, "limit": 3},
                            reductions=None if full else rotate(mod, op, i + j)))

    # Packings. The seven points of PG(2,2) in one row are the only packing of it by points; the
    # 7 x 40 member is the packing of the 35 lines of PG(3,2) by seven spreads, which costs the
    # Lean kernel a couple of seconds a member and so is claimed under one reduction.
    point_packings = [[[x for v in points for x in v]],
                      [[x for v in points[:6] + points[:1] for x in v]]]
    packings = [[[x for i in row for x in lines[i]] for row in PACKING],
                [[x for i in row for x in lines[i]] for row in PACKING[:6]] +
                [[x for i in PACKING[0] for x in lines[i]]]]
    out += [
        Case("PG(2,2) point packing", ctx.explicit(lk.matrix(2, point_packings)), "is_packing",
             {"n": 3, "h": 1, "limit": 2}),
        Case("PG(3,2) packing", ctx.explicit(lk.matrix(2, packings)), "is_packing", {"n": 4, "h": 2},
             reductions=["count"]),
    ]

    # Inputs the runtime must refuse (named by [[rejections]] in the manifest).
    candidates = families[2][1]
    out += [
        Case("ragged ambient dimension", candidates, "is_partial_spread", {"n": 3}, reductions=["count"], oracle=False),
        Case("zero ambient dimension", candidates, "is_spread", {"n": 0}, reductions=["count"], oracle=False),
        Case("words are not a field", ctx.words(3, 2), "is_partial_spread", {"n": 1}, reductions=["count"], oracle=False),
        Case("packing block too wide", candidates, "is_packing", {"n": 4, "h": 3}, reductions=["count"], oracle=False),
    ]

    # Benchmarks: requests that look like real use, sized so that the ratio shows.
    G32 = ctx.grassmannian(2, 4, 2)
    out += [
        Case("pg32_spreads", lambda: ctx.subsets_of(G32, 5), "is_spread", {"n": 4}, bench="count", oracle=False,
             what="the 56 spreads of PG(3,2): which of the 324,632 five-sets of lines partition F_2^4"),
        Case("pg32_first_spread", lambda: ctx.subsets_of(G32, 5), "is_spread", {"n": 4}, reductions=["first"],
             bench="first", oracle=False, what="the least five-set of lines of PG(3,2) that is a spread"),
        Case("pg32_partial_spreads", lambda: ctx.subsets_of(G32, 4), "is_partial_spread", {"n": 4}, bench="count",
             oracle=False, what="how many of the 52,360 four-sets of lines of PG(3,2) are partial spreads"),
        Case("pg32_meetings", lambda: ctx.subsets_of(G32, 3), "intersecting_pairs", {"n": 4}, bench="histogram",
             oracle=False, what="how many of the three pairs meet, over every three-set of lines of PG(3,2)"),
        Case("pg42_partial_spreads", lambda: ctx.subsets_of(ctx.grassmannian(2, 5, 2), 4), "is_partial_spread",
             {"n": 5}, bench="count", oracle=False,
             what="partial spreads among the 23,130,030 four-sets of the 155 lines of PG(4,2)"),
    ]
    return out


def invariants(ctx):
    """Cross-operation identities on inputs beyond the kernel oracle."""
    def value(op, fam, red="all", **args):
        return ctx.value("spreads_and_partitions." + op, fam, red, n=4, **args)

    G = ctx.grassmannian(2, 4, 2)
    five, four, three, two = (ctx.subsets_of(G, k) for k in (5, 4, 3, 2))
    # PG(3,2) has 56 spreads; with five lines of equal dimension, spread, partition and partial
    # spread are the same condition, and four lines can never cover.
    assert value("is_spread", five, "count").value == 56
    assert value("is_vector_space_partition", five, "count").value == 56
    assert (value("is_spread", five, "hits", limit=0).indices ==
            value("is_partial_spread", five, "hits", limit=0).indices)
    assert value("is_spread", four, "count").value == 0
    # "No intersecting pair" is exactly "partial spread" when every component is a line.
    for fam in (three, four):
        assert (value("intersecting_pairs", fam, "histogram").bins[0] ==
                value("is_partial_spread", fam, "count").value)
    # Each line of PG(3,2) meets 18 others, and a meeting pair lies in C(33, 1) of the triples.
    meetings = value("intersecting_pairs", two, "sum").value
    assert meetings == 35 * 18 // 2
    assert value("intersecting_pairs", three, "sum").value == meetings * 33
    # The 56 spreads cover each line 8 times: every line is in the same number of them.
    hits = value("is_spread", five, "hits", limit=56).members.tolist()
    assert len(hits) == 56 and all(sum(m.count(line) for m in hits) == 8 for line in hits[0])
