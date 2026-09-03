"""elliptic_curves_fp cases: the inputs the harness runs against the backend, the naive
implementation and the Lean reference.

A member is a 1 x 2 matrix (a, b), so every family kind that produces one row of two entries is
fair game: all p^2 pairs over F_p, a chosen batch, singletons of another family, a Grassmannian
of lines through F_p^2, and a linear reparametrisation. Oracle cases are sized for the Lean
kernel; the bench cases are the requests this module exists for.
"""
from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lemmakernel as lk  # noqa: E402
from tools.harness import Case, rotate  # noqa: E402

OPS = ["point_count", "nonsingular", "supersingular", "j_invariant", "is_canonical", "class_size",
       "group_structure"]
# Everything but the group law, which needs the point list of every member.
LIGHT_OPS = [op for op in OPS if op != "group_structure"]


def pairs(p, ab):
    return lk.matrix(p, [[list(c)] for c in ab])


def cases(ctx, rng):
    from tools.harness import module
    mod = module("elliptic_curves_fp")
    out = []

    # All 25 pairs over F_5, every operation under every reduction it allows.
    all5 = ctx.all_matrices(5, 1, 2)
    out += [Case("all pairs F_5", all5, op, {"limit": 3}) for op in OPS]

    # All 49 pairs over F_7 and all 121 over F_11, one reduction each, rotating.
    for i, (p, ops) in enumerate([(7, OPS), (11, LIGHT_OPS)]):
        fam = ctx.all_matrices(p, 1, 2)
        for j, op in enumerate(ops):
            out.append(Case(f"all pairs F_{p}", fam, op, {"limit": 2},
                            reductions=rotate(mod, op, i + j)))

    # Chosen curves: singular pairs, j = 0 (b only), j = 1728 (a only), and generic ones.
    thirteen = pairs(13, [(0, 0), (0, 1), (1, 0), (1, 1), (2, 3), (5, 7), (8, 8), (12, 12), (9, 0), (0, 6)])
    for j, op in enumerate(OPS):
        out.append(Case("curves F_13", ctx.explicit(thirteen), op, {"limit": 2}, reductions=rotate(mod, op, j)))

    # A prime that needs two bytes per entry, so the interchange width is exercised.
    wide = pairs(257, [(0, 0), (0, 1), (1, 0), (17, 42)])
    for j, op in enumerate(LIGHT_OPS):
        out.append(Case("curves F_257", ctx.explicit(wide), op, {"limit": 2}, reductions=rotate(mod, op, j + 1)))

    # The other family kinds that yield 1 x 2 members.
    dictionary = pairs(7, [(1, 1), (2, 4), (3, 0), (0, 5), (6, 6)])
    others = [
        ("chosen pairs F_7", ctx.subsets(dictionary, 1)),
        ("singleton pairs F_5", ctx.subsets_of(ctx.all_matrices(5, 1, 2), 1)),
        ("line coefficients F_5", ctx.grassmannian(5, 2, 1)),
        ("mixed pairs F_5", ctx.transform(ctx.all_matrices(5, 1, 2), lk.matrix(5, [[1, 2], [3, 1]]))),
    ]
    for i, (name, fam) in enumerate(others):
        for j, op in enumerate(OPS):
            out.append(Case(name, fam, op, {"limit": 2}, reductions=rotate(mod, op, i + j)))

    # Refusals: a reduction that cannot take the operation's values, primes below 5, and members
    # that are not a single pair.
    tiny = ctx.explicit(pairs(7, [(2, 1), (3, 5)]))
    out += [Case("reduction rejections F_7", tiny, op, reductions=["all"])
            for op in ("point_count", "nonsingular", "group_structure")]
    out += [
        Case("F_3 pairs", ctx.all_matrices(3, 1, 2), "point_count", reductions=["all"], oracle=False),
        Case("stacked pairs F_5", ctx.stack(ctx.all_matrices(5, 1, 2), lk.matrix(5, [[1, 1]])), "point_count",
             reductions=["all"], oracle=False),
        Case("plane points F_5", ctx.grassmannian(5, 3, 1), "j_invariant", reductions=["all"], oracle=False),
    ]

    # Benchmarks: the requests this module exists for, sized so that the ratio shows.
    out += [
        Case("point_counts_F_1009", lambda: ctx.all_matrices(1009, 1, 2), "point_count",
             what="the number of points on every curve over F_1009; one histogram bin per isogeny class",
             bench="histogram", oracle=False),
        Case("iso_classes_F_997", lambda: ctx.all_matrices(997, 1, 2), "is_canonical",
             what="how many curves over F_997 there are up to F_997-isomorphism",
             bench="count", oracle=False),
        Case("supersingular_F_499", lambda: ctx.all_matrices(499, 1, 2), "supersingular",
             what="how many of the 249001 pairs over F_499 give a supersingular curve",
             bench="count", oracle=False),
        Case("group_structure_F_101", lambda: ctx.all_matrices(101, 1, 2), "group_structure",
             what="the invariant factors of E(F_101) for all 10201 pairs (a, b)",
             bench="all", oracle=False),
    ]
    return out


def invariants(ctx):
    """Cross-operation identities on inputs beyond the kernel oracle."""
    p = 23
    F = ctx.all_matrices(p, 1, 2)
    name = "elliptic_curves_fp."

    # Every (x, y, a) determines one b, so the point counts of all p^2 pairs sum to p^3 affine
    # points plus one point at infinity each.
    assert ctx.value(name + "point_count", F, "sum").value == p ** 3 + p * p

    counts = ctx.value(name + "point_count", F, "all").values
    smooth = ctx.value(name + "nonsingular", F, "all").values
    supersingular = ctx.value(name + "supersingular", F, "all").values
    sizes = ctx.value(name + "class_size", F, "all").values
    canonical = ctx.value(name + "is_canonical", F, "all").values
    groups = ctx.value(name + "group_structure", F).orders

    for i, n in enumerate(counts):
        if smooth[i]:
            assert abs(n - (p + 1)) ** 2 <= 4 * p, f"Hasse bound violated: {n} points"
            assert supersingular[i] == (n == p + 1)
            n1, n2 = groups[i]
            assert n1 * n2 == n and n2 % n1 == 0, f"{(n1, n2)} is not the structure of a group of order {n}"
            assert (p - 1) % n1 == 0, "full n1-torsion needs the n1-th roots of unity"
        else:
            assert groups[i] == (0, 0)

    # The isomorphism classes partition the p^2 pairs: each contributes one canonical member and
    # `class_size` members of that size.
    assert sum(canonical) == sum(Fraction(1, s) for s in sizes)
    assert all((p - 1) % s == 0 for s in sizes)

    # (a, b) and (u^4*a, u^6*b) are the same curve seen through (x, y) -> (u^2*x, u^3*y), so they
    # have the same j-invariant and the same number of points.
    u4, u6 = pow(5, 4, p), pow(5, 6, p)
    twisted = ctx.explicit(lk.matrix(p, [[[a * u4 % p, b * u6 % p]] for a in range(p) for b in range(p)]))
    assert ctx.value(name + "point_count", twisted, "all").values == counts
    assert ctx.value(name + "j_invariant", twisted, "all").values == \
        ctx.value(name + "j_invariant", F, "all").values
