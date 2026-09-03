"""orbits cases. Kernel evaluation of the reference closes groups by list scans, so oracle cases
keep groups to a few dozen elements and families to a few dozen members; the invariants below
check larger cases against the kernel's own Burnside counts.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lemmakernel as lk  # noqa: E402
from tools.harness import (Case, companion, cyclic, dihedral, frobenius, module, projective_points,  # noqa: E402
                           random_batch, rotate, symmetric, unit_vectors)

ORBIT_OPS = ["is_canonical", "canonical_index", "orbit_size", "stabilizer_order"]

SINGER3 = [[0, 1, 0], [0, 0, 1], [1, 1, 0]]     # x^3 = x + 1 over F_2: order 7 on the 7 points of PG(2,2)
FROB3 = [[1, 0, 0], [0, 0, 1], [0, 1, 1]]       # Frobenius in the same basis; together order 21


def hypercube(n):
    """Generators of the symmetry group of the n-cube on its 2^n vertices (bit strings)."""
    flip = [v ^ 1 for v in range(1 << n)]
    swap = [(v & ~3) | ((v & 1) << 1) | ((v >> 1) & 1) for v in range(1 << n)]
    cycle = [((v << 1) | (v >> (n - 1))) & ((1 << n) - 1) for v in range(1 << n)]
    return [flip, swap, cycle]


def cases(ctx, rng):
    I6, I5, I4 = unit_vectors(2, 6), unit_vectors(3, 5), unit_vectors(7, 4)
    D4 = random_batch(rng, 5, 4, 1, 3)
    perm_actions = [
        ("D6 on 2-subsets", ctx.perms(6, dihedral(6)), ctx.subsets(I6, 2)),
        ("D6 on 3-subsets", ctx.perms(6, dihedral(6)), ctx.subsets(I6, 3)),
        ("C5 on 2-subsets", ctx.perms(5, cyclic(5)), ctx.subsets(I5, 2)),
        ("S4 on 2-subsets", ctx.perms(4, symmetric(4)), ctx.subsets(I4, 2)),
        ("S4 on random dictionary", ctx.perms(4, symmetric(4)), ctx.subsets(D4, 2)),
        ("trivial on 1-subsets", ctx.perms(4, [list(range(4))]), ctx.subsets(I4, 1)),
        ("swap on 4-subsets", ctx.perms(6, [[1, 0, 2, 3, 4, 5]]), ctx.subsets(I6, 4)),
    ]
    GL22 = lk.matrix(2, [[[1, 1], [0, 1]], [[0, 1], [1, 0]]])          # order 6
    FIB3 = lk.matrix(3, [[[0, 1], [1, 1]]])                             # order 8
    DIAG5 = lk.matrix(5, [[[2, 0], [0, 3]]])                            # order 4
    SINGER = lk.matrix(2, [SINGER3, FROB3])
    matrix_actions = [
        ("GL(2,2) on points", GL22, ctx.grassmannian(2, 2, 1)),
        ("GL(2,2) on 1x2", GL22, ctx.all_matrices(2, 1, 2)),
        ("GL(2,2) on 2x2", GL22, ctx.all_matrices(2, 2, 2)),
        ("<Fib> on PG(1,3)", FIB3, ctx.grassmannian(3, 2, 1)),
        ("<Fib> on 1x2 over F3", FIB3, ctx.all_matrices(3, 1, 2)),
        ("<diag(2,3)> on PG(1,5)", DIAG5, ctx.grassmannian(5, 2, 1)),
        ("Singer normaliser on PG(2,2) points", SINGER, ctx.grassmannian(2, 3, 1)),
        ("Singer normaliser on PG(2,2) lines", SINGER, ctx.grassmannian(2, 3, 2)),
        ("Singer normaliser on 1x3", SINGER, ctx.all_matrices(2, 1, 3)),
    ]
    # every reduction on two cases; one reduction per case, rotating, elsewhere
    mod = module("orbits")
    full = {"D6 on 2-subsets", "GL(2,2) on 2x2"}
    out = []
    for i, (name, G, F) in enumerate(perm_actions + matrix_actions):
        for j, op in enumerate(ORBIT_OPS):
            reds = None if name in full else rotate(mod, op, i + j)
            out.append(Case(name, F, op, {"group": G, "limit": 3}, reductions=reds))
    for name, G, F in perm_actions:
        out.append(Case(f"{name} fixed points", ctx.group_elements(G), "fixed_points", {"on": F}))

    # projective actions, and the induced permutations acting on pairs of points
    for p, n, mats in [(2, 3, [SINGER3, FROB3]), (3, 2, [[[0, 1], [1, 1]], [[2, 0], [0, 1]]]), (5, 2, [[[2, 0], [0, 3]], [[1, 1], [0, 1]]])]:
        pts = projective_points(ctx, p, n)
        F = ctx.explicit(lk.matrix(p, mats))
        out.append(Case(f"PG({n - 1},{p}) projective action", F, "projective_action", {"points": pts}))
        P = ctx.run("orbits.projective_action", F, points=pts)
        out.append(Case(f"PG({n - 1},{p}) pairs", ctx.subsets(pts, 2), "is_canonical", {"group": P, "limit": 3}))

    # inputs the runtime must refuse (named by [[rejections]] in the manifest)
    out += [
        Case("C5 on the hexagon", ctx.subsets(I6, 2), "is_canonical", {"group": ctx.perms(5, cyclic(5))}, reductions=["count"], oracle=False),
        Case("perms on a Grassmannian", ctx.grassmannian(2, 3, 1), "orbit_size", {"group": ctx.perms(7, cyclic(7))}, oracle=False),
        Case("2x2 matrices on PG(2,2)", ctx.grassmannian(2, 3, 1), "orbit_size", {"group": lk.matrix(2, [[[1, 0], [0, 1]]])}, oracle=False),
        Case("singular matrix on PG(1,2)", ctx.grassmannian(2, 2, 1), "orbit_size", {"group": lk.matrix(2, [[[1, 1], [1, 1]]])}, oracle=False),
        Case("points not closed", ctx.explicit(lk.matrix(2, [[[1, 1], [0, 1]]])), "projective_action", {"points": lk.matrix(2, [[1, 0], [0, 1]])}, oracle=False),
        Case("fixed_points on a subsets family", ctx.subsets(I6, 2), "fixed_points", {"on": ctx.subsets(I6, 2)}, oracle=False),
    ]

    # benchmarks, sized so that naive finishes and the ratio shows
    D16 = ctx.perms(16, dihedral(16))
    beads = ctx.subsets(unit_vectors(2, 16), 6)
    f5 = [1, 0, 1, 0, 0]     # x^5 + x^2 + 1, primitive over F_2
    f4 = [1, 1, 0, 0]        # x^4 + x + 1, primitive over F_2
    out += [
        Case("bracelets_16_6", beads, "is_canonical", {"group": D16}, bench="count", oracle=False,
             what="bracelets with 6 black and 10 white beads: orbits of D_16 on 6-subsets of 16 positions"),
        Case("bracelets_stabilizers", beads, "stabilizer_order", {"group": D16}, bench="histogram", oracle=False,
             what="how symmetric each such bracelet is: stabiliser orders in D_16"),
        Case("bracelets_burnside", ctx.group_elements(D16), "fixed_points", {"on": beads}, bench="all", oracle=False,
             what="fixed 6-subsets of every element of D_16 (Burnside count of the bracelets)"),
        Case("cube4_quadruples", ctx.subsets(unit_vectors(2, 16), 4), "is_canonical", {"group": ctx.perms(16, hypercube(4)), "limit": 8},
             bench="hits", oracle=False, what="4-vertex configurations of the 4-cube up to symmetry (group of order 384)"),
        Case("pg42_lines_singer", ctx.grassmannian(2, 5, 2), "orbit_size", {"group": lk.matrix(2, [companion(f5, 2), frobenius(f5, 2)])},
             bench="histogram", oracle=False, what="orbits of the Singer normaliser of GL(5,2) (order 155) on the 155 lines of PG(4,2)"),
        Case("all_3x4_f2_singer_normaliser", ctx.all_matrices(2, 3, 4), "canonical_index", {"group": lk.matrix(2, [companion(f4, 2), frobenius(f4, 2)])},
             bench="histogram", oracle=False, what="every 3x4 matrix over F_2 under right multiplication by a group of order 60"),
    ]
    return out


def invariants(ctx):
    """Orbit counts agree with the Cauchy-Frobenius count on cases too large for the kernel oracle."""
    I8, I7 = unit_vectors(2, 8), unit_vectors(3, 7)
    for G, F in [(ctx.perms(8, dihedral(8)), ctx.subsets(I8, 4)), (ctx.perms(7, symmetric(7)), ctx.subsets(I7, 3)),
                 (ctx.perms(8, [[1, 2, 3, 0, 5, 6, 7, 4], [4, 5, 6, 7, 0, 1, 2, 3]]), ctx.subsets(I8, 3))]:
        E = ctx.group_elements(G)
        fixed = ctx.value("orbits.fixed_points", E, on=F).values
        orbits = ctx.value("orbits.is_canonical", F, "count", group=G).value
        assert sum(fixed) == orbits * ctx.size(E)
        sizes = ctx.value("orbits.orbit_size", F, group=G).values
        stabs = ctx.value("orbits.stabilizer_order", F, group=G).values
        assert all(s * t == ctx.size(E) for s, t in zip(sizes, stabs))
        assert abs(sum(1 / s for s in sizes) - orbits) < 1e-9
    # GL(3,2) on the 35 triples of points of PG(2,2): lines and triangles, stabilisers 24 and 6
    A = lk.matrix(2, [SINGER3, [[1, 1, 0], [0, 1, 0], [0, 0, 1]]])
    pts = projective_points(ctx, 2, 3)
    P = ctx.run("orbits.projective_action", ctx.explicit(A), points=pts)
    S = ctx.subsets(pts, 3)
    assert ctx.value("orbits.is_canonical", S, "count", group=P).value == 2
    assert sorted(set(ctx.value("orbits.stabilizer_order", S, group=P).values)) == [6, 24]
    assert ctx.size(ctx.group_elements(P)) == 168
