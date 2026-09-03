"""Cases for matrix-group orbits on Grassmannian-derived families."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lemmakernel as lk  # noqa: E402
from tools.harness import Case, companion, frobenius, module, rotate  # noqa: E402

ORBIT_OPS = ["is_canonical", "canonical_index", "orbit_size", "stabilizer_order"]

GL22 = lk.matrix(2, [[[1, 1], [0, 1]], [[0, 1], [1, 0]]])
SINGER3 = [[0, 1, 0], [0, 0, 1], [1, 1, 0]]
SCALAR3 = lk.matrix(3, [[[2, 0], [0, 2]]])


def cases(ctx, rng):
    del rng
    actions = [
        ("GL(2,2) on PG(1,2)", ctx.grassmannian(2, 2, 1), GL22, 0),
        ("PGL scalar quotient on PG(1,3)", ctx.grassmannian(3, 2, 1),
         lk.matrix(3, [[[0, 1], [2, 0]], [[2, 0], [0, 2]]]), 1),
        ("GL scalar subgroup on PG(1,3)", ctx.grassmannian(3, 2, 1), SCALAR3, 0),
        ("PGL scalar subgroup on PG(1,3)", ctx.grassmannian(3, 2, 1), SCALAR3, 1),
        ("rank-changing transform", ctx.transform(ctx.grassmannian(2, 3, 1),
                                                   lk.matrix(2, [[1, 0], [0, 1], [1, 1]])), GL22, 1),
        ("stacked Grassmannian", ctx.stack(ctx.grassmannian(2, 3, 1),
                                           lk.matrix(2, [[1, 1, 1]])), lk.matrix(2, [SINGER3]), 1),
        ("stack of a transform", ctx.stack(ctx.transform(ctx.grassmannian(3, 3, 1),
                                                         lk.matrix(3, [[1, 0], [0, 1], [1, 1]])),
                                             lk.matrix(3, [[1, 0]])),
         lk.matrix(3, [[[0, 1], [2, 0]]]), 1),
    ]

    mod = module("subspace_orbits")
    out = []
    for i, (name, family, group, projective) in enumerate(actions):
        for j, op in enumerate(ORBIT_OPS):
            reductions = None if name == "GL(2,2) on PG(1,2)" else rotate(mod, op, i + j)
            out.append(Case(name, family, op,
                            {"group": group, "projective": projective, "limit": 3},
                            reductions=reductions))

    out += [
        Case("permutations on subspaces", ctx.grassmannian(2, 3, 1), "orbit_size",
             {"group": ctx.perms(7, [[1, 2, 3, 4, 5, 6, 0]]), "projective": 1}, oracle=False),
        Case("all matrices are not a subspace family", ctx.all_matrices(2, 1, 2), "orbit_size",
             {"group": GL22, "projective": 0}, oracle=False),
        Case("wrong group dimension", ctx.grassmannian(2, 3, 1), "orbit_size",
             {"group": GL22, "projective": 0}, oracle=False),
        Case("singular generator", ctx.grassmannian(2, 2, 1), "orbit_size",
             {"group": lk.matrix(2, [[[1, 1], [1, 1]]]), "projective": 0}, oracle=False),
        Case("invalid projective flag", ctx.grassmannian(2, 2, 1), "orbit_size",
             {"group": GL22, "projective": 2}, oracle=False),
    ]

    f5 = [1, 0, 1, 0, 0]
    out.append(Case(
        "PG(4,2) line orbits",
        ctx.grassmannian(2, 5, 2),
        "canonical_index",
        {"group": lk.matrix(2, [companion(f5, 2), frobenius(f5, 2)]), "projective": 1},
        bench="histogram",
        oracle=False,
        what="canonical indices for the 155 lines of PG(4,2) under the Singer normaliser",
    ))
    return out


def invariants(ctx):
    group = lk.matrix(2, [SINGER3])
    family = ctx.grassmannian(2, 3, 1)
    canonical = ctx.value("subspace_orbits.canonical_index", family, group=group, projective=1).values
    flags = ctx.value("subspace_orbits.is_canonical", family, group=group, projective=1).values
    sizes = ctx.value("subspace_orbits.orbit_size", family, group=group, projective=1).values
    stabilizers = ctx.value("subspace_orbits.stabilizer_order", family, group=group, projective=1).values
    assert canonical == [0] * 7
    assert flags == [1] + [0] * 6
    assert all(size * stabilizer == 7 for size, stabilizer in zip(sizes, stabilizers))

    scalar_family = ctx.grassmannian(3, 2, 1)
    gl_stabilizers = ctx.value("subspace_orbits.stabilizer_order", scalar_family,
                               group=SCALAR3, projective=0).values
    pgl_stabilizers = ctx.value("subspace_orbits.stabilizer_order", scalar_family,
                                group=SCALAR3, projective=1).values
    assert gl_stabilizers == [2] * 4
    assert pgl_stabilizers == [1] * 4
