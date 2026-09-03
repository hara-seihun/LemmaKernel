"""Cases for q-ary linear codes presented by generator matrices."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lemmakernel as lk  # noqa: E402
from tools.harness import Case, module, random_batch, rotate  # noqa: E402

OPS = ["minimum_distance", "weight_enumerator", "dual", "is_self_dual", "covering_radius", "is_mds", "aut_order"]


def binary_batch():
    return lk.matrix(2, [
        [[0, 0, 0, 0], [0, 0, 0, 0]],
        [[1, 1, 1, 1], [0, 0, 0, 0]],
        [[1, 1, 0, 0], [0, 0, 1, 1]],
        [[1, 0, 1, 0], [0, 1, 0, 1]],
        [[1, 0, 0, 0], [0, 1, 0, 0]],
    ])


def cases(ctx, rng):
    mod = module("linear_codes")
    explicit = ctx.explicit(binary_batch())
    out = []

    for op in OPS:
        args = {"limit": 3} if op in ("is_self_dual", "is_mds") else {}
        out.append(Case(f"explicit binary {op.replace('_', ' ')}", explicit, op, args))

    extended = ctx.transform(
        ctx.grassmannian(2, 3, 1),
        lk.matrix(2, [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]),
    )
    families = [
        ("binary Grassmannian", ctx.grassmannian(2, 3, 2)),
        ("parity extension", extended),
        ("ternary stacked generators", ctx.stack(ctx.grassmannian(3, 3, 1), lk.matrix(3, [[1, 1, 1]]))),
    ]
    for i, (name, family) in enumerate(families):
        for j, op in enumerate(OPS):
            args = {"limit": 2} if op in ("is_self_dual", "is_mds") else {}
            out.append(Case(name, family, op, args, reductions=rotate(mod, op, i + j)))

    out += [
        Case("words are not generator matrices", ctx.words(2, 3), "minimum_distance", oracle=False),
        Case(
            "grassmannian_3_7_binary",
            lambda: ctx.grassmannian(2, 7, 3),
            "minimum_distance",
            reductions=["histogram"],
            bench="histogram",
            oracle=False,
            what="minimum-distance distribution of every binary [7,3] code in canonical Grassmannian order",
        ),
    ]
    return out


def invariants(ctx):
    p, count, rows, n = 3, 12, 3, 5
    batch = random_batch(__import__("random").Random(31), p, count, rows, n)
    family = ctx.explicit(batch)
    enumerators = ctx.value("linear_codes.weight_enumerator", family)
    distances = ctx.value("linear_codes.minimum_distance", family).values
    radii = ctx.value("linear_codes.covering_radius", family).values
    ranks = ctx.value("gfp.rank", family).values
    duals = ctx.value("linear_codes.dual", family)
    mds = ctx.value("linear_codes.is_mds", family).values

    for i in range(count):
        coefficients = enumerators.member(i)
        assert sum(coefficients) == p ** ranks[i]
        assert coefficients[0] == 1
        assert distances[i] == next((w for w, a in enumerate(coefficients[1:], 1) if a), 0)
        assert len(duals.member(i)) == n - ranks[i]
        assert radii[i] <= n - ranks[i]
        assert bool(mds[i]) == (ranks[i] > 0 and distances[i] == n - ranks[i] + 1)
