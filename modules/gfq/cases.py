"""Oracle and benchmark cases for explicitly presented finite fields."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lemmakernel as lk  # noqa: E402
from tools.harness import Case, random_batch  # noqa: E402

MODULI = {
    2: [0, 1],
    4: [1, 1, 1],
    8: [1, 1, 0, 1],
    9: [1, 0, 1],
    16: [1, 1, 0, 0, 1],
    25: [2, 0, 1],
}


def polynomial(q):
    return lk.matrix(q, [MODULI[q]])


def cases(ctx, rng):
    q4 = polynomial(4)
    explicit = ctx.explicit(random_batch(rng, 4, 7, 3, 4))
    target = random_batch(rng, 4, 1, 1, 4)
    out = [
        Case("explicit GF(4)", explicit, "rank", {"modulus": q4}),
        Case("explicit GF(4)", explicit, "in_span", {"modulus": q4, "target": target, "limit": 3}),
        Case("explicit GF(4)", explicit, "rref", {"modulus": q4}),
        Case("explicit GF(4)", explicit, "nullspace", {"modulus": q4}),
    ]
    square = ctx.explicit(random_batch(rng, 4, 7, 3, 3))
    out += [
        Case("square GF(4)", square, "inverse", {"modulus": q4}),
        Case("square GF(4)", square, "solve",
             {"modulus": q4, "rhs": random_batch(rng, 4, 7, 1, 3)}),
        Case("rectangular solve GF(4)", ctx.explicit(random_batch(rng, 4, 6, 2, 3)), "solve",
             {"modulus": q4, "rhs": random_batch(rng, 4, 6, 1, 2)}),
    ]

    family_cases = [
        ("subsets GF(4)", ctx.subsets(random_batch(rng, 4, 5, 1, 4), 2), "rank", {}, ["histogram"]),
        ("Grassmannian GF(4)", ctx.grassmannian(4, 3, 1), "nullspace", {}, None),
        ("all matrices GF(4)", ctx.all_matrices(4, 1, 2), "in_span",
         {"target": random_batch(rng, 4, 1, 1, 2), "limit": 2}, ["hits"]),
        ("subsets_of GF(8)", ctx.subsets_of(ctx.all_matrices(8, 1, 1), 2), "rref", {}, None),
        ("symmetric GF(9)", ctx.symmetric_matrices(9, 1), "rank", {}, ["sum"]),
        ("stack GF(25)", ctx.stack(ctx.grassmannian(25, 2, 1), random_batch(rng, 25, 1, 1, 2)),
         "in_span", {"target": random_batch(rng, 25, 1, 1, 2)}, ["count"]),
    ]
    for name, family, op, args, reductions in family_cases:
        args = {"modulus": polynomial(family.param("p")), **args}
        out.append(Case(name, family, op, args, reductions=reductions))

    for q, op in ((8, "rref"), (9, "solve"), (16, "rank"), (25, "inverse")):
        family = ctx.explicit(random_batch(rng, q, 4, 2, 2))
        args = {"modulus": polynomial(q)}
        if op == "solve":
            args["rhs"] = random_batch(rng, q, 4, 1, 2)
        out.append(Case(f"presentation GF({q})", family, op, args,
                        reductions=["all"] if op == "rank" else None))

    out += [
        Case("reducible modulus", ctx.explicit(random_batch(rng, 4, 2, 2, 2)), "rank",
             {"modulus": lk.matrix(4, [[0, 0, 1]])}, reductions=["all"], oracle=False),
        Case("wrong field size", ctx.explicit(random_batch(rng, 4, 2, 2, 2)), "rank",
             {"modulus": lk.matrix(4, [[1, 1, 0, 1]])}, reductions=["all"], oracle=False),
        Case("transform over a prime field",
             ctx.transform(ctx.all_matrices(2, 1, 2), lk.matrix(2, [[1], [1]])), "rank",
             {"modulus": polynomial(2)}, reductions=["all"], oracle=False),
        Case("nonsquare inverse", ctx.explicit(random_batch(rng, 4, 2, 2, 3)), "inverse",
             {"modulus": q4}, oracle=False),
        Case("Grassmannian solve", ctx.grassmannian(4, 3, 1), "solve",
             {"modulus": q4, "rhs": random_batch(rng, 4, 1, 1, 1)}, oracle=False),
    ]

    out += [
        Case("gf4_subset_ranks", lambda: ctx.subsets(random_batch(rng, 4, 18, 1, 8), 5), "rank",
             {"modulus": q4}, bench="histogram", oracle=False,
             what="rank distribution of every 5-subset of 18 vectors in GF(4)^8"),
        Case("gf16_explicit_rref", lambda: ctx.explicit(random_batch(rng, 16, 2000, 6, 8)), "rref",
             {"modulus": polynomial(16)}, bench="all", oracle=False,
             what="rref of 2000 random 6x8 matrices over GF(16)"),
    ]
    return out
