"""gfp cases: the inputs the harness runs against every backend, the naive implementation and the
Lean reference. Oracle cases are sized for the Lean kernel (a few dozen members); bench cases
are sized to show what the kernel does.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lemmakernel as lk  # noqa: E402
from tools.harness import Case, random_batch, rotate  # noqa: E402

PRIMES = [2, 3, 7, 257, 65537, 4294967291]  # small primes, then one per entry width
WALK_OPS = ["rank", "nullity", "full_row_rank", "full_col_rank", "in_span", "rref", "nullspace"]


def small_families(ctx, rng, p):
    """Every family kind at a size Lean evaluates in seconds. Returns (name, handle) pairs."""
    out = [("explicit", ctx.explicit(random_batch(rng, p, 8, 3, 4))),
           ("subsets", ctx.subsets(random_batch(rng, p, 6, 1, 4), 3))]
    if p == 2:
        out += [("grassmannian", ctx.grassmannian(2, 4, 2)), ("all_matrices", ctx.all_matrices(2, 2, 2))]
        G = ctx.grassmannian(2, 4, 2)
    elif p == 3:
        out += [("grassmannian", ctx.grassmannian(3, 3, 1)), ("all_matrices", ctx.all_matrices(3, 1, 2))]
        G = ctx.subsets(random_batch(rng, p, 5, 1, 4), 2)
    elif p < 20:
        out += [("grassmannian", ctx.grassmannian(p, 2, 1)), ("all_matrices", ctx.all_matrices(p, 1, 1))]
        G = ctx.subsets(random_batch(rng, p, 5, 1, 4), 2)
    else:
        G = ctx.subsets(random_batch(rng, p, 5, 1, 4), 2)
    out += [("transform", ctx.transform(G, random_batch(rng, p, 1, 4, 3))),
            ("stack", ctx.stack(G, random_batch(rng, p, 1, 2, 4))),
            ("stack(transform)", ctx.stack(ctx.transform(G, random_batch(rng, p, 1, 4, 2)), random_batch(rng, p, 1, 1, 2)))]
    return out


def cases(ctx, rng):
    """Every reduction on the F_2 and F_257 families; one reduction per case, rotating, elsewhere."""
    from tools.harness import module
    mod = module("gfp")
    out = []
    for p in PRIMES:
        for i, (name, fam) in enumerate(small_families(ctx, rng, p)):
            cols = fam.param("cols")
            for j, op in enumerate(WALK_OPS):
                args = {"limit": 3}
                if op == "in_span":
                    args["target"] = random_batch(rng, p, 1, 1, cols)
                reds = None if p in (2, 257) else rotate(mod, op, i + j)
                out.append(Case(f"{name} F_{p}", fam, op, args, reductions=reds))
        for rows, cols in [(1, 1), (2, 3), (3, 3), (4, 2), (5, 5)]:
            fam = ctx.explicit(random_batch(rng, p, 8, rows, cols))
            out.append(Case(f"explicit {rows}x{cols} F_{p}", fam, "rref_witness"))
            out.append(Case(f"explicit {rows}x{cols} F_{p}", fam, "solve", {"rhs": random_batch(rng, p, 8, 1, rows)}))
            if rows == cols:
                out.append(Case(f"explicit {rows}x{cols} F_{p}", fam, "inverse"))

    # Benchmarks: requests that look like real use, sized so that naive finishes and the ratio shows.
    out += [
        Case("subsets_independent", lambda: ctx.subsets(random_batch(rng, 2, 20, 1, 10), 6), "full_row_rank",
             what="how many 6-subsets of 20 vectors in F_2^10 are independent", bench="count", oracle=False),
        Case("grassmannian_image_rank", lambda: ctx.transform(ctx.grassmannian(2, 7, 3), random_batch(rng, 2, 1, 7, 5)), "rank",
             what="rank distribution of the image of every 3-subspace of F_2^7 under a fixed 7x5 map", bench="histogram", oracle=False),
        Case("explicit_rref", lambda: ctx.explicit(random_batch(rng, 7, 2000, 6, 8)), "rref",
             what="rref of 2000 random 6x8 matrices over F_7", bench="all", oracle=False),
        Case("explicit_inverse", lambda: ctx.explicit(random_batch(rng, 251, 2000, 8, 8)), "inverse",
             what="inverse of 2000 random 8x8 matrices over F_251", bench="all", oracle=False),
        Case("all_matrices_invertible", lambda: ctx.all_matrices(2, 4, 4), "full_col_rank",
             what="|GL(4, F_2)| by counting every 4x4 matrix over F_2", bench="count", oracle=False),
        Case("grassmannian_span_hits", lambda: ctx.stack(ctx.grassmannian(2, 7, 3), random_batch(rng, 2, 1, 2, 7)), "in_span",
             {"target": random_batch(rng, 2, 1, 1, 7), "limit": 4}, reductions=["hits"],
             what="which 3-subspaces of F_2^7, extended by two fixed rows, span a target vector",
             bench="hits", oracle=False),
    ]
    return out


def invariants(ctx):
    """Cross-operation identities on inputs beyond the kernel oracle."""
    G = ctx.grassmannian(3, 4, 2)
    F = ctx.stack(ctx.transform(G, lk.matrix(3, [[1, 0], [0, 1], [1, 1], [2, 1]])), lk.matrix(3, [[1, 2]]))
    r = ctx.run("gfp.rref", F)
    # a result batch is a valid explicit family, and rref preserves rank
    assert ctx.value("gfp.rank", ctx.explicit(r), "histogram").bins == ctx.value("gfp.rank", F, "histogram").bins
    # the witness really transforms: T * A = R for every member
    batch = random_batch(__import__("random").Random(11), 5, 50, 4, 6)
    w = ctx.value("gfp.rref_witness", ctx.explicit(batch))
    rr = ctx.value("gfp.rref", ctx.explicit(batch))
    for i in range(50):
        R, T = w.member(i)
        A = batch.member(i)
        TA = [[sum(T[a][k] * A[k][b] for k in range(4)) % 5 for b in range(6)] for a in range(4)]
        assert TA == R == rr.member(i)
