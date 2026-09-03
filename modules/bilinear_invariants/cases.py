"""Oracle, rejection, and benchmark cases for bilinear_invariants."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lemmakernel as lk  # noqa: E402
from tools.harness import Case, module, rotate  # noqa: E402

OPS = ["rank", "radical_dimension", "determinant", "determinant_class",
       "discriminant_class", "is_nondegenerate", "is_alternating", "congruence_label"]


def cases(ctx, rng):
    mod = module("bilinear_invariants")
    alternating3 = ctx.alternating_matrices(3, 2)
    symmetric2 = ctx.symmetric_matrices(2, 3)
    symmetric3 = ctx.symmetric_matrices(3, 2)
    explicit5 = ctx.explicit(lk.matrix(5, [
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 2, 0], [0, 0, 0]],
        [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
        [[0, 1, 0], [4, 0, 0], [0, 0, 0]],
        [[0, 2, 1], [3, 0, 0], [4, 0, 0]],
    ]))

    out = []
    for i, op in enumerate(OPS):
        out.append(Case("alternating F_3 dimension 2", alternating3, op, {"limit": 2}))
        out.append(Case("symmetric F_2 dimension 3", symmetric2, op, {"limit": 3},
                        reductions=rotate(mod, op, i)))
        out.append(Case("symmetric F_3 dimension 2", symmetric3, op, {"limit": 3},
                        reductions=rotate(mod, op, i + 1)))
        out.append(Case("chosen forms over F_5", explicit5, op, {"limit": 3},
                        reductions=rotate(mod, op, i + 2)))

    out += [
        Case("nonsquare explicit", ctx.explicit(lk.matrix(3, [[[1, 0, 0], [0, 1, 0]]])),
             "rank", oracle=False),
        Case("neither form", ctx.explicit(lk.matrix(3, [[[0, 1], [0, 0]]])),
             "determinant", oracle=False),
        Case("composite field tag", ctx.explicit(lk.matrix(4, [[[1, 0], [0, 1]]])),
             "rank", oracle=False),
        Case("unrestricted matrix family", ctx.all_matrices(3, 2, 2), "rank", oracle=False),
        Case("rank census of symmetric 5x5 forms", lambda: ctx.symmetric_matrices(2, 5), "rank",
             bench="histogram", oracle=False,
             what="rank distribution of all 32,768 symmetric 5x5 matrices over F_2"),
    ]
    return out


def invariants(ctx):
    p = 5
    a = [[1, 0, 0], [0, 2, 0], [0, 0, 0]]
    change = [[1, 1, 0], [0, 1, 1], [0, 0, 1]]
    ap = [[sum(a[i][k] * change[k][j] for k in range(3)) % p for j in range(3)] for i in range(3)]
    b = [[sum(change[k][i] * ap[k][j] for k in range(3)) % p for j in range(3)] for i in range(3)]
    family = ctx.explicit(lk.matrix(p, [a, b]))
    labels = ctx.value("bilinear_invariants.congruence_label", family).tolist()
    assert labels[0] == labels[1]
    assert ctx.value("bilinear_invariants.rank", family).values == [2, 2]
    assert ctx.value("bilinear_invariants.discriminant_class", family).values == [2, 2]

    alternating = ctx.alternating_matrices(3, 4)
    assert all(r % 2 == 0 for r in ctx.value("bilinear_invariants.rank", alternating).values)
