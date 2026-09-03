"""Oracle and benchmark cases for char_poly."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lemmakernel as lk  # noqa: E402
from tools.harness import Case, module, random_batch, rotate  # noqa: E402

OPS = ["charpoly", "minpoly", "rational_canonical_form", "conjugacy_label",
       "is_regular", "is_semisimple", "element_order"]


def cases(ctx, rng):
    mod = module("char_poly")
    out = []

    all22 = ctx.all_matrices(2, 2, 2)
    for op in OPS:
        out.append(Case("all 2x2 matrices over F_2", all22, op, {"limit": 3}))

    structured = lk.matrix(3, [
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
        [[1, 1, 0], [0, 1, 0], [0, 0, 2]],
        [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 1, 0], [2, 0, 0], [0, 0, 1]],
    ])
    explicit = ctx.explicit(structured)
    for i, op in enumerate(OPS):
        out.append(Case("structured 3x3 matrices over F_3", explicit, op, {"limit": 3},
                        reductions=rotate(mod, op, i)))

    generators = lk.matrix(2, [[[1, 1], [0, 1]], [[0, 1], [1, 0]]])
    group = ctx.group_elements(generators)
    for i, op in enumerate(OPS):
        out.append(Case("GL(2,2) elements", group, op, {"limit": 3},
                        reductions=rotate(mod, op, i + 1)))

    out += [
        Case("nonsquare explicit", ctx.explicit(lk.matrix(2, [[[1, 0, 1], [0, 1, 1]]])),
             "charpoly", oracle=False),
        Case("permutation group", ctx.group_elements(ctx.perms(3, [[1, 2, 0]])),
             "charpoly", oracle=False),
        Case("explicit_6x6_charpoly", lambda: ctx.explicit(random_batch(rng, 251, 10000, 6, 6)),
             "charpoly", bench="all", oracle=False,
             what="characteristic polynomials of 10,000 dense and structured 6x6 matrices over F_251"),
    ]
    return out


def invariants(ctx):
    a = [[0, 1, 0], [0, 0, 1], [1, 1, 0]]
    p = [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
    pinv = ctx.value("gfp.inverse", ctx.explicit(lk.matrix(2, [p]))).member(0)
    conjugate = [[sum(pinv[i][k] * a[k][l] * p[l][j] for k in range(3) for l in range(3)) % 2
                  for j in range(3)] for i in range(3)]
    family = ctx.explicit(lk.matrix(2, [a, conjugate]))
    labels = ctx.value("char_poly.conjugacy_label", family).tolist()
    forms = ctx.value("char_poly.rational_canonical_form", family).tolist()
    assert labels[0] == labels[1]
    assert forms[0] == forms[1]
