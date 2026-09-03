"""Oracle and benchmark cases for binary symplectic stabiliser codes."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lemmakernel as lk  # noqa: E402
from tools.harness import Case, module, rotate  # noqa: E402

OPS = ["is_self_orthogonal", "distance", "is_css"]


def cases(ctx, rng):
    out = []

    # This Grassmannian contains all 35 two-dimensional subspaces of F_2^4. It includes
    # Lagrangian stabilisers, non-self-orthogonal spaces, CSS spaces, and non-CSS spaces.
    all_two_qubit = ctx.grassmannian(2, 4, 2)
    for op in OPS:
        out.append(Case("all two-qubit 2-spaces", all_two_qubit, op, {"limit": 4}))

    explicit = lk.matrix(2, [
        [[1, 1, 0, 0], [0, 0, 1, 1]],  # <XX, ZZ>
        [[1, 0, 0, 0], [0, 0, 0, 1]],  # X on qubit 0, Z on qubit 1
        [[1, 0, 0, 0], [0, 0, 1, 0]],  # anticommuting X and Z on qubit 0
        [[1, 0, 1, 0], [0, 0, 0, 0]],  # Y on qubit 0
        [[1, 0, 1, 0], [0, 1, 0, 1]],  # <YI, IY>
        [[1, 1, 0, 0], [1, 1, 0, 0]],  # dependent displayed generators
    ])
    for i, op in enumerate(OPS):
        out.append(Case("displayed bases and dependent rows", ctx.explicit(explicit), op,
                        {"limit": 3}, reductions=rotate(module("quantum_codes"), op, i)))

    # Exercise the runtime's compositional matrix families. These maps need not be symplectic;
    # each image is simply interpreted as a new generated subspace.
    g = ctx.grassmannian(2, 4, 1)
    families = [
        ("six-coordinate transforms", ctx.transform(g, lk.matrix(2, [
            [1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 1, 0, 1, 1],
            [0, 0, 0, 1, 1, 0],
        ]))),
        ("stacked generators", ctx.stack(g, lk.matrix(2, [[0, 0, 1, 1]]))),
        ("subsets of Pauli words", ctx.subsets(lk.matrix(2, [
            [1, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 1], [0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 0], [1, 1, 0, 0, 0, 1],
        ]), 2)),
        ("all one-row two-qubit spaces", ctx.all_matrices(2, 1, 4)),
        ("subsets of projective points", ctx.subsets_of(ctx.grassmannian(2, 2, 1), 2)),
        ("symmetric binary generators", ctx.symmetric_matrices(2, 2)),
    ]
    mod = module("quantum_codes")
    for i, (name, family) in enumerate(families):
        for j, op in enumerate(OPS):
            out.append(Case(name, family, op, {"limit": 2}, reductions=rotate(mod, op, i + j)))

    out += [
        Case("nonbinary generators", ctx.explicit(lk.matrix(3, [[[1, 0, 0, 1]]])),
             "is_self_orthogonal", reductions=["all"], oracle=False),
        Case("odd number of columns", ctx.explicit(lk.matrix(2, [[[1, 0, 1]]])),
             "distance", reductions=["all"], oracle=False),
        Case("distance census of three-spaces", lambda: ctx.grassmannian(2, 8, 3), "distance",
             reductions=["histogram"], bench="histogram", oracle=False,
             what="distance distribution of all binary symplectic 3-spaces in F_2^8"),
    ]
    return out


def invariants(ctx):
    """Check standard stabiliser codes whose parameters are known independently."""
    five_qubit = lk.matrix(2, [[
        [1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 1, 0],
        [1, 0, 1, 0, 0, 0, 0, 0, 1, 1],
        [0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
    ]])
    f = ctx.explicit(five_qubit)
    assert ctx.value("quantum_codes.is_self_orthogonal", f).values == [1]
    assert ctx.value("quantum_codes.distance", f).values == [3]
    assert ctx.value("quantum_codes.is_css", f).values == [0]

    h = [
        [1, 1, 1, 1, 0, 0, 0],
        [1, 1, 0, 0, 1, 1, 0],
        [1, 0, 1, 0, 1, 0, 1],
    ]
    zero = [0] * 7
    steane_rows = [row + zero for row in h] + [zero + row for row in h]
    steane = ctx.explicit(lk.matrix(2, [steane_rows]))
    assert ctx.value("quantum_codes.is_self_orthogonal", steane).values == [1]
    assert ctx.value("quantum_codes.distance", steane).values == [3]
    assert ctx.value("quantum_codes.is_css", steane).values == [1]

    bell = ctx.explicit(lk.matrix(2, [[
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]]))
    assert ctx.value("quantum_codes.distance", bell).values == [3]
