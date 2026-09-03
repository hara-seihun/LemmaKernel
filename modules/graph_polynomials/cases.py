"""Oracle cases for exact graph polynomials."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lemmakernel as lk  # noqa: E402
from tools.harness import Case  # noqa: E402

OPS = ["chromatic", "tutte", "characteristic", "matching"]


def edge_family(ctx, edges, k):
    return ctx.subsets(lk.naturals(edges), k)


def cases(ctx, rng):
    del rng
    samples = [
        ("empty graph on three vertices", 3, [(0, 1), (0, 2), (1, 2)], 0),
        ("two edges on three vertices", 3, [(0, 1), (0, 2), (1, 2)], 2),
        ("three edges on four vertices", 4,
         [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], 3),
        ("three edges from a five-vertex cycle with chord", 5,
         [(0, 1), (0, 4), (1, 2), (1, 3), (2, 3), (3, 4)], 3),
    ]
    out = []
    for name, vertices, edges, k in samples:
        family = edge_family(ctx, edges, k)
        out.extend(Case(name, family, op, {"vertices": vertices}) for op in OPS)

    out += [
        Case("reversed edge", edge_family(ctx, [(1, 0), (0, 2)], 1), "chromatic", {"vertices": 3}, oracle=False),
        Case("duplicate edge", edge_family(ctx, [(0, 1), (0, 1), (1, 2)], 1), "tutte", {"vertices": 3}, oracle=False),
        Case("endpoint outside graph", edge_family(ctx, [(0, 1), (1, 3)], 1), "matching", {"vertices": 3}, oracle=False),
        Case("not an edge-subset family", ctx.range(0, 4), "characteristic", {"vertices": 4}, oracle=False),
    ]
    return out


def evaluate_univariate(coefficients, x):
    return sum(coefficient * x ** degree for degree, coefficient in enumerate(coefficients))


def invariants(ctx):
    vertices = 6
    dictionary = [(u, v) for u in range(vertices) for v in range(u + 1, vertices)]
    family = edge_family(ctx, dictionary, 4)
    chromatic = ctx.value("graph_polynomials.chromatic", family, vertices=vertices).tolist()
    characteristic = ctx.value("graph_polynomials.characteristic", family, vertices=vertices).tolist()
    matching = ctx.value("graph_polynomials.matching", family, vertices=vertices).tolist()
    tutte = ctx.value("graph_polynomials.tutte", family, vertices=vertices).tolist()
    width = 5
    for chrom, char, match, tut in zip(chromatic, characteristic, matching, tutte):
        assert chrom[-1] == char[-1] == match[-1] == 1
        assert evaluate_univariate(chrom, 1) == 0
        assert char[-2] == 0 and char[-3] == -4
        assert all(coefficient == 0 for degree, coefficient in enumerate(match) if (vertices - degree) % 2)
        at_two_two = sum(tut[i * width + j] * 2 ** (i + j)
                         for i in range(vertices + 1) for j in range(width))
        assert at_two_two == 2 ** 4
