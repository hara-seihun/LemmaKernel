"""Small design families for the Lean oracle, plus one larger incidence-count benchmark."""
from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lemmakernel as lk  # noqa: E402
from tools.harness import Case, cyclic, dihedral, unit_vectors  # noqa: E402


def block(v, points):
    return [[int(i == point) for i in range(v)] for point in points]


def block_batch(v, blocks):
    return lk.matrix(2, [block(v, points) for points in blocks])


def cases(ctx, rng):
    complete_triples = ctx.subsets(unit_vectors(2, 5), 3)
    fano_blocks = [(0, 1, 3), (0, 2, 5), (0, 4, 6), (1, 2, 6), (1, 4, 5), (2, 3, 4), (3, 5, 6)]
    fano = ctx.explicit(block_batch(7, fano_blocks))
    edges4 = list(itertools.combinations(range(4), 2))
    k4 = ctx.explicit(block_batch(4, edges4))
    star = ctx.explicit(block_batch(4, [(0, 1), (0, 2), (0, 3)]))

    out = [
        Case("complete triples design", complete_triples, "is_design", {"t": 2}),
        Case("complete triples lambdas", complete_triples, "lambda_vector", {"t": 2}),
        Case("complete triples intersections", complete_triples, "intersection_numbers"),
        Case("Fano design", fano, "is_design", {"t": 2}),
        Case("Fano dual", fano, "dual_is_design"),
        Case("Fano intersections", fano, "intersection_numbers"),
        Case("Fano is not resolvable", fano, "is_resolvable"),
        Case("K4 edge resolution", k4, "is_resolvable"),
        Case("K4 dual", k4, "dual_is_design"),
        Case("star is not a design", star, "is_design", {"t": 1}),
        Case("C5 Kramer-Mesner", ctx.subsets(unit_vectors(2, 5), 2), "kramer_mesner_matrix",
             {"t": 1, "group": ctx.perms(5, cyclic(5))}),
        Case("D4 Kramer-Mesner", ctx.subsets(unit_vectors(2, 4), 3), "kramer_mesner_matrix",
             {"t": 2, "group": ctx.perms(4, dihedral(4))}),
    ]

    out += [
        Case("malformed repeated point", ctx.explicit(lk.matrix(2, [[block(3, (0, 0))[0], block(3, (0, 0))[0]]])),
             "intersection_numbers", oracle=False),
        Case("Kramer-Mesner with a matrix group", ctx.subsets(unit_vectors(2, 4), 2), "kramer_mesner_matrix",
             {"t": 1, "group": lk.matrix(2, [[[1, 0], [0, 1]]])}, oracle=False),
    ]

    out.append(Case(
        "complete 6-subsets of 12",
        lambda: ctx.subsets(unit_vectors(2, 12), 6),
        "lambda_vector",
        {"t": 3},
        bench="all",
        oracle=False,
        what="multiplicities of all triples in the complete family of 6-subsets of 12 points",
    ))
    return out


def invariants(ctx):
    v, k, t = 8, 4, 2
    complete = ctx.subsets(unit_vectors(2, v), k)
    lambdas = ctx.value("designs.lambda_vector", complete, t=t).member(0)[0]
    decision = ctx.value("designs.is_design", complete, t=t).member(0)[0]
    assert len(lambdas) == math.comb(v, t)
    assert set(lambdas) == {math.comb(v - t, k - t)}
    assert decision == [1, lambdas[0]]

    edges = list(itertools.combinations(range(4), 2))
    edge_family = ctx.explicit(block_batch(4, edges))
    witness = ctx.value("designs.is_resolvable", edge_family).member(0)
    assert sorted(i for parallel_class in witness for i in parallel_class) == list(range(len(edges)))
    for parallel_class in witness:
        assert set().union(*(set(edges[i]) for i in parallel_class)) == set(range(4))

    km = ctx.value("designs.kramer_mesner_matrix", ctx.subsets(unit_vectors(2, 6), 3),
                   t=2, group=ctx.perms(6, dihedral(6))).member(0)
    assert {sum(row) for row in km} == {math.comb(6 - 2, 3 - 2)}
