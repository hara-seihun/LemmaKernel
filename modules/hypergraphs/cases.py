"""Oracle cases and benchmarks for finite uniform hypergraphs."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lemmakernel as lk  # noqa: E402
from tools.harness import Case  # noqa: E402


def uniform_hypergraphs(ctx, vertices, uniformity, edges):
    edge_family = ctx.subsets_of(ctx.range(0, vertices), uniformity)
    return ctx.subsets_of(edge_family, edges)


def cases(ctx, rng):
    del rng
    triple_pairs = uniform_hypergraphs(ctx, 4, 3, 2)
    three_edge_graphs = uniform_hypergraphs(ctx, 4, 2, 3)

    out = [
        Case("pairs of triples on four vertices", triple_pairs, "is_linear", {"vertices": 4, "limit": 3}),
        Case("three-edge graphs on four vertices", three_edge_graphs, "colouring_number", {"vertices": 4}),
        Case("pairs of triples on four vertices", triple_pairs, "has_berge_cycle",
             {"vertices": 4, "length": 2, "limit": 3}),
        Case("pairs of triples on four vertices", triple_pairs, "berge_girth", {"vertices": 4}),
        Case("three-edge graphs on four vertices", three_edge_graphs, "is_clique_free",
             {"vertices": 4, "clique_size": 3, "limit": 3}),
        Case("three-edge graphs on four vertices", three_edge_graphs, "is_ramsey_colouring",
             {"vertices": 4, "red_clique": 3, "blue_clique": 3, "limit": 3}),
    ]

    explicit = ctx.explicit(ctx.naturals([
        [[0, 1], [1, 2], [2, 3]],
        [[0, 1], [0, 2], [0, 3]],
        [[0, 1], [0, 2], [1, 2]],
    ]))
    out += [
        Case("explicit graphs", explicit, "colouring_number", {"vertices": 4}, reductions=["histogram"]),
        Case("explicit graphs", explicit, "has_berge_cycle", {"vertices": 4, "length": 3}, reductions=["all"]),
        Case("explicit graphs", explicit, "berge_girth", {"vertices": 4}, reductions=["all"]),
    ]

    out += [
        Case("finite-field edges", ctx.subsets(lk.matrix(2, [[0, 1], [1, 0]]), 1), "is_linear",
             {"vertices": 2}, reductions=["count"], oracle=False),
        Case("noncanonical edges", ctx.explicit(ctx.naturals([[[1, 0], [1, 2]]])), "is_linear",
             {"vertices": 3}, reductions=["count"], oracle=False),
        Case("too many vertices", uniform_hypergraphs(ctx, 4, 2, 2), "is_linear",
             {"vertices": 65}, reductions=["count"], oracle=False),
        Case("Berge length one", uniform_hypergraphs(ctx, 4, 2, 2), "has_berge_cycle",
             {"vertices": 4, "length": 1}, reductions=["count"], oracle=False),
        Case("clique below uniformity", triple_pairs, "is_clique_free",
             {"vertices": 4, "clique_size": 2}, reductions=["count"], oracle=False),
    ]

    out.append(Case(
        "ramsey_R33_on_six_vertices",
        lambda: uniform_hypergraphs(ctx, 6, 2, 7),
        "is_ramsey_colouring",
        {"vertices": 6, "red_clique": 3, "blue_clique": 3, "limit": 4},
        reductions=["hits"],
        bench="hits",
        oracle=False,
        what="balanced red-blue colourings of K_6 avoiding a monochromatic triangle; R(3,3)=6 makes the hit set empty",
    ))
    return out


def invariants(ctx):
    triples = uniform_hypergraphs(ctx, 6, 3, 3)
    linear = ctx.value("hypergraphs.is_linear", triples, vertices=6).values
    two_cycles = ctx.value("hypergraphs.has_berge_cycle", triples, vertices=6, length=2).values
    assert all(bool(a) != bool(b) for a, b in zip(linear, two_cycles))

    graphs = uniform_hypergraphs(ctx, 5, 2, 5)
    ramsey = ctx.value("hypergraphs.is_ramsey_colouring", graphs, "count",
                       vertices=5, red_clique=3, blue_clique=3)
    assert ramsey.value == 12

    four_edges = uniform_hypergraphs(ctx, 4, 2, 4)
    five_edges = uniform_hypergraphs(ctx, 4, 2, 5)
    assert ctx.value("hypergraphs.is_clique_free", four_edges, "count", vertices=4, clique_size=3).value > 0
    assert ctx.value("hypergraphs.is_clique_free", five_edges, "count", vertices=4, clique_size=3).value == 0
