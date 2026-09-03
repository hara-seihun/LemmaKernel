"""Oracle and benchmark cases for matchings_and_flows.

Oracle families stay below 64 members. Benchmarks move the size into runtime enumeration and use
families that describe whole graph classes without materialising them in Python.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lemmakernel as lk  # noqa: E402
from tools.harness import Case  # noqa: E402


def natural_batch(rng, count, n, bound):
    return lk.naturals([
        [[rng.randrange(bound) for _ in range(n)] for _ in range(n)]
        for _ in range(count)
    ])


def symmetric_batch(rng, count, n, bound):
    batch = []
    for _ in range(count):
        matrix = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                matrix[i][j] = matrix[j][i] = rng.randrange(bound)
        batch.append(matrix)
    return lk.naturals(batch)


def cases(ctx, rng):
    out = [
        Case("binary 2x2 perfect matchings", ctx.all_matrices(2, 2, 2), "perfect_matching_count"),
        Case("binary three-vertex spanning trees", ctx.symmetric_matrices(2, 3), "spanning_tree_count"),
        Case("binary two-vertex flows", ctx.all_matrices(2, 2, 2), "max_flow", {"source": 0, "sink": 1}),
    ]

    weighted_permanents = ctx.explicit(natural_batch(rng, 7, 3, 5))
    weighted_trees = ctx.explicit(symmetric_batch(rng, 7, 4, 4))
    weighted_flows = ctx.explicit(natural_batch(rng, 7, 4, 6))
    out.append(Case("weighted 3x3 permanents", weighted_permanents, "perfect_matching_count"))
    out.append(Case("weighted four-vertex trees", weighted_trees, "spanning_tree_count"))
    out.append(Case("weighted four-vertex flows", weighted_flows, "max_flow", {"source": 0, "sink": 3}))

    out += [
        Case("rectangular perfect matching", ctx.explicit(lk.naturals([[1, 0, 1], [0, 1, 0]])),
             "perfect_matching_count", oracle=False),
        Case("rectangular spanning tree", ctx.explicit(lk.naturals([[1, 0, 1], [0, 1, 0]])),
             "spanning_tree_count", oracle=False),
        Case("asymmetric spanning tree", ctx.explicit(lk.naturals([[0, 1], [0, 0]])),
             "spanning_tree_count", oracle=False),
        Case("equal flow terminals", ctx.explicit(lk.naturals([[0, 2], [0, 0]])),
             "max_flow", {"source": 0, "sink": 0}, oracle=False),
        Case("flow terminal outside graph", ctx.explicit(lk.naturals([[0, 2], [0, 0]])),
             "max_flow", {"source": 0, "sink": 2}, oracle=False),
    ]

    out += [
        Case("all_bipartite_graphs_4_4", lambda: ctx.all_matrices(2, 4, 4), "perfect_matching_count",
             bench="histogram", oracle=False,
             what="perfect-matching count distribution over all 65,536 bipartite graphs on two labelled sets of four vertices"),
        Case("all_graphs_5", lambda: ctx.symmetric_matrices(2, 5), "spanning_tree_count",
             bench="histogram", oracle=False,
             what="spanning-tree count distribution over all symmetric binary 5x5 adjacency matrices, with loops ignored"),
        Case("all_unit_networks_4", lambda: ctx.all_matrices(2, 4, 4), "max_flow",
             {"source": 0, "sink": 3}, bench="histogram", oracle=False,
             what="maximum-flow distribution over all directed unit-capacity networks on four labelled vertices"),
    ]
    return out


def invariants(ctx):
    for n in range(1, 7):
        ones = lk.naturals([[[1] * n for _ in range(n)]])
        assert ctx.value("matchings_and_flows.perfect_matching_count", ctx.explicit(ones)).values == [math.factorial(n)]

    for n in range(2, 8):
        complete = lk.naturals([[[int(i != j) for j in range(n)] for i in range(n)]])
        assert ctx.value("matchings_and_flows.spanning_tree_count", ctx.explicit(complete)).values == [n ** (n - 2)]

    network = lk.naturals([[
        [0, 16, 13, 0, 0, 0],
        [0, 0, 10, 12, 0, 0],
        [0, 4, 0, 0, 14, 0],
        [0, 0, 9, 0, 0, 20],
        [0, 0, 0, 7, 0, 4],
        [0, 0, 0, 0, 0, 0],
    ]])
    assert ctx.value("matchings_and_flows.max_flow", ctx.explicit(network), source=0, sink=5).values == [23]
