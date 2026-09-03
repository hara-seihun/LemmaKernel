"""Cases for graph transitivity and Cayley recognition."""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lemmakernel as lk  # noqa: E402
from tools.harness import Case  # noqa: E402


def graph(n, edges=(), loops=()):
    adjacency = [[0] * n for _ in range(n)]
    for i, j in edges:
        adjacency[i][j] = adjacency[j][i] = 1
    for i in loops:
        adjacency[i][i] = 1
    return adjacency


def cycle(n):
    return graph(n, [(i, (i + 1) % n) for i in range(n)])


def complete(n):
    return graph(n, [(i, j) for i in range(n) for j in range(i + 1, n)])


def cases(ctx, rng):
    del rng
    graphs4 = ctx.explicit(lk.matrix(2, [
        graph(4),
        complete(4),
        cycle(4),
        graph(4, [(0, 1), (1, 2), (2, 3)]),
        graph(4, [(0, 1), (0, 2), (0, 3)]),
        graph(4, [(0, 1), (2, 3)]),
    ]))
    out = [
        Case("small simple graphs", graphs4, "is_vertex_transitive", {"limit": 3}),
        Case("small simple graphs", graphs4, "is_arc_transitive", {"limit": 3}),
        Case("small simple graphs", graphs4, "is_cayley", {"limit": 3}),
        Case("all simple graphs on three vertices", ctx.all_graphs(3), "regular_subgroups"),
    ]

    cyclic4 = ctx.perms(4, [[1, 2, 3, 0]])
    out.append(Case("C4 Cayley graph family", ctx.cayley_graphs(cyclic4),
                    "is_vertex_transitive", {"limit": 2}, reductions=["all"]))

    out += [
        Case("ternary adjacency", ctx.explicit(lk.matrix(3, [graph(3, [(0, 1)])])),
             "is_vertex_transitive", reductions=["all"], oracle=False),
        Case("rectangular adjacency", ctx.explicit(lk.matrix(2, [[[0, 1, 0], [1, 0, 1]]])),
             "is_arc_transitive", reductions=["all"], oracle=False),
        Case("directed adjacency", ctx.explicit(lk.matrix(2, [[[0, 1, 0], [0, 0, 1], [0, 0, 0]]])),
             "is_cayley", reductions=["all"], oracle=False),
        Case("looped adjacency", ctx.explicit(lk.matrix(2, [graph(3, [(0, 1)], loops=[2])])),
             "regular_subgroups", reductions=["all"], oracle=False),
        Case("eleven vertices", ctx.explicit(lk.matrix(2, [graph(11)])),
             "is_vertex_transitive", reductions=["all"], oracle=False),
        Case("all matrices family", ctx.all_matrices(2, 3, 3),
             "is_vertex_transitive", reductions=["all"], oracle=False),
    ]

    all6 = lambda: ctx.all_graphs(6)
    out += [
        Case("all_graphs_6_vertex_transitive", all6, "is_vertex_transitive",
             reductions=["count"], bench="count", oracle=False,
             what="count vertex-transitive isomorphism classes among all 156 simple graphs on six vertices"),
        Case("all_graphs_6_arc_transitive", all6, "is_arc_transitive",
             reductions=["count"], bench="count", oracle=False,
             what="count arc-transitive isomorphism classes among all 156 simple graphs on six vertices"),
        Case("all_graphs_6_cayley", all6, "is_cayley",
             reductions=["count"], bench="count", oracle=False,
             what="count Cayley isomorphism classes among all 156 simple graphs on six vertices"),
        Case("cycle_8_regular_subgroups", ctx.explicit(lk.matrix(2, [cycle(8)])),
             "regular_subgroups", reductions=["all"], bench="all", oracle=False,
             what="list every regular subgroup of the automorphism group of the 8-cycle"),
    ]
    return out


def _compose(g, h):
    return [h[g[i]] for i in range(len(g))]


def _relabel(adjacency, permutation):
    return [[adjacency[permutation[i]][permutation[j]] for j in range(len(adjacency))]
            for i in range(len(adjacency))]


def invariants(ctx):
    cyclic5 = ctx.perms(5, [[1, 2, 3, 4, 0]])
    cayley5 = ctx.cayley_graphs(cyclic5)
    assert all(ctx.value("vertex_transitive.is_vertex_transitive", cayley5).values)
    assert all(ctx.value("vertex_transitive.is_cayley", cayley5).values)

    c6 = cycle(6)
    family = ctx.explicit(lk.matrix(2, [c6]))
    groups = ctx.value("vertex_transitive.regular_subgroups", family).member(0)
    assert groups
    identity = list(range(6))
    for group in groups:
        assert len(group) == 6 and group == sorted(group) and identity in group
        assert {g[0] for g in group} == set(range(6))
        assert all(_relabel(c6, g) == c6 for g in group)
        assert all(_compose(g, h) in group for g in group for h in group)
    assert ctx.value("vertex_transitive.is_cayley", family).values == [bool(groups)]
