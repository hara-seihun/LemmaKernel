"""Graph families and requests checked against the Lean oracle."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lemmakernel as lk  # noqa: E402
from tools.harness import Case, module, rotate  # noqa: E402

SCALAR_OPS = ["connected", "girth", "diameter", "chromatic_number", "clique_number",
              "independence_number", "is_bipartite"]
ALL_OPS = SCALAR_OPS + ["degree_sequence", "canonical_form"]


def adjacency(n, edges):
    a = [[0] * n for _ in range(n)]
    for u, v in edges:
        a[u][v] = a[v][u] = 1
    return a


def complete(n):
    return adjacency(n, [(i, j) for i in range(n) for j in range(i + 1, n)])


def cycle(n):
    return adjacency(n, [(i, (i + 1) % n) for i in range(n)])


def random_graphs(rng, count, n):
    return [adjacency(n, [(i, j) for i in range(n) for j in range(i + 1, n) if rng.randrange(2)])
            for _ in range(count)]


def cases(ctx, rng):
    explicit = ctx.explicit(lk.matrix(2, [
        adjacency(5, []),
        adjacency(5, [(0, 1), (1, 2), (2, 3), (3, 4)]),
        cycle(5),
        adjacency(5, [(0, 1), (1, 2), (2, 0), (3, 4)]),
        complete(5),
        adjacency(5, [(0, 1), (0, 2), (0, 3), (0, 4)]),
    ]))
    all_three = ctx.all_graphs(3)
    host = lk.matrix(2, cycle(5))
    edge_families = [ctx.edge_subgraphs(host, k) for k in (0, 2, 4)]
    cayley = ctx.cayley_graphs(ctx.perms(4, [[1, 2, 3, 0]]))

    out = []
    for op in ALL_OPS:
        out.append(Case(f"all graphs on three vertices {op}", all_three, op, {"limit": 3}))

    mod = module("graphs")
    families = [("explicit examples", explicit),
                ("0-edge subgraphs of C5", edge_families[0]),
                ("2-edge subgraphs of C5", edge_families[1]),
                ("4-edge subgraphs of C5", edge_families[2]),
                ("Cayley graphs of C4", cayley)]
    for i, (name, family) in enumerate(families):
        for j, op in enumerate(ALL_OPS):
            out.append(Case(name, family, op, {"limit": 2}, reductions=rotate(mod, op, i + j)))

    out += [
        Case("directed adjacency", ctx.explicit(lk.matrix(2, [[0, 1], [0, 0]])), "connected",
             reductions=["all"], oracle=False),
        Case("matrix family is not a graph family", ctx.all_matrices(2, 2, 2), "connected",
             reductions=["all"], oracle=False),
    ]

    out += [
        Case("unlabelled_graphs_6", lambda: ctx.all_graphs(6), "connected", bench="count", oracle=False,
             what="connected isomorphism classes of simple graphs on six vertices"),
        Case("K7_six_edge_subgraphs", lambda: ctx.edge_subgraphs(lk.matrix(2, complete(7)), 6),
             "diameter", bench="histogram", oracle=False,
             what="diameters of all six-edge spanning subgraphs of K7"),
        Case("random_graph_chromatic", lambda: ctx.explicit(lk.matrix(2, random_graphs(rng, 120, 7))),
             "chromatic_number", bench="histogram", oracle=False,
             what="chromatic numbers of 120 random seven-vertex graphs"),
        Case("random_graph_canonical", lambda: ctx.explicit(lk.matrix(2, random_graphs(rng, 120, 8))),
             "canonical_form", bench="all", oracle=False,
             what="canonical forms of 120 random eight-vertex graphs"),
    ]
    return out


def invariants(ctx):
    import random

    graphs = random_graphs(random.Random(24), 80, 8)
    family = ctx.explicit(lk.matrix(2, graphs))
    connected_values = ctx.value("graphs.connected", family).values
    diameters = ctx.value("graphs.diameter", family).values
    chromatic = ctx.value("graphs.chromatic_number", family).values
    clique = ctx.value("graphs.clique_number", family).values
    bipartite = ctx.value("graphs.is_bipartite", family).values
    assert all(bool(c) == (d < 8) for c, d in zip(connected_values, diameters))
    assert all(w <= chi for w, chi in zip(clique, chromatic))
    assert all(bool(b) == (chi <= 2) for b, chi in zip(bipartite, chromatic))

    canonical = ctx.value("graphs.canonical_form", family)
    twice = ctx.value("graphs.canonical_form", ctx.explicit(canonical))
    assert canonical.encode() == twice.encode()
