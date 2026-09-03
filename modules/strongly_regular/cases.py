"""Strongly regular graph cases for the native backend, naive implementation, and Lean oracle."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lemmakernel as lk  # noqa: E402
from tools.harness import Case  # noqa: E402

OPS = ["srg_params", "spectrum", "is_srg", "krein_bound", "absolute_bound"]


def graph(n, edges):
    a = [[0] * n for _ in range(n)]
    for i, j in edges:
        a[i][j] = a[j][i] = 1
    return a


def cycle(n):
    return graph(n, [(i, (i + 1) % n) for i in range(n)])


def rook_graph(n):
    return graph(n * n, [(n * i + j, n * x + y)
                         for i in range(n) for j in range(n)
                         for x in range(n) for y in range(n)
                         if (i, j) < (x, y) and (i == x or j == y)])


def petersen():
    edges = [(i, (i + 1) % 5) for i in range(5)]
    edges += [(5 + i, 5 + (i + 2) % 5) for i in range(5)]
    edges += [(i, 5 + i) for i in range(5)]
    return graph(10, edges)


def cases(ctx, rng):
    c5 = ctx.explicit(lk.matrix(2, cycle(5)))
    out = [Case("C5 Cayley adjacency", c5, op, {"limit": 1}) for op in OPS]

    mixed = [cycle(5),
             graph(5, [(0, 1), (1, 2), (2, 3), (3, 4)]),
             graph(5, []),
             graph(5, [(i, j) for i in range(5) for j in range(i + 1, 5)])]
    looped = cycle(5)
    looped[0][0] = 1
    directed = cycle(5)
    directed[0][1] = 0
    mixed += [looped, directed]
    batch = ctx.explicit(lk.matrix(2, mixed))
    out += [Case("mixed explicit adjacency", batch, op, {"limit": 2}, reductions=["all"])
            for op in OPS]

    point_graph = ctx.explicit(lk.matrix(2, rook_graph(3)))
    out += [Case("point graph of the 3x3 grid", point_graph, op, {"limit": 1}, reductions=["all"])
            for op in OPS]

    imprimitive = ctx.explicit(lk.matrix(2, [cycle(4), graph(4, [(0, 1), (2, 3)])]))
    out += [Case("imprimitive strongly regular graphs", imprimitive, op, {"limit": 2}, reductions=["all"])
            for op in OPS]

    all_small = ctx.symmetric_matrices(2, 3)
    out += [Case("all simple candidates on three vertices", all_small, op, {"limit": 2}, reductions=["all"])
            for op in OPS]

    out += [
        Case("ternary square matrices", ctx.explicit(lk.matrix(3, cycle(5))), "is_srg",
             reductions=["all"], oracle=False),
        Case("rectangular binary matrices", ctx.all_matrices(2, 2, 3), "srg_params",
             reductions=["all"], oracle=False),
        Case("six-vertex graph census", lambda: ctx.symmetric_matrices(2, 6), "is_srg",
             bench="count", oracle=False,
             what="count labelled strongly regular graphs among all 2^21 symmetric 6x6 binary matrices"),
    ]
    return out


def invariants(ctx):
    for adjacency in [petersen(), rook_graph(3), cycle(5), cycle(4)]:
        fam = ctx.explicit(lk.matrix(2, adjacency))
        flag = ctx.value("strongly_regular.is_srg", fam).values[0]
        krein = ctx.value("strongly_regular.krein_bound", fam).values[0]
        absolute = ctx.value("strongly_regular.absolute_bound", fam).values[0]
        params = ctx.value("strongly_regular.srg_params", fam).member(0)
        spectrum = ctx.value("strongly_regular.spectrum", fam).member(0)
        assert flag == krein == absolute
        assert bool(flag) == (params is not None) == (spectrum is not None)
        assert spectrum is None or spectrum[4] + spectrum[5] + 1 == params[0]
