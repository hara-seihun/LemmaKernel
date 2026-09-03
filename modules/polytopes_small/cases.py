"""Oracle and benchmark cases for small lattice polytopes."""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lemmakernel as lk  # noqa: E402
from tools.harness import Case  # noqa: E402


def cube(d):
    return lk.matrix(2, [list(bits) for bits in itertools.product(range(2), repeat=d)])


def cases(ctx, rng):
    triangles = ctx.subsets(cube(2), 3)
    lattice = lk.matrix(7, [[0, 0], [1, 0], [2, 0], [0, 2], [2, 2]])
    four_lattice_points = ctx.subsets(lattice, 4)
    cube3 = ctx.subsets(cube(3), 8)
    out = [
        Case("triangles in the square", triangles, "vertex_count"),
        Case("triangles in the square", triangles, "is_simplicial", {"limit": 3}),
        Case("triangles in the square", triangles, "f_vector"),
        Case("triangles in the square", triangles, "ehrhart_polynomial"),
        Case("four of five lattice points", four_lattice_points, "vertex_count", reductions=["all", "histogram"]),
        Case("four of five lattice points", four_lattice_points, "f_vector"),
        Case("four of five lattice points", four_lattice_points, "ehrhart_polynomial"),
        Case("the 3-cube", cube3, "is_simplicial", {"limit": 1}, reductions=["all"]),
        Case("the 3-cube", cube3, "f_vector"),
        Case("the 3-cube", cube3, "ehrhart_polynomial"),
        Case("explicit family is unsupported", ctx.explicit(lk.matrix(2, [[[0, 0], [1, 0], [0, 1]]])),
             "vertex_count", oracle=False),
        Case("five-subsets of the 5-cube", lambda: ctx.subsets(cube(5), 5), "vertex_count",
             reductions=["histogram"], bench="histogram", oracle=False,
             what="vertex counts for all 201376 five-point selections from the vertices of the 5-cube"),
    ]
    return out


def invariants(ctx):
    square = ctx.subsets(cube(2), 4)
    cube_family = ctx.subsets(cube(3), 8)
    simplex = ctx.subsets(lk.matrix(2, [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]), 4)
    assert ctx.value("polytopes_small.f_vector", square).tolist() == [[4, 4, 1]]
    assert ctx.value("polytopes_small.ehrhart_polynomial", square).tolist() == [[1, 1, 0]]
    assert ctx.value("polytopes_small.f_vector", cube_family).tolist() == [[8, 12, 6, 1]]
    assert ctx.value("polytopes_small.ehrhart_polynomial", cube_family).tolist() == [[1, 4, 1, 0]]
    assert ctx.value("polytopes_small.is_simplicial", cube_family).values == [0]
    assert ctx.value("polytopes_small.f_vector", simplex).tolist() == [[4, 6, 4, 1]]
    assert ctx.value("polytopes_small.is_simplicial", simplex).values == [1]
