"""Plain Python implementation of the hypergraphs module."""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
from lemmakernel import naive as rt  # noqa: E402
from lemmakernel.interchange import NATURALS, Family  # noqa: E402


def validate(family: Family, members, p: int, vertices: int) -> int:
    if family.kind not in ("explicit", "subsets", "subsets_of"):
        raise ValueError("hypergraph operations require explicit, subsets, or subsets_of families")
    if p != NATURALS:
        raise ValueError("hypergraph edges must be natural-number matrices")
    if not 0 <= vertices <= 64:
        raise ValueError("vertices must be at most 64")
    if not members or not members[0]:
        raise ValueError("hypergraph families must contain nonempty edge sets")
    uniformity, edge_count = len(members[0][0]), len(members[0])
    if not 2 <= uniformity <= vertices:
        raise ValueError("uniformity must satisfy 2 <= uniformity <= vertices")
    for hypergraph in members:
        if len(hypergraph) != edge_count:
            raise ValueError("edge counts differ")
        for edge in hypergraph:
            if len(edge) != uniformity or edge != sorted(set(edge)) or any(v >= vertices for v in edge):
                raise ValueError("each edge must be strictly increasing with vertices in range")
        if any(a >= b for a, b in zip(hypergraph, hypergraph[1:])):
            raise ValueError("edges must be in strict lexicographic order")
    return uniformity


def is_linear(hypergraph) -> bool:
    return all(len(set(a) & set(b)) <= 1 for a, b in itertools.combinations(hypergraph, 2))


def proper_colouring(hypergraph, colouring) -> bool:
    return all(any(colouring[v] != colouring[edge[0]] for v in edge[1:]) for edge in hypergraph)


def colouring_number(vertices: int, hypergraph) -> int:
    for colours in range(1, vertices + 1):
        for tail in itertools.product(range(colours), repeat=vertices - 1):
            if proper_colouring(hypergraph, (0,) + tail):
                return colours
    return vertices + 1


def has_berge_cycle(vertices: int, hypergraph, length: int) -> bool:
    if length < 2 or length > min(vertices, len(hypergraph)):
        return False
    used_edges = [False] * len(hypergraph)

    def extend(start, current, depth, used_vertices):
        if depth == length - 1:
            return any(not used_edges[i] and start in edge and current in edge
                       for i, edge in enumerate(hypergraph))
        for i, edge in enumerate(hypergraph):
            if used_edges[i] or current not in edge:
                continue
            used_edges[i] = True
            for following in edge:
                if following not in used_vertices:
                    used_vertices.add(following)
                    if extend(start, following, depth + 1, used_vertices):
                        used_vertices.remove(following)
                        used_edges[i] = False
                        return True
                    used_vertices.remove(following)
            used_edges[i] = False
        return False

    return any(extend(start, start, 0, {start}) for start in range(vertices))


def berge_girth(vertices: int, hypergraph) -> int:
    for length in range(2, min(vertices, len(hypergraph)) + 1):
        if has_berge_cycle(vertices, hypergraph, length):
            return length
    return 0


def contains_clique(vertices: int, hypergraph, uniformity: int, size: int, red: bool) -> bool:
    edge_set = {tuple(edge) for edge in hypergraph}
    for selected in itertools.combinations(range(vertices), size):
        present = (tuple(edge) in edge_set for edge in itertools.combinations(selected, uniformity))
        if all(present) if red else not any(present):
            return True
    return False


def run(op: str, family: Family, reduction: str = "all", prefix: int | None = None, **args):
    op = op.removeprefix("hypergraphs.")
    members, p = rt.members(family)
    vertices = args["vertices"]
    uniformity = validate(family, members, p, vertices)
    selected = members if prefix is None else members[:prefix]

    if op == "is_linear":
        flags = [is_linear(h) for h in selected]
        return rt.reduce_bool(reduction, flags, selected, p, **args)
    if op == "colouring_number":
        values = [colouring_number(vertices, h) for h in selected]
        return rt.reduce_int(reduction, values, selected, p)
    if op == "has_berge_cycle":
        length = args["length"]
        if length < 2:
            raise ValueError("length must be at least 2")
        flags = [has_berge_cycle(vertices, h, length) for h in selected]
        return rt.reduce_bool(reduction, flags, selected, p, **args)
    if op == "berge_girth":
        values = [berge_girth(vertices, h) for h in selected]
        return rt.reduce_int(reduction, values, selected, p)
    if op == "is_clique_free":
        size = args["clique_size"]
        if size < uniformity:
            raise ValueError("clique_size must be at least the uniformity")
        flags = [not contains_clique(vertices, h, uniformity, size, True) for h in selected]
        return rt.reduce_bool(reduction, flags, selected, p, **args)
    if op == "is_ramsey_colouring":
        red, blue = args["red_clique"], args["blue_clique"]
        if red < uniformity or blue < uniformity:
            raise ValueError("clique sizes must be at least the uniformity")
        flags = [not contains_clique(vertices, h, uniformity, red, True) and
                 not contains_clique(vertices, h, uniformity, blue, False) for h in selected]
        return rt.reduce_bool(reduction, flags, selected, p, **args)
    raise ValueError(f"unknown hypergraphs operation {op}")
