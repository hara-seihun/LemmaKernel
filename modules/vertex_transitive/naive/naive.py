"""Brute-force baseline for graph transitivity and Cayley recognition.

Every vertex permutation is tested to form the automorphism group. Regular subgroups are generated
by closing canonical sequences of fixed-point-free automorphisms; the native backend uses the same
mathematical search with colour refinement and indexed group elements.
"""
from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
from lemmakernel import naive as rt  # noqa: E402
from lemmakernel.interchange import Family, RegularSubgroups  # noqa: E402

SUPPORTED = {"explicit", "all_graphs", "edge_subgraphs", "cayley_graphs"}


def compose(g, h):
    return tuple(h[x] for x in g)


def relabel(graph, permutation):
    return [[graph[permutation[i]][permutation[j]] for j in range(len(graph))]
            for i in range(len(graph))]


def automorphisms(graph):
    return [g for g in itertools.permutations(range(len(graph))) if relabel(graph, g) == graph]


def permutation_order(g):
    seen = [False] * len(g)
    order = 1
    for start in range(len(g)):
        if seen[start]:
            continue
        length = 0
        x = start
        while not seen[x]:
            seen[x] = True
            length += 1
            x = g[x]
        order = math.lcm(order, length)
    return order


def regular_subgroups(graph):
    n = len(graph)
    elements = automorphisms(graph)
    index = {g: i for i, g in enumerate(elements)}
    identity = tuple(range(n))
    candidates = [i for i, g in enumerate(elements)
                  if g != identity and all(g[x] != x for x in range(n)) and n % permutation_order(g) == 0]
    closure_cache = {}

    def closure(generators):
        key = tuple(generators)
        if key in closure_cache:
            return closure_cache[key]
        seen = {0}
        queue = [0]
        for x in queue:
            for generator in generators:
                y = index[compose(elements[x], elements[generator])]
                if y not in seen:
                    if y and any(elements[y][v] == v for v in range(n)):
                        closure_cache[key] = None
                        return None
                    seen.add(y)
                    if len(seen) > n:
                        closure_cache[key] = None
                        return None
                    queue.append(y)
        result = tuple(sorted(seen))
        closure_cache[key] = result
        return result

    found = []

    def search(subgroup, generators, after):
        members = set(subgroup)
        for g in candidates:
            if g <= after or g in members:
                continue
            larger = closure((*generators, g))
            if larger is None:
                continue
            added = [x for x in larger if x not in members]
            if not added or added[0] != g:
                continue
            if len(larger) == n:
                found.append(larger)
            else:
                search(larger, (*generators, g), g)

    search((0,), (), 0)
    found.sort()
    assert len(found) == len(set(found))
    return [[[int(x) for x in elements[i]] for i in subgroup] for subgroup in found]


def vertex_transitive(elements, n):
    return len({g[0] for g in elements}) == n


def arc_transitive(graph, elements):
    arcs = [(i, j) for i in range(len(graph)) for j in range(len(graph)) if graph[i][j]]
    if not arcs:
        return True
    first = arcs[0]
    return len({(g[first[0]], g[first[1]]) for g in elements}) == len(arcs)


def validate(family: Family, graphs, p):
    if family.kind not in SUPPORTED:
        raise ValueError("vertex_transitive operations are defined on explicit and graph families only")
    if p != 2:
        raise ValueError("graph adjacency matrices must be over F_2")
    if not graphs:
        raise ValueError("need at least one graph")
    n = len(graphs[0])
    if n == 0:
        raise ValueError("graphs must have at least one vertex")
    if n > 10:
        raise ValueError("the generic backend accepts at most 10 vertices")
    for graph in graphs:
        if len(graph) != n or any(len(row) != n for row in graph):
            raise ValueError("graph adjacency matrices must be square")
        if any(graph[i][j] != graph[j][i] for i in range(n) for j in range(n)):
            raise ValueError("graph adjacency matrices must be symmetric")
        if any(graph[i][i] for i in range(n)):
            raise ValueError("simple graph adjacency matrices must have zero diagonal")
    return n


def run(op: str, family: Family, reduction: str = "all", prefix: int | None = None, **args):
    op = op.removeprefix("vertex_transitive.")
    graphs = list(itertools.islice(rt.iter_members(family), prefix))
    p = rt.prime(family)
    n = validate(family, graphs, p)

    if op == "regular_subgroups":
        groups = [regular_subgroups(graph) for graph in graphs]
        offsets = [0]
        for member in groups:
            offsets.append(offsets[-1] + len(member))
        value = RegularSubgroups(len(graphs), n, offsets,
                                 [x for member in groups for subgroup in member for g in subgroup for x in g])
        return rt.reduce_values(reduction, value)

    elements = [automorphisms(graph) for graph in graphs]
    if op == "is_vertex_transitive":
        flags = [vertex_transitive(group, n) for group in elements]
    elif op == "is_arc_transitive":
        flags = [arc_transitive(graph, group) for graph, group in zip(graphs, elements)]
    elif op == "is_cayley":
        flags = [bool(regular_subgroups(graph)) for graph in graphs]
    else:
        raise ValueError(f"unknown operation {op}")
    return rt.reduce_bool(reduction, flags, graphs, p, **args)
