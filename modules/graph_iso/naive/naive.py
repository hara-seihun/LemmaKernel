"""Brute-force reference implementation of graph_iso.

Every graph is materialised, then every element of S_n is tested. Oracle cases keep n at most 5;
the benchmark uses n = 8. The generic backend uses colour refinement and backtracking instead.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))
from lemmakernel import naive as rt  # noqa: E402
from lemmakernel.interchange import Family, GraphGroups, Matrix, Perms  # noqa: E402


def relabel(graph: list[list[int]], order: tuple[int, ...] | list[int]) -> list[list[int]]:
    return [[graph[i][j] for j in order] for i in order]


def canonical_label(graph: list[list[int]]) -> list[int]:
    n = len(graph)
    best_code = None
    best_order = None
    for order in itertools.permutations(range(n)):
        code = tuple(graph[i][j] for i in order for j in order)
        if best_code is None or (code, order) < (best_code, best_order):
            best_code, best_order = code, order
    return list(best_order)


def automorphisms(graph: list[list[int]]) -> list[list[int]]:
    n = len(graph)
    return [list(g) for g in itertools.permutations(range(n)) if relabel(graph, g) == graph]


def validate(family: Family, graphs: list[list[list[int]]], p: int) -> int:
    if family.kind not in ("explicit", "symmetric_matrices"):
        raise ValueError("graph_iso operations are defined on explicit, symmetric_matrices families only")
    if p != 2:
        raise ValueError("graph adjacency matrices must be over F_2")
    if not graphs:
        raise ValueError("graphs must have at least one vertex")
    n = len(graphs[0])
    if n == 0:
        raise ValueError("graphs must have at least one vertex")
    for graph in graphs:
        if len(graph) != n or any(len(row) != n for row in graph):
            raise ValueError("graph adjacency matrices must be square with one common size")
        if any(graph[i][j] != graph[j][i] for i in range(n) for j in range(n)):
            raise ValueError("graph adjacency matrices must be symmetric")
    return n


def run(op: str, family: Family, reduction: str = "all", prefix: int | None = None, **_args):
    op = op.removeprefix("graph_iso.")
    if reduction != "all":
        raise ValueError(f"{op} values only reduce with `all`")
    graphs = list(itertools.islice(rt.iter_members(family), prefix))
    p = rt.prime(family)
    n = validate(family, graphs, p)

    if op == "canonical_label":
        labels = [canonical_label(graph) for graph in graphs]
        return Perms(n, len(labels), [v for label in labels for v in label])
    if op == "canonical_form":
        forms = [relabel(graph, canonical_label(graph)) for graph in graphs]
        return Matrix(2, len(forms), n, n, [x for form in forms for row in form for x in row])
    if op == "automorphism_group":
        groups = [automorphisms(graph) for graph in graphs]
        offsets = [0]
        for group in groups:
            offsets.append(offsets[-1] + len(group))
        return GraphGroups(len(groups), n, offsets, [v for group in groups for g in group for v in g])
    raise ValueError(f"unknown operation {op}")
