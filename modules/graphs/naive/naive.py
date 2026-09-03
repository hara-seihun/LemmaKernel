"""Direct Python implementation of finite simple graph invariants."""
from __future__ import annotations

import itertools
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
from lemmakernel import naive as rt  # noqa: E402
from lemmakernel.interchange import DegreeSequences, Family, Matrix  # noqa: E402


def validate(g):
    n = len(g)
    if n == 0 or any(len(row) != n for row in g):
        raise ValueError("graphs needs a nonempty square adjacency matrix over F_2")
    if any(g[i][i] or any(g[i][j] != g[j][i] for j in range(n)) for i in range(n)):
        raise ValueError("graphs needs a simple undirected graph: adjacency must be symmetric with zero diagonal")


def distances(g, source):
    d = [-1] * len(g)
    d[source] = 0
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v, edge in enumerate(g[u]):
            if edge and d[v] < 0:
                d[v] = d[u] + 1
                queue.append(v)
    return d


def connected(g):
    return all(d >= 0 for d in distances(g, 0))


def girth(g):
    n = len(g)
    best = n + 1
    for root in range(n):
        d = [-1] * n
        parent = [-1] * n
        d[root] = 0
        queue = deque([root])
        while queue:
            u = queue.popleft()
            for v, edge in enumerate(g[u]):
                if not edge:
                    continue
                if d[v] < 0:
                    d[v] = d[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif parent[u] != v:
                    best = min(best, d[u] + d[v] + 1)
    return 0 if best == n + 1 else best


def diameter(g):
    rows = [distances(g, u) for u in range(len(g))]
    return len(g) if any(-1 in row for row in rows) else max(max(row) for row in rows)


def colorable(g, colors):
    n = len(g)
    return any(all(c[u] != c[v] for u in range(n) for v in range(u + 1, n) if g[u][v])
               for c in itertools.product(range(colors), repeat=n))


def chromatic_number(g):
    return next(k for k in range(1, len(g) + 1) if colorable(g, k))


def clique_number(g, complement=False):
    vertices = range(len(g))
    for k in range(len(g), 0, -1):
        for chosen in itertools.combinations(vertices, k):
            if all((not g[u][v] if complement else g[u][v]) for u, v in itertools.combinations(chosen, 2)):
                return k
    return 0


def degree_sequence(g):
    return sorted((sum(row) for row in g), reverse=True)


def run(op: str, family: Family, reduction: str = "all", prefix: int | None = None, **args):
    op = op.removeprefix("graphs.")
    allowed = {"explicit", "all_graphs", "edge_subgraphs", "cayley_graphs"}
    if family.kind not in allowed:
        raise ValueError(f"{op} is defined on {', '.join(sorted(allowed))} families only")
    members = list(itertools.islice(rt.iter_members(family), prefix))
    for g in members:
        validate(g)
    n = len(members[0]) if members else family.params.get("n", 0)

    if op == "degree_sequence":
        values = [degree_sequence(g) for g in members]
        return rt.reduce_values(reduction, DegreeSequences(len(values), n, [x for row in values for x in row]))
    if op == "canonical_form":
        values = [rt.canonical_graph(g) for g in members]
        return rt.reduce_values(reduction, Matrix(2, len(values), n, n, [x for g in values for row in g for x in row]))

    if op == "connected":
        return rt.reduce_bool(reduction, [connected(g) for g in members], members, 2, **args)
    if op == "is_bipartite":
        return rt.reduce_bool(reduction, [colorable(g, 2) for g in members], members, 2, **args)
    if op == "girth":
        values = [girth(g) for g in members]
    elif op == "diameter":
        values = [diameter(g) for g in members]
    elif op == "chromatic_number":
        values = [chromatic_number(g) for g in members]
    elif op == "clique_number":
        values = [clique_number(g) for g in members]
    elif op == "independence_number":
        values = [clique_number(g, True) for g in members]
    else:
        raise ValueError(f"unknown operation {op}")
    return rt.reduce_int(reduction, values, members, 2)
