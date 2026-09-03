"""Direct Python implementation of simple undirected Cayley graph operations."""
from __future__ import annotations

import itertools
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
from lemmakernel import naive as rt  # noqa: E402
from lemmakernel.interchange import Family, Perms  # noqa: E402


def compose(g, h):
    return [h[g[x]] for x in range(len(g))]


def group_model(group: Perms):
    if not isinstance(group, Perms):
        raise ValueError("group must be a permutation group")
    elements = rt.perm_closure(group.tolist())
    index = {tuple(g): i for i, g in enumerate(elements)}
    identity = index[tuple(range(group.n))]
    table = [[index[tuple(compose(a, b))] for b in elements] for a in elements]
    inverse = []
    for x in range(len(elements)):
        inverse.append(next(y for y in range(len(elements)) if table[x][y] == identity and table[y][x] == identity))
    return elements, index, identity, table, inverse


def selected_set(member, index, identity):
    selected = []
    for row in member:
        key = tuple(row)
        if key not in index:
            raise ValueError("connection row is not an element of G")
        x = index[key]
        if x == identity:
            raise ValueError("the identity is not allowed in a connection set")
        if x in selected:
            raise ValueError("connection rows must be distinct")
        selected.append(x)
    return frozenset(selected)


def graph(table, inverse, selected):
    n = len(table)
    return [[x != y and (table[inverse[x]][y] in selected or table[inverse[y]][x] in selected)
             for y in range(n)] for x in range(n)]


def distances(adjacency, start, blocked=None):
    n = len(adjacency)
    out = [-1] * n
    out[start] = 0
    queue = deque([start])
    while queue:
        x = queue.popleft()
        for y in range(n):
            if not adjacency[x][y] or out[y] >= 0:
                continue
            if blocked is not None and {x, y} == set(blocked):
                continue
            out[y] = out[x] + 1
            queue.append(y)
    return out


def connected(adjacency):
    return all(d >= 0 for d in distances(adjacency, 0))


def regular_of_degree(adjacency, degree):
    return all(sum(row) == degree for row in adjacency)


def girth(adjacency):
    cycles = []
    for x in range(len(adjacency)):
        for y in range(x + 1, len(adjacency)):
            if adjacency[x][y]:
                d = distances(adjacency, x, (x, y))[y]
                if d >= 0:
                    cycles.append(d + 1)
    return min(cycles, default=0)


def diameter(adjacency):
    value = 0
    for x in range(len(adjacency)):
        ds = distances(adjacency, x)
        if any(d < 0 for d in ds):
            return 0
        value = max(value, max(ds))
    return value


def preserves_graph(a, b, permutation):
    n = len(a)
    return all(a[x][y] == b[permutation[x]][permutation[y]] for x in range(n) for y in range(n))


def graph_isomorphic(a, b):
    return any(preserves_graph(a, b, p) for p in itertools.permutations(range(len(a))))


def aut_order(adjacency):
    return sum(preserves_graph(adjacency, adjacency, p) for p in itertools.permutations(range(len(adjacency))))


def group_automorphisms(table, identity):
    n = len(table)
    rest = [x for x in range(n) if x != identity]
    out = []
    for image_rest in itertools.permutations(rest):
        phi = [0] * n
        phi[identity] = identity
        for x, y in zip(rest, image_rest):
            phi[x] = y
        if all(phi[table[x][y]] == table[phi[x]][phi[y]] for x in range(n) for y in range(n)):
            out.append(phi)
    return out


def inverse_connection_sets(inverse, identity, k):
    atoms = []
    seen = {identity}
    for x in range(len(inverse)):
        if x in seen:
            continue
        atom = frozenset({x, inverse[x]})
        seen.update(atom)
        atoms.append(atom)
    out = []
    for mask in range(1 << len(atoms)):
        selected = frozenset().union(*(atoms[i] for i in range(len(atoms)) if mask >> i & 1))
        if len(selected) == k:
            out.append(selected)
    return out


def is_ci_set(table, inverse, identity, selected):
    if any(inverse[x] not in selected for x in selected):
        return False
    source = graph(table, inverse, selected)
    automorphisms = group_automorphisms(table, identity)
    images = {frozenset(phi[x] for x in selected) for phi in automorphisms}
    for target in inverse_connection_sets(inverse, identity, len(selected)):
        if target not in images and graph_isomorphic(source, graph(table, inverse, target)):
            return False
    return True


def run(op: str, family: Family, reduction: str = "all", prefix: int | None = None, **args):
    op = op.removeprefix("cayley.")
    elements, index, identity, table, inverse = group_model(args["group"])
    members, p = rt.members(family)
    selected = [selected_set(member, index, identity) for member in members]
    if prefix is not None:
        members = members[:prefix]
        selected = selected[:prefix]
    graphs = [graph(table, inverse, s) for s in selected]

    if op == "connected":
        flags = [connected(g) for g in graphs]
    elif op == "is_regular_of_degree":
        flags = [regular_of_degree(g, args["degree"]) for g in graphs]
    elif op == "is_ci_set":
        flags = [is_ci_set(table, inverse, identity, s) for s in selected]
    elif op == "girth":
        return rt.reduce_int(reduction, [girth(g) for g in graphs], members, p)
    elif op == "diameter":
        return rt.reduce_int(reduction, [diameter(g) for g in graphs], members, p)
    elif op == "aut_order":
        return rt.reduce_int(reduction, [aut_order(g) for g in graphs], members, p)
    else:
        raise ValueError(f"unknown operation {op}")
    return rt.reduce_bool(reduction, flags, members, p, **args)
