"""Readable polynomial algorithms for finite simple graphs.

A member is an edge list from a `subsets` family. Its dictionary rows are canonical pairs
``[u, v]`` with ``u < v < vertices``. Every operation materialises one signed coefficient vector
per member.
"""
from __future__ import annotations

import functools
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
from lemmakernel import naive as rt  # noqa: E402
from lemmakernel.interchange import Coefficients, Family  # noqa: E402


def _canonical(n, edges):
    return n, tuple(sorted((min(u, v), max(u, v)) for u, v in edges))


def _contract(n, edges, edge):
    u, v = edge

    def rename(x):
        y = u if x == v else x
        return y - int(y > v)

    return _canonical(n - 1, [(rename(a), rename(b)) for a, b in edges])


def _remove_endpoints(n, edges, edge):
    u, v = edge
    kept = [(a, b) for a, b in edges if u not in (a, b) and v not in (a, b)]

    def rename(x):
        return x - int(u < x) - int(v < x)

    return _canonical(n - 2, [(rename(a), rename(b)) for a, b in kept])


def _add(a, b, length, sign=1):
    return tuple((a[i] if i < len(a) else 0) + sign * (b[i] if i < len(b) else 0)
                 for i in range(length))


@functools.lru_cache(maxsize=None)
def _chromatic(state):
    n, edges = state
    if any(u == v for u, v in edges):
        return (0,) * (n + 1)
    if not edges:
        return tuple(int(i == n) for i in range(n + 1))
    edge, rest = edges[-1], edges[:-1]
    return _add(_chromatic((n, rest)), _chromatic(_contract(n, rest, edge)), n + 1, -1)


@functools.lru_cache(maxsize=None)
def _matching(state):
    n, edges = state
    if not edges:
        return tuple(int(i == n) for i in range(n + 1))
    edge, rest = edges[-1], edges[:-1]
    return _add(_matching((n, rest)), _matching(_remove_endpoints(n, rest, edge)), n + 1, -1)


def _reachable(edges, source, target):
    seen = {source}
    todo = [source]
    for u in todo:
        for a, b in edges:
            if a == u and b not in seen:
                seen.add(b)
                todo.append(b)
            elif b == u and a not in seen:
                seen.add(a)
                todo.append(a)
    return target in seen


@functools.lru_cache(maxsize=None)
def _tutte(state):
    n, edges = state
    if not edges:
        return {(0, 0): 1}
    edge, rest = edges[-1], edges[:-1]
    u, v = edge
    if u == v:
        child = _tutte((n, rest))
        return {(i, j + 1): c for (i, j), c in child.items()}
    contracted = _contract(n, rest, edge)
    if not _reachable(rest, u, v):
        child = _tutte(contracted)
        return {(i + 1, j): c for (i, j), c in child.items()}
    out = dict(_tutte((n, rest)))
    for degree, coefficient in _tutte(contracted).items():
        out[degree] = out.get(degree, 0) + coefficient
    return out


def _characteristic(n, edges):
    adjacency = [[False] * n for _ in range(n)]
    for u, v in edges:
        adjacency[u][v] = adjacency[v][u] = True
    dp = {0: (1,)}
    for mask_size in range(n):
        for mask in [m for m in list(dp) if m.bit_count() == mask_size]:
            poly = dp.pop(mask)
            row = mask_size
            for col in range(n):
                if mask >> col & 1:
                    continue
                inversions = sum(mask >> k & 1 for k in range(col + 1, n))
                sign = -1 if inversions % 2 else 1
                if row == col:
                    term = (0,) + tuple(sign * c for c in poly)
                elif adjacency[row][col]:
                    term = tuple(-sign * c for c in poly)
                else:
                    continue
                next_mask = mask | 1 << col
                dp[next_mask] = _add(dp.get(next_mask, ()), term, max(len(dp.get(next_mask, ())), len(term)))
    return dp.get((1 << n) - 1, (0,) * (n + 1)) + (0,) * max(0, n + 1 - len(dp.get((1 << n) - 1, ())))


def _validate(family: Family, vertices: int):
    if family.kind != "subsets":
        raise ValueError("graph polynomial operations are defined on subsets families only")
    dictionary = rt.dictionary(family)
    if any(len(edge) != 2 or not (edge[0] < edge[1] < vertices) for edge in dictionary):
        raise ValueError("edge dictionary rows must satisfy u < v < vertices")
    if len({tuple(edge) for edge in dictionary}) != len(dictionary):
        raise ValueError("edge dictionary contains a duplicate edge")


def run(op: str, family: Family, reduction: str = "all", prefix: int | None = None, **args):
    op = op.removeprefix("graph_polynomials.")
    if reduction != "all":
        raise ValueError("coefficient vectors only reduce with `all`")
    vertices = int(args["vertices"])
    _validate(family, vertices)
    edge_count = family.params["k"]
    members = list(itertools.islice(rt.iter_members(family), prefix))
    vectors = []
    for member in members:
        state = _canonical(vertices, map(tuple, member))
        if op == "chromatic":
            values = _chromatic(state)
        elif op == "matching":
            values = _matching(state)
        elif op == "tutte":
            terms = _tutte(state)
            values = tuple(terms.get((i, j), 0)
                           for i in range(vertices + 1) for j in range(edge_count + 1))
        elif op == "characteristic":
            values = _characteristic(vertices, state[1])
        else:
            raise ValueError(f"unknown operation {op}")
        vectors.extend(values)
    length = (vertices + 1) * (edge_count + 1) if op == "tutte" else vertices + 1
    return Coefficients(len(members), length, vectors)
