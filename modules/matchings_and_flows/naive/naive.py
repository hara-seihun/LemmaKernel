"""Readable baseline for exact matching, spanning-tree, and flow counts.

Families are materialised by ``lemmakernel.naive``. The permanent tries every permutation,
spanning trees try every edge subset of size n-1, and maximum flow tries every source-side cut.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))
from lemmakernel import naive as rt  # noqa: E402
from lemmakernel.interchange import Family  # noqa: E402


def _square(matrix):
    return all(len(row) == len(matrix) for row in matrix)


def permanent(matrix):
    n = len(matrix)
    return sum(
        __import__("math").prod(matrix[i][permutation[i]] for i in range(n))
        for permutation in itertools.permutations(range(n))
    )


def _connected(n, edges):
    if n == 0:
        return False
    seen = {0}
    changed = True
    while changed:
        changed = False
        for u, v, _ in edges:
            if u in seen and v not in seen:
                seen.add(v)
                changed = True
            elif v in seen and u not in seen:
                seen.add(u)
                changed = True
    return len(seen) == n


def spanning_tree_count(matrix):
    n = len(matrix)
    if n == 0:
        return 0
    edges = [(i, j, matrix[i][j]) for i in range(n) for j in range(i + 1, n) if matrix[i][j]]
    total = 0
    for tree in itertools.combinations(edges, n - 1):
        if _connected(n, tree):
            total += __import__("math").prod(weight for _, _, weight in tree)
    return total


def max_flow(matrix, source, sink):
    n = len(matrix)
    best = None
    for mask in range(1 << n):
        if not (mask >> source) & 1 or (mask >> sink) & 1:
            continue
        capacity = sum(
            matrix[i][j]
            for i in range(n)
            if (mask >> i) & 1
            for j in range(n)
            if not (mask >> j) & 1
        )
        best = capacity if best is None else min(best, capacity)
    return best


def run(op: str, family: Family, reduction: str = "all", prefix: int | None = None, **args):
    op = op.removeprefix("matchings_and_flows.")
    matrices = list(itertools.islice(rt.iter_members(family), prefix))
    if not all(_square(matrix) for matrix in matrices):
        raise ValueError("matchings_and_flows operations require square matrices")

    if op == "perfect_matching_count":
        values = [permanent(matrix) for matrix in matrices]
    elif op == "spanning_tree_count":
        if not all(matrix[i][j] == matrix[j][i] for matrix in matrices for i in range(len(matrix)) for j in range(len(matrix))):
            raise ValueError("spanning_tree_count requires symmetric matrices")
        values = [spanning_tree_count(matrix) for matrix in matrices]
    elif op == "max_flow":
        source, sink = args["source"], args["sink"]
        n = len(matrices[0]) if matrices else family.params.get("rows", 0)
        if source == sink:
            raise ValueError("source and sink must be distinct")
        if source >= n or sink >= n:
            raise ValueError("source and sink must be vertex indices")
        values = [max_flow(matrix, source, sink) for matrix in matrices]
    else:
        raise ValueError(f"unknown operation {op}")
    return rt.reduce_int(reduction, values, matrices, rt.prime(family))
