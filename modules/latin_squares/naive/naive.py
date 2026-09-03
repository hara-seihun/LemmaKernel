"""Readable baseline for Latin-square operations.

The runtime enumerates families. This file computes each requested property independently and
uses the shared reductions in ``lemmakernel.naive``.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))
from lemmakernel import naive as rt  # noqa: E402
from lemmakernel.interchange import NATURALS, Family, Matrix  # noqa: E402


def is_latin(square):
    n = len(square)
    symbols = list(range(n))
    return n > 0 and all(len(row) == n and sorted(row) == symbols for row in square) and all(
        sorted(square[i][j] for i in range(n)) == symbols for j in range(n)
    )


def orthogonal(a, b):
    return len({(x, y) for ar, br in zip(a, b) for x, y in zip(ar, br)}) == len(a) ** 2


def has_orthogonal_mate(square):
    if not is_latin(square):
        return False
    n = len(square)
    rows = list(itertools.permutations(range(n)))
    columns = [set() for _ in range(n)]
    pairs = set()

    def search(r):
        if r == n:
            return True
        candidates = [tuple(range(n))] if r == 0 else rows
        for row in candidates:
            if any(row[c] in columns[c] or (square[r][c], row[c]) in pairs for c in range(n)):
                continue
            for c in range(n):
                columns[c].add(row[c])
                pairs.add((square[r][c], row[c]))
            if search(r + 1):
                return True
            for c in range(n):
                columns[c].remove(row[c])
                pairs.remove((square[r][c], row[c]))
        return False

    return search(0)


def transversal_count(square):
    n = len(square)
    return sum(
        len({square[i][columns[i]] for i in range(n)}) == n
        for columns in itertools.permutations(range(n))
    )


def is_group_table(square):
    if not is_latin(square):
        return False
    n = len(square)
    identity = next((e for e in range(n) if all(square[e][x] == x and square[x][e] == x for x in range(n))), None)
    return identity is not None and all(
        square[square[x][y]][z] == square[x][square[y][z]]
        for x in range(n) for y in range(n) for z in range(n)
    )


def normalise_symbols(square):
    labels = {}
    return [[labels.setdefault(x, len(labels)) for x in row] for row in square]


def isotopy_canonical_form(square):
    n = len(square)
    permutations = list(itertools.permutations(range(n)))
    best = None
    for rows in permutations:
        for columns in permutations:
            candidate = normalise_symbols([[square[i][j] for j in columns] for i in rows])
            if best is None or candidate < best:
                best = candidate
    return best or []


def _flat(squares):
    return [x for square in squares for row in square for x in row]


def run(op: str, family: Family, reduction: str = "all", prefix: int | None = None, **args):
    op = op.removeprefix("latin_squares.")
    members = list(itertools.islice(rt.iter_members(family), prefix))
    if any(not square or any(len(row) != len(square) for row in square) for square in members):
        raise ValueError("Latin-square operations need nonempty square arrays")
    n = len(members[0]) if members else family.params.get("n", 0)
    member_prime = rt.prime(family)

    if op == "isotopy_canonical_form":
        values = [isotopy_canonical_form(square) for square in members]
        return rt.reduce_values(reduction, Matrix(NATURALS, len(values), n, n, _flat(values)))
    if op == "is_latin":
        return rt.reduce_bool(reduction, [is_latin(square) for square in members], members, member_prime, **args)
    if op == "has_orthogonal_mate":
        return rt.reduce_bool(reduction, [has_orthogonal_mate(square) for square in members], members, member_prime, **args)
    if op == "transversal_count":
        return rt.reduce_int(reduction, [transversal_count(square) for square in members], members, member_prime)
    if op == "is_group_table":
        return rt.reduce_bool(reduction, [is_group_table(square) for square in members], members, member_prime, **args)
    raise ValueError(f"unknown operation {op}")
