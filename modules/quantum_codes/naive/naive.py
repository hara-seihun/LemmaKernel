"""Readable stabiliser-code operations on binary symplectic generator matrices.

A row is ``(x_0,...,x_{n-1} | z_0,...,z_{n-1})``. It corresponds to the additive
GF(4) word ``x + omega*z``. Families and reductions come from ``lemmakernel.naive``;
this file only computes one member's mathematical value.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))
from lemmakernel import naive as rt  # noqa: E402
from lemmakernel.interchange import Family  # noqa: E402


def rank(rows: list[list[int]]) -> int:
    """Binary row rank."""
    if not rows:
        return 0
    a = [list(row) for row in rows]
    r = 0
    for c in range(len(a[0])):
        pivot = next((i for i in range(r, len(a)) if a[i][c]), None)
        if pivot is None:
            continue
        a[r], a[pivot] = a[pivot], a[r]
        for i in range(len(a)):
            if i != r and a[i][c]:
                a[i] = [x ^ y for x, y in zip(a[i], a[r])]
        r += 1
        if r == len(a):
            break
    return r


def symplectic_inner(a: list[int], b: list[int]) -> int:
    n = len(a) // 2
    return sum(a[i] * b[n + i] + a[n + i] * b[i] for i in range(n)) % 2


def is_self_orthogonal(rows: list[list[int]]) -> bool:
    return all(symplectic_inner(a, b) == 0 for a in rows for b in rows)


def span(rows: list[list[int]], cols: int) -> set[tuple[int, ...]]:
    words = {(0,) * cols}
    for row in rows:
        words |= {tuple(x ^ y for x, y in zip(word, row)) for word in list(words)}
    return words


def symplectic_vector(word: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(a & 1 for a in word) + tuple(a >> 1 for a in word)


def distance(rows: list[list[int]]) -> int:
    """Least symplectic weight in C^perp_s outside C, or n+1 if there is none."""
    cols = len(rows[0]) if rows else 0
    n = cols // 2
    code = span(rows, cols)
    best = n + 1
    for word in itertools.product(range(4), repeat=n):
        weight = sum(a != 0 for a in word)
        if weight >= best:
            continue
        v = symplectic_vector(word)
        if v not in code and all(symplectic_inner(v, row) == 0 for row in rows):
            best = weight
    return best


def is_css(rows: list[list[int]]) -> bool:
    cols = len(rows[0]) if rows else 0
    n = cols // 2
    return rank([row[:n] for row in rows]) + rank([row[n:] for row in rows]) == rank(rows)


def run(op: str, family: Family, reduction: str = "all", prefix: int | None = None, **args):
    """Materialise the family, compute each member from scratch, then use the shared reduction."""
    op = op.removeprefix("quantum_codes.")
    ms = list(itertools.islice(rt.iter_members(family), prefix))
    p = rt.prime(family)
    cols = len(ms[0][0]) if ms and ms[0] else family.params.get("cols", family.params.get("n", 0))
    if p != 2:
        raise ValueError("quantum_codes is defined over F_2")
    if cols % 2:
        raise ValueError("quantum_codes needs an even number 2n of columns")

    if op == "is_self_orthogonal":
        return rt.reduce_bool(reduction, [is_self_orthogonal(m) for m in ms], ms, p, **args)
    if op == "is_css":
        return rt.reduce_bool(reduction, [is_css(m) for m in ms], ms, p, **args)
    if op == "distance":
        return rt.reduce_int(reduction, [distance(m) for m in ms], ms, p)
    raise ValueError(f"unknown operation {op}")
