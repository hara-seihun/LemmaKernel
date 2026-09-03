"""Direct Python implementation of sign-matrix predicates and canonicalisation.

Every member is materialised. Bit zero denotes +1 and bit one denotes -1. Canonicalisation tries
every row order and every first column, normalises the first row and column to +1, sorts columns,
and keeps the least row-major bit matrix.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))
from lemmakernel import naive as rt  # noqa: E402
from lemmakernel.interchange import Family, Matrix  # noqa: E402


def _square(m):
    return all(len(row) == len(m) for row in m)


def is_hadamard(m):
    if not _square(m):
        return False
    n = len(m)
    return all(sum(a != b for a, b in zip(m[i], m[j])) * 2 == n
               for i in range(n) for j in range(i))


def is_skew(m):
    if not _square(m):
        return False
    n = len(m)
    return all(m[i][i] == 0 for i in range(n)) and all(
        m[i][j] != m[j][i] for i in range(n) for j in range(i))


def is_regular(m):
    if not _square(m):
        return False
    n = len(m)
    target = sum(m[0]) if n else 0
    return (all(sum(row) == target for row in m) and
            all(sum(m[i][j] for i in range(n)) == target for j in range(n)))


def is_conference(m):
    if not _square(m) or len(m) < 2:
        return False
    n = len(m)
    if any(m[i][i] != 0 for i in range(n)):
        return False
    return all(sum(m[i][k] != m[j][k] for k in range(n) if k not in (i, j)) * 2 == n - 2
               for i in range(n) for j in range(i))


def canonical_form(m):
    rows = len(m)
    cols = len(m[0]) if rows else 0
    if rows == 0 or cols == 0:
        return [list(row) for row in m]
    best = None
    for row_order in itertools.permutations(range(rows)):
        first_row = row_order[0]
        for first_col in range(cols):
            columns = sorted([
                [m[r][c] ^ m[r][first_col] ^ m[first_row][c] ^ m[first_row][first_col]
                 for r in row_order]
                for c in range(cols)
            ])
            candidate = [[columns[c][r] for c in range(cols)] for r in range(rows)]
            if best is None or candidate < best:
                best = candidate
    return best


def _flat(mats):
    return [x for m in mats for row in m for x in row]


def run(op: str, family: Family, reduction: str = "all", prefix: int | None = None, **args):
    op = op.removeprefix("hadamard.")
    ms = list(itertools.islice(rt.iter_members(family), prefix))
    p = rt.prime(family)
    if p != 2:
        raise ValueError("hadamard operations accept F_2 matrix families only")

    if op == "canonical_form":
        forms = [canonical_form(m) for m in ms]
        rows = len(ms[0]) if ms else family.params.get("rows", 0)
        cols = len(ms[0][0]) if ms and rows else family.params.get("cols", 0)
        return rt.reduce_values(reduction, Matrix(2, len(forms), rows, cols, _flat(forms)))

    predicates = {
        "is_hadamard": is_hadamard,
        "is_skew": is_skew,
        "is_regular": is_regular,
        "is_conference": is_conference,
    }
    try:
        flags = [predicates[op](m) for m in ms]
    except KeyError as error:
        raise ValueError(f"unknown operation {op}") from error
    return rt.reduce_bool(reduction, flags, ms, p, **args)
