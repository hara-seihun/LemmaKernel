"""Plain Python baseline for symmetric forms over odd prime fields.

Each member is diagonalized independently by symmetric congruence. The implementation then reads
rank, discriminant square class, type, Witt index, exact isometry, and isotropic-point count from
the diagonal form. Radicals use the canonical gfp nullspace basis.
"""
from __future__ import annotations

import importlib.util
import itertools
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
from lemmakernel import naive as rt  # noqa: E402
from lemmakernel.interchange import Basis, Family  # noqa: E402

_spec = importlib.util.spec_from_file_location("gfp_naive", ROOT / "modules" / "gfp" / "naive" / "naive.py")
gfp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gfp)

SUPPORTED_FAMILIES = {"explicit", "symmetric_matrices"}


def is_prime(p: int) -> bool:
    return p >= 2 and all(p % d for d in range(2, math.isqrt(p) + 1))


def square_symmetric(matrix: list[list[int]]) -> bool:
    n = len(matrix)
    return all(len(row) == n for row in matrix) and all(
        matrix[i][j] == matrix[j][i] for i in range(n) for j in range(n)
    )


def swap_coordinates(matrix: list[list[int]], i: int, j: int) -> None:
    matrix[i], matrix[j] = matrix[j], matrix[i]
    for row in matrix:
        row[i], row[j] = row[j], row[i]


def add_coordinate(matrix: list[list[int]], i: int, j: int, p: int) -> None:
    matrix[i] = [(x + y) % p for x, y in zip(matrix[i], matrix[j])]
    for row in matrix:
        row[i] = (row[i] + row[j]) % p


def diagonal_entries(matrix: list[list[int]], p: int) -> list[int]:
    """Nonzero diagonal entries after symmetric congruence elimination."""
    a = [list(row) for row in matrix]
    n = len(a)
    diagonal = []
    for k in range(n):
        pivot = next((i for i in range(k, n) if a[i][i]), None)
        if pivot is None:
            pair = next(((i, j) for i in range(k, n) for j in range(i + 1, n) if a[i][j]), None)
            if pair is None:
                break
            i, j = pair
            add_coordinate(a, i, j, p)
            pivot = i
        swap_coordinates(a, k, pivot)
        d = a[k][k]
        diagonal.append(d)
        inverse = pow(d, p - 2, p)
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                a[i][j] = (a[i][j] - a[i][k] * inverse * a[k][j]) % p
        for i in range(k + 1, n):
            a[i][k] = 0
            a[k][i] = 0
    return diagonal


def discriminant(diagonal: list[int], p: int) -> int:
    return math.prod(diagonal) % p


def square_sign(a: int, p: int) -> int:
    return pow(a, (p - 1) // 2, p)


def is_hyperbolic(diagonal: list[int], p: int) -> bool:
    r = len(diagonal)
    signed_discriminant = discriminant(diagonal, p)
    if (r // 2) % 2:
        signed_discriminant = -signed_discriminant % p
    return square_sign(signed_discriminant, p) == 1


def form_type(matrix: list[list[int]], p: int) -> int:
    diagonal = diagonal_entries(matrix, p)
    if len(diagonal) % 2:
        return 2
    return 0 if is_hyperbolic(diagonal, p) else 1


def witt_index(matrix: list[list[int]], p: int) -> int:
    diagonal = diagonal_entries(matrix, p)
    half = len(diagonal) // 2
    if len(diagonal) % 2 or is_hyperbolic(diagonal, p):
        return half
    return half - 1


def is_isometric(left: list[list[int]], right: list[list[int]], p: int) -> bool:
    a = diagonal_entries(left, p)
    b = diagonal_entries(right, p)
    return len(a) == len(b) and square_sign(discriminant(a, p), p) == square_sign(discriminant(b, p), p)


def projective_point_count(p: int, n: int) -> int:
    return sum(p**i for i in range(n))


def isotropic_point_count(matrix: list[list[int]], p: int) -> int:
    n = len(matrix)
    diagonal = diagonal_entries(matrix, p)
    r = len(diagonal)
    if r == 0:
        return projective_point_count(p, n)
    base = projective_point_count(p, n - 1)
    if r % 2:
        return base
    correction = p ** (n - r + r // 2 - 1)
    return base + correction if is_hyperbolic(diagonal, p) else base - correction


def _flat(rows):
    return [x for row in rows for x in row]


def run(op: str, family: Family, reduction: str = "all", prefix: int | None = None, **args):
    op = op.removeprefix("quadratic_forms.")
    if family.kind not in SUPPORTED_FAMILIES:
        raise ValueError("quadratic_forms operations need explicit or symmetric_matrices families")
    p = rt.prime(family)
    if p >= 2**32 or p % 2 == 0 or not is_prime(p):
        raise ValueError("quadratic_forms requires an odd prime below 2^32")

    members = list(itertools.islice(rt.iter_members(family), prefix))
    if any(not square_symmetric(member) for member in members):
        rows = family.params.get("rows", 0)
        cols = family.params.get("cols", 0)
        if rows != cols:
            raise ValueError("quadratic_forms requires square members")
        raise ValueError("quadratic_forms requires symmetric members")

    if op == "radical":
        values = [gfp.nullspace(member, p) for member in members]
        offsets = [0]
        for value in values:
            offsets.append(offsets[-1] + len(value))
        n = len(members[0]) if members else family.params.get("cols", family.params.get("n", 0))
        return rt.reduce_values(reduction, Basis(p, len(values), n, offsets, [x for value in values for x in _flat(value)]))

    if op == "is_isometric":
        other_object = args["other"]
        other = rt.vectors_of(other_object)
        n = len(members[0]) if members else family.params.get("cols", family.params.get("n", 0))
        if other_object.p != p or len(other) != n or not square_symmetric(other):
            raise ValueError("other must be one symmetric matrix over the same field with the same n x n shape")
        flags = [is_isometric(member, other, p) for member in members]
        return rt.reduce_bool(reduction, flags, members, p, **args)

    operations = {
        "form_type": form_type,
        "rank": lambda matrix, field: len(diagonal_entries(matrix, field)),
        "witt_index": witt_index,
        "isotropic_point_count": isotropic_point_count,
    }
    if op not in operations:
        raise ValueError(f"unknown operation {op}")
    values = [operations[op](member, p) for member in members]
    return rt.reduce_int(reduction, values, members, p)
