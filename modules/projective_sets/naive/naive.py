"""Direct point-set computations over prime fields.

The shared runtime materialises the family. This module then rebuilds the ambient projective lines
or hyperplanes and computes every member independently.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
from lemmakernel import naive as rt  # noqa: E402
from lemmakernel.interchange import Family  # noqa: E402


def rank(rows: list[list[int]], p: int) -> int:
    if not rows:
        return 0
    a = [list(row) for row in rows]
    n = len(a[0])
    r = 0
    for c in range(n):
        pivot = next((i for i in range(r, len(a)) if a[i][c]), None)
        if pivot is None:
            continue
        a[r], a[pivot] = a[pivot], a[r]
        inv = pow(a[r][c], p - 2, p)
        a[r] = [(x * inv) % p for x in a[r]]
        for i in range(len(a)):
            if i != r and a[i][c]:
                factor = a[i][c]
                a[i] = [(x - factor * y) % p for x, y in zip(a[i], a[r])]
        r += 1
        if r == len(a):
            break
    return r


def projective_forms(p: int, d: int):
    """Normalised nonzero rows, one for each hyperplane equation."""
    for lead in range(d):
        tail = d - lead - 1
        for digits in itertools.product(range(p), repeat=tail):
            yield [0] * lead + [1] + list(digits)


def projective_lines(p: int, d: int):
    """Rref bases of all two-dimensional subspaces."""
    for pivots in itertools.combinations(range(d), 2):
        free = [(i, c) for i in range(2) for c in range(d) if pivots[i] < c and c not in pivots]
        for digits in itertools.product(range(p), repeat=len(free)):
            line = [[0] * d for _ in range(2)]
            for i, c in enumerate(pivots):
                line[i][c] = 1
            for (i, c), value in zip(free, digits):
                line[i][c] = value
            yield line


def normalise(point, p):
    lead = next((x for x in point if x), None)
    if lead is None:
        return None
    inverse = pow(lead, p - 2, p)
    return [(x * inverse) % p for x in point]


def validate_dictionary(family, p):
    dictionary = rt.dictionary(family)
    if any(normalise(point, p) is None for point in dictionary):
        raise ValueError("projective point dictionary contains zero")
    if any(normalise(point, p) != point for point in dictionary):
        raise ValueError("projective point dictionary is not normalised")
    if len({tuple(point) for point in dictionary}) != len(dictionary):
        raise ValueError("projective point dictionary contains a duplicate")


def dot(a, b, p):
    return sum(x * y for x, y in zip(a, b)) % p


def hyperplane_intersections(points, forms, p):
    return [sum(dot(form, point, p) == 0 for point in points) for form in forms]


def line_intersections(points, lines, p):
    return [sum(rank(line + [point], p) == 2 for point in points) for line in lines]


def run(op: str, family: Family, reduction: str = "all", prefix: int | None = None, **args):
    op = op.removeprefix("projective_sets.")
    if family.kind != "subsets":
        raise ValueError("projective_sets operations are defined on subsets families only")

    p = rt.prime(family)
    validate_dictionary(family, p)
    points = list(itertools.islice(rt.iter_members(family), prefix))
    d = family.children[0].cols
    forms = list(projective_forms(p, d)) if op in ("is_arc", "is_blocking_set") else None
    lines = list(projective_lines(p, d)) if op in (
        "is_cap", "is_hyperoval", "is_ovoid", "max_collinear",
        "secant_count", "tangent_count", "passant_count",
    ) else None

    values = []
    for member in points:
        if op == "spanned_rank":
            value = rank(member, p)
        elif op in ("is_arc", "is_blocking_set"):
            intersections = hyperplane_intersections(member, forms, p)
            value = all(x < d for x in intersections) if op == "is_arc" else all(x != 0 for x in intersections)
        else:
            intersections = line_intersections(member, lines, p)
            if op == "is_cap":
                value = all(x <= 2 for x in intersections)
            elif op == "is_hyperoval":
                value = p == 2 and d == 3 and len(member) == 4 and all(x <= 2 for x in intersections)
            elif op == "is_ovoid":
                value = d == 4 and len(member) == p * p + 1 and all(x <= 2 for x in intersections)
            elif op == "max_collinear":
                value = max(intersections, default=0)
            elif op == "secant_count":
                value = intersections.count(2)
            elif op == "tangent_count":
                value = intersections.count(1)
            elif op == "passant_count":
                value = intersections.count(0)
            else:
                raise ValueError(f"unknown operation {op}")
        values.append(value)

    if op in ("is_arc", "is_cap", "is_blocking_set", "is_hyperoval", "is_ovoid"):
        return rt.reduce_bool(reduction, values, points, p, **args)
    return rt.reduce_int(reduction, values, points, p)
