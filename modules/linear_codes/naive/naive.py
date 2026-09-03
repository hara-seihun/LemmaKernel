"""Plain Python implementation of linear codes from generator matrices.

Every member is reduced independently. Codewords use `itertools.product`, and covering radii and
coordinate automorphisms use exhaustive search. The generic backend keeps the same definitions but
uses incremental q-ary Gray code enumeration for codewords.
"""
from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
from lemmakernel import naive as rt  # noqa: E402
from lemmakernel.interchange import Basis, Family, WeightEnumerators  # noqa: E402

_spec = importlib.util.spec_from_file_location("gfp_naive", ROOT / "modules" / "gfp" / "naive" / "naive.py")
gfp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gfp)


def code_basis(rows, p):
    return [row for row in gfp.rref(rows, p)[0] if any(row)]


def linear_combination(coefficients, basis, n, p):
    return [sum(a * row[j] for a, row in zip(coefficients, basis)) % p for j in range(n)]


def codewords(rows, p):
    n = len(rows[0]) if rows else 0
    basis = code_basis(rows, p)
    return [linear_combination(a, basis, n, p) for a in itertools.product(range(p), repeat=len(basis))]


def weight(word):
    return sum(x != 0 for x in word)


def weight_enumerator(rows, p):
    n = len(rows[0]) if rows else 0
    coefficients = [0] * (n + 1)
    for word in codewords(rows, p):
        coefficients[weight(word)] += 1
    return coefficients


def minimum_distance(rows, p):
    weights = [weight(word) for word in codewords(rows, p) if any(word)]
    return min(weights, default=0)


def dual(rows, p):
    return gfp.nullspace(rows, p)


def is_self_dual(rows, p):
    return code_basis(rows, p) == code_basis(dual(rows, p), p)


def distance(left, right):
    return sum(x != y for x, y in zip(left, right))


def covering_radius(rows, p):
    n = len(rows[0]) if rows else 0
    code = codewords(rows, p)
    return max((min(distance(word, c) for c in code) for word in itertools.product(range(p), repeat=n)), default=0)


def is_mds(rows, p):
    n = len(rows[0]) if rows else 0
    k = len(code_basis(rows, p))
    return k > 0 and minimum_distance(rows, p) == n - k + 1


def aut_order(rows, p):
    n = len(rows[0]) if rows else 0
    basis = code_basis(rows, p)
    total = 0
    for perm in itertools.permutations(range(n)):
        image = [[row[j] for j in perm] for row in basis]
        total += code_basis(image, p) == basis
    return total


def _flat(rows):
    return [x for row in rows for x in row]


def run(op: str, family: Family, reduction: str = "all", prefix: int | None = None, **args):
    op = op.removeprefix("linear_codes.")
    members = list(itertools.islice(rt.iter_members(family), prefix))
    p = rt.prime(family)
    n = len(members[0][0]) if members else family.params.get("cols", 0)

    if op == "weight_enumerator":
        values = [weight_enumerator(m, p) for m in members]
        return rt.reduce_values(reduction, WeightEnumerators(len(values), n, [x for v in values for x in v]))
    if op == "dual":
        values = [dual(m, p) for m in members]
        offsets = [0]
        for value in values:
            offsets.append(offsets[-1] + len(value))
        return rt.reduce_values(reduction, Basis(p, len(values), n, offsets, [x for value in values for x in _flat(value)]))

    if op == "minimum_distance":
        values = [minimum_distance(m, p) for m in members]
        return rt.reduce_int(reduction, values, members, p)
    if op == "covering_radius":
        values = [covering_radius(m, p) for m in members]
        return rt.reduce_int(reduction, values, members, p)
    if op == "aut_order":
        values = [aut_order(m, p) for m in members]
        return rt.reduce_int(reduction, values, members, p)
    if op == "is_self_dual":
        values = [is_self_dual(m, p) for m in members]
        return rt.reduce_bool(reduction, values, members, p, **args)
    if op == "is_mds":
        values = [is_mds(m, p) for m in members]
        return rt.reduce_bool(reduction, values, members, p, **args)
    raise ValueError(f"unknown operation {op}")
