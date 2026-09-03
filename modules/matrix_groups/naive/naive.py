"""Plain Python baseline for matrix_groups.

A member is a list of flattened square generators. The implementation materializes each member,
closes groups and vector orbits with Python sets, and performs every finite-field elimination from
scratch. It is deliberately direct so the benchmark has a readable baseline.
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
from lemmakernel.interchange import Family  # noqa: E402

_spec = importlib.util.spec_from_file_location("gfp_naive", ROOT / "modules" / "gfp" / "naive" / "naive.py")
gfp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gfp)


def unflatten(v, n):
    return [list(v[i * n:(i + 1) * n]) for i in range(n)]


def decode(member, n):
    return [unflatten(row, n) for row in member]


def matrix_key(a):
    return tuple(tuple(row) for row in a)


def identity(n):
    return [[int(i == j) for j in range(n)] for i in range(n)]


def group_elements(gens, p):
    one = identity(len(gens[0]))
    seen = {matrix_key(one)}
    queue = [one]
    for a in queue:
        for gen in gens:
            b = gfp.matmul(a, gen, p)
            key = matrix_key(b)
            if key not in seen:
                seen.add(key)
                queue.append(b)
    return queue


def group_order(gens, p):
    return len(group_elements(gens, p))


def projective_vectors(p, n):
    return [w[0] for w in rt.grassmannian_members(p, n, 1)]


def orbit_span(v, gens, p):
    seen = {tuple(v)}
    queue = [v]
    for w in queue:
        for a in gens:
            image = gfp.matmul([w], a, p)[0]
            key = tuple(image)
            if key not in seen:
                seen.add(key)
                queue.append(image)
    return queue


def is_irreducible(gens, p):
    n = len(gens[0])
    return all(gfp.rank(orbit_span(v, gens, p), p) == n for v in projective_vectors(p, n))


def centralizer_rows(gens, p):
    n = len(gens[0])
    rows = []
    for a in gens:
        for i in range(n):
            for j in range(n):
                row = []
                for r in range(n):
                    for c in range(n):
                        left = a[c][j] if r == i else 0
                        right = a[i][r] if c == j else 0
                        row.append((left - right) % p)
                rows.append(row)
    return rows


def is_absolutely_irreducible(gens, p):
    n = len(gens[0])
    return is_irreducible(gens, p) and n * n - gfp.rank(centralizer_rows(gens, p), p) == 1


def invariant_form_rows(gens, p):
    n = len(gens[0])
    rows = []
    for a in gens:
        for i in range(n):
            for j in range(n):
                row = []
                for r in range(n):
                    for c in range(n):
                        row.append((a[i][r] * a[j][c] - int(i == r and j == c)) % p)
                rows.append(row)
    return rows


def linear_combination(basis, coeffs, p):
    return [sum(c * v[j] for c, v in zip(coeffs, basis)) % p for j in range(len(basis[0]))]


def preserves_form(gens, p):
    n = len(gens[0])
    basis = gfp.nullspace(invariant_form_rows(gens, p), p)
    for coeffs in itertools.product(range(p), repeat=len(basis)):
        if not any(coeffs):
            continue
        form = unflatten(linear_combination(basis, coeffs, p), n)
        if gfp.rank(form, p) == n:
            return True
    return False


def subspace_orbit(w, gens, p):
    seen = {matrix_key(w)}
    queue = [w]
    for u in queue:
        for a in gens:
            image = gfp.rref(gfp.matmul(u, a, p), p)[0]
            key = matrix_key(image)
            if key not in seen:
                seen.add(key)
                queue.append(image)
    return queue


def is_imprimitive(gens, p):
    n = len(gens[0])
    if not is_irreducible(gens, p):
        return False
    for d in range(1, n):
        if n % d:
            continue
        for w in rt.grassmannian_members(p, n, d):
            orbit = subspace_orbit(w, gens, p)
            if len(orbit) == n // d and gfp.rank([row for u in orbit for row in u], p) == n:
                return True
    return False


def run(op: str, family: Family, reduction: str = "all", prefix: int | None = None, **args):
    op = op.removeprefix("matrix_groups.")
    if family.kind != "subsets":
        raise ValueError("matrix_groups operations are defined on subsets families only")
    dictionary = rt.dictionary(family)
    n = math.isqrt(len(dictionary[0]))
    if n == 0 or n * n != len(dictionary[0]):
        raise ValueError("generator rows must have positive perfect square length n*n")
    p = rt.prime(family)
    for i, row in enumerate(dictionary):
        if len(row) != n * n or gfp.rank(unflatten(row, n), p) != n:
            raise ValueError(f"generator dictionary row {i} is not an invertible n x n matrix")

    members = list(itertools.islice(rt.iter_members(family), prefix))
    groups = [decode(member, n) for member in members]
    if op == "order":
        return rt.reduce_int(reduction, [group_order(gens, p) for gens in groups], members, p)

    predicates = {
        "is_irreducible": is_irreducible,
        "is_absolutely_irreducible": is_absolutely_irreducible,
        "preserves_form": preserves_form,
        "is_imprimitive": is_imprimitive,
    }
    if op not in predicates:
        raise ValueError(f"unknown operation {op}")
    flags = [predicates[op](gens, p) for gens in groups]
    return rt.reduce_bool(reduction, flags, members, p, **args)
