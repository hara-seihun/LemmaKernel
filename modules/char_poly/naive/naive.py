"""Obvious per-matrix implementation of the char_poly module."""
from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
from lemmakernel import naive as rt  # noqa: E402
from lemmakernel.interchange import Family, Matrix  # noqa: E402

_spec = importlib.util.spec_from_file_location("gfp_naive", ROOT / "modules" / "gfp" / "naive" / "naive.py")
gfp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gfp)


def trim(f, p=None):
    f = list(f)
    if p is not None:
        f = [x % p for x in f]
    while f and f[-1] == 0:
        f.pop()
    return f


def coeff(f, i):
    return f[i] if i < len(f) else 0


def poly_add(f, g, p):
    return trim([(coeff(f, i) + coeff(g, i)) % p for i in range(max(len(f), len(g)))])


def poly_neg(f, p):
    return [(-x) % p for x in f]


def poly_sub(f, g, p):
    return poly_add(f, poly_neg(g, p), p)


def poly_mul(f, g, p):
    if not f or not g:
        return []
    out = [0] * (len(f) + len(g) - 1)
    for i, x in enumerate(f):
        for j, y in enumerate(g):
            out[i + j] = (out[i + j] + x * y) % p
    return trim(out)


def poly_pow(f, exponent, p):
    out = [1]
    for _ in range(exponent):
        out = poly_mul(out, f, p)
    return out


def monic(f, p):
    f = trim(f, p)
    if not f:
        return []
    inv = pow(f[-1], p - 2, p)
    return trim([(x * inv) % p for x in f])


def poly_divmod(f, g, p):
    g = monic(g, p)
    remainder = trim(f, p)
    quotient = []
    while g and len(remainder) >= len(g):
        shift = len(remainder) - len(g)
        term = [0] * shift + [remainder[-1]]
        quotient = poly_add(quotient, term, p)
        remainder = poly_sub(remainder, poly_mul(term, g, p), p)
    return quotient, remainder


def derivative(f, p):
    return trim([(i * f[i]) % p for i in range(1, len(f))])


def poly_gcd(f, g, p):
    f, g = trim(f, p), trim(g, p)
    while g:
        f, g = g, poly_divmod(f, g, p)[1]
    return monic(f, p)


def determinant_poly(matrix, p):
    n = len(matrix)
    if n == 0:
        return [1]
    total = []
    for j, entry in enumerate(matrix[0]):
        minor = [[x for k, x in enumerate(row) if k != j] for row in matrix[1:]]
        term = poly_mul(entry, determinant_poly(minor, p), p)
        total = poly_add(total, term, p) if j % 2 == 0 else poly_sub(total, term, p)
    return total


def characteristic_polynomial(a, p):
    n = len(a)
    xi_minus_a = [[([(-a[i][j]) % p, 1] if i == j else trim([(-a[i][j]) % p]))
                   for j in range(n)] for i in range(n)]
    return trim(determinant_poly(xi_minus_a, p))


def identity(n):
    return [[int(i == j) for j in range(n)] for i in range(n)]


def minimal_polynomial(a, p):
    n = len(a)
    powers = [identity(n)]
    for _ in range(n):
        powers.append(gfp.matmul(powers[-1], a, p))
    for degree in range(1, n + 1):
        previous = [[x for row in power for x in row] for power in powers[:degree]]
        target = [x for row in powers[degree] for x in row]
        equations = [[v[i] for v in previous] for i in range(n * n)]
        relation = gfp.solve(equations, [(-x) % p for x in target], p)
        if relation is not None:
            return relation + [1]
    raise AssertionError("Cayley-Hamilton relation was not found")


def irreducible_factors(f, p):
    def factor(g):
        degree = len(g) - 1
        for d in range(1, degree // 2 + 1):
            for cs in itertools.product(range(p), repeat=d):
                candidate = list(cs) + [1]
                quotient, remainder = poly_divmod(g, candidate, p)
                if not remainder:
                    return factor(candidate) + factor(quotient)
        return [monic(g, p)]
    return sorted(factor(monic(f, p)))


def zero_matrix(n):
    return [[0] * n for _ in range(n)]


def matrix_add(a, b, p):
    return [[(x + y) % p for x, y in zip(r, s)] for r, s in zip(a, b)]


def scalar_identity(n, c, p):
    return [[c % p if i == j else 0 for j in range(n)] for i in range(n)]


def evaluate_at_matrix(f, a, p):
    value = zero_matrix(len(a))
    for c in reversed(f):
        value = matrix_add(gfp.matmul(value, a, p), scalar_identity(len(a), c, p), p)
    return value


def matrix_power(a, exponent, p):
    out = identity(len(a))
    for _ in range(exponent):
        out = gfp.matmul(out, a, p)
    return out


def block_exponents(a, f, multiplicity, p):
    b = evaluate_at_matrix(f, a, p)
    degree = len(f) - 1

    def at_least(k):
        if k == 0 or k > multiplicity:
            return 0
        now = len(a) - gfp.rank(matrix_power(b, k, p), p)
        before = len(a) - gfp.rank(matrix_power(b, k - 1, p), p)
        return (now - before) // degree

    return [k for k in range(1, multiplicity + 1) for _ in range(at_least(k) - at_least(k + 1))]


def invariant_factors(a, p):
    factors = irreducible_factors(characteristic_polynomial(a, p), p)
    unique = []
    for f in factors:
        if f not in unique:
            unique.append(f)
    data = [(f, block_exponents(a, f, factors.count(f), p)) for f in unique]
    count = max((len(exponents) for _, exponents in data), default=0)
    out = []
    for j in range(count):
        q = [1]
        for f, exponents in data:
            offset = count - len(exponents)
            if j >= offset:
                q = poly_mul(q, poly_pow(f, exponents[j - offset], p), p)
        out.append(q)
    return out


def companion(f, p):
    n = len(f) - 1
    out = [[0] * n for _ in range(n)]
    for i in range(n - 1):
        out[i][i + 1] = 1
    out[-1] = [(-f[j]) % p for j in range(n)]
    return out


def block_sum(a, b):
    ca = len(a[0]) if a else 0
    cb = len(b[0]) if b else 0
    return [row + [0] * cb for row in a] + [[0] * ca + row for row in b]


def rational_canonical_form(a, p):
    out = []
    for f in invariant_factors(a, p):
        out = block_sum(out, companion(f, p))
    return out


def conjugacy_label(a, p):
    n = len(a)
    factors = invariant_factors(a, p)
    rows = [f + [0] * (n + 1 - len(f)) for f in factors]
    return rows + [[0] * (n + 1) for _ in range(n - len(rows))]


def element_order(a, p):
    n = len(a)
    if gfp.rank(a, p) != n:
        return 0
    power = identity(n)
    for order in range(1, p ** n + 1):
        power = gfp.matmul(power, a, p)
        if power == identity(n):
            return order
    raise AssertionError("invertible matrix order exceeded p^n")


def _flat(matrices):
    return [x for m in matrices for row in m for x in row]


def run(op: str, family: Family, reduction: str = "all", prefix: int | None = None, **args):
    op = op.removeprefix("char_poly.")
    matrices = list(itertools.islice(rt.iter_members(family), prefix))
    p = rt.prime(family)
    if p in (0, rt.NATURALS) or not matrices or any(len(a) != len(a[0]) for a in matrices):
        raise ValueError("char_poly operations need nonempty square matrices over a prime field")
    n = len(matrices[0])

    if op in ("charpoly", "minpoly", "rational_canonical_form", "conjugacy_label"):
        if reduction != "all":
            raise ValueError(f"{op} values only reduce with `all`")
        if op == "charpoly":
            polynomials = [characteristic_polynomial(a, p) for a in matrices]
            values = [[f + [0] * (n + 1 - len(f))] for f in polynomials]
            return Matrix(p, len(values), 1, n + 1, _flat(values))
        if op == "minpoly":
            polynomials = [minimal_polynomial(a, p) for a in matrices]
            values = [[f + [0] * (n + 1 - len(f))] for f in polynomials]
            return Matrix(p, len(values), 1, n + 1, _flat(values))
        values = ([rational_canonical_form(a, p) for a in matrices] if op == "rational_canonical_form"
                  else [conjugacy_label(a, p) for a in matrices])
        rows, cols = (n, n) if op == "rational_canonical_form" else (n, n + 1)
        return Matrix(p, len(values), rows, cols, _flat(values))

    minpolys = [minimal_polynomial(a, p) for a in matrices]
    if op == "is_regular":
        return rt.reduce_bool(reduction, [len(f) == n + 1 for f in minpolys], matrices, p, **args)
    if op == "is_semisimple":
        return rt.reduce_bool(reduction, [poly_gcd(f, derivative(f, p), p) == [1] for f in minpolys], matrices, p, **args)
    if op == "element_order":
        return rt.reduce_int(reduction, [element_order(a, p) for a in matrices], matrices, p)
    raise ValueError(f"unknown operation {op}")
