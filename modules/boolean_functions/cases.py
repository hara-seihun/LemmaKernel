"""Oracle cases and benchmarks for Boolean truth-table operations."""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lemmakernel as lk  # noqa: E402
from tools.harness import Case, module, rotate  # noqa: E402

SCALAR_INTEGER_OPS = ["nonlinearity", "algebraic_degree"]
SCALAR_BOOLEAN_OPS = ["is_bent"]


def monomials(n, degrees):
    return [subset for degree in degrees for subset in itertools.combinations(range(n), degree)]


def evaluation_matrix(n, terms):
    return [[int(all((x >> variable) & 1 for variable in term)) for x in range(1 << n)] for term in terms]


def degree_at_most(ctx, n, degree):
    """Every scalar function of degree at most `degree`, in ANF coefficient order."""
    terms = monomials(n, range(min(n, degree) + 1))
    return ctx.transform(ctx.all_matrices(2, 1, len(terms)), lk.matrix(2, evaluation_matrix(n, terms)))


def quadratic_forms(ctx, n):
    """Every zero-constant quadratic form, including the diagonal terms x_i^2 = x_i."""
    terms = monomials(n, [1, 2])
    return ctx.transform(ctx.all_matrices(2, 1, len(terms)), lk.matrix(2, evaluation_matrix(n, terms)))


def gf8_multiply(a, b):
    product = 0
    while b:
        if b & 1:
            product ^= a
        carry = a & 4
        a = (a << 1) & 7
        if carry:
            a ^= 3
        b >>= 1
    return product


def gold_cube_table():
    values = [gf8_multiply(gf8_multiply(x, x), x) for x in range(8)]
    return [[(value >> bit) & 1 for value in values] for bit in range(3)]


def cases(ctx, rng):
    mod = module("boolean_functions")
    scalar_tables = []
    for function in (
        lambda x: 0,
        lambda x: x.bit_count() & 1,
        lambda x: ((x >> 0) & 1) & ((x >> 1) & 1),
        lambda x: ((x >> 0) & 1) & ((x >> 1) & 1) & ((x >> 2) & 1),
        lambda x: rng.randrange(2),
    ):
        scalar_tables.append([[function(x) for x in range(8)]])
    explicit_scalar = ctx.explicit(lk.matrix(2, scalar_tables))

    out = [
        Case("named scalar functions", explicit_scalar, "nonlinearity", reductions=None),
        Case("named scalar functions", explicit_scalar, "algebraic_degree", reductions=None),
        Case("named scalar functions", explicit_scalar, "walsh_spectrum"),
        Case("named scalar functions", explicit_scalar, "is_bent", {"limit": 3}, reductions=None),
        Case("named scalar functions", explicit_scalar, "affine_class"),
    ]

    affine = degree_at_most(ctx, 3, 1)
    quadratics = quadratic_forms(ctx, 2)
    for i, family in enumerate((affine, quadratics)):
        name = "all degree <= 1 functions in 3 variables" if i == 0 else "all quadratic forms in 2 variables"
        for j, op in enumerate(SCALAR_INTEGER_OPS + SCALAR_BOOLEAN_OPS):
            args = {"limit": 2}
            out.append(Case(name, family, op, args, reductions=rotate(mod, op, i + j)))
        out.append(Case(name, family, "walsh_spectrum"))
        out.append(Case(name, family, "affine_class"))

    vectorial = ctx.explicit(lk.matrix(2, [
        gold_cube_table(),
        [[0] * 8 for _ in range(3)],
        [[(x >> bit) & 1 for x in range(8)] for bit in range(3)],
    ]))
    out.append(Case("vectorial functions on F_2^3", vectorial, "algebraic_degree", reductions=["all", "histogram"]))
    out.append(Case("vectorial functions on F_2^3", vectorial, "is_apn", {"limit": 2}, reductions=None))

    out += [
        Case("ternary truth table", ctx.explicit(lk.matrix(3, [[[0, 1, 2]]])), "algebraic_degree", oracle=False),
        Case("three-value truth table", ctx.explicit(lk.matrix(2, [[[0, 1, 0]]])), "algebraic_degree", oracle=False),
        Case("vectorial scalar operation", ctx.explicit(lk.matrix(2, [[[0, 1, 0, 1], [0, 0, 1, 1]]])), "nonlinearity", oracle=False),
        Case("vectorial scalar operation", ctx.explicit(lk.matrix(2, [[[0, 1, 0, 1], [0, 0, 1, 1]]])), "walsh_spectrum", oracle=False),
        Case("vectorial scalar operation", ctx.explicit(lk.matrix(2, [[[0, 1, 0, 1], [0, 0, 1, 1]]])), "is_bent", oracle=False),
        Case("vectorial scalar operation", ctx.explicit(lk.matrix(2, [[[0, 1, 0, 1], [0, 0, 1, 1]]])), "affine_class", oracle=False),
        Case("spectrum with count", explicit_scalar, "walsh_spectrum", reductions=["count"], oracle=False),
        Case("affine class with count", explicit_scalar, "affine_class", reductions=["count"], oracle=False),
    ]

    out += [
        Case(
            "degree2_n5_nonlinearity",
            lambda: degree_at_most(ctx, 5, 2),
            "nonlinearity",
            bench="max",
            oracle=False,
            what="largest nonlinearity among all 65,536 Boolean functions of degree at most two in five variables",
        ),
        Case(
            "quadratic_n6_bent",
            lambda: quadratic_forms(ctx, 6),
            "is_bent",
            bench="count",
            oracle=False,
            what="bent functions among all 2,097,152 zero-constant quadratic forms in six variables",
        ),
    ]
    return out


def invariants(ctx):
    family = degree_at_most(ctx, 4, 2)
    degrees = ctx.value("boolean_functions.algebraic_degree", family).values
    assert max(degrees) == 2

    sample = ctx.explicit(lk.matrix(2, [
        [[((x >> 0) & 1) & ((x >> 1) & 1) for x in range(16)]],
        [[x.bit_count() & 1 for x in range(16)]],
    ]))
    spectra = ctx.value("boolean_functions.walsh_spectrum", sample).tolist()
    nonlinearities = ctx.value("boolean_functions.nonlinearity", sample).values
    assert nonlinearities == [(16 - max(map(abs, spectrum[0]))) // 2 for spectrum in spectra]

    vectorial = ctx.explicit(lk.matrix(2, [gold_cube_table(), [[0] * 8 for _ in range(3)]]))
    assert ctx.value("boolean_functions.is_apn", vectorial).values == [1, 0]

    base = [(((x >> 0) & 1) & ((x >> 1) & 1)) ^ ((x >> 2) & 1) for x in range(8)]
    transformed = []
    for x in range(8):
        swapped = ((x & 1) << 1) | ((x & 2) >> 1) | (x & 4)
        transformed.append(base[swapped ^ 5])
    equivalent = ctx.explicit(lk.matrix(2, [[base], [transformed]]))
    forms = ctx.value("boolean_functions.affine_class", equivalent).tolist()
    assert forms[0] == forms[1]
