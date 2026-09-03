"""Oracle and benchmark inputs for arithmetic in Z/n.

Oracle cases are small moduli where the Lean reference can walk every power and every candidate
square root in a few seconds; they cover a prime, a composite whose unit group is not cyclic, an
odd prime power, and the degenerate modulus 1. Bench cases are sized so that the naive
implementation still finishes, or nearly.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lemmakernel as lk  # noqa: E402
from tools.harness import Case  # noqa: E402

RESIDUE_OPS = ["multiplicative_order", "is_primitive_root", "is_quadratic_residue",
               "discrete_log", "legendre", "jacobi"]


def cases(ctx, rng):
    out = []

    # A prime modulus, every operation, every reduction the operation allows: 13 residues, four
    # of which (2, 6, 7, 11) are primitive roots.
    every = ctx.range(0, 13)
    for op in RESIDUE_OPS:
        out.append(Case("prime modulus 13", every, op, {"modulus": 13, "base": 2, "limit": 3}))

    # A composite modulus with no primitive root: nothing is a primitive root, so `count` is zero
    # and `first` finds nothing. The residues run past the modulus and wrap.
    wrapped = ctx.range(1, 22)
    for op, reds in [("multiplicative_order", ["all", "histogram"]),
                     ("is_primitive_root", ["count", "first"]),
                     ("is_quadratic_residue", ["all", "hits"]),
                     ("discrete_log", ["all", "max"]),
                     ("jacobi", ["all", "histogram"])]:
        out.append(Case("composite modulus 21", wrapped, op, {"modulus": 21, "base": 2, "limit": 4},
                        reductions=reds))

    # An odd prime power, whose unit group is cyclic but not of prime order, on residues that all
    # exceed the modulus.
    high = ctx.range(20, 41)
    for op, reds in [("multiplicative_order", ["all", "max"]),
                     ("is_primitive_root", ["all", "hits"]),
                     ("is_quadratic_residue", ["count", "first"]),
                     ("discrete_log", ["all", "min"]),
                     ("jacobi", ["all", "sum"])]:
        out.append(Case("prime power modulus 27", high, op, {"modulus": 27, "base": 2, "limit": 3},
                        reductions=reds))

    # Z/1: one element, which is its own unit and its own square root.
    trivial = ctx.range(0, 3)
    for op, reds in [("multiplicative_order", ["all"]), ("is_primitive_root", ["all"]),
                     ("is_quadratic_residue", ["all"]), ("discrete_log", ["all"]),
                     ("jacobi", ["all"])]:
        out.append(Case("modulus one", trivial, op, {"modulus": 1, "base": 0}, reductions=reds))

    # A base that is not a unit: the powers of 6 modulo 21 are eventually periodic rather than
    # cyclic, so most residues have no logarithm at all.
    out.append(Case("non-unit base modulo 21", ctx.range(0, 21), "discrete_log",
                    {"modulus": 21, "base": 6}, reductions=["all", "histogram"]))
    out.append(Case("non-unit base modulo 16", ctx.range(0, 16), "discrete_log",
                    {"modulus": 16, "base": 4}, reductions=["all", "sum"]))

    # The moduli sweep: 8 and 12 have no primitive root, the rest do.
    out += [Case("moduli 1 to 14", ctx.range(1, 15), "least_primitive_root")]

    # Requests the runtime must refuse.
    out += [
        Case("not a range", ctx.explicit(lk.matrix(2, [[[1, 0]]])), "multiplicative_order",
             {"modulus": 5}, reductions=["all"], oracle=False),
        Case("modulus zero", ctx.range(0, 3), "multiplicative_order", {"modulus": 0},
             reductions=["all"], oracle=False),
        Case("modulus too large", ctx.range(0, 3), "multiplicative_order", {"modulus": 1 << 32},
             reductions=["all"], oracle=False),
        Case("even legendre modulus", ctx.range(0, 3), "legendre", {"modulus": 8},
             reductions=["all"], oracle=False),
        Case("composite legendre modulus", ctx.range(0, 3), "legendre", {"modulus": 9},
             reductions=["all"], oracle=False),
        Case("even jacobi modulus", ctx.range(0, 3), "jacobi", {"modulus": 10},
             reductions=["all"], oracle=False),
        Case("moduli from zero", ctx.range(0, 4), "least_primitive_root",
             reductions=["all"], oracle=False),
    ]

    # Benchmarks: requests that look like real use, sized so that naive finishes and the ratio
    # shows. 10007 is prime, so its unit group is cyclic of order 10006 = 2 * 5003.
    units = ctx.range(1, 10007)
    out += [
        Case("orders_mod_10007", units, "multiplicative_order", {"modulus": 10007},
             what="the multiplicative order of every unit modulo 10007",
             bench="histogram", oracle=False),
        Case("primitive_roots_mod_10007", units, "is_primitive_root", {"modulus": 10007},
             what="how many units modulo 10007 generate the whole unit group",
             bench="count", oracle=False),
        Case("discrete_logs_mod_10007", units, "discrete_log", {"base": 5, "modulus": 10007},
             what="the largest discrete logarithm to base 5 modulo 10007, over every unit",
             bench="max", oracle=False),
        Case("jacobi_255255", lambda: ctx.range(0, 100000), "jacobi", {"modulus": 255255},
             what="the Jacobi symbol (a/255255) for the first hundred thousand residues",
             bench="histogram", oracle=False),
        Case("least_primitive_roots", ctx.range(100, 1000), "least_primitive_root",
             what="the least primitive root of every modulus from 100 to 999, and the largest of them",
             bench="max", oracle=False),
    ]
    return out


def invariants(ctx):
    """Cross-operation identities on a prime modulus beyond the kernel oracle."""
    p = 1009
    units = ctx.range(1, p)
    orders = ctx.value("residues.multiplicative_order", units, "histogram", modulus=p)
    assert sum(orders.bins) == p - 1                      # every residue below p is a unit
    # a primitive root is exactly a unit of full order
    primitive = ctx.value("residues.is_primitive_root", units, "count", modulus=p)
    assert orders.bins[p - 1] == primitive.value
    # modulo a prime the Jacobi symbol is the Legendre symbol
    assert (ctx.value("residues.jacobi", units, "all", modulus=p).values ==
            ctx.value("residues.legendre", units, "all", modulus=p).values)
    # half the units are squares, and the symbol says which
    assert ctx.value("residues.legendre", units, "histogram", modulus=p).bins == [0, (p - 1) // 2, (p - 1) // 2]
    assert ctx.value("residues.is_quadratic_residue", units, "count", modulus=p).value == (p - 1) // 2
    # the powers of a primitive root run through every unit exactly once
    g = ctx.value("residues.least_primitive_root", ctx.range(p, p + 1), "all").values[0]
    assert ctx.value("residues.discrete_log", units, "histogram", base=g, modulus=p).bins == [1] * (p - 1)
