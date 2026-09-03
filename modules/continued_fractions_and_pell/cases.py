"""Oracle and benchmark inputs for continued fractions, fundamental units and class numbers.

Oracle cases are small radicands where the Lean reference can walk every period, build every
convergent and count every reduced form in a few seconds: a plain sweep, a sweep that starts at
the perfect squares 0 and 1, and an explicit batch of the radicands whose behaviour is worth
naming (the ones with long periods, the ones where the negative Pell equation is solvable, and
a few perfect squares). Bench cases are sized so that the naive implementation still finishes.

The two ends of the 64-bit boundary are cases too: d = 277 has a fundamental unit that fits and
a fundamental Pell solution that does not, and d = 526 is the least radicand whose fundamental
unit itself is too wide. Both must be refused rather than truncated.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lemmakernel as lk  # noqa: E402
from tools.harness import Case  # noqa: E402

PERIOD_OPS = ["cf_period", "cf_period_max", "cf_period_sum", "negative_pell"]
VALUE_OPS = ["cf_expansion", "fundamental_unit", "pell_fundamental"]

# 94 and 46 have the longest periods below 100 (16 and 12); 2, 5, 10, 13, 29, 41 and 61 solve
# x^2 - d y^2 = -1; 0, 1, 4 and 49 are squares, with no period, no unit and no Pell solution.
NAMED = [0, 1, 2, 4, 5, 10, 13, 29, 41, 46, 49, 61, 94]


def numbers(ds):
    """A batch of 1 x 1 natural-number members, for an `explicit` family."""
    return lk.naturals([[[d]] for d in ds])


def cases(ctx, rng):
    out = []

    # The plain sweep: every period statistic and every reduction it allows, over radicands whose
    # periods are short enough for the reference and whose units stay well inside 64 bits.
    sweep = ctx.range(2, 40)
    for op in PERIOD_OPS:
        out.append(Case("radicands 2 to 40", sweep, op, {"limit": 4}))
    for op in VALUE_OPS:
        out.append(Case("expansions and units to 40", sweep, op))

    # From zero: 0, 1, 4 and 9 are perfect squares, so their expansion is one term, their period
    # is empty and they have neither a unit nor a Pell solution.
    squares = ctx.range(0, 12)
    for op, reds in [("cf_period", ["all", "histogram"]), ("cf_period_max", ["all", "min"]),
                     ("cf_period_sum", ["all", "sum"]), ("negative_pell", ["all", "count"]),
                     ("cf_expansion", None), ("fundamental_unit", None), ("pell_fundamental", None)]:
        out.append(Case("radicands from zero", squares, op, {"limit": 2}, reductions=reds))

    # An explicit batch rather than a range: the same operations on the radicands worth naming.
    named = ctx.explicit(numbers(NAMED))
    for op, reds in [("cf_period", ["all", "max"]), ("cf_period_max", ["all", "histogram"]),
                     ("cf_period_sum", ["all", "max"]), ("negative_pell", ["all", "hits"]),
                     ("cf_expansion", None), ("fundamental_unit", None), ("pell_fundamental", None)]:
        out.append(Case("named radicands", named, op, {"limit": 8}, reductions=reds))

    # Class numbers read the member as |D|: h(-n) for every n up to 60, so both the n that are not
    # discriminants (h = 0) and the imaginary quadratic orders of class number 1, 2, 3 and 4.
    out.append(Case("discriminants 1 to 61", ctx.range(1, 61), "class_number"))
    out.append(Case("discriminants 100 to 140", ctx.range(100, 140), "class_number",
                    reductions=["all", "histogram", "max"]))

    # Requests the runtime must refuse.
    out += [
        Case("words are not numbers", ctx.words(4, 3), "cf_period", reductions=["all"], oracle=False),
        Case("explicit pairs", ctx.explicit(lk.naturals([[[2, 3]]])), "cf_period",
             reductions=["all"], oracle=False),
        Case("residues are not numbers", ctx.explicit(lk.matrix(7, [[[3]]])), "cf_period",
             reductions=["all"], oracle=False),
        Case("expansion does not reduce", ctx.range(2, 8), "cf_expansion",
             reductions=["all"], oracle=False),
        Case("period does not count", ctx.range(2, 8), "cf_period", reductions=["all"], oracle=False),
        # d = 526 is the least radicand whose fundamental unit passes 2^64 - 1.
        Case("unit past 64 bits", ctx.range(526, 527), "fundamental_unit",
             reductions=["all"], oracle=False),
        # d = 277 has a unit that fits (8920484118 + 535979945 sqrt 277, of norm -1) and a Pell
        # solution, its square, that does not.
        Case("pell past 64 bits", ctx.range(277, 278), "pell_fundamental",
             reductions=["all"], oracle=False),
    ]

    # Benchmarks: requests that look like real use, sized so that naive finishes. The period of
    # sqrt(d) averages about 120 terms for d below 200000, so a sweep of that width is a few
    # hundred million integer operations either way.
    sweep = lambda: ctx.range(2, 200002)  # noqa: E731
    out += [
        Case("periods_to_200000", sweep, "cf_period",
             what="the period length of sqrt(d) for every d below 200000",
             bench="histogram", oracle=False),
        Case("negative_pell_to_200000", sweep, "negative_pell",
             what="for how many d below 200000 the equation x^2 - d y^2 = -1 is solvable",
             bench="count", oracle=False),
        Case("largest_partial_quotient", sweep, "cf_period_max",
             what="the largest partial quotient in any period of sqrt(d) for d below 200000",
             bench="max", oracle=False),
        Case("class_numbers_to_50000", lambda: ctx.range(3, 50003), "class_number",
             what="the class number of every imaginary quadratic order of discriminant above -50000",
             bench="histogram", oracle=False),
    ]
    # `fundamental_unit` and `pell_fundamental` are not benched: a request has to refuse the whole
    # family when one member's answer passes 2^64 - 1, so the widest sweep either can take is
    # d < 526, which is far too small to time against anything.
    return out


def invariants(ctx):
    """Cross-operation identities on radicands beyond what the Lean kernel can afford."""
    # 277 is the first radicand whose fundamental Pell solution passes 2^64 - 1, so it is where a
    # sweep that asks for every operation at once has to stop.
    lo, hi = 2, 277
    ds = list(range(lo, hi))
    sweep = ctx.range(lo, hi)

    periods = ctx.value("continued_fractions_and_pell.cf_period", sweep, "all").values
    negative = ctx.value("continued_fractions_and_pell.negative_pell", sweep, "all").values
    # x^2 - d y^2 = -1 is solvable exactly when d is not a square and the period is odd
    assert negative == [p % 2 for p in periods]
    # a perfect square has no period, and nothing else has an empty one
    squares = {d * d for d in range(1, hi) if d * d < hi}
    assert [d for d, p in zip(ds, periods) if p == 0] == [d for d in ds if d in squares]

    expansions = ctx.value("continued_fractions_and_pell.cf_expansion", sweep, "all")
    units = ctx.value("continued_fractions_and_pell.fundamental_unit", sweep, "all")
    pells = ctx.value("continued_fractions_and_pell.pell_fundamental", sweep, "all")
    maxima = ctx.value("continued_fractions_and_pell.cf_period_max", sweep, "all").values
    sums = ctx.value("continued_fractions_and_pell.cf_period_sum", sweep, "all").values
    for i, d in enumerate(ds):
        terms = expansions.member(i)
        # one period after a_0, ending at 2*a_0, and the statistics read off it
        assert len(terms) == periods[i] + 1
        assert terms[0] * terms[0] <= d < (terms[0] + 1) * (terms[0] + 1)
        assert periods[i] == 0 or terms[-1] == 2 * terms[0]
        assert maxima[i] == max(terms[1:], default=0) and sums[i] == sum(terms[1:])

        unit, pell = units.member(i), pells.member(i)
        assert (unit is None) == (d in squares) and (pell is None) == (d in squares)
        if unit is None:
            continue
        # the unit has the norm its flag claims, and the flag is the parity of the period
        sign, x, y = unit
        assert x * x - d * y * y == (-1 if sign else 1) and sign == negative[i]
        # the Pell solution solves the +1 equation, and is the unit or its square
        sign, X, Y = pell
        assert sign == 0 and X * X - d * Y * Y == 1
        assert (X, Y) == ((2 * x * x + 1, 2 * x * y) if negative[i] else (x, y))

    # class numbers: the nine imaginary quadratic fields of class number one are the discriminants
    # -3, -4, -7, -8, -11, -19, -43, -67, -163 (Baker, Heegner, Stark).
    h = ctx.value("continued_fractions_and_pell.class_number", ctx.range(1, 200), "all").values
    fundamental_ones = [3, 4, 7, 8, 11, 19, 43, 67, 163]
    assert all(h[n - 1] == 1 for n in fundamental_ones)
    # every n that is not 0 or 3 mod 4 is not a discriminant
    assert all(h[n - 1] == 0 for n in range(1, 200) if n % 4 in (1, 2))
    # and every discriminant has a class, so h >= 1 there
    assert all(h[n - 1] >= 1 for n in range(1, 200) if n % 4 in (0, 3))
