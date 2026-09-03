"""The obvious implementation of the continued_fractions_and_pell module, in plain Python.

The shared runtime materialises the family; every member is then one natural number answered from
scratch. The continued fraction of sqrt(d) is walked term by term with the textbook (m, q)
recurrence, the convergents are accumulated with Python's unbounded integers, and class numbers
come from a double loop over reduced forms. Nothing is precomputed and nothing is shared between
members.

This is what the fast backend is tested against, byte for byte on the interchange encoding, and
what the benchmark reports the speed-up against. Keep it readable before keeping it quick.

    naive.run(op, family, reduction, **args) -> interchange object

`family` is a lemmakernel.interchange.Family (what a family handle exports).
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))
from lemmakernel import naive as rt  # noqa: E402  (families and reductions, shared by every module)
from lemmakernel.interchange import NATURALS, Expansions, Family, QuadraticUnits  # noqa: E402

U64 = 1 << 64


# ---- the mathematics, one member at a time ------------------------------------------------------

def isqrt(n: int) -> int:
    """floor(sqrt(n)): the largest x with x*x <= n, by Newton's method from x = n."""
    if n == 0:
        return 0
    x = n
    while True:
        y = (x + n // x) // 2
        if y >= x:
            return x
        x = y


def period_terms(d: int) -> list[int]:
    """One period a_1 .. a_L of the continued fraction of sqrt(d); empty for a perfect square.

    State (m, q) is the complete quotient (m + sqrt d)/q, whose integer part is (a_0 + m)//q. The
    period ends at the first k >= 1 with q_k = 1, where a_k = 2*a_0."""
    a0 = isqrt(d)
    if a0 * a0 == d:
        return []
    m, q, out = a0, d - a0 * a0, []
    while True:
        a = (a0 + m) // q
        out.append(a)
        if q == 1:
            return out
        m = q * a - m
        q = (d - m * m) // q


def expansion_terms(d: int) -> list[int]:
    """a_0 followed by exactly one period."""
    return [isqrt(d)] + period_terms(d)


def convergent(terms: list[int]) -> tuple[int, int]:
    """(p, q) of the finite continued fraction `terms`."""
    p, q, pp, qq = 1, 0, 0, 1
    for a in terms:
        p, q, pp, qq = a * p + pp, a * q + qq, p, q
    return p, q


def unit(d: int):
    """The fundamental unit of Z[sqrt d] as (norm is -1, x, y), or None for a perfect square.

    It is the convergent p_{L-1}/q_{L-1}, which solves x^2 - d y^2 = (-1)^L."""
    period = period_terms(d)
    if not period:
        return None
    x, y = convergent([isqrt(d)] + period[:-1])
    return (1 if len(period) % 2 else 0, x, y)


def pell(d: int):
    """The fundamental solution of x^2 - d y^2 = 1: the unit, squared when its norm is -1."""
    u = unit(d)
    if u is None:
        return None
    negative, x, y = u
    return (0, 2 * x * x + 1, 2 * x * y) if negative else (0, x, y)


def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def class_number(n: int) -> int:
    """h(-n): the primitive reduced positive definite forms (a, b, c) of discriminant -n.

    A reduced form has -a < b <= a <= c, with b >= 0 when a = c; that pairs (a, b, c) with
    (a, -b, c), so an interior form counts twice. n = 4ac - b^2 >= 3a^2 bounds a."""
    if n == 0 or n % 4 not in (0, 3):
        return 0
    total = 0
    for a in range(1, isqrt(n // 3) + 1):
        for b in range(a + 1):
            t = b * b + n
            if t % (4 * a):
                continue
            c = t // (4 * a)
            if c < a or gcd(gcd(a, b), c) != 1:
                continue
            total += 1 if (b == 0 or b == a or c == a) else 2
    return total


# ---- operations and reductions -------------------------------------------------------------------

def _numbers(family: Family, prefix):
    """The natural number each member carries, in canonical order, with the members themselves."""
    if family.kind not in ("range", "explicit") or rt.prime(family) != NATURALS:
        raise ValueError("continued_fractions_and_pell needs 1 x 1 natural-number members")
    ms = list(itertools.islice(rt.iter_members(family), prefix))
    if any(len(m) != 1 or len(m[0]) != 1 for m in ms):
        raise ValueError("continued_fractions_and_pell needs 1 x 1 natural-number members")
    return [m[0][0] for m in ms], ms


def _units(op: str, values) -> QuadraticUnits:
    """The unit kind, refusing the whole request rather than truncating any member."""
    for u in values:
        if u is not None and (u[1] >= U64 or u[2] >= U64):
            raise ValueError(f"{op}: a member's answer exceeds 2^64 - 1")
    pairs = [x for u in values for x in ((u[1], u[2]) if u else (0, 0))]
    return QuadraticUnits(len(values), [int(u is not None) for u in values],
                          [int(bool(u and u[0])) for u in values], pairs)


def run(op: str, family: Family, reduction: str = "all", prefix: int | None = None, **args):
    """`prefix`: answer for the first `prefix` members only (the benchmark's timing sample)."""
    op = op.removeprefix("continued_fractions_and_pell.")
    ns, ms = _numbers(family, prefix)

    if op == "cf_expansion":
        terms = [expansion_terms(d) for d in ns]
        offsets = [0]
        for t in terms:
            offsets.append(offsets[-1] + len(t))
        return rt.reduce_values(reduction, Expansions(len(ns), offsets, [a for t in terms for a in t]))
    if op == "fundamental_unit":
        return rt.reduce_values(reduction, _units(op, [unit(d) for d in ns]))
    if op == "pell_fundamental":
        return rt.reduce_values(reduction, _units(op, [pell(d) for d in ns]))
    if op == "negative_pell":
        flags = [len(period_terms(d)) % 2 == 1 for d in ns]
        return rt.reduce_bool(reduction, flags, ms, NATURALS, **args)

    if op == "cf_period":
        values = [len(period_terms(d)) for d in ns]
    elif op == "cf_period_max":
        values = [max(period_terms(d), default=0) for d in ns]
    elif op == "cf_period_sum":
        values = [sum(period_terms(d)) for d in ns]
    elif op == "class_number":
        values = [class_number(n) for n in ns]
    else:
        raise ValueError(f"unknown operation {op}")
    return rt.reduce_int(reduction, values, ms, NATURALS)
