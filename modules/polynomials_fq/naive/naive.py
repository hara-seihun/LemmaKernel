"""The obvious implementation of the polynomials_fq module, in plain Python.

A member is one row of d entries over F_p, read as the monic polynomial x^d + sum a_i x^i with
a_i the entry at index i. Inside this file a polynomial is the full ascending coefficient list
with no trailing zero, so [] is 0 and [1] is 1.

Everything is done the obvious way: polynomials are factorised by trying every monic divisor of
degree 1, then 2, and so on; roots are found by evaluating at every element of the field; the
order of x is found by multiplying by x until the answer is 1. Nothing is shared between members.

This is what the fast backends are tested against, byte for byte on the interchange encoding, and
what the benchmark reports the speed-up against. Keep it readable before keeping it quick.

    naive.run(op, family, reduction, **args) -> interchange object
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))
from lemmakernel import naive as rt  # noqa: E402  (families and reductions, shared by every module)
from lemmakernel.interchange import Degrees, Elements, Family, Matrix  # noqa: E402

WALK_LIMIT = 1 << 32  # `order` and `is_primitive` walk the powers of x; the ring must be small


# ---- polynomial arithmetic ----------------------------------------------------------------------

def trim(f: list[int]) -> list[int]:
    while f and f[-1] == 0:
        f = f[:-1]
    return f


def divmod_poly(a, b, p):
    """Quotient and remainder of a by a nonzero b."""
    a, b = trim(a), trim(b)
    if not b:
        raise ZeroDivisionError("division by the zero polynomial")
    q = [0] * max(0, len(a) - len(b) + 1)
    r = list(a)
    binv = pow(b[-1], p - 2, p)
    while len(r) >= len(b):
        c = r[-1] * binv % p
        k = len(r) - len(b)
        q[k] = c
        for i, y in enumerate(b):
            r[k + i] = (r[k + i] - c * y) % p
        r = trim(r)
    return trim(q), r


def pmod(a, b, p):
    return divmod_poly(a, b, p)[1]


def pdiv(a, b, p):
    return divmod_poly(a, b, p)[0]


def monicise(f, p):
    if not f:
        return []
    c = pow(f[-1], p - 2, p)
    return [x * c % p for x in f]


def pgcd(a, b, p):
    """The monic greatest common divisor."""
    a, b = trim(a), trim(b)
    while b:
        a, b = b, pmod(a, b, p)
    return monicise(a, p)


def peval(f, x, p):
    acc = 0
    for c in reversed(f):
        acc = (c + x * acc) % p
    return acc


# ---- the operations, one member at a time --------------------------------------------------------

def monics(p, d):
    """Every monic polynomial of degree d."""
    for t in itertools.product(range(p), repeat=d):
        yield list(t) + [1]


def first_factor(f, p):
    """A monic divisor of least degree, 1 <= degree <= deg f / 2, or None."""
    for d in range(1, len(f) // 2 + 1):
        for g in monics(p, d):
            if not pmod(f, g, p):
                return g
    return None


def factor_degrees(f, p):
    """Degrees of the monic irreducible factors with multiplicity, non-decreasing."""
    out = []
    while len(f) > 1:
        g = first_factor(f, p)
        if g is None:  # nothing of half the degree or less divides it, so it is irreducible
            out.append(len(f) - 1)
            break
        out.append(len(g) - 1)
        f = pdiv(f, g, p)
    return sorted(out)


def irreducible(f, p):
    return len(f) >= 2 and first_factor(f, p) is None


def roots(f, p):
    return [x for x in range(p) if peval(f, x, p) == 0]


def strip_x(f):
    """Drop the largest power of x dividing f."""
    i = 0
    while i < len(f) and f[i] == 0:
        i += 1
    return f[i:]


def order(f, p):
    """The least e >= 1 with g | x^e - 1, where f = x^h g and g(0) != 0."""
    g = strip_x(f)
    one = pmod([1], g, p)
    cur, e = pmod([0, 1], g, p), 1
    while cur != one:
        cur = pmod([0] + cur if cur else [], g, p)
        e += 1
    return e


def primitive(f, p):
    return irreducible(f, p) and f[0] != 0 and order(f, p) == p ** (len(f) - 1) - 1


# ---- operations and reductions -------------------------------------------------------------------

def ragged(lists):
    offsets = [0]
    for xs in lists:
        offsets.append(offsets[-1] + len(xs))
    return offsets, [x for xs in lists for x in xs]


def run(op: str, family: Family, reduction: str = "all", prefix: int | None = None, **args):
    """`prefix`: answer for the first `prefix` members only (the benchmark's timing sample)."""
    op = op.removeprefix("polynomials_fq.")
    p = rt.prime(family)
    if p < 2 or p >= 1 << 32:
        raise ValueError("polynomials_fq needs a prime field with p < 2^32")
    ms = list(itertools.islice(rt.iter_members(family), prefix))
    if any(len(m) != 1 for m in ms):
        raise ValueError("polynomials_fq needs one row per member: one polynomial each")
    size = len(ms)
    d = len(ms[0][0]) if ms else family.params.get("cols", 0)
    # a member row is the monic polynomial of degree d whose lower coefficients it lists
    fs = [list(m[0]) + [1] for m in ms]

    if op in ("order", "is_primitive") and p ** d >= WALK_LIMIT:
        raise ValueError(f"{op} walks the powers of x, and the residue ring is too large")

    if op == "factorisation_degrees":
        offsets, values = ragged([factor_degrees(f, p) for f in fs])
        return rt.reduce_values(reduction, Degrees(size, offsets, values))
    if op == "roots":
        offsets, values = ragged([roots(f, p) for f in fs])
        return rt.reduce_values(reduction, Elements(p, size, offsets, values))
    if op == "gcd":
        other: Matrix = args["other"]
        g = list(other.member(0)[0]) + [1]
        offsets, values = ragged([pgcd(f, g, p)[:-1] for f in fs])
        return rt.reduce_values(reduction, Elements(p, size, offsets, values))

    if op == "order":
        return rt.reduce_int(reduction, [order(f, p) for f in fs], ms, p)
    if op == "root_count":
        return rt.reduce_int(reduction, [len(roots(f, p)) for f in fs], ms, p)

    if op == "is_irreducible":
        flags = [irreducible(f, p) for f in fs]
    elif op == "is_primitive":
        flags = [primitive(f, p) for f in fs]
    else:
        raise ValueError(f"unknown operation {op}")
    return rt.reduce_bool(reduction, flags, ms, p, **args)
