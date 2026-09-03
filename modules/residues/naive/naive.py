"""The obvious implementation of the residues module, in plain Python.

The shared runtime materialises the family; every member is then answered from scratch. Orders
count up through the powers, quadratic residues look for a square root, discrete logarithms walk
the powers one at a time, and the Jacobi symbol is the product of Legendre symbols over the
prime factorisation. Nothing is precomputed, nothing is shared between members.

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
from lemmakernel.interchange import Family  # noqa: E402


# ---- arithmetic ---------------------------------------------------------------------------------

def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def totient(n: int) -> int:
    """The number of units of Z/n, counted."""
    return sum(1 for a in range(n) if gcd(a, n) == 1)


def order(n: int, a: int) -> int:
    """The least k >= 1 with a^k = 1 in Z/n; 0 when a is not a unit."""
    v, one = 1 % n, 1 % n
    for k in range(1, n + 1):
        v = v * a % n
        if v == one:
            return k
    return 0


def is_primitive_root(n: int, a: int) -> bool:
    return gcd(a, n) == 1 and order(n, a) == totient(n)


def is_quadratic_residue(n: int, a: int) -> bool:
    return gcd(a, n) == 1 and any(x * x % n == a % n for x in range(n))


def discrete_log(n: int, base: int, a: int) -> int:
    """The least x < n with base^x = a in Z/n; n when there is none."""
    v, target = 1 % n, a % n
    for x in range(n):
        if v == target:
            return x
        v = v * base % n
    return n


def factors(n: int) -> list[int]:
    """Prime factors with multiplicity, ascending."""
    out, d = [], 2
    while d * d <= n:
        while n % d == 0:
            out.append(d)
            n //= d
        d += 1
    if n > 1:
        out.append(n)
    return out


def legendre(p: int, a: int) -> int:
    """(a/p) for an odd prime p by Euler's criterion, encoded 0, 1, 2 with 2 for -1."""
    a %= p
    if a == 0:
        return 0
    return 1 if pow(a, (p - 1) // 2, p) == 1 else 2


def mul_symbol(x: int, y: int) -> int:
    if x == 0 or y == 0:
        return 0
    return 1 if x == y else 2


def jacobi(n: int, a: int) -> int:
    """(a/n) for odd n as the product of the Legendre symbols of the prime factors of n."""
    s = 1
    for p in factors(n):
        s = mul_symbol(s, legendre(p, a))
    return s


def least_primitive_root(n: int) -> int:
    """The least g >= 1 generating the units of Z/n; 0 when there is no primitive root."""
    phi = totient(n)
    for g in range(1, n + 1):
        if gcd(g, n) == 1 and order(n, g) == phi:
            return g
    return 0


# ---- operations and reductions -------------------------------------------------------------------

RESIDUE_OPS = ("multiplicative_order", "is_primitive_root", "is_quadratic_residue",
               "discrete_log", "legendre", "jacobi")


def run(op: str, family: Family, reduction: str = "all", prefix: int | None = None, **args):
    """`prefix`: answer for the first `prefix` members only (the benchmark's timing sample)."""
    op = op.removeprefix("residues.")
    if family.kind != "range":
        raise ValueError(f"{op} is defined on range families only")
    ms = list(itertools.islice(rt.iter_members(family), prefix))
    xs = [m[0][0] for m in ms]
    p = rt.prime(family)

    if op == "least_primitive_root":
        if family.params["a"] < 1:
            raise ValueError("least_primitive_root needs moduli of at least 1")
        return rt.reduce_int(reduction, [least_primitive_root(x) for x in xs], ms, p)

    if op not in RESIDUE_OPS:
        raise ValueError(f"unknown operation {op}")
    n = args["modulus"]
    if n < 1:
        raise ValueError("residues needs a modulus of at least 1")
    if n >= 1 << 32:
        raise ValueError("residues needs a modulus less than 2^32")

    if op == "multiplicative_order":
        return rt.reduce_int(reduction, [order(n, x) for x in xs], ms, p)
    if op == "discrete_log":
        base = args["base"] % n
        return rt.reduce_int(reduction, [discrete_log(n, base, x) for x in xs], ms, p)
    if op == "legendre":
        if n % 2 == 0 or factors(n) != [n]:
            raise ValueError("legendre needs an odd prime modulus")
        return rt.reduce_int(reduction, [legendre(n, x) for x in xs], ms, p)
    if op == "jacobi":
        if n % 2 == 0:
            raise ValueError("jacobi needs an odd modulus")
        return rt.reduce_int(reduction, [jacobi(n, x) for x in xs], ms, p)

    if op == "is_primitive_root":
        flags = [is_primitive_root(n, x) for x in xs]
    else:
        flags = [is_quadratic_residue(n, x) for x in xs]
    return rt.reduce_bool(reduction, flags, ms, p, **{k: v for k, v in args.items() if k == "limit"})
