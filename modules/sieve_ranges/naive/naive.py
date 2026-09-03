"""The obvious implementation of the sieve_ranges module, in plain Python.

Every member is factored on its own by trial division up to its square root, and every arithmetic
function is read off that factorisation. Nothing is shared between members, which is exactly what
the segmented sieve in the backend exists to avoid, so this is also the benchmark baseline.

    naive.run(op, family, reduction, **args) -> interchange object

`family` is a lemmakernel.interchange.Family (what a family handle exports).
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))
from lemmakernel import naive as rt  # noqa: E402  (families and reductions, shared by every module)
from lemmakernel.interchange import NATURALS, Factorisation, Family  # noqa: E402


# ---- arithmetic ---------------------------------------------------------------------------------

def factorise(n: int) -> list[tuple[int, int]]:
    """(prime, exponent) pairs with primes increasing. 0 and 1 have none."""
    out = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            e = 0
            while n % d == 0:
                n //= d
                e += 1
            out.append((d, e))
        d += 1
    if n > 1:
        out.append((n, 1))
    return out


def omega(f):
    return len(f)


def big_omega(f):
    return sum(e for _, e in f)


def totient(n, f):
    if n == 0:
        return 0
    value = 1
    for p, e in f:
        value *= p ** (e - 1) * (p - 1)
    return value


def sigma(n, f):
    if n == 0:
        return 0
    value = 1
    for p, e in f:
        value *= (p ** (e + 1) - 1) // (p - 1)
    return value


def divisor_count(n, f):
    if n == 0:
        return 0
    value = 1
    for _, e in f:
        value *= e + 1
    return value


def is_squarefree(n, f):
    return n != 0 and all(e == 1 for _, e in f)


def is_prime(n, f):
    return f == [(n, 1)]


def mobius(n, f):
    """mu(n) + 1: 0 for mu = -1, 1 for mu = 0, 2 for mu = +1."""
    if not is_squarefree(n, f):
        return 1
    return 2 if len(f) % 2 == 0 else 0


def largest_prime_factor(f):
    return max((p for p, _ in f), default=0)


# ---- operations and reductions -------------------------------------------------------------------

def numbers(family: Family, prefix: int | None):
    """The members as (matrices, numbers); fails on families that are not single naturals."""
    ms = list(itertools.islice(rt.iter_members(family), prefix))
    if rt.prime(family) != NATURALS:
        raise ValueError("sieve_ranges operations need natural numbers, not elements of a field")
    if any(len(m) != 1 or len(m[0]) != 1 for m in ms):
        raise ValueError("sieve_ranges operations need 1 x 1 members")
    return ms, [m[0][0] for m in ms]


def run(op: str, family: Family, reduction: str = "all", prefix: int | None = None, **args):
    """`prefix`: answer for the first `prefix` members only (the benchmark's timing sample)."""
    op = op.removeprefix("sieve_ranges.")
    ms, ns = numbers(family, prefix)
    fs = [factorise(n) for n in ns]

    if op == "factorisation":
        offsets = [0]
        for f in fs:
            offsets.append(offsets[-1] + len(f))
        pairs = [x for f in fs for pe in f for x in pe]
        return rt.reduce_values(reduction, Factorisation(len(ns), offsets, pairs))

    if op == "is_prime":
        flags = [is_prime(n, f) for n, f in zip(ns, fs)]
    elif op == "is_squarefree":
        flags = [is_squarefree(n, f) for n, f in zip(ns, fs)]
    else:
        if op == "omega":
            values = [omega(f) for f in fs]
        elif op == "big_omega":
            values = [big_omega(f) for f in fs]
        elif op == "totient":
            values = [totient(n, f) for n, f in zip(ns, fs)]
        elif op == "sigma":
            values = [sigma(n, f) for n, f in zip(ns, fs)]
        elif op == "divisor_count":
            values = [divisor_count(n, f) for n, f in zip(ns, fs)]
        elif op == "mobius":
            values = [mobius(n, f) for n, f in zip(ns, fs)]
        elif op == "largest_prime_factor":
            values = [largest_prime_factor(f) for f in fs]
        else:
            raise ValueError(f"unknown operation {op}")
        return rt.reduce_int(reduction, values, ms, NATURALS)

    return rt.reduce_bool(reduction, flags, ms, NATURALS, **args)
