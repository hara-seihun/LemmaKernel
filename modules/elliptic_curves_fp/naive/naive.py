"""The obvious implementation of the elliptic_curves_fp module, in plain Python.

Every member (a, b) is taken on its own: the curve y^2 = x^3 + a*x + b over F_p is walked point
by point, its isomorphism class is listed one u at a time, and the order of a point is found by
adding the point to itself until it reaches the point at infinity. Nothing is tabulated, nothing
is shared between members.

This is what the fast backend is tested against, byte for byte on the interchange encoding, and
what the benchmark reports the speed-up against.

    naive.run(op, family, reduction, **args) -> interchange object
"""
from __future__ import annotations

import itertools
import sys
from math import gcd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))
from lemmakernel import naive as rt  # noqa: E402  (families and reductions, shared by every module)
from lemmakernel.interchange import CurveGroups, Family  # noqa: E402

OPS = ("point_count", "nonsingular", "supersingular", "j_invariant", "is_canonical", "class_size",
       "group_structure")


# ---- one curve ----------------------------------------------------------------------------------

def rhs(x, a, b, p):
    """x^3 + a*x + b in F_p."""
    return (x * x % p * x + a * x + b) % p


def root_count(v, p):
    """How many y in F_p have y^2 = v, by Euler's criterion."""
    if v == 0:
        return 1
    return 2 if pow(v, (p - 1) // 2, p) == 1 else 0


def point_count(a, b, p):
    """#E(F_p): the affine solutions plus the point at infinity."""
    return 1 + sum(root_count(rhs(x, a, b, p), p) for x in range(p))


def singular_form(a, b, p):
    """4*a^3 + 27*b^2 in F_p; zero exactly on the singular pairs."""
    return (4 * (a * a % p * a % p) + 27 * (b * b % p)) % p


def j_invariant(a, b, p):
    """1728*4*a^3 / (4*a^3 + 27*b^2), and p for a singular pair."""
    d = singular_form(a, b, p)
    if d == 0:
        return p
    return 6912 % p * (a * a % p * a % p) % p * pow(d, p - 2, p) % p


def iso_orbit(a, b, p):
    """(u^4*a, u^6*b) for u = 1, ..., p-1, repeats kept."""
    out = []
    for u in range(1, p):
        u2 = u * u % p
        u4 = u2 * u2 % p
        out.append((u4 * a % p, u4 * u2 % p * b % p))
    return out


def affine_points(a, b, p):
    """The affine points of the curve, ordered by x then y."""
    return [(x, y) for x in range(p) for y in range(p) if y * y % p == rhs(x, a, b, p)]


def add_points(P, Q, a, p):
    """The chord-and-tangent law; None is the point at infinity."""
    if P is None:
        return Q
    if Q is None:
        return P
    (x1, y1), (x2, y2) = P, Q
    if x1 == x2 and (y1 + y2) % p == 0:
        return None
    if x1 == x2:
        l = (3 * (x1 * x1 % p) + a) % p * pow(2 * y1 % p, p - 2, p) % p
    else:
        l = (y2 - y1) % p * pow((x2 - x1) % p, p - 2, p) % p
    x3 = (l * l - x1 - x2) % p
    return (x3, (l * (x1 - x3) - y1) % p)


def point_order(P, a, p):
    """The least m >= 1 with m*P = O, by repeated addition."""
    m, cur = 1, P
    while cur is not None:
        cur = add_points(cur, P, a, p)
        m += 1
    return m


def group_structure(a, b, p):
    """(n1, n2) with n1 | n2 and E(F_p) = Z/n1 x Z/n2; (0, 0) for a singular pair.

    The exponent of a finite abelian group is the order of one of its elements, so the least
    common multiple of the point orders is the larger invariant factor."""
    if singular_form(a, b, p) == 0:
        return (0, 0)
    points = affine_points(a, b, p)
    exponent = 1
    for P in points:
        order = point_order(P, a, p)
        exponent = exponent * order // gcd(exponent, order)
    return ((len(points) + 1) // exponent, exponent)


# ---- operations and reductions -------------------------------------------------------------------

def curves(family: Family, prefix: int | None):
    """The members of the family as (a, b) pairs, with the rows they came from."""
    members = list(itertools.islice(rt.iter_members(family), prefix))
    for m in members:
        if len(m) != 1 or len(m[0]) != 2:
            raise ValueError(f"elliptic_curves_fp needs members that are 1 x 2 matrices (a, b), got {len(m)} x {len(m[0])}")
    return members, [(m[0][0], m[0][1]) for m in members]


def run(op: str, family: Family, reduction: str = "all", prefix: int | None = None, **args):
    """`prefix`: answer for the first `prefix` members only (the benchmark's timing sample)."""
    op = op.removeprefix("elliptic_curves_fp.")
    if op not in OPS:
        raise ValueError(f"unknown operation {op}")
    p = rt.prime(family)
    if p <= 3 or p >= 1 << 32:
        raise ValueError(f"elliptic_curves_fp needs a prime p > 3 (short Weierstrass form), got p = {p}")
    members, pairs = curves(family, prefix)

    if op == "group_structure":
        if reduction != "all":
            raise ValueError("group_structure values only reduce with `all`")
        return CurveGroups([group_structure(a, b, p) for a, b in pairs])

    if op == "point_count":
        return rt.reduce_int(reduction, [point_count(a, b, p) for a, b in pairs], members, p)
    if op == "j_invariant":
        return rt.reduce_int(reduction, [j_invariant(a, b, p) for a, b in pairs], members, p)
    if op == "class_size":
        return rt.reduce_int(reduction, [len(set(iso_orbit(a, b, p))) for a, b in pairs], members, p)

    if op == "nonsingular":
        flags = [singular_form(a, b, p) != 0 for a, b in pairs]
    elif op == "supersingular":
        flags = [singular_form(a, b, p) != 0 and point_count(a, b, p) == p + 1 for a, b in pairs]
    else:
        flags = [all((a, b) <= q for q in iso_orbit(a, b, p)) for a, b in pairs]
    return rt.reduce_bool(reduction, flags, members, p, **args)
