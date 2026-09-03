"""Readable strongly regular graph tests, with one fresh O(v^3) scan per adjacency matrix."""
from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))
from lemmakernel import naive as rt  # noqa: E402
from lemmakernel.interchange import (  # noqa: E402
    Family,
    StronglyRegularParams,
    StronglyRegularSpectra,
)


def srg_params(a: list[list[int]]):
    v = len(a)
    if any(len(row) != v for row in a):
        return None
    if any(a[i][i] or any(a[i][j] != a[j][i] for j in range(v)) for i in range(v)):
        return None
    degrees = [sum(row) for row in a]
    k = degrees[0] if degrees else 0
    if not 0 < k < v - 1 or any(d != k for d in degrees):
        return None
    adjacent, nonadjacent = [], []
    for i in range(v):
        for j in range(i + 1, v):
            common = sum(a[i][x] and a[j][x] for x in range(v))
            (adjacent if a[i][j] else nonadjacent).append(common)
    lam, mu = adjacent[0], nonadjacent[0]
    if any(x != lam for x in adjacent) or any(x != mu for x in nonadjacent):
        return None
    return v, k, lam, mu


def spectrum_of(params):
    v, k, lam, mu = params
    delta = lam - mu
    discriminant = delta * delta + 4 * (k - mu)
    if discriminant <= 0:
        return None
    imbalance = 2 * k + (v - 1) * delta
    q = math.isqrt(discriminant)
    if q * q == discriminant:
        denominator = 2 * q
        numerator_plus = (v - 1) * q - imbalance
        numerator_minus = (v - 1) * q + imbalance
        if numerator_plus < 0 or numerator_minus < 0:
            return None
        if numerator_plus % denominator or numerator_minus % denominator:
            return None
        f, g = numerator_plus // denominator, numerator_minus // denominator
    elif imbalance == 0 and (v - 1) % 2 == 0:
        f = g = (v - 1) // 2
    else:
        return None
    return k, delta, discriminant, f, g


def _mul(x, y, discriminant):
    a, b = x
    c, d = y
    return a * c + b * d * discriminant, a * d + b * c


def _cube(x, discriminant):
    return _mul(_mul(x, x, discriminant), x, discriminant)


def _nonnegative(x, discriminant):
    a, b = x
    if a >= 0:
        return b >= 0 or a * a >= b * b * discriminant
    return b > 0 and b * b * discriminant >= a * a


def _krein_one(params, sign):
    v, k, lam, mu = params
    delta = lam - mu
    discriminant = delta * delta + 4 * (k - mu)
    c = (v - k - 1) ** 2
    kk = k * k
    x3 = _cube((delta, sign), discriminant)
    x_plus_two3 = _cube((delta + 2, sign), discriminant)
    value = (c * (8 * kk + x3[0]) - kk * x_plus_two3[0],
             c * x3[1] - kk * x_plus_two3[1])
    return _nonnegative(value, discriminant)


def krein_bound(params):
    return _krein_one(params, 1) and _krein_one(params, -1)


def absolute_bound(params):
    v, k, lam, mu = params
    if mu == 0 or v + lam == 2 * k:
        return True
    spectrum = spectrum_of(params)
    if spectrum is None:
        return False
    _, _, _, f, g = spectrum
    return 2 * v <= f * (f + 3) and 2 * v <= g * (g + 3)


def run(op: str, family: Family, reduction: str = "all", prefix: int | None = None, **args):
    op = op.removeprefix("strongly_regular.")
    ms = list(itertools.islice(rt.iter_members(family), prefix))
    if rt.prime(family) != 2 or any(any(len(row) != len(m) for row in m) for m in ms):
        raise ValueError("strongly_regular needs square matrices over F_2")
    params = [srg_params(m) for m in ms]

    if op == "srg_params":
        values = [x for p in params for x in (p or (0, 0, 0, 0))]
        return rt.reduce_values(reduction, StronglyRegularParams(len(ms), [int(p is not None) for p in params], values))
    if op == "spectrum":
        spectra = [spectrum_of(p) if p is not None else None for p in params]
        records = [(s[0], int(s[1] < 0), abs(s[1]), s[2], s[3], s[4])
                   if s is not None else (0, 0, 0, 0, 0, 0) for s in spectra]
        return rt.reduce_values(reduction, StronglyRegularSpectra(len(ms), [int(s is not None) for s in spectra], records))
    if op == "is_srg":
        flags = [p is not None for p in params]
    elif op == "krein_bound":
        flags = [p is not None and krein_bound(p) for p in params]
    elif op == "absolute_bound":
        flags = [p is not None and absolute_bound(p) for p in params]
    else:
        raise ValueError(f"unknown operation {op}")
    return rt.reduce_bool(reduction, flags, ms, 2, **args)
