"""Plain Python reference implementation for circulants."""
from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
from lemmakernel import naive as rt  # noqa: E402
from lemmakernel.interchange import Family, Matrix, NATURALS, Spectra  # noqa: E402


def _connection(member):
    return [x for row in member for x in row]


def _valid_connection(n: int, connection: list[int]) -> bool:
    return 0 < n < 1 << 32 and 0 not in connection and len(set(connection)) == len(connection) and all(0 <= x < n for x in connection)


def _effective(n: int, directed: int, connection: list[int]) -> list[int]:
    if directed:
        return sorted(connection)
    return sorted((set(connection) | {(-x) % n for x in connection}) - {0})


def _adam_order(n: int) -> bool:
    if n <= 0 or n % 8 == 0:
        return False
    return not any(n % (d * d) == 0 for d in range(3, math.isqrt(n) + 1, 2))


def _ci_order(n: int, directed: int) -> bool:
    return _adam_order(n) or (not directed and n in (8, 9, 18))


def _units(n: int):
    return [a for a in range(n) if math.gcd(a, n) == 1]


def _image(n: int, multiplier: int, connection: list[int]) -> list[int]:
    return sorted(multiplier * x % n for x in connection)


def _equivalent(n: int, left: list[int], right: list[int]) -> bool:
    return any(_image(n, a, left) == right for a in _units(n))


def _canonical(n: int, connection: list[int]) -> bool:
    return all(connection <= _image(n, a, connection) for a in _units(n))


def _validate_mode(directed: int):
    if directed not in (0, 1):
        raise ValueError("directed must be 0 or 1")


def _members(family: Family, prefix: int | None):
    return list(itertools.islice(rt.iter_members(family), prefix))


def _validate_family(family: Family, members, n: int):
    if rt.prime(family) != NATURALS:
        raise ValueError("connection sets must contain natural-number residues")
    if not all(_valid_connection(n, _connection(member)) for member in members):
        raise ValueError(f"connection sets must be identity-free sets of residues below {n}")


def _target(obj, n: int) -> list[int]:
    if not isinstance(obj, Matrix) or obj.p != NATURALS or obj.count != 1 or obj.rows != 1:
        raise ValueError("target must be one natural-number vector")
    target = obj.member(0)[0]
    if not _valid_connection(n, target):
        raise ValueError(f"target must be an identity-free set of residues below {n}")
    return target


def _spectrum(n: int, directed: int, connection: list[int]):
    effective = _effective(n, directed, connection)
    return [sorted(j * value % n for value in effective) for j in range(n)]


def run(op: str, family: Family, reduction: str = "all", prefix: int | None = None, **args):
    op = op.removeprefix("circulants.")
    directed = int(args["directed"])
    _validate_mode(directed)
    members = _members(family, prefix)

    if op == "is_ci":
        if family.kind != "range":
            raise ValueError("is_ci is defined on range families only")
        orders = [_connection(member)[0] for member in members]
        if not all(order > 0 for order in orders):
            raise ValueError("is_ci needs positive orders")
        return rt.reduce_bool(reduction, [_ci_order(n, directed) for n in orders], members, NATURALS, **args)

    n = int(args["n"])
    if not 0 < n < 1 << 32:
        raise ValueError("n must satisfy 1 <= n < 2^32")
    _validate_family(family, members, n)

    if op == "spectrum":
        if reduction != "all":
            raise ValueError("spectrum values only reduce with `all`")
        offsets = [0]
        exponents = []
        for member in members:
            for row in _spectrum(n, directed, _connection(member)):
                exponents.extend(row)
                offsets.append(len(exponents))
        return Spectra(n, len(members), offsets, exponents)

    if not _ci_order(n, directed):
        raise ValueError(f"Adam's multiplier criterion is not complete for n={n} and directed={directed}")

    if op == "isomorphic":
        target = _effective(n, directed, _target(args["target"], n))
        flags = [_equivalent(n, _effective(n, directed, _connection(member)), target) for member in members]
    elif op == "is_canonical":
        flags = []
        for member in members:
            raw = sorted(_connection(member))
            effective = _effective(n, directed, raw)
            flags.append((directed or raw == effective) and _canonical(n, effective))
    else:
        raise ValueError(f"unknown operation {op}")
    return rt.reduce_bool(reduction, flags, members, NATURALS, **args)
