"""Readable baseline for classical permutation statistics.

The input family is materialised in canonical order. Each member is a one-row zero-based image
list, either from an explicit ``orbits.perms`` batch or from ``group_elements``.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))
from lemmakernel import naive as rt  # noqa: E402
from lemmakernel.interchange import Family, Perms  # noqa: E402


def inversions(perm: list[int]) -> int:
    return sum(perm[i] > perm[j] for i in range(len(perm)) for j in range(i + 1, len(perm)))


def descents(perm: list[int]) -> int:
    return sum(a > b for a, b in zip(perm, perm[1:]))


def major_index(perm: list[int]) -> int:
    return sum(i + 1 for i, (a, b) in enumerate(zip(perm, perm[1:])) if a > b)


def cycle_lengths(perm: list[int]) -> list[int]:
    seen: set[int] = set()
    lengths = []
    for start in range(len(perm)):
        if start in seen:
            continue
        current = start
        length = 0
        while current not in seen:
            seen.add(current)
            current = perm[current]
            length += 1
        lengths.append(length)
    return sorted(lengths, reverse=True)


def integer_partitions(n: int, maximum: int | None = None):
    if n == 0:
        yield []
        return
    maximum = n if maximum is None else min(n, maximum)
    for part in range(maximum, 0, -1):
        for tail in integer_partitions(n - part, part):
            yield [part, *tail]


def cycle_type_code(perm: list[int]) -> int:
    target = cycle_lengths(perm)
    return next(i for i, part in enumerate(integer_partitions(len(perm))) if part == target)


def contains_pattern(perm: list[int], pattern: list[int]) -> bool:
    k = len(pattern)
    for positions in itertools.combinations(range(len(perm)), k):
        values = [perm[i] for i in positions]
        if all((values[i] < values[j]) == (pattern[i] < pattern[j])
               for i in range(k) for j in range(i + 1, k)):
            return True
    return False


def pattern_avoids(perm: list[int], patterns: list[list[int]]) -> bool:
    return all(not contains_pattern(perm, pattern) for pattern in patterns)


def upper_rank(perm: list[int], prefix: int, threshold: int) -> int:
    return sum(value >= threshold for value in perm[:prefix])


def bruhat_leq(lower: list[int], upper: list[int]) -> bool:
    n = len(lower)
    return len(upper) == n and all(
        upper_rank(lower, prefix, threshold) <= upper_rank(upper, prefix, threshold)
        for prefix in range(n + 1) for threshold in range(n + 1)
    )


def _permutations(family: Family, prefix: int | None):
    if family.kind not in ("explicit", "group_elements") or rt.prime(family) != 0:
        raise ValueError("permutation_statistics needs an explicit or group_elements permutation family")
    members = list(itertools.islice(rt.iter_members(family), prefix))
    perms = [member[0] for member in members]
    if any(sorted(perm) != list(range(len(perm))) for perm in perms):
        raise ValueError("permutation family contains an invalid permutation")
    return members, perms


def run(op: str, family: Family, reduction: str = "all", prefix: int | None = None, **args):
    op = op.removeprefix("permutation_statistics.")
    members, perms = _permutations(family, prefix)

    if op == "inversions":
        return rt.reduce_int(reduction, [inversions(perm) for perm in perms], members, 0)
    if op == "descents":
        return rt.reduce_int(reduction, [descents(perm) for perm in perms], members, 0)
    if op == "major_index":
        return rt.reduce_int(reduction, [major_index(perm) for perm in perms], members, 0)
    if op == "cycle_type":
        return rt.reduce_int(reduction, [cycle_type_code(perm) for perm in perms], members, 0)
    if op == "pattern_avoids":
        patterns = args["patterns"]
        if not isinstance(patterns, Perms):
            raise ValueError("patterns must be an orbits.perms batch")
        pattern_lists = patterns.tolist()
        flags = [pattern_avoids(perm, pattern_lists) for perm in perms]
        return rt.reduce_bool(reduction, flags, members, 0, **args)
    if op == "bruhat_leq":
        upper = args["upper"]
        if not isinstance(upper, Perms):
            raise ValueError("upper must be an orbits.perms batch")
        if upper.count != 1:
            raise ValueError("bruhat_leq needs exactly one upper permutation")
        bound = upper.member(0)
        if any(len(perm) != len(bound) for perm in perms):
            raise ValueError("bruhat_leq upper must have the same degree as the family")
        return rt.reduce_bool(reduction, [bruhat_leq(perm, bound) for perm in perms], members, 0, **args)
    raise ValueError(f"unknown operation {op}")
