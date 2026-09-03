"""Oracle and benchmark cases for permutation_statistics."""
from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lemmakernel as lk  # noqa: E402
from tools.harness import Case  # noqa: E402


def symmetric_group(ctx, n: int):
    if n == 1:
        generators = [[0]]
    elif n == 2:
        generators = [[1, 0]]
    else:
        generators = [list(range(1, n)) + [0], [1, 0] + list(range(2, n))]
    return ctx.group_elements(ctx.perms(n, generators))


def avoids(perm, pattern):
    k = len(pattern)
    return not any(
        all((values[i] < values[j]) == (pattern[i] < pattern[j])
            for i in range(k) for j in range(i + 1, k))
        for positions in itertools.combinations(range(len(perm)), k)
        for values in [[perm[i] for i in positions]]
    )


def avoidance_class(ctx, n: int, pattern: list[int]):
    members = [list(perm) for perm in itertools.permutations(range(n)) if avoids(perm, pattern)]
    return ctx.explicit(lk.perms(n, members))


def cases(ctx, rng):
    del rng
    s4 = symmetric_group(ctx, 4)
    p123 = ctx.perms(3, [[0, 1, 2]])
    two_patterns = ctx.perms(3, [[0, 1, 2], [2, 1, 0]])
    bound = ctx.perms(4, [[2, 0, 3, 1]])

    out = [
        Case("S4 inversions", s4, "inversions"),
        Case("S4 descents", s4, "descents"),
        Case("S4 major index", s4, "major_index"),
        Case("S4 cycle types", s4, "cycle_type"),
        Case("S4 avoids 123", s4, "pattern_avoids", {"patterns": p123, "limit": 5}),
        Case("S4 avoids 123 and 321", s4, "pattern_avoids", {"patterns": two_patterns, "limit": 5},
             reductions=["count", "hits"]),
        Case("S4 below 2031", s4, "bruhat_leq", {"upper": bound, "limit": 5}),
    ]

    av123 = avoidance_class(ctx, 5, [0, 1, 2])
    out += [
        Case("Av_5(123) inversions", av123, "inversions", reductions=["histogram"]),
        Case("Av_5(123) descents", av123, "descents", reductions=["histogram"]),
        Case("Av_5(123) major index", av123, "major_index", reductions=["histogram"]),
        Case("Av_5(123) cycle types", av123, "cycle_type", reductions=["histogram"]),
    ]

    out += [
        Case("matrices are not permutations", ctx.explicit(lk.matrix(2, [[1, 0], [0, 1]])),
             "inversions", oracle=False),
        Case("Bruhat degree mismatch", s4, "bruhat_leq", {"upper": ctx.perms(3, [[0, 1, 2]])},
             reductions=["count"], oracle=False),
        Case("Bruhat needs one upper bound", s4, "bruhat_leq",
             {"upper": ctx.perms(4, [list(range(4)), [1, 0, 2, 3]])}, reductions=["count"], oracle=False),
    ]

    bench_family = {}

    def explicit_s8():
        if "family" not in bench_family:
            batch = lk.perms(8, itertools.permutations(range(8)))
            bench_family["family"] = ctx.explicit(batch)
        return bench_family["family"]

    out += [
        Case("S8 Mahonian", explicit_s8, "inversions", bench="histogram", oracle=False,
             what="Mahonian inversion distribution over all 8! permutations"),
        Case("S8 Eulerian", explicit_s8, "descents", bench="histogram", oracle=False,
             what="Eulerian descent distribution over all 8! permutations"),
        Case("S8 major index", explicit_s8, "major_index", bench="histogram", oracle=False,
             what="major-index distribution over all 8! permutations"),
        Case("Av_8(123) count", explicit_s8, "pattern_avoids", {"patterns": p123}, bench="count", oracle=False,
             what="the 1430 permutations of length 8 avoiding the classical pattern 123"),
        Case("S8 cycle types", explicit_s8, "cycle_type", bench="histogram", oracle=False,
             what="conjugacy-class distribution of S_8 by dense complete-cycle-partition code"),
    ]
    return out


def invariants(ctx):
    s7 = symmetric_group(ctx, 7)
    inv = ctx.value("permutation_statistics.inversions", s7, "histogram").bins
    major = ctx.value("permutation_statistics.major_index", s7, "histogram").bins
    desc = ctx.value("permutation_statistics.descents", s7, "histogram").bins
    assert inv == major
    assert desc == list(reversed(desc))
    expected_sum = math.factorial(7) * 7 * 6 // 4
    assert ctx.value("permutation_statistics.inversions", s7, "sum").value == expected_sum
    p123 = ctx.perms(3, [[0, 1, 2]])
    assert ctx.value("permutation_statistics.pattern_avoids", s7, "count", patterns=p123).value == 429
