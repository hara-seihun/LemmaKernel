"""Oracle-sized cases and benchmarks for the young module."""
from __future__ import annotations

import math
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lemmakernel as lk  # noqa: E402
from tools.harness import Case  # noqa: E402


def vector(xs):
    return lk.naturals([list(xs)])


def cases(ctx, rng):
    del rng
    out = [
        Case("Kostka n=5", ctx.partitions(5), "kostka", {"weight": vector([2, 2, 1])}),
        Case("hook counts n=6", ctx.partitions(6), "hook_length_count"),
        Case("tableaux of shape 3,2", ctx.standard_tableaux([3, 2]), "hook_length_count"),
        Case("S6 character at 3,2,1", ctx.partitions(6), "murnaghan_nakayama",
             {"cycle_type": vector([3, 2, 1])}),
        Case("binary words length 3", ctx.words(2, 3), "rsk"),
        Case("bad Kostka weight", ctx.partitions(4), "kostka", {"weight": vector([2, 1])},
             reductions=["all"], oracle=False),
        Case("bad cycle type", ctx.partitions(5), "murnaghan_nakayama",
             {"cycle_type": vector([2, 3])}, oracle=False),
        Case("RSK on partitions", ctx.partitions(4), "rsk", oracle=False),
    ]

    out += [
        Case("hook_counts_n30", lambda: ctx.partitions(30), "hook_length_count", bench="all", oracle=False,
             what="the dimensions of every irreducible representation of S_30"),
        Case("ternary_rsk_length7", lambda: ctx.words(3, 7), "rsk", bench="all", oracle=False,
             what="RSK insertion and recording tableaux for all 2,187 ternary words of length 7"),
    ]
    return out


def invariants(ctx):
    for shape in ([4, 2, 1], [3, 3], [5]):
        family = ctx.standard_tableaux(shape)
        expected = ctx.value("young.hook_length_count", family).values[0]
        assert ctx.size(family) == expected

    n = 9
    dimensions = ctx.value("young.hook_length_count", ctx.partitions(n)).values
    involutions = [1, 1]
    for k in range(2, n + 1):
        involutions.append(involutions[-1] + (k - 1) * involutions[-2])
    assert sum(dimensions) == involutions[n]

    n = 6
    partitions = ctx.partitions(n)
    cycle_types = []
    columns = []
    for i in range(ctx.size(partitions)):
        cycle = [x for x in ctx.member(partitions, i).value().member(0)[0] if x]
        cycle_types.append(cycle)
        columns.append(ctx.value("young.murnaghan_nakayama", partitions,
                                 cycle_type=vector(cycle)).values)
    table = list(map(list, zip(*columns)))
    for i, row in enumerate(table):
        for j, other in enumerate(table):
            inner = 0
            for cycle, a, b in zip(cycle_types, row, other):
                multiplicities = Counter(cycle)
                centralizer = math.prod(part ** count * math.factorial(count)
                                        for part, count in multiplicities.items())
                inner += Fraction(a * b, centralizer)
            assert inner == int(i == j)
    assert table[0] == [1] * len(cycle_types)
    dimensions = ctx.value("young.hook_length_count", partitions).values
    assert [row[-1] for row in table] == dimensions
    assert ctx.value("young.kostka", partitions, weight=vector([1] * n)).values == dimensions

    pairs = ctx.value("young.rsk", ctx.words(3, 4))
    for i in range(pairs.count):
        shape, insertion, recording = pairs.member(i)
        rows = [x for x in shape if x]
        assert sum(rows) == 4
        assert all(rows[j] >= rows[j + 1] for j in range(len(rows) - 1))
        for table, strict_rows in ((insertion, False), (recording, True)):
            entries = []
            for r, width in enumerate(rows):
                row = table[r][:width]
                entries.extend(row)
                relation = (lambda a, b: a < b) if strict_rows else (lambda a, b: a <= b)
                assert all(relation(a, b) for a, b in zip(row, row[1:]))
                if r:
                    assert all(table[r - 1][c] < table[r][c] for c in range(width))
            if strict_rows:
                assert sorted(entries) == list(range(1, 5))
