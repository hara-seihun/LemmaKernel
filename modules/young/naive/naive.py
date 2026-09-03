"""Plain Python implementation of partitions, tableaux, RSK, Kostka numbers and characters."""
from __future__ import annotations

import itertools
import math
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))
from lemmakernel import naive as rt  # noqa: E402
from lemmakernel.interchange import Characters, Family, RskPairs  # noqa: E402


def shape_of(member):
    shape = list(member[0])
    while shape and shape[-1] == 0:
        shape.pop()
    return shape


def hook_length_count(shape):
    n = sum(shape)
    hooks = 1
    for i, width in enumerate(shape):
        for j in range(width):
            hooks *= width - j + sum(other > j for other in shape[i + 1:])
    return math.factorial(n) // hooks


def kostka(shape, weight):
    cells = [(i, j) for i, width in enumerate(shape) for j in range(width)]
    filling = {}
    remaining = list(weight)

    def count(pos):
        if pos == len(cells):
            return 1
        i, j = cells[pos]
        lower = filling.get((i, j - 1), 0) if j else 0
        if i and j < shape[i - 1]:
            lower = max(lower, filling[(i - 1, j)] + 1)
        total = 0
        for value in range(lower, len(remaining)):
            if not remaining[value]:
                continue
            remaining[value] -= 1
            filling[i, j] = value
            total += count(pos + 1)
            del filling[i, j]
            remaining[value] += 1
        return total

    return count(0)


def subpartitions(outer, row=0, bound=None, prefix=()):
    if row == len(outer):
        yield list(prefix)
        return
    largest = min(outer[row], outer[0] if bound is None else bound)
    for width in range(largest, -1, -1):
        yield from subpartitions(outer, row + 1, width, prefix + (width,))


def skew_cells(outer, inner):
    return {(i, j) for i, width in enumerate(outer) for j in range(inner[i], width)}


def border_strip(outer, inner, length):
    skew = skew_cells(outer, inner)
    if len(skew) != length or not skew:
        return None
    if any((i, j + 1) in skew and (i + 1, j) in skew and (i + 1, j + 1) in skew for i, j in skew):
        return None
    seen = {next(iter(skew))}
    queue = list(seen)
    for i, j in queue:
        for cell in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
            if cell in skew and cell not in seen:
                seen.add(cell)
                queue.append(cell)
    if seen != skew:
        return None
    return len({i for i, _ in skew})


def murnaghan_nakayama(shape, cycle_type):
    cycle_type = tuple(cycle_type)

    @lru_cache(maxsize=None)
    def character(current, pos):
        current = list(current)
        if pos == len(cycle_type):
            return int(sum(current) == 0)
        strip = cycle_type[pos]
        total = 0
        target = sum(current) - strip
        if target < 0:
            return 0
        for inner in subpartitions(current):
            if sum(inner) != target:
                continue
            height = border_strip(current, inner, strip)
            if height is not None:
                total += (1 if height % 2 else -1) * character(tuple(inner), pos + 1)
        return total

    return character(tuple(shape), 0)


def rsk(word):
    insertion = []
    recording = []
    for label, letter in enumerate(word, 1):
        carry = letter
        row = 0
        while True:
            if row == len(insertion):
                insertion.append([carry])
                recording.append([label])
                break
            bumped = next((j for j, value in enumerate(insertion[row]) if value > carry), None)
            if bumped is None:
                insertion[row].append(carry)
                recording[row].append(label)
                break
            insertion[row][bumped], carry = carry, insertion[row][bumped]
            row += 1
    n = len(word)
    shape = [len(row) for row in insertion] + [0] * (n - len(insertion))
    pad = lambda tableau: [row + [0] * (n - len(row)) for row in tableau] + [[0] * n for _ in range(n - len(tableau))]
    return shape, pad(insertion), pad(recording)


def run(op: str, family: Family, reduction: str = "all", prefix: int | None = None, **args):
    op = op.removeprefix("young.")
    members = list(itertools.islice(rt.iter_members(family), prefix))

    if op == "hook_length_count":
        if family.kind == "standard_tableaux":
            shape = family.children[0].member(0)[0]
            values = [hook_length_count(shape)] * len(members)
        else:
            values = [hook_length_count(shape_of(member)) for member in members]
        return rt.reduce_int(reduction, values, members, rt.prime(family))

    if op == "kostka":
        weight = args["weight"].member(0)[0]
        if not weight or sum(weight) != family.params["total"]:
            raise ValueError("weight must sum to the partition size")
        values = [kostka(shape_of(member), weight) for member in members]
        return rt.reduce_int(reduction, values, members, rt.prime(family))

    if op == "murnaghan_nakayama":
        if reduction != "all":
            raise ValueError("young.characters values only reduce with `all`")
        cycle_type = args["cycle_type"].member(0)[0]
        valid = cycle_type and all(x > 0 for x in cycle_type) and all(a >= b for a, b in itertools.pairwise(cycle_type))
        if not valid or sum(cycle_type) != family.params["total"]:
            raise ValueError("cycle_type must be a positive weakly decreasing partition of n")
        return Characters([murnaghan_nakayama(shape_of(member), cycle_type) for member in members])

    if op == "rsk":
        if reduction != "all":
            raise ValueError("young.rsk_pairs values only reduce with `all`")
        n = family.params["length"]
        pairs = [rsk(member[0]) for member in members]
        return RskPairs(len(pairs), n,
                        [x for shape, _, _ in pairs for x in shape],
                        [x for _, insertion, _ in pairs for row in insertion for x in row],
                        [x for _, _, recording in pairs for row in recording for x in row])

    raise ValueError(f"unknown operation {op}")
