"""Readable baseline for constrained partition and composition statistics."""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))
from lemmakernel import naive as rt  # noqa: E402
from lemmakernel.interchange import Family  # noqa: E402


def positive_parts(member):
    return [x for x in member[0] if x]


def rank_offset(total, parts):
    return total + parts[0] - len(parts)


def crank_offset(total, parts):
    ones = parts.count(1)
    if ones == 0:
        return total + parts[0]
    return total + sum(x > ones for x in parts) - ones


def run(op: str, family: Family, reduction: str = "all", prefix: int | None = None, **_):
    op = op.removeprefix("integer_partitions.")
    members = list(itertools.islice(rt.iter_members(family), prefix))
    rows = [positive_parts(member) for member in members]
    if op == "number_of_parts":
        values = [len(parts) for parts in rows]
    elif op == "largest_part":
        values = [parts[0] for parts in rows]
    elif op == "rank":
        if family.kind != "partitions":
            raise ValueError("rank is defined on partitions families only")
        values = [rank_offset(family.params["total"], parts) for parts in rows]
    elif op == "crank":
        if family.kind != "partitions":
            raise ValueError("crank is defined on partitions families only")
        values = [crank_offset(family.params["total"], parts) for parts in rows]
    else:
        raise ValueError(f"unknown operation {op}")
    return rt.reduce_int(reduction, values, members, rt.prime(family))
