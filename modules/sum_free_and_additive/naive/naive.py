"""Direct additive-combinatorics predicates on finite sets of naturals.

The shared runtime materialises the family; this module reads each member as a set of elements
and computes every quantity from scratch, the obvious way: all ordered pairs, all candidate
common differences. It is the benchmark baseline and a second opinion, and it is held to the
same Lean oracle as the kernel.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
from lemmakernel import naive as rt  # noqa: E402
from lemmakernel.interchange import NATURALS, Family  # noqa: E402

BOOLEAN_OPS = ("is_sum_free", "is_sidon", "is_ap_free", "is_small_doubling")


def check_rows(rows, modulus: int, what: str) -> None:
    if any(len(r) != 1 for r in rows):
        raise ValueError(f"{what} must have one column: one element per row")
    values = [r[0] for r in rows]
    if len(set(values)) != len(values):
        raise ValueError(f"{what} contains a duplicate element")
    if modulus and any(v >= modulus for v in values):
        raise ValueError(f"{what} has an element that is not below the modulus")


def elements(family: Family, modulus: int, prefix: int | None):
    """Members as lists of elements, after the same validation the backend does."""
    if family.kind not in ("explicit", "subsets", "subsets_of"):
        raise ValueError("sum_free_and_additive is defined on explicit, subsets, subsets_of families only")
    if rt.prime(family) != NATURALS:
        raise ValueError("sum_free_and_additive members must be lk.naturals")
    if family.kind == "explicit":
        for member in rt.iter_members(family):
            check_rows(member, modulus, "member")
    else:
        check_rows(rt.dictionary(family), modulus, "dictionary")
    members = list(itertools.islice(rt.iter_members(family), prefix))
    return members, [[r[0] for r in m] for m in members]


def sums(xs, modulus: int):
    return [(x + y) % modulus if modulus else x + y for x in xs for y in xs]


def difference_codes(xs, modulus: int, distinct_only: bool):
    """x - y over ordered pairs, coded so that it is a natural: shifted over the integers."""
    shift = max(xs, default=0)
    return [(x + modulus - y) % modulus if modulus else x + shift - y
            for x in xs for y in xs if not (distinct_only and x == y)]


def max_difference_multiplicity(xs, modulus: int) -> int:
    codes = difference_codes(xs, modulus, True)
    return max((codes.count(c) for c in set(codes)), default=0)


def has_ap(xs, modulus: int, length: int) -> bool:
    candidates = range(1, modulus) if modulus else range(1, max(xs, default=0) + 1)
    members = set(xs)
    for a in xs:
        for d in candidates:
            terms = [(a + d * i) % modulus if modulus else a + d * i for i in range(length)]
            if all(t in members for t in terms):
                return True
    return False


def value(op: str, xs: list[int], modulus: int, args: dict):
    if op == "sumset_size":
        return len(set(sums(xs, modulus)))
    if op == "difference_set_size":
        return len(set(difference_codes(xs, modulus, False)))
    if op == "schur_triple_count":
        members = set(xs)
        return sum(1 for s in sums(xs, modulus) if s in members)
    if op == "is_sum_free":
        members = set(xs)
        return not any(s in members for s in sums(xs, modulus))
    if op == "max_difference_multiplicity":
        return max_difference_multiplicity(xs, modulus)
    if op == "is_sidon":
        return max_difference_multiplicity(xs, modulus) <= 1
    if op == "is_ap_free":
        return not has_ap(xs, modulus, int(args["length"]))
    if op == "is_small_doubling":
        return len(set(sums(xs, modulus))) * int(args["bound_den"]) <= int(args["bound_num"]) * len(xs)
    raise ValueError(f"unknown operation {op}")


def run(op: str, family: Family, reduction: str = "all", prefix: int | None = None, **args):
    op = op.removeprefix("sum_free_and_additive.")
    modulus = int(args["modulus"])
    if op == "is_ap_free" and int(args["length"]) < 2:
        raise ValueError("is_ap_free needs a progression length of at least 2")
    if op == "is_small_doubling" and int(args["bound_den"]) < 1:
        raise ValueError("is_small_doubling needs bound_den >= 1")
    members, sets = elements(family, modulus, prefix)
    values = [value(op, xs, modulus, args) for xs in sets]
    if op in BOOLEAN_OPS:
        return rt.reduce_bool(reduction, values, members, NATURALS, limit=int(args.get("limit", 0)))
    return rt.reduce_int(reduction, values, members, NATURALS)
