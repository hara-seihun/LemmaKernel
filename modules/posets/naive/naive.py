"""Obvious finite-poset algorithms, run independently for every materialised family member."""
from __future__ import annotations

import functools
import itertools
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
from lemmakernel import naive as rt  # noqa: E402
from lemmakernel.interchange import Family, MobiusMatrices  # noqa: E402


def relation_matrix(member):
    n = len(member)
    if any(len(row) != n for row in member) or any(x not in (0, 1) for row in member for x in row):
        raise ValueError("explicit member is not a square 0/1 partial order relation")
    r = [[bool(x) for x in row] for row in member]
    if any(not r[i][i] for i in range(n)):
        raise ValueError("relation is not a partial order: it is not reflexive")
    if any(i != j and r[i][j] and r[j][i] for i in range(n) for j in range(n)):
        raise ValueError("relation is not a partial order: it is not antisymmetric")
    if any(r[i][j] and r[j][k] and not r[i][k] for i in range(n) for j in range(n) for k in range(n)):
        raise ValueError("relation is not a partial order: it is not transitive")
    return r


def subset_relation(member):
    supports = [tuple(bool(x) for x in row) for row in member]
    if len(set(supports)) != len(supports):
        raise ValueError("subset member has duplicate supports")
    return [[all(not x or y for x, y in zip(a, b)) for b in supports] for a in supports]


def divisors(x):
    if x == 0:
        raise ValueError("divisor posets need a positive integer")
    low, high = [], []
    for d in range(1, math.isqrt(x) + 1):
        if x % d == 0:
            low.append(d)
            if d * d != x:
                high.append(x // d)
    return low + high[::-1]


def divisor_relation(x):
    ds = divisors(x)
    return [[b % a == 0 for b in ds] for a in ds]


def relation(family: Family, member):
    if family.kind == "explicit":
        return relation_matrix(member)
    if family.kind in ("subsets", "subsets_of"):
        return subset_relation(member)
    if family.kind == "range":
        return divisor_relation(member[0][0])
    raise ValueError(f"{family.kind} families do not present finite posets")


def mobius(r):
    n = len(r)

    @functools.cache
    def mu(i, j):
        if not r[i][j]:
            return 0
        if i == j:
            return 1
        return -sum(mu(i, k) for k in range(n) if r[i][k] and r[k][j] and k != j)

    return [[mu(i, j) for j in range(n)] for i in range(n)]


def linear_extension_count(r):
    n = len(r)
    predecessors = [sum((1 << y) for y in range(n) if y != x and r[y][x]) for x in range(n)]
    dp = [0] * (1 << n)
    dp[0] = 1
    for chosen in range(1 << n):
        for x in range(n):
            bit = 1 << x
            if not chosen & bit and not predecessors[x] & ~chosen:
                dp[chosen | bit] += dp[chosen]
    return dp[-1]


def is_chain(r, subset):
    return all(r[a][b] or r[b][a] for a, b in itertools.combinations(subset, 2))


def is_antichain(r, subset):
    return all(not r[a][b] and not r[b][a] for a, b in itertools.combinations(subset, 2))


def maximum_subset(r, predicate):
    n = len(r)
    return max((len(s) for k in range(n + 1) for s in itertools.combinations(range(n), k) if predicate(r, s)), default=0)


def meet(r, a, b):
    lower = [x for x in range(len(r)) if r[x][a] and r[x][b]]
    return next((x for x in lower if all(r[y][x] for y in lower)), None)


def join(r, a, b):
    upper = [x for x in range(len(r)) if r[a][x] and r[b][x]]
    return next((x for x in upper if all(r[x][y] for y in upper)), None)


def lattice_tables(r):
    n = len(r)
    meets = [[meet(r, a, b) for b in range(n)] for a in range(n)]
    joins = [[join(r, a, b) for b in range(n)] for a in range(n)]
    return meets, joins


def is_lattice(r):
    meets, joins = lattice_tables(r)
    return all(x is not None for row in meets + joins for x in row)


def is_distributive(r):
    meets, joins = lattice_tables(r)
    if any(x is None for row in meets + joins for x in row):
        return False
    n = len(r)
    for x, y, z in itertools.product(range(n), repeat=3):
        if meets[x][joins[y][z]] != joins[meets[x][y]][meets[x][z]]:
            return False
        if joins[x][meets[y][z]] != meets[joins[x][y]][joins[x][z]]:
            return False
    return True


def order_polynomial(r, t):
    n = len(r)
    if t == 0:
        return int(n == 0)
    predecessor_masks = [sum((1 << y) for y in range(n) if r[y][x]) for x in range(n)]
    size = 1 << n
    ideals = [all(not mask & (1 << x) or predecessor_masks[x] & mask == predecessor_masks[x]
                  for x in range(n)) for mask in range(size)]
    dp = [int(ideal) for ideal in ideals]
    for _ in range(1, t):
        sums = dp[:]
        for bit in range(n):
            b = 1 << bit
            for mask in range(size):
                if mask & b:
                    sums[mask] += sums[mask ^ b]
        dp = [value if ideals[mask] else 0 for mask, value in enumerate(sums)]
    return dp[-1]


def run(op: str, family: Family, reduction: str = "all", prefix: int | None = None, **args):
    op = op.removeprefix("posets.")
    members = list(itertools.islice(rt.iter_members(family), prefix))
    relations = [relation(family, member) for member in members]

    if op == "mobius_function":
        if reduction != "all":
            raise ValueError("mobius values only reduce with `all`")
        matrices = [mobius(r) for r in relations]
        offsets = [0]
        entries = []
        for matrix in matrices:
            entries.extend(x for row in matrix for x in row)
            offsets.append(len(entries))
        return MobiusMatrices(len(matrices), offsets, entries)

    if op in ("linear_extension_count", "order_polynomial") and any(len(r) > 24 for r in relations):
        raise ValueError(f"{op} accepts posets with at most 24 elements")

    if op == "linear_extension_count":
        values = [linear_extension_count(r) for r in relations]
    elif op == "width":
        values = [maximum_subset(r, is_antichain) for r in relations]
    elif op == "height":
        values = [maximum_subset(r, is_chain) for r in relations]
    elif op == "order_polynomial":
        values = [order_polynomial(r, args["t"]) for r in relations]
    elif op == "is_lattice":
        return rt.reduce_bool(reduction, [is_lattice(r) for r in relations], members, rt.prime(family), **args)
    elif op == "is_distributive":
        return rt.reduce_bool(reduction, [is_distributive(r) for r in relations], members, rt.prime(family), **args)
    else:
        raise ValueError(f"unknown operation {op}")
    return rt.reduce_int(reduction, values, members, rt.prime(family))
