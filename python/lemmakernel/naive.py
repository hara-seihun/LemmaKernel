"""The runtime's concepts, in plain Python, for a module's naive implementation.

A naive implementation only has to know its own mathematics: it materialises the family with
`members`, computes one value per member, and hands the values to `reduce_int`, `reduce_bool`,
or `reduce_values`. Families and reductions are runtime concepts, so their naive form lives
here, once, and matches the runtime's canonical orders and the Lean reference exactly.

    members(family)          -> (list of members as lists of rows, p)
    dictionary(family)       -> the rows a subsets/subsets_of family draws from
    reduce_int(reduction, values, members, p)          -> interchange object
    reduce_bool(reduction, flags, members, p, limit=0) -> interchange object
    reduce_values(reduction, obj)                      -> obj (`all` only)

`p` is the prime of the members, 0 for permutations, and NATURALS for integer members.
"""
from __future__ import annotations

import itertools

from .interchange import NATURALS, Count, Extremum, Family, First, Histogram, Hits, Integers, Matrix, Perms


# ---- arithmetic the families need ---------------------------------------------------------------

def matmul(a, b, p):
    return [[sum(x * y for x, y in zip(row, col)) % p for col in zip(*b)] for row in a]


def compose(g, h):
    """Permutation x -> h(g(x))."""
    return [h[g[x]] for x in range(len(g))]


def perm_closure(gens: list[list[int]]) -> list[list[int]]:
    """Every element of the generated permutation group, sorted lexicographically."""
    n = len(gens[0])
    identity = list(range(n))
    seen = {tuple(identity)}
    queue = [identity]
    for g in queue:
        for gen in gens:
            h = compose(g, gen)
            if tuple(h) not in seen:
                seen.add(tuple(h))
                queue.append(h)
    return sorted(queue)


# ---- families -----------------------------------------------------------------------------------

def batch_members(m: Matrix):
    return [m.member(i) for i in range(m.count)]


def vectors_of(m):
    """A batch of 1 x n rows, one k x n matrix, or a permutation batch as vectors."""
    if isinstance(m, Perms):
        return m.tolist()
    return [r[0] for r in batch_members(m)] if m.rows == 1 else m.member(0)


def grassmannian_members(p, n, h):
    """Every h-dimensional subspace of F_p^n as its rref basis: pivot sets in lexicographic
    order, then free entries in row-major lexicographic order."""
    for piv in itertools.combinations(range(n), h):
        free = [(i, c) for i in range(h) for c in range(piv[i] + 1, n) if c not in piv]
        for digits in itertools.product(range(p), repeat=len(free)):
            rows = [[0] * n for _ in range(h)]
            for i, pc in enumerate(piv):
                rows[i][pc] = 1
            for (i, c), d in zip(free, digits):
                rows[i][c] = d
            yield rows


def symmetric_members(p, n):
    """Every symmetric n x n matrix over F_p; the upper triangle, row-major, are the base-p
    digits of the index."""
    upper = [(i, j) for i in range(n) for j in range(i, n)]
    for digits in itertools.product(range(p), repeat=len(upper)):
        m = [[0] * n for _ in range(n)]
        for (i, j), d in zip(upper, digits):
            m[i][j] = m[j][i] = d
        yield m


def prime(f: Family) -> int:
    if f.kind in ("range", "words"):
        return NATURALS
    if f.kind == "group_elements":
        return 0
    if "p" in f.params:
        return f.params["p"]
    child = f.children[0]
    return prime(child) if isinstance(child, Family) else getattr(child, "p", 0)


def dictionary(f: Family) -> list[list[int]]:
    """The rows a `subsets` family chooses from, or the flattened members of a `subsets_of`
    family's inner family."""
    if f.kind == "subsets":
        (d,) = f.children
        return vectors_of(d)
    if f.kind == "subsets_of":
        (inner,) = f.children
        return [[x for r in m for x in r] for m in iter_members(inner)]
    raise ValueError(f"{f.kind} has no dictionary")


def iter_members(f: Family):
    """Yield the members of a family in canonical order as lists of rows."""
    if f.kind == "explicit":
        (batch,) = f.children
        yield from batch_members(batch)
    elif f.kind in ("subsets", "subsets_of"):
        vecs = dictionary(f)
        for idx in itertools.combinations(range(len(vecs)), f.params["k"]):
            yield [vecs[i] for i in idx]
    elif f.kind == "grassmannian":
        yield from grassmannian_members(f.params["p"], f.params["n"], f.params["h"])
    elif f.kind == "all_matrices":
        p, rows, cols = f.params["p"], f.params["rows"], f.params["cols"]
        for digits in itertools.product(range(p), repeat=rows * cols):
            yield [list(digits[r * cols:(r + 1) * cols]) for r in range(rows)]
    elif f.kind == "symmetric_matrices":
        yield from symmetric_members(f.params["p"], f.params["n"])
    elif f.kind == "transform":
        inner, C = f.children
        c, p = C.member(0), C.p
        for m in iter_members(inner):
            yield matmul(m, c, p)
    elif f.kind == "stack":
        inner, rows = f.children
        extra = rows.member(0)
        for m in iter_members(inner):
            yield m + extra
    elif f.kind == "group_elements":
        (gens,) = f.children
        for g in perm_closure(gens.tolist()):
            yield [g]
    elif f.kind == "range":
        for x in range(f.params["a"], f.params["b"]):
            yield [[x]]
    elif f.kind == "words":
        for w in itertools.product(range(f.params["alphabet"]), repeat=f.params["length"]):
            yield [list(w)]
    else:
        raise ValueError(f"unknown family {f.kind}")


def members(f: Family):
    """Materialise a family in canonical order as a list of matrices (lists of rows), plus p."""
    return list(iter_members(f)), prime(f)


# ---- reductions ---------------------------------------------------------------------------------

def _shape(members, p):
    rows = len(members[0]) if members else 0
    cols = len(members[0][0]) if members else 0
    return rows, cols


def _one(p, rows, cols, m):
    return Matrix(p, 1, rows, cols, [x for r in m for x in r])


def reduce_int(reduction: str, values: list[int], members, p: int):
    """Integer values of `members` (in family order) under an integer reduction."""
    size = len(values)
    rows, cols = _shape(members, p)
    if reduction == "all":
        return Integers(values)
    if reduction == "histogram":
        bins = [0] * (max(values) + 1 if values else 0)
        for v in values:
            bins[v] += 1
        return Histogram(size, size, bins)
    if reduction == "sum":
        return Count(sum(values), size, size)
    if reduction in ("max", "min"):
        if not values:
            raise ValueError(f"{reduction} of an empty family")
        v = max(values) if reduction == "max" else min(values)
        i = values.index(v)
        return Extremum(p, rows, cols, v, i, size, size, _one(p, rows, cols, members[i]))
    raise ValueError(f"reduction {reduction} does not accept integer values")


def reduce_bool(reduction: str, flags: list[bool], members, p: int, limit: int = 0, **_):
    """Boolean values of `members` (in family order) under a boolean reduction."""
    size = len(flags)
    rows, cols = _shape(members, p)
    if reduction == "all":
        return Integers([int(f) for f in flags])
    if reduction == "count":
        return Count(sum(flags), size, size)
    if reduction == "hits":
        idx = [i for i, f in enumerate(flags) if f]
        n = min(limit, len(idx))
        mem = Matrix(p, n, rows, cols, [x for i in idx[:n] for r in members[i] for x in r])
        return Hits(p, rows, cols, len(idx), size, size, idx, mem)
    if reduction == "first":
        i = next((i for i, f in enumerate(flags) if f), None)
        if i is None:
            return First(p, rows, cols, 0, 0, size, size, Matrix(p, 0, rows, cols, []))
        return First(p, rows, cols, 1, i, i + 1, size, _one(p, rows, cols, members[i]))
    raise ValueError(f"reduction {reduction} does not accept boolean values")


def reduce_values(reduction: str, obj):
    """A materialised per-member value only reduces with `all`."""
    if reduction != "all":
        raise ValueError(f"{type(obj).__name__} values only reduce with `all`")
    return obj
