"""The obvious implementation of the simplicial_complexes module, in plain Python.

Every member is materialised, its complex is built by testing every nonempty subset of the
vertex set for membership, and each operation is computed from that list of faces from scratch:
the f-vector by counting, the Betti numbers by eliminating the boundary matrices over F_p, and
shellability by searching the orders of the facets. Nothing is shared between members and
nothing is pruned.

This is what the fast backend is tested against, byte for byte on the interchange encoding, and
what the benchmark reports the speed-up against. Keep it readable before keeping it quick.

    naive.run(op, family, reduction, **args) -> interchange object

`family` is a lemmakernel.interchange.Family (what a family handle exports).
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))
from lemmakernel import naive as rt  # noqa: E402  (families and reductions, shared by every module)
from lemmakernel.interchange import Family  # noqa: E402


# ---- the complex of a member --------------------------------------------------------------------

def supports(member: list[list[int]]) -> list[frozenset[int]]:
    """The vertex set of each row: the columns where it is 1."""
    return [frozenset(j for j, x in enumerate(row) if x) for row in member]


def faces(member: list[list[int]], nonfaces: int) -> list[frozenset[int]]:
    """Every nonempty face, by increasing size and lexicographically within a size.

    `nonfaces = 0`: the rows generate the complex, so a set is a face when some row contains it.
    `nonfaces = 1`: the rows are forbidden, so a set is a face when it contains no row.
    """
    n = len(member[0]) if member else 0
    gens = supports(member)
    out = []
    for k in range(1, n + 1):
        for combination in itertools.combinations(range(n), k):
            T = frozenset(combination)
            keep = any(T <= S for S in gens) if nonfaces == 0 else all(not S <= T for S in gens)
            if keep:
                out.append(T)
    return out


def f_count(fs: list[frozenset[int]], dim: int) -> int:
    return sum(1 for F in fs if len(F) == dim + 1)


def euler_mod(fs: list[frozenset[int]], p: int) -> int:
    """The alternating sum of the f-vector modulo p: faces of even dimension have odd size."""
    even = sum(1 for F in fs if len(F) % 2 == 1)
    odd = sum(1 for F in fs if len(F) % 2 == 0)
    return (even - odd) % p


# ---- homology -----------------------------------------------------------------------------------

def rank_mod_p(rows: list[list[int]], p: int) -> int:
    """Gaussian elimination over F_p."""
    rows = [list(r) for r in rows]
    cols = len(rows[0]) if rows else 0
    r = 0
    for c in range(cols):
        pivot = next((i for i in range(r, len(rows)) if rows[i][c]), None)
        if pivot is None:
            continue
        rows[r], rows[pivot] = rows[pivot], rows[r]
        inv = pow(rows[r][c], p - 2, p)
        rows[r] = [(x * inv) % p for x in rows[r]]
        for i in range(r + 1, len(rows)):
            if rows[i][c]:
                f = rows[i][c]
                rows[i] = [(a - f * b) % p for a, b in zip(rows[i], rows[r])]
        r += 1
    return r


def boundary(fs: list[frozenset[int]], dim: int, p: int) -> list[list[int]]:
    """The matrix of boundary_dim: one row per dim-face, one column per (dim-1)-face, with
    (-1)^j where the j-th vertex of the row's face is dropped. Empty for dim = 0."""
    if dim == 0:
        return []
    upper = [sorted(F) for F in fs if len(F) == dim + 1]
    lower = [sorted(F) for F in fs if len(F) == dim]
    index = {tuple(G): j for j, G in enumerate(lower)}
    matrix = []
    for F in upper:
        row = [0] * len(lower)
        for j in range(len(F)):
            row[index[tuple(F[:j] + F[j + 1:])]] = 1 if j % 2 == 0 else p - 1
        matrix.append(row)
    return matrix


def betti(fs: list[frozenset[int]], dim: int, p: int) -> int:
    """f_dim - rank(boundary_dim) - rank(boundary_{dim+1}), the dimension of H_dim over F_p."""
    return f_count(fs, dim) - rank_mod_p(boundary(fs, dim, p), p) - rank_mod_p(boundary(fs, dim + 1, p), p)


# ---- shellability -------------------------------------------------------------------------------

def facets(fs: list[frozenset[int]]) -> list[frozenset[int]]:
    """The maximal faces."""
    return [F for F in fs if not any(len(G) > len(F) and F <= G for G in fs)]


def is_shellable(fs: list[frozenset[int]]) -> bool:
    """Bjorner-Wachs, non-pure allowed: is there an order of the facets in which every earlier
    facet meets the next one inside a chosen facet that misses exactly one of its vertices?

    Whether a facet may come next depends only on the set of facets already placed, not on their
    order, so the search is over subsets of the facets."""
    fac = facets(fs)
    total = len(fac)

    def may_follow(chosen: tuple[int, ...], j: int) -> bool:
        return all(any(fac[i] & fac[j] <= fac[k] and len(fac[j] - fac[k]) == 1 for k in chosen)
                   for i in chosen)

    seen = set()

    def search(chosen: frozenset[int]) -> bool:
        if len(chosen) == total:
            return True
        if chosen in seen:
            return False
        seen.add(chosen)
        order = tuple(sorted(chosen))
        return any(search(chosen | {j}) for j in range(total)
                   if j not in chosen and (not chosen or may_follow(order, j)))

    return search(frozenset())


# ---- operations and reductions -------------------------------------------------------------------

def run(op: str, family: Family, reduction: str = "all", prefix: int | None = None, **args):
    """`prefix`: answer for the first `prefix` members only (the benchmark's timing sample)."""
    op = op.removeprefix("simplicial_complexes.")
    p_family = rt.prime(family)
    if p_family != 2:
        raise ValueError("simplicial_complexes: members must be 0/1 matrices over F_2")
    nonfaces = int(args["nonfaces"])
    if nonfaces > 1:
        raise ValueError("nonfaces must be 0 or 1")
    ms = list(itertools.islice(rt.iter_members(family), prefix))
    complexes = [faces(m, nonfaces) for m in ms]

    if op == "is_shellable":
        return rt.reduce_bool(reduction, [is_shellable(fs) for fs in complexes], ms, p_family, **args)

    if op == "faces":
        values = [len(fs) for fs in complexes]
    elif op == "f_count":
        dim = int(args["dim"])
        values = [f_count(fs, dim) for fs in complexes]
    elif op in ("betti", "euler_characteristic"):
        p = int(args["p"])
        if p < 2 or p >= 2 ** 32 or any(p % d == 0 for d in range(2, int(p ** 0.5) + 1)):
            raise ValueError(f"p = {p} is not a prime below 2^32")
        if op == "betti":
            values = [betti(fs, int(args["dim"]), p) for fs in complexes]
        else:
            values = [euler_mod(fs, p) for fs in complexes]
    else:
        raise ValueError(f"unknown operation {op}")

    return rt.reduce_int(reduction, values, ms, p_family)
