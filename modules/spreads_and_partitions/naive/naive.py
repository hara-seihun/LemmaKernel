"""The obvious implementation of the spreads_and_partitions module, in plain Python.

Every family is materialised member by member in its canonical order; every member is split into
its components, each component is put in reduced row echelon form from scratch (gfp's naive
elimination), and the predicate is read off the ranks. Nothing is shared between members and
nothing is pruned: this is the benchmark baseline and the readable second opinion, held to the
same Lean oracle as the kernel.

    naive.run(op, family, reduction, n=..., **args) -> interchange object

`family` is a lemmakernel.interchange.Family (what a family handle exports).
"""
from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
from lemmakernel import naive as rt  # noqa: E402  (families and reductions, shared by every module)
from lemmakernel.interchange import Family  # noqa: E402

_spec = importlib.util.spec_from_file_location("gfp_naive", ROOT / "modules" / "gfp" / "naive" / "naive.py")
gfp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gfp)


# ---- components ---------------------------------------------------------------------------------

def basis(block: list[list[int]], p: int) -> list[list[int]]:
    """The reduced row echelon basis of the row space of `block`: its length is the dimension,
    and it is the canonical form of the subspace."""
    R, piv = gfp.rref(block, p)
    return R[:len(piv)]


def component(row: list[int], start: int, h: int, n: int, p: int) -> list[list[int]]:
    """The subspace spanned by `h * n` entries of `row` from `start`, read row-major."""
    return basis([row[start + i * n:start + (i + 1) * n] for i in range(h)], p)


def components(member: list[list[int]], n: int, p: int) -> list[list[list[int]]]:
    """One component per row of the member: an h x n matrix with h = cols / n, as its basis."""
    return [component(row, 0, len(row) // n, n, p) for row in member]


def meets(a, b, p) -> bool:
    """Do two subspaces, given by bases, intersect in more than 0? Their ranks add exactly when
    they do not."""
    return gfp.rank(a + b, p) != len(a) + len(b)


def intersecting(cs, p) -> int:
    return sum(1 for a, b in itertools.combinations(cs, 2) if meets(a, b, p))


def covered(cs, p) -> int:
    """The nonzero vectors of the components, summed; the number covered when they are disjoint."""
    return sum(p ** len(c) - 1 for c in cs)


def disjoint(cs, p) -> bool:
    return all(cs) and intersecting(cs, p) == 0


def is_partial_spread(cs, p) -> bool:
    return disjoint(cs, p) and all(len(c) == len(cs[0]) for c in cs)


def is_spread(cs, n, p) -> bool:
    return is_partial_spread(cs, p) and covered(cs, p) == p ** n - 1


def is_partition(cs, n, p) -> bool:
    return disjoint(cs, p) and covered(cs, p) == p ** n - 1


def gauss_binomial(p: int, n: int, h: int) -> int:
    """[n choose h]_p: the number of h-dimensional subspaces of F_p^n."""
    c = 1
    for i in range(h):
        c = c * (p ** (n - i) - 1) // (p ** (i + 1) - 1)
    return c


def is_packing(member, n: int, h: int, p: int) -> bool:
    """Every row a spread of F_p^n by h-subspaces, and the components together every h-subspace
    of F_p^n exactly once."""
    canon = []
    for row in member:
        cs = [component(row, b * h * n, h, n, p) for b in range(len(row) // (h * n))]
        if not all(len(c) == h for c in cs) or not is_spread(cs, n, p):
            return False
        canon += cs
    return len(canon) == gauss_binomial(p, n, h) and len({tuple(map(tuple, c)) for c in canon}) == len(canon)


# ---- operations and reductions -------------------------------------------------------------------

def run(op: str, family: Family, reduction: str = "all", prefix: int | None = None, **args):
    """`prefix`: answer for the first `prefix` members only (the benchmark's timing sample)."""
    op = op.removeprefix("spreads_and_partitions.")
    p = rt.prime(family)
    ms = list(itertools.islice(rt.iter_members(family), prefix))
    cols = len(ms[0][0]) if ms and ms[0] else family.params.get("cols", 0)
    n = args["n"]
    h = args.get("h", 0) if op == "is_packing" else (cols // n if n else 0)
    if n == 0:
        raise ValueError("the ambient dimension n must be at least 1")
    if op == "is_packing" and h == 0:
        raise ValueError("the component dimension h must be at least 1")
    unit = n * h if op == "is_packing" else n
    if cols % unit:
        raise ValueError(f"{unit} does not divide the {cols} columns of a member")

    if op == "is_packing":
        flags = [is_packing(m, n, h, p) for m in ms]
    elif op == "intersecting_pairs":
        return rt.reduce_int(reduction, [intersecting(components(m, n, p), p) for m in ms], ms, p)
    elif op == "is_partial_spread":
        flags = [is_partial_spread(components(m, n, p), p) for m in ms]
    elif op == "is_spread":
        flags = [is_spread(components(m, n, p), n, p) for m in ms]
    elif op == "is_vector_space_partition":
        flags = [is_partition(components(m, n, p), n, p) for m in ms]
    else:
        raise ValueError(f"unknown operation {op}")
    return rt.reduce_bool(reduction, flags, ms, p, **args)
