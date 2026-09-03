"""orbits: every backend's answer must be accepted by Lean's `decide +kernel` against the reference.

As in gfp, no expected answers live here. Each case builds a group and a family, runs the kernel,
and states the answer as a Lean `example` over `Orbits.run` (lean/Orbits/Reference.lean). The
naive Python implementation is held to the same oracle.

Kernel evaluation of the reference costs about a microsecond per reduction step, and group
closure in the reference is a list scan per product, so groups here have order at most a few
dozen and families at most a few dozen members. The Burnside test checks larger cases against
the kernel's own fixed-point counts instead.

Run from the repository root after building: `pytest -n auto modules/orbits`.
"""
from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT))
import lemmakernel as lk  # noqa: E402
from lemmakernel import interchange as ic  # noqa: E402
from tools.leancheck import LeanCheck  # noqa: E402
from tools.leanterms import L, lean_family, lean_perms, lean_red, random_batch  # noqa: E402

_spec = importlib.util.spec_from_file_location("orbits_naive", ROOT / "modules" / "orbits" / "naive" / "naive.py")
naive = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(naive)

BACKENDS = [b for b in lk.describe()["available_backends"] if b.startswith("orbits.")]
IMPORTS, OPENS = ["Orbits.Reference"], ["Orbits", "Gfp"]


# ---- Lean terms ---------------------------------------------------------------------------------

def lean_group(g) -> str:
    if isinstance(g, ic.Perms):
        return f"(.perms {lean_perms(g)})"
    return f"(.mats {g.p} {L(g.tolist())})"


def lean_orbits_family(f: ic.Family) -> str:
    if f.kind == "group_elements":
        (gens,) = f.children
        return f"(.groupElements {lean_perms(gens)})"
    return f"(.gfp {lean_family(f)})"


def lean_op(op: str, args: dict) -> str:
    name = op.removeprefix("orbits.")
    if name == "fixed_points":
        return f"(.fixedPoints {lean_family(args['on'])})"
    if name == "projective_action":
        pts = args["points"]
        vecs = [m[0] for m in pts.tolist()] if pts.rows == 1 else pts.member(0)
        return f"(.projectiveAction {L(vecs)})"
    ctor = {"is_canonical": "isCanonical", "canonical_index": "canonicalIndex", "orbit_size": "orbitSize",
            "stabilizer_order": "stabilizerOrder"}[name]
    return f"(.{ctor} {lean_group(args['group'])})"


def lean_result(r) -> str:
    if isinstance(r, ic.Integers):
        return f".integers {L(r.values)}"
    if isinstance(r, ic.Count):
        assert r.visited == r.family_size
        return f".count {r.value} {r.family_size}"
    if isinstance(r, ic.Histogram):
        assert r.visited == r.family_size
        return f".histogram {r.family_size} {L(r.bins)}"
    if isinstance(r, ic.Hits):
        assert r.visited == r.family_size and r.total == len(r.indices)
        return f".hits {r.family_size} {L(r.indices)} {L(r.members.tolist())}"
    if isinstance(r, ic.Perms):
        return f".perms {lean_perms(r)}"
    raise TypeError(type(r))


def run_claim(op, family_desc, red, args):
    return f"run {lean_op(op, args)} {lean_orbits_family(family_desc)} {lean_red(red, args)}"


# ---- groups and families small enough for the kernel --------------------------------------------

def cyclic(n):
    return [[(i + 1) % n for i in range(n)]]


def dihedral(n):
    return [[(i + 1) % n for i in range(n)], [(-i) % n for i in range(n)]]


def symmetric(n):
    return [[(i + 1) % n for i in range(n)], [1, 0] + list(range(2, n))]


def unit_vectors(p, n):
    return lk.matrix(p, [[int(i == j) for j in range(n)] for i in range(n)])


def perm_cases(ctx, rng):
    """(name, group handle, family handle) with permutation groups on subsets."""
    I6, I5, I4 = unit_vectors(2, 6), unit_vectors(3, 5), unit_vectors(7, 4)
    D4 = random_batch(rng, 5, 4, 1, 3)
    return [
        ("D6 on 2-subsets", ctx.perms(6, dihedral(6)), ctx.subsets(I6, 2)),
        ("D6 on 3-subsets", ctx.perms(6, dihedral(6)), ctx.subsets(I6, 3)),
        ("C5 on 2-subsets", ctx.perms(5, cyclic(5)), ctx.subsets(I5, 2)),
        ("S4 on 2-subsets", ctx.perms(4, symmetric(4)), ctx.subsets(I4, 2)),
        ("S4 on random dictionary", ctx.perms(4, symmetric(4)), ctx.subsets(D4, 2)),
        ("trivial on 1-subsets", ctx.perms(4, [list(range(4))]), ctx.subsets(I4, 1)),
        ("swap on 4-subsets", ctx.perms(6, [[1, 0, 2, 3, 4, 5]]), ctx.subsets(I6, 4)),
    ]


SINGER3 = [[0, 1, 0], [0, 0, 1], [1, 1, 0]]     # x^3 = x + 1 over F_2: order 7 on the 7 points
FROB3 = [[1, 0, 0], [0, 0, 1], [0, 1, 1]]       # Frobenius in the same basis; together order 21


def matrix_cases(ctx):
    """(name, generator batch, family handle) with matrix groups acting on the right."""
    GL22 = lk.matrix(2, [[[1, 1], [0, 1]], [[0, 1], [1, 0]]])          # order 6
    FIB3 = lk.matrix(3, [[[0, 1], [1, 1]]])                             # order 8
    DIAG5 = lk.matrix(5, [[[2, 0], [0, 3]]])                            # order 4
    SINGER = lk.matrix(2, [SINGER3, FROB3])
    return [
        ("GL(2,2) on points", GL22, ctx.grassmannian(2, 2, 1)),
        ("GL(2,2) on 1x2", GL22, ctx.all_matrices(2, 1, 2)),
        ("GL(2,2) on 2x2", GL22, ctx.all_matrices(2, 2, 2)),
        ("<Fib> on PG(1,3)", FIB3, ctx.grassmannian(3, 2, 1)),
        ("<Fib> on 1x2 over F3", FIB3, ctx.all_matrices(3, 1, 2)),
        ("<diag(2,3)> on PG(1,5)", DIAG5, ctx.grassmannian(5, 2, 1)),
        ("Singer normaliser on PG(2,2) points", SINGER, ctx.grassmannian(2, 3, 1)),
        ("Singer normaliser on PG(2,2) lines", SINGER, ctx.grassmannian(2, 3, 2)),
        ("Singer normaliser on 1x3", SINGER, ctx.all_matrices(2, 1, 3)),
    ]


OPS = [("orbits.is_canonical", ["all", "count", "hits"]), ("orbits.canonical_index", ["all", "histogram"]),
       ("orbits.orbit_size", ["all", "histogram"]), ("orbits.stabilizer_order", ["all", "histogram"])]


def args_for(red, group):
    return {"group": group, "limit": 3} if red == "hits" else {"group": group}


# ---- tests --------------------------------------------------------------------------------------

@pytest.mark.parametrize("backend", BACKENDS)
def test_perm_actions(backend):
    ctx = lk.Context(backend)
    lc = LeanCheck(f"orbits_perms_{backend}", IMPORTS, OPENS)
    for name, G, F in perm_cases(ctx, random.Random(1)):
        desc, grp = F.value(), G.value()
        for op, reductions in OPS:
            for red in reductions:
                args = args_for(red, grp)
                got = ctx.run(op, F, red, **args).value()
                lc.claim(run_claim(op, desc, red, args), lean_result(got), f"{name} {op}/{red}")
    lc.verify()


@pytest.mark.parametrize("backend", BACKENDS)
def test_matrix_actions(backend):
    ctx = lk.Context(backend)
    lc = LeanCheck(f"orbits_mats_{backend}", IMPORTS, OPENS)
    for name, A, F in matrix_cases(ctx):
        desc = F.value()
        for op, reductions in OPS:
            for red in reductions:
                args = args_for(red, A)
                got = ctx.run(op, F, red, **args).value()
                lc.claim(run_claim(op, desc, red, args), lean_result(got), f"{name} {op}/{red}")
    lc.verify()


@pytest.mark.parametrize("backend", BACKENDS)
def test_group_elements_and_fixed_points(backend):
    ctx = lk.Context(backend)
    lc = LeanCheck(f"orbits_fixed_{backend}", IMPORTS, OPENS)
    for name, G, F in perm_cases(ctx, random.Random(2)):
        E = ctx.group_elements(G)
        elems = [ctx.member(E, i).value().member(0) for i in range(ctx.size(E))]
        lc.claim(f"permElements {lean_perms(G.value())}", L(elems), f"elements of {name}")
        desc = E.value()
        for red in ("all", "histogram"):
            got = ctx.run("orbits.fixed_points", E, red, on=F).value()
            lc.claim(run_claim("orbits.fixed_points", desc, red, {"on": F.value()}), lean_result(got), f"{name} fixed_points/{red}")
    lc.verify()


@pytest.mark.parametrize("backend", BACKENDS)
def test_projective_action(backend):
    ctx = lk.Context(backend)
    lc = LeanCheck(f"orbits_projective_{backend}", IMPORTS, OPENS)
    for p, n, mats in [(2, 3, [SINGER3, FROB3]), (3, 2, [[[0, 1], [1, 1]], [[2, 0], [0, 1]]]), (5, 2, [[[2, 0], [0, 3]], [[1, 1], [0, 1]]])]:
        pts = lk.matrix(p, [m[0] for m in ctx.value("gfp.rref", ctx.grassmannian(p, n, 1)).tolist()])
        F = ctx.explicit(lk.matrix(p, mats))
        got = ctx.run("orbits.projective_action", F, points=pts).value()
        lc.claim(run_claim("orbits.projective_action", F.value(), "all", {"points": pts}), lean_result(got), f"PG({n - 1},{p})")
        # the induced permutations act on subsets of the points exactly as the matrices do on the Grassmannian
        P = ctx.put(got)
        S = ctx.subsets(pts, 2)
        c = ctx.run("orbits.is_canonical", S, "count", group=P).value()
        lc.claim(run_claim("orbits.is_canonical", S.value(), "count", {"group": got}), lean_result(c), f"PG({n - 1},{p}) pairs")
    lc.verify()


def test_naive_matches_lean():
    """The benchmark baseline is held to the same oracle."""
    ctx = lk.Context()
    lc = LeanCheck("orbits_naive", IMPORTS, OPENS)
    for name, G, F in perm_cases(ctx, random.Random(3))[:4] + [(n, A, F) for n, A, F in matrix_cases(ctx)][:4]:
        desc, grp = F.value(), G.value() if isinstance(G, lk.Handle) else G
        for op, reductions in OPS:
            for red in reductions:
                args = args_for(red, grp)
                lc.claim(run_claim(op, desc, red, args), lean_result(naive.run(op, desc, red, **args)), f"naive {name} {op}/{red}")
    G, F = perm_cases(ctx, random.Random(3))[0][1:]
    E = ctx.group_elements(G)
    lc.claim(run_claim("orbits.fixed_points", E.value(), "all", {"on": F.value()}),
             lean_result(naive.run("orbits.fixed_points", E.value(), "all", on=F.value())), "naive fixed_points")
    pts = lk.matrix(2, [m[0] for m in ctx.value("gfp.rref", ctx.grassmannian(2, 3, 1)).tolist()])
    F = ctx.explicit(lk.matrix(2, [SINGER3, FROB3]))
    lc.claim(run_claim("orbits.projective_action", F.value(), "all", {"points": pts}),
             lean_result(naive.run("orbits.projective_action", F.value(), "all", points=pts)), "naive projective_action")
    lc.verify()


@pytest.mark.parametrize("backend", BACKENDS)
def test_burnside(backend):
    """Orbit counts agree with the Cauchy-Frobenius count on cases too large for the kernel oracle."""
    ctx = lk.Context(backend)
    I8, I7 = unit_vectors(2, 8), unit_vectors(3, 7)
    for G, F in [(ctx.perms(8, dihedral(8)), ctx.subsets(I8, 4)), (ctx.perms(7, symmetric(7)), ctx.subsets(I7, 3)),
                 (ctx.perms(8, [[1, 2, 3, 0, 5, 6, 7, 4], [4, 5, 6, 7, 0, 1, 2, 3]]), ctx.subsets(I8, 3))]:
        E = ctx.group_elements(G)
        fixed = ctx.value("orbits.fixed_points", E, on=F).values
        orbits = ctx.value("orbits.is_canonical", F, "count", group=G).value
        assert sum(fixed) == orbits * ctx.size(E)
        sizes = ctx.value("orbits.orbit_size", F, group=G).values
        stabs = ctx.value("orbits.stabilizer_order", F, group=G).values
        assert all(s * t == ctx.size(E) for s, t in zip(sizes, stabs))
        assert sum(1 / s for s in sizes) == pytest.approx(orbits)
    # GL(3,2) on the 35 lines-or-triangles of PG(2,2): 2 orbits, stabilisers 24 and 6
    A = lk.matrix(2, [SINGER3, [[1, 1, 0], [0, 1, 0], [0, 0, 1]]])
    pts = lk.matrix(2, [m[0] for m in ctx.value("gfp.rref", ctx.grassmannian(2, 3, 1)).tolist()])
    P = ctx.put(ctx.value("orbits.projective_action", ctx.explicit(A), points=pts))
    S = ctx.subsets(pts, 3)
    assert ctx.value("orbits.is_canonical", S, "count", group=P).value == 2
    assert sorted(set(ctx.value("orbits.stabilizer_order", S, group=P).values)) == [6, 24]
    assert ctx.size(ctx.group_elements(P)) == 168


def test_errors_are_specific():
    ctx = lk.Context()
    I6 = unit_vectors(2, 6)
    S = ctx.subsets(I6, 2)
    with pytest.raises(lk.Error, match="cannot act on a dictionary"):
        ctx.run("orbits.is_canonical", S, "count", group=ctx.perms(5, cyclic(5)))
    with pytest.raises(lk.Error, match="subsets"):
        ctx.run("orbits.orbit_size", ctx.grassmannian(2, 3, 1), group=ctx.perms(7, cyclic(7)))
    with pytest.raises(lk.Error, match="n x n"):
        ctx.run("orbits.orbit_size", ctx.grassmannian(2, 3, 1), group=lk.matrix(2, [[[1, 0], [0, 1]]]))
    with pytest.raises(ValueError, match="not a permutation"):
        ctx.perms(3, [[0, 0, 1]])
    with pytest.raises(lk.Error, match="singular"):
        ctx.run("orbits.orbit_size", ctx.grassmannian(2, 2, 1), group=lk.matrix(2, [[[1, 1], [1, 1]]]))
    with pytest.raises(lk.Error, match="outside the dictionary"):
        pts = lk.matrix(2, [[1, 0], [0, 1]])
        ctx.run("orbits.projective_action", ctx.explicit(lk.matrix(2, [[[1, 1], [0, 1]]])), points=pts)
    with pytest.raises(lk.Error, match="does not accept"):
        ctx.run("orbits.is_canonical", S, "histogram", group=ctx.perms(6, cyclic(6)))
    with pytest.raises(lk.Error, match="families"):
        ctx.run("orbits.fixed_points", S, on=S)


def test_threads_do_not_change_answers():
    ctx = lk.Context()
    I9 = unit_vectors(2, 9)
    S = ctx.subsets(I9, 4)
    G = ctx.perms(9, symmetric(9))
    answers = set()
    for threads in (1, 3, 32):
        ctx.threads = threads
        answers.add(ctx.run("orbits.is_canonical", S, "hits", group=G, limit=5).export())
        answers.add(ctx.run("orbits.orbit_size", S, group=G).export())
    assert len(answers) == 2
