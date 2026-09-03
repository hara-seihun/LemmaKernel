"""gfp: every backend's answer must be accepted by Lean's `decide +kernel` against the reference.

No expected answers live in this file. Each test builds inputs, runs the kernel, and states the
kernel's answer as a Lean `example` over `Gfp.run` (lean/Gfp/Reference.lean); Lean evaluates the
reference and accepts or rejects. The naive Python implementation is checked the same way, so the
benchmark baseline is also verified.

Run from the repository root after building: `pytest -n auto modules/gfp`.
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

_spec = importlib.util.spec_from_file_location("gfp_naive", ROOT / "modules" / "gfp" / "naive" / "naive.py")
naive = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(naive)

BACKENDS = [b for b in lk.describe()["available_backends"] if b.startswith("gfp.")]


# ---- interchange objects as Lean terms ----------------------------------------------------------

def L(x) -> str:
    """Lean literal for nested lists of ints / Options / tuples."""
    if x is None:
        return "none"
    if isinstance(x, tuple):
        return "(" + ", ".join(L(v) for v in x) + ")"
    if isinstance(x, (list,)):
        return "[" + ", ".join(L(v) for v in x) + "]"
    if hasattr(x, "tolist"):
        return L(x.tolist()) if not isinstance(x, ic.Matrix) else L(x.tolist())
    return str(int(x))


def lean_family(f: ic.Family) -> str:
    q = f.params
    if f.kind == "explicit":
        (b,) = f.children
        return f"(.explicit {b.p} {L(b.tolist())})"
    if f.kind == "subsets":
        (d,) = f.children
        vecs = [m[0] for m in d.tolist()] if d.rows == 1 else d.member(0)
        return f"(.subsets {d.p} {L(vecs)} {q['k']})"
    if f.kind == "grassmannian":
        return f"(.grassmannian {q['p']} {q['n']} {q['h']})"
    if f.kind == "all_matrices":
        return f"(.allMatrices {q['p']} {q['rows']} {q['cols']})"
    if f.kind == "transform":
        inner, c = f.children
        return f"(.transform {lean_family(inner)} {L(c.member(0))})"
    if f.kind == "stack":
        inner, rows = f.children
        return f"(.stack {lean_family(inner)} {L(rows.member(0))})"
    raise ValueError(f.kind)


def lean_op(op: str, args: dict) -> str:
    name = op.removeprefix("gfp.")
    if name == "in_span":
        return f"(.inSpan {L(args['target'].member(0)[0])})"
    if name == "solve":
        return f"(.solve {L([m[0] for m in args['rhs'].tolist()])})"
    return {"rank": ".rank", "nullity": ".nullity", "full_row_rank": ".fullRowRank", "full_col_rank": ".fullColRank",
            "rref": ".rref", "nullspace": ".nullspace", "inverse": ".inverse", "rref_witness": ".rrefWitness"}[name]


def lean_red(red: str, args: dict) -> str:
    return f"(.hits {args['limit']})" if red == "hits" else "." + red


def lean_result(r) -> str:
    if isinstance(r, ic.Integers):
        return f".integers {L(r.values)}"
    if isinstance(r, ic.Count):
        assert r.visited == r.family_size, "incomplete enumeration reported"
        return f".count {r.value} {r.family_size}"
    if isinstance(r, ic.Histogram):
        assert r.visited == r.family_size, "incomplete enumeration reported"
        return f".histogram {r.family_size} {L(r.bins)}"
    if isinstance(r, ic.Hits):
        assert r.visited == r.family_size and r.total == len(r.indices)
        return f".hits {r.family_size} {L(r.indices)} {L(r.members.tolist())}"
    if isinstance(r, ic.Matrix):
        return f".matrices {L(r.tolist())}"
    if isinstance(r, ic.Basis):
        return f".bases {L([r.member(i) for i in range(r.count)])}"
    if isinstance(r, ic.Solutions):
        return f".solutions {L([r.member(i) for i in range(r.count)])}"
    if isinstance(r, ic.Inverses):
        return f".inverses {L([r.member(i) for i in range(r.count)])}"
    if isinstance(r, ic.Witness):
        return f".witnesses {L([r.member(i) for i in range(r.count)])}"
    raise TypeError(type(r))


def run_claim(op, family_desc, red, args):
    return f"run {lean_op(op, args)} {lean_family(family_desc)} {lean_red(red, args)}"


# ---- inputs -------------------------------------------------------------------------------------

def random_batch(rng, p, count, rows, cols):
    """Batches with a mix of generic, singular and structured members."""
    mats = []
    for i in range(count):
        m = [[rng.randrange(p) for _ in range(cols)] for _ in range(rows)]
        if i % 4 == 1 and rows > 1:
            m[-1] = list(m[0])
        if i % 4 == 2:
            m[0] = [0] * cols
            for r in m:
                r[-1] = 0
        if i % 4 == 3:
            m = [[rng.randrange(p) if rng.random() < 0.3 else 0 for _ in range(cols)] for _ in range(rows)]
        mats.append(m)
    return lk.matrix(p, mats)


PRIMES = [2, 3, 5, 7, 251, 257, 65521, 65537, 4294967291]


def small_families(ctx, rng, p):
    """Every family kind at a size Lean evaluates in seconds. Returns (name, handle) pairs."""
    out = [("explicit", ctx.explicit(random_batch(rng, p, 8, 3, 4))),
           ("subsets", ctx.subsets(random_batch(rng, p, 6, 1, 4), 3))]
    if p == 2:
        out += [("grassmannian", ctx.grassmannian(2, 4, 2)), ("all_matrices", ctx.all_matrices(2, 2, 2))]
        G = ctx.grassmannian(2, 4, 2)
    elif p == 3:
        out += [("grassmannian", ctx.grassmannian(3, 3, 1)), ("all_matrices", ctx.all_matrices(3, 1, 2))]
        G = ctx.subsets(random_batch(rng, p, 5, 1, 4), 2)
    elif p < 20:
        out += [("grassmannian", ctx.grassmannian(p, 2, 1)), ("all_matrices", ctx.all_matrices(p, 1, 1))]
        G = ctx.subsets(random_batch(rng, p, 5, 1, 4), 2)
    else:
        G = ctx.subsets(random_batch(rng, p, 5, 1, 4), 2)
    out += [("transform", ctx.transform(G, random_batch(rng, p, 1, 4, 3))),
            ("stack", ctx.stack(G, random_batch(rng, p, 1, 2, 4))),
            ("stack(transform)", ctx.stack(ctx.transform(G, random_batch(rng, p, 1, 4, 2)), random_batch(rng, p, 1, 1, 2)))]
    return out


WALK_OPS = [("gfp.rank", ["all", "histogram"]), ("gfp.nullity", ["all", "histogram"]),
            ("gfp.full_row_rank", ["all", "count", "hits"]), ("gfp.full_col_rank", ["all", "count", "hits"]),
            ("gfp.in_span", ["all", "count", "hits"]), ("gfp.rref", ["all"]), ("gfp.nullspace", ["all"])]


def walk_args(rng, op, red, p, cols):
    args = {}
    if op == "gfp.in_span":
        args["target"] = random_batch(rng, p, 1, 1, cols)
    if red == "hits":
        args["limit"] = 3
    return args


# ---- tests --------------------------------------------------------------------------------------

@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("p", PRIMES)
def test_walk_operations(backend, p):
    rng = random.Random(p)
    ctx = lk.Context(backend)
    lc = LeanCheck(f"gfp_walk_{backend}_{p}", ["Gfp.Reference"], ["Gfp"])
    for name, fam in small_families(ctx, rng, p):
        desc = fam.value()
        cols = fam.param("cols")
        for op, reductions in WALK_OPS:
            for red in reductions:
                args = walk_args(rng, op, red, p, cols)
                got = ctx.run(op, fam, red, **args).value()
                lc.claim(run_claim(op, desc, red, args), lean_result(got), f"{name} {op}/{red}")
    lc.verify()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("p", PRIMES)
def test_explicit_operations(backend, p):
    rng = random.Random(p * 7)
    ctx = lk.Context(backend)
    lc = LeanCheck(f"gfp_explicit_{backend}_{p}", ["Gfp.Reference"], ["Gfp"])
    for rows, cols in [(1, 1), (2, 3), (3, 3), (4, 2), (5, 5)]:
        batch = random_batch(rng, p, 8, rows, cols)
        fam = ctx.explicit(batch)
        desc = fam.value()
        lc.claim(run_claim("gfp.rref_witness", desc, "all", {}), lean_result(ctx.run("gfp.rref_witness", fam).value()), f"{rows}x{cols} witness")
        if rows == cols:
            lc.claim(run_claim("gfp.inverse", desc, "all", {}), lean_result(ctx.run("gfp.inverse", fam).value()), f"{rows}x{cols} inverse")
        rhs = random_batch(rng, p, 8, 1, rows)
        lc.claim(run_claim("gfp.solve", desc, "all", {"rhs": rhs}), lean_result(ctx.run("gfp.solve", fam, rhs=rhs).value()), f"{rows}x{cols} solve")
    lc.verify()


@pytest.mark.parametrize("p", [2, 3, 7, 251])
def test_naive_matches_lean(p):
    """The benchmark baseline is held to the same oracle."""
    rng = random.Random(p)
    ctx = lk.Context()
    lc = LeanCheck(f"gfp_naive_{p}", ["Gfp.Reference"], ["Gfp"])
    for name, fam in small_families(ctx, rng, p):
        desc = fam.value()
        cols = fam.param("cols")
        for op, reductions in WALK_OPS:
            for red in reductions:
                args = walk_args(rng, op, red, p, cols)
                lc.claim(run_claim(op, desc, red, args), lean_result(naive.run(op, desc, red, **args)), f"naive {name} {op}/{red}")
    batch = random_batch(rng, p, 6, 3, 3)
    desc = ctx.explicit(batch).value()
    rhs = random_batch(rng, p, 6, 1, 3)
    for op, args in [("gfp.rref_witness", {}), ("gfp.inverse", {}), ("gfp.solve", {"rhs": rhs})]:
        lc.claim(run_claim(op, desc, "all", args), lean_result(naive.run(op, desc, "all", **args)), f"naive {op}")
    lc.verify()


@pytest.mark.parametrize("p", [2, 3, 5])
def test_member_order(p):
    """lk_family_member enumerates in the order the reference defines."""
    rng = random.Random(p)
    ctx = lk.Context()
    lc = LeanCheck(f"gfp_members_{p}", ["Gfp.Reference"], ["Gfp"])
    for name, fam in small_families(ctx, rng, p):
        members = [ctx.member(fam, i).value().member(0) for i in range(ctx.size(fam))]
        lc.claim(f"Family.members {lean_family(fam.value())}", L(members), f"members of {name}")
    lc.verify()


def test_rejections_agree_with_reference():
    """What the runtime refuses, the reference calls invalid, and vice versa."""
    ctx = lk.Context()
    G = ctx.grassmannian(2, 4, 2)
    desc = G.value()
    lc = LeanCheck("gfp_rejections", ["Gfp.Reference"], ["Gfp"])
    cases = [("gfp.rank", "count", {}), ("gfp.rref", "histogram", {}), ("gfp.inverse", "all", {}),
             ("gfp.rref_witness", "all", {}), ("gfp.nullspace", "count", {})]
    for op, red, args in cases:
        with pytest.raises(lk.Error):
            ctx.run(op, G, red, **args)
        lc.claim(run_claim(op, desc, red, args), ".invalid", f"{op}/{red} rejected")
    lc.verify()


def test_errors_are_specific():
    ctx = lk.Context()
    G = ctx.grassmannian(2, 4, 2)
    with pytest.raises(lk.Error, match="does not accept integer"):
        ctx.run("gfp.rank", G, "count")
    with pytest.raises(lk.Error, match="missing argument target"):
        ctx.run("gfp.in_span", G, "count")
    with pytest.raises(lk.Error, match="explicit families only"):
        ctx.run("gfp.inverse", G)
    with pytest.raises(lk.Error, match="unknown operation"):
        ctx.run("gfp.determinant", G)
    with pytest.raises(lk.Error, match="unexpected argument"):
        ctx.run("gfp.rank", G, "histogram", limit=3)
    with pytest.raises(lk.Error, match="not prime"):
        ctx.matrix(4, [[1, 2]])
    with pytest.raises(lk.Error):
        lk.Context("gfp.nonexistent")


def test_roundtrip_and_composition():
    ctx = lk.Context()
    G = ctx.grassmannian(3, 4, 2)
    F = ctx.stack(ctx.transform(G, lk.matrix(3, [[1, 0], [0, 1], [1, 1], [2, 1]])), lk.matrix(3, [[1, 2]]))
    again = ctx.load(F.export())
    assert again.export() == F.export()
    assert ctx.run("gfp.rank", again, "histogram").export() == ctx.run("gfp.rank", F, "histogram").export()
    r = ctx.run("gfp.rref", F)
    assert ctx.load(r.export()).export() == r.export()
    # a result batch is a valid explicit family, and rref is idempotent on ranks
    assert ctx.value("gfp.rank", ctx.explicit(r), "histogram").bins == ctx.value("gfp.rank", F, "histogram").bins


def test_threads_do_not_change_answers():
    ctx = lk.Context()
    F = ctx.transform(ctx.grassmannian(2, 8, 3), lk.matrix(2, [[1, 0, 1, 1, 0], [0, 1, 1, 0, 0], [1, 1, 0, 0, 1], [0, 0, 1, 1, 1],
                                                              [1, 0, 0, 0, 1], [0, 1, 0, 1, 0], [1, 1, 1, 0, 0], [0, 0, 0, 1, 1]]))
    target = lk.matrix(2, [[1, 1, 0, 1, 0]])
    answers = set()
    for threads in (1, 3, 32):
        ctx.threads = threads
        answers.add(ctx.run("gfp.in_span", F, "hits", target=target, limit=5).export())
        answers.add(ctx.run("gfp.rank", F, "all").export())
    assert len(answers) == 2


def test_describe_lists_this_module():
    d = lk.describe()
    gfp = next(m for m in d["modules"] if m["module"]["name"] == "gfp")
    assert {o["name"] for o in gfp["operations"]} >= {"rank", "rref", "nullspace", "solve", "inverse", "rref_witness", "in_span"}
    assert "gfp.generic" in d["available_backends"]
