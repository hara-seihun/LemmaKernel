"""gfp: every backend must reproduce the naive implementation byte for byte.

Run from the repository root with `pytest modules/gfp` after building (see README).
"""
from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
import lemmakernel as lk  # noqa: E402

_spec = importlib.util.spec_from_file_location("gfp_naive", ROOT / "modules" / "gfp" / "naive" / "naive.py")
naive = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(naive)

BACKENDS = [b for b in lk.describe()["available_backends"] if b.startswith("gfp.")]
PRIMES = [2, 3, 5, 7, 251, 257, 65521, 65537, 4294967291]


def random_batch(rng, p, count, rows, cols):
    """Batches with a mix of generic, singular and structured members."""
    mats = []
    for i in range(count):
        m = [[rng.randrange(p) for _ in range(cols)] for _ in range(rows)]
        if i % 4 == 1 and rows > 1:  # duplicate a row
            m[-1] = list(m[0])
        if i % 4 == 2:  # a zero row and a zero column
            m[0] = [0] * cols
            for r in m:
                r[-1] = 0
        if i % 4 == 3:  # sparse
            m = [[rng.randrange(p) if rng.random() < 0.3 else 0 for _ in range(cols)] for _ in range(rows)]
        mats.append(m)
    return lk.matrix(p, mats)


def families(ctx, rng, p):
    """(name, handle, naive Family) triples covering every family kind, small enough to enumerate."""
    out = []
    batch = random_batch(rng, p, 12, 3, 4)
    out.append(("explicit", ctx.explicit(batch)))
    dictionary = random_batch(rng, p, 7, 1, 4)
    out.append(("subsets", ctx.subsets(dictionary, 3)))
    # Sizes stay under ~10^5 so the naive side finishes; the shapes shrink as p grows.
    if p <= 3:
        out.append(("grassmannian", ctx.grassmannian(p, 5, 2)))
        out.append(("all_matrices", ctx.all_matrices(p, 2, 3)))
        G = ctx.grassmannian(p, 4, 2)
    elif p <= 7:
        out.append(("grassmannian", ctx.grassmannian(p, 4, 2)))
        out.append(("all_matrices", ctx.all_matrices(p, 2, 2)))
        G = ctx.grassmannian(p, 4, 2)
    elif p < 1 << 12:
        out.append(("grassmannian", ctx.grassmannian(p, 2, 1)))
        out.append(("all_matrices", ctx.all_matrices(p, 1, 1)))
        G = ctx.subsets(random_batch(rng, p, 8, 1, 4), 2)
    else:
        G = ctx.subsets(random_batch(rng, p, 8, 1, 4), 2)
    C = random_batch(rng, p, 1, 4, 3)
    out.append(("transform", ctx.transform(G, C)))
    extra = random_batch(rng, p, 1, 2, 4)
    out.append(("stack", ctx.stack(G, extra)))
    out.append(("stack(transform)", ctx.stack(ctx.transform(G, C), random_batch(rng, p, 1, 1, 3))))
    return [(name, h, h.value()) for name, h in out]


def check_same(kernel: lk.Handle, expected) -> None:
    got = kernel.export()
    want = expected.encode()
    if got != want:
        pytest.fail(f"kernel {kernel.value()!r}\n!= naive {expected!r}")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("p", PRIMES)
def test_family_operations_match_naive(backend, p):
    rng = random.Random(p)
    ctx = lk.Context(backend)
    for name, fam, desc in families(ctx, rng, p):
        cols = fam.param("cols")
        target = random_batch(rng, p, 1, 1, cols)
        for op, reductions, args in [
            ("gfp.rank", ["all", "histogram"], {}),
            ("gfp.nullity", ["all", "histogram"], {}),
            ("gfp.full_row_rank", ["all", "count", "hits"], {}),
            ("gfp.full_col_rank", ["all", "count", "hits"], {}),
            ("gfp.in_span", ["all", "count", "hits"], {"target": target}),
            ("gfp.rref", ["all"], {}),
            ("gfp.nullspace", ["all"], {}),
        ]:
            for red in reductions:
                extra = {"limit": 3} if red == "hits" else {}
                got = ctx.run(op, fam, red, **args, **extra)
                want = naive.run(op, desc, red, **args, **extra)
                assert got.export() == want.encode(), f"{backend} {name} {op}/{red} p={p}: {got.value()!r} != {want!r}"


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("p", PRIMES)
def test_explicit_operations_match_naive(backend, p):
    rng = random.Random(p * 7)
    ctx = lk.Context(backend)
    for rows, cols in [(1, 1), (2, 3), (3, 3), (4, 2), (5, 5)]:
        batch = random_batch(rng, p, 16, rows, cols)
        fam = ctx.explicit(batch)
        desc = fam.value()
        check_same(ctx.run("gfp.rref_witness", fam), naive.run("gfp.rref_witness", desc))
        if rows == cols:
            check_same(ctx.run("gfp.inverse", fam), naive.run("gfp.inverse", desc))
        rhs = random_batch(rng, p, 16, 1, rows)
        check_same(ctx.run("gfp.solve", fam, rhs=rhs), naive.run("gfp.solve", desc, rhs=rhs))


@pytest.mark.parametrize("p", [2, 3, 7])
def test_member_order_matches_naive(p):
    """lk_family_member(i) is the i-th member of the naive enumeration."""
    ctx = lk.Context()
    rng = random.Random(p)
    for name, fam, desc in families(ctx, rng, p):
        ms, _ = naive.members(desc)
        assert ctx.size(fam) == len(ms), name
        for i in range(len(ms)):
            assert ctx.member(fam, i).value().member(0) == ms[i], f"{name} member {i}"


def test_witness_and_inverse_are_verified_by_multiplication():
    p = 7
    rng = random.Random(1)
    ctx = lk.Context()
    batch = random_batch(rng, p, 20, 4, 4)
    fam = ctx.explicit(batch)
    w = ctx.value("gfp.rref_witness", fam)
    inv = ctx.value("gfp.inverse", fam)
    for i in range(20):
        A = batch.member(i)
        R, T = w.member(i)
        assert naive.matmul(T, A, p) == R
        assert naive.inverse(T, p) is not None
        B = inv.member(i)
        if B is not None:
            assert naive.matmul(A, B, p) == [[int(a == b) for b in range(4)] for a in range(4)]


def test_solutions_satisfy_the_system():
    p = 5
    rng = random.Random(2)
    ctx = lk.Context()
    batch = random_batch(rng, p, 30, 3, 4)
    rhs = random_batch(rng, p, 30, 1, 3)
    sol = ctx.value("gfp.solve", ctx.explicit(batch), rhs=rhs)
    for i in range(30):
        x = sol.member(i)
        if x is not None:
            A = batch.member(i)
            assert [sum(a * b for a, b in zip(r, x)) % p for r in A] == rhs.member(i)[0]


def test_completeness_fields():
    ctx = lk.Context()
    G = ctx.grassmannian(2, 7, 3)
    c = ctx.value("gfp.full_col_rank", G, "count")
    assert c.value == 0 and c.visited == c.family_size == 11811
    h = ctx.value("gfp.rank", G, "histogram")
    assert h.bins == [0, 0, 0, 11811]
    hits = ctx.value("gfp.in_span", G, "hits", target=lk.matrix(2, [[1, 1, 1, 1, 1, 1, 1]]), limit=2)
    assert hits.total == 651 and hits.visited == 11811 and hits.members.count == 2
    for k, i in enumerate(hits.indices[:2]):
        assert hits.members.member(k) == ctx.member(G, i).value().member(0)


def test_roundtrip_of_families_and_results():
    ctx = lk.Context()
    G = ctx.grassmannian(3, 4, 2)
    F = ctx.stack(ctx.transform(G, lk.matrix(3, [[1, 0], [0, 1], [1, 1], [2, 1]])), lk.matrix(3, [[1, 2]]))
    again = ctx.load(F.export())
    assert again.export() == F.export()
    assert ctx.run("gfp.rank", again, "histogram").export() == ctx.run("gfp.rank", F, "histogram").export()
    r = ctx.run("gfp.rref", F)
    assert ctx.load(r.export()).export() == r.export()
    # a result batch is a valid explicit family
    assert ctx.value("gfp.rank", ctx.explicit(r), "histogram").bins == ctx.value("gfp.rank", F, "histogram").bins


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


def test_describe_lists_this_module():
    d = lk.describe()
    gfp = next(m for m in d["modules"] if m["module"]["name"] == "gfp")
    assert {o["name"] for o in gfp["operations"]} >= {"rank", "rref", "nullspace", "solve", "inverse", "rref_witness", "in_span"}
    assert "gfp.generic" in d["available_backends"]
