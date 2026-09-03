"""Runtime behaviour that belongs to no module: argument checking, backend pinning, describe."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import lemmakernel as lk  # noqa: E402


def test_request_checking_is_specific():
    ctx = lk.Context()
    G = ctx.grassmannian(2, 4, 2)
    with pytest.raises(lk.Error, match="missing argument target"):
        ctx.run("gfp.in_span", G, "count")
    with pytest.raises(lk.Error, match="unknown operation"):
        ctx.run("gfp.determinant", G)
    with pytest.raises(lk.Error, match="unexpected argument"):
        ctx.run("gfp.rank", G, "histogram", limit=3)
    assert ctx.matrix(4, [[1, 2]]).value().p == 4
    with pytest.raises(lk.Error, match="field-size tag"):
        ctx.matrix(1, [[0]])
    with pytest.raises(ValueError, match="not a permutation"):
        ctx.perms(3, [[0, 0, 1]])
    with pytest.raises(lk.Error):
        lk.Context("gfp.nonexistent")


def test_pinned_backend_applies_to_its_module_only():
    ctx = lk.Context("orbits.generic")
    assert ctx.value("gfp.rank", ctx.grassmannian(2, 3, 1), "histogram").bins == [0, 7]


def test_describe_lists_every_module():
    d = lk.describe()
    names = {m["module"]["name"] for m in d["modules"]}
    assert names == {m["module"]["name"] for m in lk.MODULES}
    for m in d["modules"]:
        assert any(b.startswith(m["module"]["name"] + ".") for b in d["available_backends"]), m["module"]["name"]
    assert {f["name"] for f in d["runtime"]["families"]} >= {"explicit", "subsets", "grassmannian", "all_matrices", "group_elements"}


def test_natural_number_families_enumerate_like_the_naive_layer():
    """Natural-number families agree with the shared naive enumeration member for member, and
    `explicit`, `subsets` and `subsets_of` carry naturals onwards to sum_free_and_additive."""
    from lemmakernel import naive as rt
    ctx = lk.Context()
    families = [ctx.range(10, 15), ctx.words(3, 3), ctx.range(0, 1), ctx.latin_squares(3),
                ctx.partitions(7), ctx.partitions(9, distinct=True, odd=True), ctx.compositions(6),
                ctx.compositions(8, parts=3, max_part=4),
                ctx.explicit(lk.naturals([[[3], [5]], [[4], [9]]])),
                ctx.subsets(lk.naturals([[2], [3], [5]]), 2),
                ctx.subsets_of(ctx.range(0, 4), 2)]
    for fam in families:
        desc = fam.value()
        expected, p = rt.members(desc)
        assert p == lk.NATURALS
        assert ctx.size(fam) == len(expected)
        for i, m in enumerate(expected):
            got = ctx.member(fam, i).value()
            assert got.p == lk.NATURALS and got.member(0) == m
        assert ctx.put(desc).export() == desc.encode()
    with pytest.raises(lk.Error, match="a < b"):
        ctx.range(5, 5)
    with pytest.raises(lk.Error, match="alphabet"):
        ctx.words(1, 3)
    with pytest.raises(lk.Error, match="1 <= n <= 5"):
        ctx.latin_squares(6)
    with pytest.raises(lk.Error, match="no partition"):
        ctx.partitions(3, max_part=2, max_parts=1)
    with pytest.raises(lk.Error, match="no composition"):
        ctx.compositions(5, parts=2, max_part=2)
    # naturals never enter F_p arithmetic
    with pytest.raises(lk.Error, match="no available backend"):
        ctx.run("gfp.rank", ctx.words(2, 3))
    with pytest.raises(lk.Error, match="over a prime"):
        ctx.transform(ctx.words(2, 3), ctx.naturals([[1, 0], [0, 1], [1, 1]]))


def test_first_stops_early_and_is_thread_invariant():
    """The first invertible 4x4 over F_2 is lexicographically the reversed identity, at index 4680
    of 65536; `visited` is index + 1 whatever the thread count, and a family without a hit reports
    found = 0 after visiting everything."""
    results = []
    for t in (1, 4, 32):
        c = lk.Context(threads=t)
        results.append(c.value("gfp.full_col_rank", c.all_matrices(2, 4, 4), "first"))
    for r in results:
        assert (r.found, r.index, r.visited, r.family_size) == (1, 4680, 4681, 65536)
        assert r.member.member(0) == [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]
    assert len({r.encode() for r in results}) == 1
    ctx = lk.Context()
    none = ctx.value("gfp.full_col_rank", ctx.grassmannian(2, 4, 1), "first")
    assert (none.found, none.visited, none.family_size) == (0, 15, 15)


def test_sum_max_min_reductions():
    ctx = lk.Context()
    G = ctx.grassmannian(2, 4, 2)
    assert ctx.value("gfp.rank", G, "sum").value == 2 * 35
    hi = ctx.value("gfp.nullity", ctx.all_matrices(3, 2, 2), "max")
    assert (hi.value, hi.index, hi.member.member(0)) == (2, 0, [[0, 0], [0, 0]])
    lo = ctx.value("gfp.nullity", ctx.all_matrices(3, 2, 2), "min")
    assert (lo.value, lo.index) == (0, 12)  # [[0,1],[1,0]] is the first invertible 2x2 over F_3
    assert lo.member.member(0) == [[0, 1], [1, 0]]


def test_families_past_two_to_the_64():
    """Family sizes, member indices and everything counted over a family are 128-bit. C(152, 15)
    is 2.0e20: the family constructs, unranks and ranks at the top end, its size survives a
    round trip through the encoding, a backend that walks 128-bit indices settles a `first` and
    a `count` over such a family, and a backend that keeps 64-bit indices is refused rather
    than handed a truncated family."""
    from math import comb

    ctx = lk.Context(threads=8)
    fam = ctx.subsets_of(ctx.range(0, 152), 15)
    size = comb(152, 15)
    assert size > 1 << 64
    assert ctx.size(fam) == size
    assert fam.params["size"] == size
    assert ctx.load(fam.export()).params["size"] == size
    last = ctx.member(fam, size - 1)
    assert last.value().tolist() == [[[v] for v in range(137, 152)]]
    with pytest.raises(lk.Error):
        ctx.member(fam, size)

    # A 2-term progression is any pair, so no 20-subset of [0, 150] is 2-AP-free: every subtree
    # is refused at its second element and the count decides 4.2e24 members without a visit.
    none = ctx.value("sum_free_and_additive.is_ap_free", ctx.subsets_of(ctx.range(0, 151), 20), "count",
                     modulus=0, length=2)
    assert (none.value, none.visited, none.family_size) == (0, comb(151, 20), comb(151, 20))
    hit = ctx.value("sum_free_and_additive.is_sidon", ctx.subsets_of(ctx.range(0, 221), 13), "first", modulus=0)
    assert hit.found == 1 and hit.family_size == comb(221, 13) and hit.visited == hit.index + 1
    assert ctx.load(hit.encode()).value().index == hit.index
    with pytest.raises(lk.Error, match="64 bits"):
        ctx.run("gfp.full_col_rank", ctx.all_matrices(2, 9, 8), "count")
