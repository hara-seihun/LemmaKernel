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
    with pytest.raises(lk.Error, match="not prime"):
        ctx.matrix(4, [[1, 2]])
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
