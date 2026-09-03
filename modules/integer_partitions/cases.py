"""Oracle and benchmark cases for constrained partitions and compositions."""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.harness import Case, module, rotate  # noqa: E402

COMMON_OPS = ["number_of_parts", "largest_part"]
PARTITION_OPS = COMMON_OPS + ["rank", "crank"]


def cases(ctx, rng):
    del rng
    partition_families = [
        ("partitions of 6", ctx.partitions(6)),
        ("distinct partitions of 9", ctx.partitions(9, distinct=True)),
        ("odd partitions of 9", ctx.partitions(9, odd=True)),
        ("parts at most 4", ctx.partitions(10, max_part=4)),
        ("at most 3 parts", ctx.partitions(10, max_parts=3)),
        ("multiplicity at most 2", ctx.partitions(9, max_multiplicity=2)),
        ("distinct odd partitions", ctx.partitions(15, distinct=True, odd=True, max_parts=4)),
    ]
    composition_families = [
        ("compositions of 6", ctx.compositions(6)),
        ("three-part compositions", ctx.compositions(8, parts=3)),
        ("bounded compositions", ctx.compositions(9, parts=4, max_part=4)),
    ]
    mod = module("integer_partitions")
    out = []
    for i, (name, family) in enumerate(partition_families):
        for j, op in enumerate(PARTITION_OPS):
            reductions = None if i == 0 else rotate(mod, op, i + j)
            out.append(Case(name, family, op, reductions=reductions))
    for i, (name, family) in enumerate(composition_families):
        for j, op in enumerate(COMMON_OPS):
            reductions = None if i == 0 else rotate(mod, op, i + j)
            out.append(Case(name, family, op, reductions=reductions))

    out.append(
        Case("partition_rank_50", lambda: ctx.partitions(50), "rank", reductions=["histogram"],
             bench="histogram", oracle=False,
             what="Dyson rank distribution of all 204,226 partitions of 50")
    )
    return out


def invariants(ctx):
    for total in (9, 15, 25):
        assert ctx.size(ctx.partitions(total, distinct=True)) == ctx.size(ctx.partitions(total, odd=True))
    assert ctx.size(ctx.compositions(20)) == 1 << 19
    assert ctx.size(ctx.compositions(20, parts=7)) == math.comb(19, 6)
    for op in ("rank", "crank"):
        bins = ctx.value(f"integer_partitions.{op}", ctx.partitions(25), "histogram").bins
        bins += [0] * (51 - len(bins))
        assert bins == list(reversed(bins))
