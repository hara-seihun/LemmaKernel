"""Oracle and benchmark inputs for additive-combinatorics predicates on sets of naturals."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lemmakernel as lk  # noqa: E402
from tools.harness import Case  # noqa: E402

BOOLEAN_OPS = ["is_sum_free", "is_sidon", "is_ap_free", "is_small_doubling"]
INTEGER_OPS = ["sumset_size", "difference_set_size", "schur_triple_count", "max_difference_multiplicity"]
OPS = BOOLEAN_OPS + INTEGER_OPS


def elements(values) -> lk.Matrix:
    """A dictionary of naturals, one element per row."""
    return lk.naturals([[v] for v in values])


def cases(ctx, rng):
    # Z/7, every 3-subset: 35 members, small enough for every operation under every reduction.
    # {0,1,3} is a Sidon set and a perfect difference set is out of reach at this size, {1,2,3}
    # is a progression, {0,2,4} wraps to one, so every predicate is exercised both ways.
    z7 = ctx.subsets_of(ctx.range(0, 7), 3)
    args = {"modulus": 7, "length": 3, "bound_num": 5, "bound_den": 1, "limit": 3}
    out = [Case("triples in Z/7", z7, op, args) for op in OPS]

    # The same shapes over the integers, where sums do not wrap: 4-subsets of 1..7.
    interval = ctx.subsets(elements(range(1, 8)), 4)
    iargs = {"modulus": 0, "length": 3, "bound_num": 5, "bound_den": 2, "limit": 2}
    for op in OPS:
        out.append(Case("quadruples in 1..7", interval, op, iargs,
                        reductions=["all", "first"] if op in BOOLEAN_OPS else ["all", "min"]))

    # Explicit sets of the same size, chosen for what they are: a planar difference set in Z/13,
    # a geometric progression, an arithmetic progression, the odd residues, and a sum-free set.
    explicit = ctx.explicit(lk.naturals([[[0], [1], [3], [9]], [[1], [2], [4], [8]],
                                         [[0], [3], [6], [9]], [[1], [3], [5], [7]],
                                         [[5], [6], [7], [8]]]))
    eargs = {"modulus": 13, "length": 4, "bound_num": 3, "bound_den": 1, "limit": 5}
    out += [Case("five sets in Z/13", explicit, op, eargs, reductions=["all"]) for op in OPS]

    # Long progressions in Z/8, where {0,2,4,6} is a 4-term progression and also a 5-term one,
    # because the fifth term wraps onto the first: only a nonzero difference is required.
    z8 = ctx.subsets_of(ctx.range(0, 8), 4)
    out += [
        Case("quadruples in Z/8", z8, "is_ap_free", {"modulus": 8, "length": 4, "limit": 2},
             reductions=["count", "first"]),
        Case("quadruples in Z/8", z8, "is_ap_free", {"modulus": 8, "length": 5, "limit": 2},
             reductions=["count"]),
        Case("quadruples in Z/8", z8, "max_difference_multiplicity", {"modulus": 8}, reductions=["max"]),
    ]

    # Dictionaries that take the other code paths of the generic backend: increasing but with
    # gaps (the span table applies, the allowed set is kept by index), increasing and symmetric
    # under reflection (the mirror rule counts each set with its reflection), and unsorted
    # (nothing about order can be assumed). Then longer progressions over an interval, where
    # only the last term of a progression is ever new.
    gapped = ctx.subsets(elements([0, 1, 3, 4, 6, 9, 10, 12]), 4)
    for op in BOOLEAN_OPS:
        out.append(Case("quadruples with gaps", gapped, op, iargs, reductions=["count", "hits", "first"]))
    symmetric = ctx.subsets(elements([0, 1, 3, 8, 10, 11]), 3)
    for op in ["is_sidon", "is_ap_free"]:
        out.append(Case("triples of a symmetric set", symmetric, op, iargs, reductions=["count", "all"]))
    unsorted = ctx.subsets(elements([5, 2, 7, 1, 4, 3, 6]), 3)
    for op in OPS:
        out.append(Case("triples, unsorted dictionary", unsorted, op, iargs,
                        reductions=["all", "first"] if op in BOOLEAN_OPS else ["all", "max"]))
    twelve = ctx.subsets_of(ctx.range(0, 12), 5)
    out += [
        Case("quintuples in 0..11", twelve, "is_ap_free", {"modulus": 0, "length": 4, "limit": 2},
             reductions=["count", "hits"]),
        Case("quintuples in 0..11", twelve, "is_ap_free", {"modulus": 0, "length": 5, "limit": 2},
             reductions=["count"]),
        Case("quintuples in 0..11", twelve, "is_sidon", {"modulus": 0, "limit": 2}, reductions=["count", "first"]),
        Case("quintuples in 0..11", twelve, "is_sum_free", {"modulus": 0, "limit": 2}, reductions=["count", "hits"]),
    ]

    # Extending a fixed context: sorted dictionaries above the context (the sorted walk with the
    # context pushed below level 0), a context above the dictionary and residues (the general
    # update), explicit members, and a context that already fails the predicate.
    ext = {"is_sum_free": "extends_sum_free", "is_sidon": "extends_sidon", "is_ap_free": "extends_ap_free"}
    low = ctx.naturals([[1, 2, 5]])
    above = ctx.subsets_of(ctx.range(6, 14), 3)
    for op in ext.values():
        out.append(Case("triples above {1,2,5}", above, op, {"modulus": 0, "length": 3, "context": low, "limit": 2},
                        reductions=["count", "hits", "first"]))
    high = ctx.naturals([[13, 7]])
    below = ctx.subsets_of(ctx.range(0, 7), 3)
    for op in ext.values():
        out.append(Case("triples below {7,13}", below, op, {"modulus": 0, "length": 3, "context": high, "limit": 2},
                        reductions=["count", "all"]))
        out.append(Case("triples in Z/7 with {1,3}", ctx.subsets(elements([0, 2, 4, 5, 6]), 3), op,
                        {"modulus": 7, "length": 3, "context": ctx.naturals([[1, 3]]), "limit": 2},
                        reductions=["count", "first"]))
        out.append(Case("five sets in Z/13 with {10}", explicit, op,
                        {"modulus": 13, "length": 4, "context": ctx.naturals([[10]])}, reductions=["all"]))
    out += [
        Case("context with a progression", above, "extends_ap_free",
             {"modulus": 0, "length": 3, "context": ctx.naturals([[1, 2, 3]]), "limit": 2},
             reductions=["count", "hits"]),
        Case("context that is not sum-free", above, "extends_sum_free",
             {"modulus": 0, "context": ctx.naturals([[1, 2, 3]])}, reductions=["count"]),
        Case("context meeting the dictionary", above, "extends_sidon",
             {"modulus": 0, "context": ctx.naturals([[2, 8]])}, ["count"], oracle=False),
        Case("context meeting a member", explicit, "extends_sidon",
             {"modulus": 13, "context": ctx.naturals([[2]])}, ["all"], oracle=False),
        Case("context with a repeat", above, "extends_sidon",
             {"modulus": 0, "context": ctx.naturals([[2, 2]])}, ["count"], oracle=False),
        Case("context outside Z/7", z7, "extends_sidon",
             {"modulus": 7, "context": ctx.naturals([[9]])}, ["count"], oracle=False),
    ]

    # Refusals: the family kind, the member shape, the dictionary, and the arguments.
    reject = {"modulus": 7, "length": 3, "bound_num": 5, "bound_den": 1}
    out += [
        Case("range of naturals", ctx.range(0, 5), "is_sum_free", {"modulus": 0}, ["count"], oracle=False),
        Case("residues are not naturals", ctx.subsets(lk.matrix(2, [[0], [1], [1]]), 2),
             "is_sum_free", {"modulus": 0}, ["count"], oracle=False),
        Case("two-column dictionary", ctx.subsets(lk.naturals([[1, 2], [3, 4], [5, 6]]), 2),
             "is_sum_free", {"modulus": 0}, ["count"], oracle=False),
        Case("repeated element", ctx.subsets(elements([1, 1, 2]), 2), "is_sum_free", {"modulus": 0},
             ["count"], oracle=False),
        Case("element outside Z/7", ctx.subsets(elements([1, 9]), 2), "is_sum_free", reject,
             ["count"], oracle=False),
        Case("progression shorter than two", z7, "is_ap_free", {"modulus": 7, "length": 1},
             ["count"], oracle=False),
        Case("zero denominator", z7, "is_small_doubling", {"modulus": 7, "bound_num": 5, "bound_den": 0},
             ["count"], oracle=False),
    ]

    # Benchmarks. The hereditary predicates are searches with pruning, and the three frontier
    # shapes are here: the 12-mark Golomb rulers near the optimum (a nonexistence proof would be
    # answered from the span table alone), the extremal count behind r_3(n), and sum-free sets
    # in a huge family. Sumset sizes are not
    # hereditary, so that case is the honest per-member cost.
    out += [
        Case("sum_free_10_subsets_of_30",
             lambda: ctx.subsets_of(ctx.range(1, 31), 10), "is_sum_free", {"modulus": 0},
             what="how many of the 30 million 10-subsets of [1,30] are sum-free",
             bench="count", oracle=False),
        Case("sum_free_24_subsets_of_64",
             lambda: ctx.subsets_of(ctx.range(0, 64), 24), "is_sum_free", {"modulus": 0},
             what="sum-free 24-subsets of [0,63], among 2.5e17 subsets",
             bench="count", oracle=False),
        Case("golomb_12_marks_up_to_92",
             lambda: ctx.subsets_of(ctx.range(0, 93), 12), "is_sidon", {"modulus": 0},
             what="12-mark Golomb rulers of length at most 92, with their reflections: 60 among 3e14 subsets",
             bench="count", oracle=False),
        Case("three_ap_free_16_subsets_of_60",
             lambda: ctx.subsets_of(ctx.range(0, 60), 16), "is_ap_free", {"modulus": 0, "length": 3},
             what="3-AP-free 16-subsets of [0,59], among 1.5e14 subsets (r_3(60) = 20)",
             bench="count", oracle=False),
        Case("sumset_sizes_6_subsets_of_40",
             lambda: ctx.subsets(elements(range(1, 41)), 6), "sumset_size", {"modulus": 0},
             what="the distribution of |S+S| over every 6-subset of [1,40]",
             bench="histogram", oracle=False),
    ]
    return out


def invariants(ctx):
    # {0,1,3,9} is a planar difference set in Z/13: each of the twelve nonzero differences occurs
    # exactly once, so the difference set is everything and the sumset has the Sidon size C(5,2).
    planar = ctx.explicit(lk.naturals([[[0], [1], [3], [9]]]))
    assert ctx.value("sum_free_and_additive.is_sidon", planar, modulus=13).values == [1]
    assert ctx.value("sum_free_and_additive.difference_set_size", planar, modulus=13).values == [13]
    assert ctx.value("sum_free_and_additive.max_difference_multiplicity", planar, modulus=13).values == [1]
    assert ctx.value("sum_free_and_additive.sumset_size", planar, modulus=13).values == [10]

    # Sum-free is exactly "no Schur triple", and Sidon is exactly "no difference twice".
    z13 = ctx.subsets_of(ctx.range(0, 13), 4)
    free = ctx.value("sum_free_and_additive.is_sum_free", z13, "count", modulus=13)
    schur = ctx.value("sum_free_and_additive.schur_triple_count", z13, "histogram", modulus=13)
    assert free.value == schur.bins[0]
    sidon = ctx.value("sum_free_and_additive.is_sidon", z13, "count", modulus=13)
    mult = ctx.value("sum_free_and_additive.max_difference_multiplicity", z13, "histogram", modulus=13)
    assert sidon.value == sum(mult.bins[:2])

    # Over the integers |S+S| <= C(k+1,2), with equality exactly for the Sidon sets.
    interval = ctx.subsets(elements(range(1, 13)), 4)
    sizes = ctx.value("sum_free_and_additive.sumset_size", interval, "histogram", modulus=0)
    assert len(sizes.bins) == 11 and sizes.bins[10] > 0
    assert ctx.value("sum_free_and_additive.is_sidon", interval, "count", modulus=0).value == sizes.bins[10]

    # An increasing dictionary takes the pruned path of the generic backend (span table, mirror
    # rule, gap bound, candidate prefilter); a shuffled dictionary of the same values takes the
    # general one. Deep enough that every bound fires; the counts must agree.
    import random
    rng = random.Random(7)
    for op, n, k, args in [("is_sidon", 36, 7, {}), ("is_sidon", 44, 8, {}),
                           ("is_ap_free", 30, 8, {"length": 3}), ("is_ap_free", 24, 8, {"length": 4}),
                           ("is_sum_free", 30, 10, {})]:
        values = list(range(n))
        shuffled = values[:]
        rng.shuffle(shuffled)
        sorted_count = ctx.value(f"sum_free_and_additive.{op}", ctx.subsets_of(ctx.range(0, n), k),
                                 "count", modulus=0, **args).value
        shuffled_count = ctx.value(f"sum_free_and_additive.{op}", ctx.subsets(elements(shuffled), k),
                                   "count", modulus=0, **args).value
        assert sorted_count == shuffled_count, (op, n, k, sorted_count, shuffled_count)
        assert sorted_count > 0

    # Splitting a search by its first elements: the sets counted by is_* over [0, n) are the
    # sets counted by extends_* under each prefix, so the pruned walk with a context below level
    # 0 (span bounds and the gap bound included) must add up to the plain count.
    for op, n, k, args in [("is_sidon", 40, 7, {}), ("is_ap_free", 40, 9, {"length": 3}),
                           ("is_sum_free", 24, 8, {})]:
        whole = ctx.value(f"sum_free_and_additive.{op}", ctx.subsets_of(ctx.range(0, n), k),
                          "count", modulus=0, **args).value
        split = 0
        for a in range(n):
            for b in range(a + 1, n - (k - 2)):
                split += ctx.value(f"sum_free_and_additive.extends_{op[3:]}",
                                   ctx.subsets_of(ctx.range(b + 1, n), k - 2), "count",
                                   modulus=0, context=ctx.naturals([[a, b]]), **args).value
        assert whole == split, (op, n, k, whole, split)

    # The backend seeds its span tables with the literature (optimal Golomb ruler lengths, and
    # G(t) from r_3(n)); with LK_SUM_FREE_SPANS=search it ignores them, and its own ladder must
    # reproduce the head of each table. A separate process, since the tables are read once.
    import os
    import subprocess
    import sys
    script = """
import sys
sys.path.insert(0, %r)
import lemmakernel as lk
ctx = lk.Context()
for op, args, top in [("is_sidon", {}, 12), ("is_ap_free", {"length": 3}, 26)]:
    spans = [0, 0]
    for t in range(2, top + 1):
        L = spans[-1] + 1
        while ctx.value("sum_free_and_additive." + op, ctx.subsets_of(ctx.range(0, L + 1), t), "count",
                        modulus=0, **args).value == 0:
            L += 1
        spans.append(L)
    print(op, spans)
"""
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    out = subprocess.run([sys.executable, "-c", script % str(root / "python")], capture_output=True, text=True,
                         env={**os.environ, "LK_SUM_FREE_SPANS": "search"}, check=True, timeout=600).stdout
    assert out.splitlines() == [
        "is_sidon [0, 0, 1, 3, 6, 11, 17, 25, 34, 44, 55, 72, 85]",
        "is_ap_free [0, 0, 1, 3, 4, 8, 10, 12, 13, 19, 23, 25, 29, 31, 35, 39, 40, 50, 53, 57, 62, 70, 73, 81, 83, 91, 94]",
    ], out

    # A 2-term progression is any pair, so only sets of at most one element are 2-AP-free; the
    # span table must stay out of the way (it has no entries for that length).
    two = ctx.value("sum_free_and_additive.is_ap_free", ctx.subsets_of(ctx.range(0, 20), 3), "count",
                    modulus=0, length=2)
    assert (two.value, two.visited) == (0, 1140)
    one = ctx.value("sum_free_and_additive.is_ap_free", ctx.subsets_of(ctx.range(0, 20), 1), "count",
                    modulus=0, length=2)
    assert one.value == 20
