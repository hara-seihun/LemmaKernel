"""Cases for canonical graph labelling and automorphism groups."""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lemmakernel as lk  # noqa: E402
from tools.harness import Case  # noqa: E402


def graph(n, edges=(), loops=()):
    a = [[0] * n for _ in range(n)]
    for i, j in edges:
        a[i][j] = a[j][i] = 1
    for i in loops:
        a[i][i] = 1
    return a


def cycle(n):
    return graph(n, [(i, (i + 1) % n) for i in range(n)])


def random_graph(rng, n, probability=0.45):
    return graph(n, [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < probability])


def relabel(a, order):
    return [[a[i][j] for j in order] for i in order]


def cases(ctx, rng):
    graphs4 = [
        graph(4),
        graph(4, [(i, j) for i in range(4) for j in range(i + 1, 4)]),
        graph(4, [(0, 1), (1, 2), (2, 3)]),
        cycle(4),
        graph(4, [(0, 1), (0, 2), (0, 3)]),
        graph(4, [(0, 1), (1, 2)], loops=[0, 2]),
    ]
    graphs5 = [
        graph(5, [(0, 1), (1, 2), (2, 3), (3, 4)]),
        cycle(5),
        graph(5, [(0, 1), (0, 2), (0, 3), (1, 2), (2, 4)]),
    ]
    families = [
        ("standard graphs on four vertices", ctx.explicit(lk.matrix(2, graphs4))),
        ("graphs on five vertices", ctx.explicit(lk.matrix(2, graphs5))),
        ("every looped graph on three vertices", ctx.symmetric_matrices(2, 3)),
    ]
    out = [Case(name, family, op) for name, family in families
           for op in ("canonical_form", "canonical_label", "automorphism_group")]

    out += [
        Case("ternary adjacency", ctx.explicit(lk.matrix(3, [graph(3, [(0, 1)])])), "canonical_form", oracle=False),
        Case("rectangular adjacency", ctx.explicit(lk.matrix(2, [[[0, 1, 0], [1, 0, 1]]])), "canonical_label", oracle=False),
        Case("directed adjacency", ctx.explicit(lk.matrix(2, [[[0, 1, 0], [0, 0, 1], [0, 0, 0]]])), "automorphism_group", oracle=False),
        Case("all matrices family", ctx.all_matrices(2, 2, 2), "canonical_form", oracle=False),
    ]

    bench_rng = random.Random(2501)
    random8 = ctx.explicit(lk.matrix(2, [random_graph(bench_rng, 8)]))
    out += [
        Case("random_graph_8_form", random8, "canonical_form", bench="all", oracle=False,
             what="lexicographically canonicalise one random graph on 8 vertices"),
        Case("random_graph_8_label", random8, "canonical_label", bench="all", oracle=False,
             what="find the canonical vertex order of one random graph on 8 vertices"),
        Case("cycle_8_automorphisms", ctx.explicit(lk.matrix(2, [cycle(8)])), "automorphism_group",
             bench="all", oracle=False, what="enumerate all 16 automorphisms of the 8-cycle"),
    ]
    return out


def invariants(ctx):
    rng = random.Random(2517)
    a = random_graph(rng, 9)
    order = [7, 2, 5, 0, 8, 3, 1, 6, 4]
    family = ctx.explicit(lk.matrix(2, [a, relabel(a, order)]))
    forms = ctx.value("graph_iso.canonical_form", family)
    labels = ctx.value("graph_iso.canonical_label", family)
    assert forms.member(0) == forms.member(1)
    for i in range(2):
        assert relabel(family.value().children[0].member(i), labels.member(i)) == forms.member(i)

    c10 = cycle(10)
    groups = ctx.value("graph_iso.automorphism_group", ctx.explicit(lk.matrix(2, [c10])))
    automorphisms = groups.member(0)
    assert len(automorphisms) == 20
    assert automorphisms == sorted(automorphisms)
    assert all(relabel(c10, g) == c10 for g in automorphisms)
