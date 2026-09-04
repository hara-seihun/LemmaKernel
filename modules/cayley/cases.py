"""Oracle and benchmark cases for finite simple undirected Cayley graphs."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lemmakernel as lk  # noqa: E402
from lemmakernel import naive as rt  # noqa: E402
from tools.harness import Case  # noqa: E402


def cyclic(n):
    return [[(i + 1) % n for i in range(n)]]


def dihedral(n):
    return [[(i + 1) % n for i in range(n)], [(-i) % n for i in range(n)]]


def c4xc2():
    a = [2 * ((x // 2 + 1) % 4) + x % 2 for x in range(8)]
    b = [2 * (x // 2) + (x % 2 + 1) % 2 for x in range(8)]
    return [a, b]


def connection_family(ctx, generators, k):
    elements = rt.perm_closure(generators)
    identity = list(range(len(generators[0])))
    return ctx.subsets(lk.perms(len(identity), [g for g in elements if g != identity]), k)


def cases(ctx, rng):
    del rng
    C4 = ctx.perms(4, cyclic(4))
    singles = connection_family(ctx, cyclic(4), 1)
    pairs = connection_family(ctx, cyclic(4), 2)
    out = [
        Case("C4 singletons", singles, "connected", {"group": C4, "limit": 3}),
        Case("C4 singletons", singles, "is_regular_of_degree", {"group": C4, "degree": 2, "limit": 3}),
        Case("C4 singletons", singles, "girth", {"group": C4}),
        Case("C4 singletons", singles, "diameter", {"group": C4}),
        Case("C4 singletons", singles, "aut_order", {"group": C4}),
        Case("C4 singletons", singles, "is_ci_set", {"group": C4, "limit": 3}),
        Case(
            "C4 automorphic witness",
            ctx.explicit(lk.perms(4, [[2, 3, 0, 1]])),
            "is_non_ci_witness",
            {
                "group": C4,
                "target": ctx.perms(4, [[2, 3, 0, 1]]),
                "isomorphism": ctx.perms(4, [list(range(4))]),
                "limit": 1,
            },
        ),
        Case(
            "C4 separated witness",
            ctx.explicit(lk.perms(4, [[2, 3, 0, 1]])),
            "is_separated_witness",
            {
                "group": C4,
                "target": ctx.perms(4, [[2, 3, 0, 1]]),
                "isomorphism": ctx.perms(4, [list(range(4))]),
                "automorphisms": ctx.perms(4, [list(range(4))]),
                "limit": 1,
            },
        ),
        Case(
            "C4xC2 non-CI witness",
            ctx.explicit(lk.perms(8, [c4xc2()[1]])),
            "is_non_ci_witness",
            {
                "group": ctx.perms(8, c4xc2()),
                "target": ctx.perms(8, [[c4xc2()[0][c4xc2()[0][x]] for x in range(8)]]),
                "isomorphism": ctx.perms(8, [[0, 4, 1, 5, 2, 6, 3, 7]]),
                "limit": 1,
            },
            oracle=False,
            bench="all",
            what="a supplied non-CI witness for C4 x C2, including graph isomorphism and full automorphism separation",
        ),
        Case("C4 pairs", pairs, "connected", {"group": C4, "limit": 2}, reductions=["all", "hits"]),
        Case("C4 pairs", pairs, "girth", {"group": C4}, reductions=["all", "histogram"]),
    ]

    bad_group = lk.matrix(2, [[[1, 0], [0, 1]]])
    identity_family = ctx.subsets(lk.perms(4, [list(range(4)), cyclic(4)[0]]), 1)
    outside_family = ctx.subsets(lk.perms(4, [[1, 0, 2, 3]]), 1)
    out += [
        Case("matrix group is unsupported", singles, "connected", {"group": bad_group}, reductions=["count"], oracle=False),
        Case("identity is not a connection", identity_family, "connected", {"group": C4}, reductions=["count"], oracle=False),
        Case("element outside G", outside_family, "connected", {"group": C4}, reductions=["count"], oracle=False),
        Case("range is not a connection family", ctx.range(0, 3), "connected", {"group": C4}, reductions=["count"], oracle=False),
        Case(
            "two witness isomorphisms",
            ctx.explicit(lk.perms(8, [c4xc2()[1]])),
            "is_non_ci_witness",
            {
                "group": ctx.perms(8, c4xc2()),
                "target": ctx.perms(8, [[c4xc2()[0][c4xc2()[0][x]] for x in range(8)]]),
                "isomorphism": ctx.perms(8, [list(range(8)), list(range(8))]),
            },
            reductions=["all"], oracle=False,
        ),
    ]

    D6gens = dihedral(6)
    C8gens = cyclic(8)
    Xgens = c4xc2()
    out += [
        Case("D6 triples", lambda: connection_family(ctx, D6gens, 3), "connected",
             {"group": ctx.perms(6, D6gens)}, bench="count", oracle=False,
             what="connected simple Cayley graphs from all 3-subsets of the 11 nonidentity elements of D_12"),
        Case("C8 automorphisms", lambda: connection_family(ctx, C8gens, 1), "aut_order",
             {"group": ctx.perms(8, C8gens)}, bench="histogram", oracle=False,
             what="full graph automorphism orders for singleton connection sets of C_8"),
        Case("C4xC2 singleton CI", lambda: connection_family(ctx, Xgens, 1), "is_ci_set",
             {"group": ctx.perms(8, Xgens)}, bench="all", oracle=False,
             what="which singleton connection sets of C_4 x C_2 satisfy the undirected CI condition"),
    ]
    return out


def invariants(ctx):
    C4 = ctx.perms(4, cyclic(4))
    family = connection_family(ctx, cyclic(4), 1)
    connected = ctx.value("cayley.connected", family, group=C4).values
    girths = ctx.value("cayley.girth", family, group=C4).values
    diameters = ctx.value("cayley.diameter", family, group=C4).values
    assert connected == [1, 0, 1]
    assert girths == [4, 0, 4]
    assert diameters == [2, 0, 2]

    generators = c4xc2()
    group = ctx.perms(8, generators)
    ci = ctx.value("cayley.is_ci_set", connection_family(ctx, generators, 1), group=group).values
    assert ci == [0] * 7

    a, b = generators
    source = ctx.explicit(lk.perms(8, [b]))
    target = ctx.perms(8, [[a[a[x]] for x in range(8)]])
    mapping = ctx.perms(8, [[0, 4, 1, 5, 2, 6, 3, 7]])
    assert ctx.value(
        "cayley.is_non_ci_witness", source,
        group=group, target=target, isomorphism=mapping,
    ).values == [1]
