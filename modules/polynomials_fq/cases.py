"""polynomials_fq cases: the inputs the harness runs against every backend, the naive
implementation and the Lean reference.

A family of one-row members over F_p is a family of monic polynomials: `all_matrices(p, 1, d)` is
every monic polynomial of degree d. Oracle cases are sized for the Lean kernel, which factorises
by trial division and walks the powers of x, so they keep p^d small; bench cases are sized to
show what the kernel does.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lemmakernel as lk  # noqa: E402
from tools.harness import Case, module, random_batch, rotate  # noqa: E402

OPS = ["is_irreducible", "factorisation_degrees", "is_primitive", "order", "roots", "root_count", "gcd"]
# `order` and `is_primitive` multiply by x until they come back to 1, in the kernel too
WALK_OPS = ("is_primitive", "order")
WALK_BUDGET = 128  # p^d above which an oracle case leaves the walking operations out


def cyclotomic_like(p, k):
    """x^k - 1 as a member row: the constant coefficient is -1 and the rest are zero."""
    return lk.matrix(p, [[p - 1] + [0] * (k - 1)])


def small_families(ctx, rng):
    """Families of monic polynomials the Lean kernel can afford: (name, handle, p, degree, ops).
    Trial division and evaluation at every field element are what the kernel pays for, so the
    larger the field the fewer operations a family carries."""
    out = []
    for p, d in [(2, 4), (2, 5), (3, 3), (5, 2)]:
        out.append((f"monic degree {d} over F_{p}", ctx.all_matrices(p, 1, d), p, d, OPS))
    for p, d, count in [(2, 6, 8), (3, 4, 8), (11, 2, 8)]:
        out.append((f"explicit degree {d} over F_{p}", ctx.explicit(random_batch(rng, p, count, 1, d)), p, d, OPS))
    # wider entries: two bytes at F_257, four at F_65537, where only the cheap operations fit
    out.append(("explicit degree 2 over F_257", ctx.explicit(random_batch(rng, 257, 4, 1, 2)), 257, 2,
                ["is_irreducible", "roots", "root_count", "gcd"]))
    out.append(("explicit degree 3 over F_65537", ctx.explicit(random_batch(rng, 65537, 4, 1, 3)), 65537, 3, ["gcd"]))
    out.append(("chosen degree 3 over F_3", ctx.subsets(random_batch(rng, 3, 6, 1, 3), 1), 3, 3, OPS))
    out.append(("mapped degree 4 over F_2",
                ctx.transform(ctx.all_matrices(2, 1, 4), random_batch(rng, 2, 1, 4, 4)), 2, 4, OPS))
    out.append(("chosen degree 2 over F_3", ctx.subsets_of(ctx.all_matrices(3, 1, 2), 1), 3, 2, OPS))
    return out


def cases(ctx, rng):
    """Every reduction on the smallest family; one reduction per case, rotating, elsewhere."""
    mod = module("polynomials_fq")
    out = []
    for i, (name, fam, p, d, ops) in enumerate(small_families(ctx, rng)):
        for j, op in enumerate(ops):
            if op in WALK_OPS and p ** d > WALK_BUDGET:
                continue
            args = {"limit": 3, "other": cyclotomic_like(p, d) if j % 2 else random_batch(rng, p, 1, 1, max(1, d - 1))}
            reds = None if i == 0 else rotate(mod, op, i + j)
            out.append(Case(name, fam, op, args, reductions=reds))

    # Requests the runtime must refuse; the manifest names them. Never run as ordinary cases.
    out += [
        Case("two rows per member F_2", ctx.stack(ctx.all_matrices(2, 1, 3), lk.matrix(2, [[1, 0, 1]])),
             "is_irreducible", reductions=["all"], oracle=False),
        Case("naturals family", ctx.range(0, 8), "root_count", reductions=["all"], oracle=False),
        Case("degree 32 over F_2", ctx.explicit(random_batch(rng, 2, 4, 1, 32)), "order",
             reductions=["all"], oracle=False),
    ]

    # Benchmarks: requests that look like real use, sized so that naive finishes and the ratio shows.
    out += [
        Case("irreducible_degree_14_F2", lambda: ctx.all_matrices(2, 1, 14), "is_irreducible",
             what="how many of the 16384 monic degree-14 polynomials over F_2 are irreducible",
             bench="count", oracle=False),
        Case("primitive_degree_12_F2", lambda: ctx.all_matrices(2, 1, 12), "is_primitive",
             what="the primitive polynomials of degree 12 over F_2, of which there are phi(4095)/12",
             bench="count", oracle=False),
        Case("order_degree_11_F2", lambda: ctx.all_matrices(2, 1, 11), "order",
             what="the order of x modulo every monic polynomial of degree 11 over F_2",
             bench="histogram", oracle=False),
        Case("factor_degrees_degree_7_F3", lambda: ctx.all_matrices(3, 1, 7), "factorisation_degrees",
             what="the factorisation pattern of every monic degree-7 polynomial over F_3",
             bench="all", oracle=False),
        Case("root_count_degree_5_F7", lambda: ctx.all_matrices(7, 1, 5), "root_count",
             what="how many roots in F_7 each of the 16807 monic degree-5 polynomials has",
             bench="histogram", oracle=False),
        Case("gcd_degree_4_F65537", lambda: ctx.explicit(random_batch(rng, 65537, 20000, 1, 4)), "gcd",
             {"other": cyclotomic_like(65537, 4)},
             what="gcd of 20000 random monic quartics over F_65537 with x^4 - 1",
             bench="all", oracle=False),
        Case("first_primitive_degree_14_F2", lambda: ctx.all_matrices(2, 1, 14), "is_primitive",
             what="the least monic primitive polynomial of degree 14 over F_2, in family order",
             bench="first", oracle=False),
    ]
    return out


def invariants(ctx):
    """Cross-operation identities on inputs beyond the kernel oracle."""
    # Over F_2 a monic polynomial of degree d is irreducible exactly when its factorisation has
    # one factor, and there are (1/d) sum_{e | d} mu(e) 2^(d/e) of them.
    for d, expected in [(6, 9), (7, 18), (8, 30)]:
        fam = ctx.all_matrices(2, 1, d)
        irreducible = ctx.value("polynomials_fq.is_irreducible", fam, "count").value
        assert irreducible == expected, (d, irreducible, expected)
        degrees = ctx.value("polynomials_fq.factorisation_degrees", fam, "all")
        one_factor = sum(1 for i in range(2 ** d) if len(degrees.member(i)) == 1)
        assert one_factor == expected, (d, one_factor, expected)

    # Every primitive polynomial is irreducible, and its order is 2^d - 1.
    d = 10
    fam = ctx.all_matrices(2, 1, d)
    primitive = ctx.value("polynomials_fq.is_primitive", fam, "hits", limit=0).indices
    irreducible = set(ctx.value("polynomials_fq.is_irreducible", fam, "hits", limit=0).indices)
    orders = ctx.value("polynomials_fq.order", fam, "all").values
    assert set(primitive) <= irreducible
    assert all(orders[i] == 2 ** d - 1 for i in primitive)
    assert len(primitive) == 60  # phi(1023) / 10

    # The factor degrees of a member sum to its degree, and a root is a factor of degree 1.
    fam = ctx.all_matrices(3, 1, 4)
    degrees = ctx.value("polynomials_fq.factorisation_degrees", fam, "all")
    counts = ctx.value("polynomials_fq.root_count", fam, "all").values
    for i in range(ctx.size(fam)):
        ds = degrees.member(i)
        assert sum(ds) == 4
        assert counts[i] <= ds.count(1)
