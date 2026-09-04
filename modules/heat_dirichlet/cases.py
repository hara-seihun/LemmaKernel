"""heat_dirichlet cases: the inputs the harness runs against the backend, the naive implementation
and the Lean reference.

The parameters are those of the Dirichlet-polynomial barrier at t = 0.1579 with cutoff
N_- = 7 * 10^5 and mollifier primes {2, 3}: t_num/t_den = 1579/10000, y in a narrow band around
0.2, C = 1.03. `sigma_num / sigma_den` is what `sigma_lower` returns for that cutoff at scale 40,
so a case chain reproduces how a caller uses the module. The oracle cases are a few members each
because every member costs the Lean kernel a few hundred big-integer operations per exponential.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lemmakernel as lk  # noqa: E402
from tools.harness import Case  # noqa: E402

SIGMA = 1822769433543  # sigma_lower at N = 700000, y_lo = 0.199, scale 40: about 1.6576
BASE = {"t_num": 1579, "t_den": 10000, "sigma_num": SIGMA, "sigma_den": 1 << 40, "scale": 48}
CELL = {**BASE, "y_lo_num": 199, "y_hi_num": 201, "y_den": 1000, "n_minus": 700_000, "n_plus": 707_000,
        "primes": 0b11, "c_num": 103, "c_den": 100}
SIGMA_ARGS = {"t_num": 1579, "t_den": 10000, "y_lo_num": 199, "y_hi_num": 201, "y_den": 1000, "scale": 40}


def cases(ctx, rng):
    out = []
    small = ctx.range(1, 9)               # 1 (beta = 1), 2, 3, 6 (every divisor of D), 4, 5, 7, 8
    # Past N_- the truncation at N bites: 1_400_001 loses d = 1 for every cutoff, 2_121_000 = 3 N_+
    # keeps d = 3 for N = N_+ only, and beyond 6 N_+ nothing is left.
    picked = [1, 2, 3, 6, 12, 97, 100_000, 700_000, 1_400_001, 2_121_000, 3_000_000, 4_242_001]
    explicit = ctx.explicit(lk.naturals([[[n]] for n in picked]))

    out += [
        Case("range 1..9", small, "weight_upper", BASE, reductions=["all", "sum"]),
        Case("range 1..9", small, "mollified_term_upper", CELL, reductions=["all", "sum", "max"]),
        Case("explicit naturals", explicit, "weight_upper", BASE, reductions=["all", "max", "min"]),
        Case("explicit naturals", explicit, "mollified_term_upper", CELL, reductions=["all", "min"]),
        # Blocks of width 1 are the per-member bound; wider blocks exercise the residue classes.
        Case("unit blocks", ctx.range(0, 8), "block_term_upper", {**CELL, "n0": 1, "width": 1},
             reductions=["all", "sum"]),
        Case("wide blocks", ctx.range(0, 3), "block_term_upper", {**CELL, "n0": 1000, "width": 100, "scale": 40},
             reductions=["all", "sum", "max", "min"]),
        Case("blocks across the cutoff", ctx.range(0, 4), "block_term_upper",
             {**CELL, "n0": 690_000, "width": 500_000, "scale": 40}, reductions=["all"]),
        Case("blocks with every prime", ctx.range(0, 2), "block_term_upper",
             {**CELL, "primes": 0b1111, "n0": 2, "width": 250, "scale": 40}, reductions=["all"]),
        Case("cutoffs", ctx.explicit(lk.naturals([[[2]], [[700_000]], [[10 ** 9]]])), "sigma_lower", SIGMA_ARGS,
             reductions=["all", "max", "min", "sum"]),
        # A histogram of these values is one bin per value; it is allowed but only sensible when
        # the values are tiny, so a low scale keeps the bins few.
        Case("coarse histogram", small, "weight_upper", {**BASE, "scale": 4}, reductions=["histogram"]),
        Case("coarse histogram", small, "mollified_term_upper", {**CELL, "scale": 4}, reductions=["histogram"]),
        Case("coarse histogram", ctx.range(0, 4), "block_term_upper", {**CELL, "n0": 1, "width": 2, "scale": 4},
             reductions=["histogram"]),
        Case("coarse histogram", ctx.explicit(lk.naturals([[[2]], [[3]]])), "sigma_lower", {**SIGMA_ARGS, "scale": 2},
             reductions=["histogram"]),
    ]

    # Requests the manifest says are refused.
    out += [
        Case("grassmannian is not a number family", ctx.grassmannian(2, 3, 1), "weight_upper", BASE,
             reductions=["sum"], oracle=False),
        Case("matrices over F_5 are not numbers", ctx.explicit(lk.matrix(5, [[[1, 2], [3, 4]]])), "weight_upper",
             BASE, reductions=["sum"], oracle=False),
        Case("pairs of numbers are not numbers", ctx.explicit(lk.naturals([[[3, 4]], [[5, 6]]])), "weight_upper",
             BASE, reductions=["sum"], oracle=False),
        Case("weight of 0", ctx.range(0, 3), "weight_upper", BASE, reductions=["sum"], oracle=False),
        Case("sigma at 1", ctx.range(1, 3), "sigma_lower", SIGMA_ARGS, reductions=["sum"], oracle=False),
        Case("t above one half", small, "weight_upper", {**BASE, "t_num": 6, "t_den": 10}, reductions=["sum"],
             oracle=False),
        # With sigma = 0 the weight of 10^9 is exp(0.039 ln^2 10^9) = exp(17): past the exp limit.
        Case("exponent too large", ctx.explicit(lk.naturals([[[10 ** 9]]])), "weight_upper",
             {**BASE, "sigma_num": 0}, reductions=["sum"], oracle=False),
    ]

    # Benchmarks: one canopy cell, as the barrier programme sums it: every n from 2 to D N_+, then
    # the same range in blocks whose width grows with n.
    out += [
        Case("cell_terms", lambda: ctx.range(2, 6 * 707_000 + 1), "mollified_term_upper", {**CELL, "scale": 40},
             what="the mollified summands of one canopy cell, n from 2 to D N_+ = 4242000, summed",
             bench="sum", oracle=False),
        Case("cell_blocks", lambda: ctx.range(0, 5_060), "block_term_upper",
             {**CELL, "n0": 700_000, "width": 700, "scale": 40},
             what="the part of that cell above N_- in blocks of 700 (a thousandth of n, which costs the bound well under a percent)",
             bench="sum", oracle=False),
    ]
    return out


def invariants(ctx):
    """Identities the kernel must satisfy far beyond what the Lean kernel evaluates."""
    interval = ctx.range(2, 20_002)
    args = {**CELL, "scale": 40}
    # A block of width 1 bounds the same summand as the per-member operation, from the same
    # enclosures, so the two agree to the last bit.
    per_member = ctx.value("heat_dirichlet.mollified_term_upper", interval, "sum", **args).value
    blocks = ctx.value("heat_dirichlet.block_term_upper", ctx.range(0, 20_000), "sum",
                       **args, n0=2, width=1).value
    assert per_member == blocks, (per_member, blocks)
    # A wide block bounds the sum of its members, and does not lose more than the width allows.
    wide = ctx.value("heat_dirichlet.block_term_upper", ctx.range(0, 1), "sum", **args, n0=10_002, width=10_000).value
    exact = ctx.value("heat_dirichlet.mollified_term_upper", ctx.range(10_002, 20_002), "sum", **args).value
    assert exact <= wide <= 2 * exact, (exact, wide)
    # Beyond D N_+ no divisor survives the truncation and the summand is 0.
    beyond = ctx.value("heat_dirichlet.mollified_term_upper", ctx.range(6 * 707_000 + 1, 6 * 707_000 + 101), "max",
                       **args).value
    assert beyond == 0
    # The mollified summand never exceeds a small multiple of the bare weight (|beta| and |alpha|
    # are bounded by the sums of |lambda_d|), so the mollified sum sits below that multiple.
    weights = ctx.value("heat_dirichlet.weight_upper", interval, "sum", **{**BASE, "scale": 40}).value
    assert per_member <= 64 * weights
    # sigma_lower increases with the cutoff.
    sig = ctx.value("heat_dirichlet.sigma_lower", ctx.explicit(lk.naturals([[[n]] for n in [10, 100, 1000]])),
                    "all", **SIGMA_ARGS).values
    assert sig == sorted(sig)
