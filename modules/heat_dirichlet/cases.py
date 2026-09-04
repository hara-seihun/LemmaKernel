"""heat_dirichlet cases: the inputs the harness runs against the backend, the naive implementation
and the Lean reference.

The parameters are those of a canopy cell at t = 0.1579 with cutoffs from N_- = 7 * 10^5 and
mollifier primes {2, 3}: t_num/t_den = 1579/10000, height y = 0.2, C = 1.03. `sigma_num /
sigma_den` is what `sigma_lower` returns for that cutoff and height at scale 40, so a case chain
reproduces how a caller uses the module. The oracle cases are a few members each
because every member costs the Lean kernel a few hundred big-integer operations per exponential.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lemmakernel as lk  # noqa: E402
from tools.harness import Case  # noqa: E402

SIGMA = 1828020623414  # sigma_lower at N = 700000, y = 0.2, scale 40: about 1.6624
BASE = {"t_num": 1579, "t_den": 10000, "sigma_num": SIGMA, "sigma_den": 1 << 40, "scale": 48}
CELL = {**BASE, "y_num": 2, "y_den": 10, "n_minus": 700_000, "n_plus": 707_000,
        "primes": 0b11, "c_num": 103, "c_den": 100, "plain": 0}
PLAIN = {**CELL, "plain": 1}
SIGMA_ARGS = {"t_num": 1579, "t_den": 10000, "y_num": 2, "y_den": 10, "scale": 40}
# A toy phase-aware cell, N in [20, 21] at y = 0.1 with the Euler mollifier on {2, 3} (rows are
# d, sign, num, den) and two bins of rough k, on a 4 x 3 x 2 x 2 theta grid with 4 psi samples,
# the low/high split at m0 = 12 and Taylor order 2 in eps; sigma spans a small range as it does
# across a cell. Tiny, because the Lean kernel pays for every exponential and every point of the
# cosine table it touches.
PHASE = {"t_num": 1579, "t_den": 10000, "sigma_num": 1_500_000_000_000, "sigma_hi_num": 1_505_000_000_000,
         "sigma_den": 1 << 40, "y_num": 1, "y_den": 10, "n_minus": 20, "n_plus": 21, "c_num": 103, "c_den": 100,
         "g2": 4, "g3": 3, "g5": 2, "g7": 2, "npsi": 4, "m0": 12, "order": 2, "prune": 0, "offset": 4, "scale": 40}
MOLLIFIER = [[1, 0, 1, 1], [2, 1, 1067, 1000], [3, 1, 1130, 1000], [6, 0, 1206, 1000]]
BINS = [2, 11, 22]


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
        # the plain mode: |beta| + alpha_abs with the capped pi, at the same members
        Case("range 1..9 plain", small, "mollified_term_upper", PLAIN, reductions=["all", "sum"]),
        Case("explicit naturals plain", explicit, "mollified_term_upper", PLAIN, reductions=["all", "min"]),
        Case("plain blocks across the cutoff", ctx.range(0, 4), "block_term_upper",
             {**PLAIN, "n0": 699_990, "width": 10}, reductions=["all", "sum"]),
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

    phase = {**PHASE, "mollifier": ctx.naturals([MOLLIFIER]), "bins": ctx.naturals([[BINS]])}
    boxes = ctx.explicit(lk.naturals([[[0, 0, 0, 0]], [[1, 2, 0, 1]], [[3, 1, 1, 0]]]))
    out += [
        Case("boxes by index", ctx.range(0, 6), "phase_bound", phase, reductions=["all", "min", "max"]),
        Case("boxes by coordinates", boxes, "phase_bound", phase, reductions=["all", "min", "sum"]),
        Case("every term low, order 3, pruned", ctx.range(0, 3), "phase_bound",
             {**phase, "m0": 21, "order": 3, "prune": 1 << 20, "bins": ctx.naturals([[[2, 5, 11, 22]]])},
             reductions=["all"]),
        Case("order 0 with a coarse split", ctx.range(0, 3), "phase_bound", {**phase, "m0": 6, "order": 0},
             reductions=["all"]),
        Case("coarse histogram", ctx.range(0, 4), "phase_bound", {**phase, "scale": 2}, reductions=["histogram"]),
    ]

    # Requests the manifest says are refused.
    out += [
        Case("box beyond the grid", ctx.range(190, 194), "phase_bound", phase, reductions=["all"], oracle=False),
        Case("bins not ending above n_plus", ctx.range(0, 2), "phase_bound",
             {**phase, "bins": ctx.naturals([[[2, 11, 21]]])}, reductions=["all"], oracle=False),
        Case("mollifier support not smooth", ctx.range(0, 2), "phase_bound",
             {**phase, "mollifier": ctx.naturals([[[1, 0, 1, 1], [11, 1, 1, 1]]])}, reductions=["all"], oracle=False),
        Case("sigma range reversed", ctx.range(0, 2), "phase_bound",
             {**phase, "sigma_hi_num": 1_400_000_000_000}, reductions=["all"], oracle=False),
    ]

    # The barrier: the box [X, X+1] x [0.1, 1] x [0, t0] at the row's X, split coarsely, with a
    # small cutoff and few moments: the Lean kernel spends about 2 ms per interval product, and
    # a request costs N x jmax x 5 of them in the setup and a few thousand per box (N = 60 with
    # jmax = 14 is 18 s for one box; N = 12 with jmax = 8 is under a second).  The ln recurrence's
    # restart at 2^16 is beyond any oracle case; the backend and the naive were checked against
    # each other at N = 70000 by hand (30 s of naive setup).  The x numerators are two 32-bit
    # words each.
    X = 6_000_000_185_827
    words = lambda v: [v >> 32, v & 0xFFFFFFFF]
    bottom = ctx.naturals([[words(X) + words(X + 1) + [1, 1, 1, 10, 0, 1579, 10000]]])
    left = ctx.naturals([[words(X) + words(X) + [1, 1, 10, 10, 0, 1579, 10000]]])
    top_t0 = ctx.naturals([[words(X) + words(X + 1) + [1, 1, 1, 10, 1579, 1579, 10000]]])
    BARRIER = {"box": bottom, "gx": 10, "gy": 1, "gt": 8, "n": 12, "jmax": 8, "real": 0, "offset": 8, "scale": 40}
    out += [
        Case("bottom edge, |f|", ctx.range(0, 3), "barrier_lower", BARRIER, reductions=["all", "min", "sum"]),
        Case("bottom edge, Re f", ctx.range(77, 80), "barrier_lower", {**BARRIER, "real": 1}, reductions=["all", "max"]),
        Case("left edge by coordinates", ctx.explicit(lk.naturals([[[0, 0, 0]], [[0, 5, 3]], [[0, 8, 7]]])),
             "barrier_lower", {**BARRIER, "box": left, "gx": 1, "gy": 9}, reductions=["all", "min"]),
        Case("the t0 face, Re f", ctx.range(4, 7), "barrier_lower", {**BARRIER, "box": top_t0, "gt": 1, "real": 1},
             reductions=["all", "min"]),
        Case("coarse histogram", ctx.range(0, 3), "barrier_lower", {**BARRIER, "scale": 1},
             reductions=["histogram"]),
        Case("box beyond the grid", ctx.range(79, 82), "barrier_lower", BARRIER, reductions=["all"], oracle=False),
        Case("box too large for the Taylor remainder", ctx.range(0, 2), "barrier_lower",
             {**BARRIER, "n": 70_000, "gx": 2, "gt": 1}, reductions=["all"], oracle=False),
        Case("box with x reversed", ctx.range(0, 2), "barrier_lower",
             {**BARRIER, "box": ctx.naturals([[words(X + 1) + words(X) + [1, 1, 1, 10, 0, 1579, 10000]]])},
             reductions=["all"], oracle=False),
        Case("box row of the wrong width", ctx.range(0, 2), "barrier_lower",
             {**BARRIER, "box": ctx.naturals([[[1, 2, 3]]])}, reductions=["all"], oracle=False),
    ]

    out += [
        Case("grassmannian is not a number family", ctx.grassmannian(2, 3, 1), "weight_upper", BASE,
             reductions=["sum"], oracle=False),
        Case("matrices over F_5 are not numbers", ctx.explicit(lk.matrix(5, [[[1, 2], [3, 4]]])), "weight_upper",
             BASE, reductions=["sum"], oracle=False),
        Case("pairs of numbers are not numbers", ctx.explicit(lk.naturals([[[3, 4]], [[5, 6]]])), "weight_upper",
             BASE, reductions=["sum"], oracle=False),
        Case("weight of 0", ctx.range(0, 3), "weight_upper", BASE, reductions=["sum"], oracle=False),
        Case("sigma at 1", ctx.range(1, 3), "sigma_lower", SIGMA_ARGS, reductions=["sum"], oracle=False),
        Case("plain of 2", small, "mollified_term_upper", {**CELL, "plain": 2}, reductions=["sum"], oracle=False),
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
        # The phase-aware bound on one cell: every box of a 16 x 8 x 4 x 4 theta grid with 16 psi
        # samples, minimised.
        Case("cell_phase", lambda: ctx.range(0, 16 * 8 * 4 * 4), "phase_bound",
             {**PHASE, "n_minus": 7000, "n_plus": 7070, "mollifier": ctx.naturals([MOLLIFIER]),
              "bins": ctx.naturals([[[2] + list(range(11, 7071, 233)) + [7071]]]),
              "g2": 16, "g3": 8, "g5": 4, "g7": 4, "npsi": 16, "m0": 300},
             what="the phase-aware bound over every box of a 16 x 8 x 4 x 4 theta grid with 16 psi samples for one cell with the {2, 3} Euler mollifier and 31 bins of rough k, terms above 300 as loss, minimised",
             bench="min", oracle=False),
        # The barrier's bottom edge at the row's cutoff: the setup (ln, weights, moments for every
        # n up to N) and 8000 boxes of 10^-2 in x and 10^-3 in t over the first half of [0, t0].
        Case("barrier_edge", lambda: ctx.range(0, 8000), "barrier_lower",
             {**BARRIER, "box": ctx.naturals([[words(X) + words(X + 1) + [1, 1, 1, 10, 0, 1579, 20000]]]),
              "gx": 100, "gt": 80, "n": 690_988, "jmax": 14},
             what="the lower bound on |f_t| over 8000 boxes of the barrier's bottom edge (x in [X, X+1], y = 0.1, t in [0, t0/2]) with N = 690988, minimised",
             bench="min", oracle=False),
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
