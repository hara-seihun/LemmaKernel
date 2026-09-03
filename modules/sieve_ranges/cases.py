"""sieve_ranges cases: the inputs the harness runs against the backend, the naive implementation
and the Lean reference.

The oracle cases are small intervals that contain everything the conventions turn on: 0 and 1, the
primes, prime powers, squarefuls and a highly composite number. The bench cases are windows high
up, where a member costs the naive implementation a trial division to ten thousand and the kernel
a handful of divisions.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lemmakernel as lk  # noqa: E402
from tools.harness import Case  # noqa: E402

OPS = ["is_prime", "is_squarefree", "factorisation", "omega", "big_omega", "totient", "sigma",
       "divisor_count", "mobius", "largest_prime_factor"]
# A histogram has one bin per value, so it belongs to the operations whose values are small; on
# sigma, totient or the largest prime factor of a large member it is refused by the runtime.
SMALL_VALUED = ["omega", "big_omega", "divisor_count", "mobius"]


def one_reduction(op: str, i: int) -> list[str]:
    """One reduction per case, rotating, so a run of cases covers them all cheaply."""
    if op == "factorisation":
        return ["all"]
    if op.startswith("is_"):
        return [["all", "count", "hits", "first"][i % 4]]
    choices = ["all", "sum", "max", "min"] + (["histogram"] if op in SMALL_VALUED else [])
    return [choices[i % len(choices)]]


def cases(ctx, rng):
    out = []

    # 0 and 1 (where every convention lives), the primes below 32, 16 = 2^4, 12 = 2^2*3, 30 = 2*3*5.
    interval = ctx.range(0, 32)
    for op in OPS:
        out.append(Case("range 0..32", interval, op, {"limit": 4}))

    # Numbers a range cannot reach: a large prime, a prime square, a primorial, a power of two.
    picked = [1, 2, 97, 128, 360, 5040, 65537, 69696, 223092870, 999983]
    explicit = ctx.explicit(lk.naturals([[[n]] for n in picked]))
    for i, op in enumerate(OPS):
        out.append(Case("explicit naturals", explicit, op, {"limit": 3}, reductions=one_reduction(op, i)))

    # An interval away from the small cases, where the leftover cofactor is usually a large prime.
    window = ctx.range(9_973, 10_009)
    for i, op in enumerate(OPS):
        out.append(Case("interval at 10^4", window, op, {"limit": 2}, reductions=one_reduction(op, i + 2)))

    # Requests the manifest says are refused.
    out += [
        Case("grassmannian is not a number family", ctx.grassmannian(2, 3, 1), "is_prime",
             reductions=["count"], oracle=False),
        Case("matrices over F_5 are not numbers", ctx.explicit(lk.matrix(5, [[[1, 2], [3, 4]]])), "totient",
             reductions=["sum"], oracle=False),
        Case("pairs of numbers are not numbers", ctx.explicit(lk.naturals([[[3, 4]], [[5, 6]]])), "omega",
             reductions=["sum"], oracle=False),
        Case("reduction mismatch", interval, "is_prime", reductions=["count"], oracle=False),
        Case("reduction mismatch", interval, "factorisation", reductions=["all"], oracle=False),
        # sigma(999999937) = 999999938, which is not a number of bins anyone wants.
        Case("histogram of a divisor sum", ctx.explicit(lk.naturals([[[999999937]]])), "sigma",
             reductions=["histogram"], oracle=False),
    ]

    # Benchmarks. The windows start high so that the naive time measured on a prefix of members
    # is the time of a typical member, not of a two-digit one.
    out += [
        Case("primes_in_a_window", lambda: ctx.range(1_000_000_000, 1_010_000_000), "is_prime",
             what="how many primes in the ten million numbers above 10^9", bench="count", oracle=False),
        Case("totient_sum", lambda: ctx.range(1, 2_000_001), "totient",
             what="sum of Euler's phi below 2*10^6, which counts the coprime pairs under it",
             bench="sum", oracle=False),
        Case("omega_distribution", lambda: ctx.range(1_000_000_000, 1_002_000_000), "omega",
             what="how many distinct prime factors the two million numbers above 10^9 have",
             bench="histogram", oracle=False),
        Case("factorise_an_interval", lambda: ctx.range(1_000_000, 1_200_000), "factorisation",
             what="the full prime factorisation of every number in an interval of 200000 at 10^6",
             bench="all", oracle=False),
        Case("smoothest_number", lambda: ctx.range(1_000_000_001, 1_001_000_001), "largest_prime_factor",
             what="the least largest-prime-factor in a million numbers above 10^9 (the smoothest one)",
             bench="min", oracle=False),
    ]
    return out


def invariants(ctx):
    """Cross-operation identities on intervals far beyond what the Lean kernel can evaluate."""
    interval = ctx.range(1, 200_001)

    # mu is +-1 exactly on the squarefree numbers, and 0 elsewhere.
    bins = ctx.value("sieve_ranges.mobius", interval, "histogram").bins
    assert bins[0] + bins[2] == ctx.value("sieve_ranges.is_squarefree", interval, "count").value
    assert bins[1] + bins[0] + bins[2] == ctx.size(interval)

    # A number is prime exactly when it has one prime factor counted with multiplicity.
    big = ctx.value("sieve_ranges.big_omega", interval, "histogram").bins
    assert big[1] == ctx.value("sieve_ranges.is_prime", interval, "count").value

    # phi and sigma bracket n: phi(n) < n < sigma(n) for n > 1, with equality of the extremes at
    # the primes, where phi(p) = p - 1 and sigma(p) = p + 1.
    primes = ctx.value("sieve_ranges.is_prime", interval, "hits", limit=1)
    p = primes.members.member(0)[0][0]
    single = ctx.range(p, p + 1)
    assert ctx.value("sieve_ranges.totient", single, "all").values == [p - 1]
    assert ctx.value("sieve_ranges.sigma", single, "all").values == [p + 1]

    # Every factorisation multiplies back to its member, and agrees with omega, Omega and tau.
    small = ctx.range(1, 5_000)
    factorisations = ctx.value("sieve_ranges.factorisation", small)
    omega = ctx.value("sieve_ranges.omega", small, "all").values
    big_omega = ctx.value("sieve_ranges.big_omega", small, "all").values
    tau = ctx.value("sieve_ranges.divisor_count", small, "all").values
    for i in range(factorisations.count):
        pairs = factorisations.member(i)
        product = 1
        divisors = 1
        for q, e in pairs:
            product *= q ** e
            divisors *= e + 1
        assert product == 1 + i
        assert len(pairs) == omega[i]
        assert sum(e for _, e in pairs) == big_omega[i]
        assert divisors == tau[i]
