"""The heat_dirichlet module in plain Python: rigorous fixed-point upper bounds on the summands of
heat-weighted Dirichlet polynomials.

Every real quantity is a Python integer at scale 2^48, and every rounding is a floor or a ceiling
chosen so the final value is an upper bound of the real number it stands for. There is no
floating point anywhere. The same algorithm, step for step, is the C++ backend and the Lean
reference; the three must return the same integer, so nothing here is "approximately" anything.

    naive.run(op, family, reduction, **args) -> interchange object

Notation, for a fixed rational t = t_num/t_den in (0, 1/2], a lower bound sigma on Re s_*, and
the height y = y_num/y_den in [0, 1]:

    b_n        = exp((t/4) ln^2 n)                      the heat weight
    g(n)       = b_n n^(-sigma)                         weight_upper
    rho_d(n)   = exp(-(t/4) ln d (2 ln n - ln d))       = b_{n/d} / b_n for d | n, in (0, 1]
    pi_d(n)    = min(1, n / (d N_-))^y                  the partner sum's weight, in (0, 1]
    lambda_d   = prod_{p | d} (-b_p)                    d squarefree over the chosen primes, D = prod p
    beta(n)    = sum_{d | gcd(n, D), n/d <= N} lambda_d rho_d(n)         mollified coefficient over b_n
    alpha(n)   = C sum_{d | gcd(n, D), n/d <= N} lambda_d pi_d(n) rho_d(n)
    r          = (1 - C N_-^(-y)) / (1 + C N_-^(-y))
    term(n)    = max(|beta - alpha|, r |beta + alpha|) g(n)

Everything is evaluated at the one height y; the caller's argument (the partner sum's weight at a
height y' >= y is at most pi_d(n) with y, and sigma only grows with the height) is what makes one
value cover every height above y. `mollified_term_upper` bounds sup over cutoffs N in [N_-, N_+]
of term(n); `block_term_upper` bounds the sum of that over a block of consecutive n using only that
rho_d decreases in n, pi_d increases, and g is quasi-convex; `sigma_lower` gives the lower bound
on Re s_* that the other operations consume.
"""
from __future__ import annotations

import itertools
import sys
from math import gcd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))
from lemmakernel import naive as rt  # noqa: E402
from lemmakernel.interchange import NATURALS, Family  # noqa: E402

K = 48
S = 1 << K
PRIMES = [2, 3, 5, 7]
LN_TERMS = 30   # atanh series terms; z < 1/3 so the tail is below 3^-61
EXP_TERMS = 24  # Taylor terms on [0, ln 2); tail below 0.7^25/25! ~ 10^-27
# exp is refused above this argument (7, so every exp value is below 1097 < 2^11): the C++ backend
# keeps scale-48 values in 128-bit words and this is what keeps its products from overflowing.
EXP_LIMIT = 7 * S
# ln 2 at scale 2^48, rounded down and up.
LN2_L = 195103586505167
LN2_U = 195103586505168


def cdiv(a: int, b: int) -> int:
    """Ceiling of a / b for b > 0 (a may be negative)."""
    return -((-a) // b)


# ---- ln and exp with directed rounding ----------------------------------------------------------

def ln_bounds(n: int) -> tuple[int, int]:
    """(lower, upper) bounds on ln n at scale S, for n >= 1."""
    if n < 1:
        raise ValueError("ln of a number below 1")
    e = n.bit_length() - 1
    m = (n << K) >> e if e <= K else n >> (e - K)   # n / 2^e in [1, 2); exact when e <= K
    m_hi = m + (1 if e > K and (n & ((1 << (e - K)) - 1)) else 0)
    lo = _ln_mantissa(m, True)
    hi = _ln_mantissa(m_hi, False)
    return 2 * lo + e * LN2_L, 2 * hi + e * LN2_U + 1


def _ln_mantissa(m: int, lower: bool) -> int:
    """sum_{k<T} z^(2k+1)/(2k+1) with z = (m-1)/(m+1), rounded one way; ln m is twice this."""
    num = (m - S) * S
    den = m + S
    z = num // den if lower else cdiv(num, den)
    z2 = (z * z) // S if lower else cdiv(z * z, S)
    acc = 0
    power = z
    for k in range(LN_TERMS):
        acc += power // (2 * k + 1) if lower else cdiv(power, 2 * k + 1)
        power = (power * z2) // S if lower else cdiv(power * z2, S)
    return acc


def exp_upper(x: int) -> int:
    """Upper bound on exp(x / S) at scale S, for x at most EXP_LIMIT."""
    if x > EXP_LIMIT:
        raise ValueError("exponent exceeds 7")
    ell = LN2_L if x >= 0 else LN2_U
    q = x // ell
    r = x - q * ell
    acc = S
    term = S
    for k in range(1, EXP_TERMS + 1):
        term = cdiv(term * r, S * k)
        acc += term
    acc += 1
    if q >= 0:
        return acc << q
    return cdiv(acc, 1 << (-q))


def exp_lower(x: int) -> int:
    """Lower bound on exp(x / S) at scale S, for x at most EXP_LIMIT."""
    if x > EXP_LIMIT:
        raise ValueError("exponent exceeds 7")
    ell = LN2_U if x >= 0 else LN2_L
    q = x // ell
    r = x - q * ell
    acc = S
    term = S
    for k in range(1, EXP_TERMS + 1):
        term = (term * r) // (S * k)
        acc += term
    if q >= 0:
        return acc << q
    return acc >> (-q)


# ---- signed intervals at scale S ------------------------------------------------------------------

def imul(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    ps = [a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]]
    return min(ps) // S, cdiv(max(ps), S)


def iadd(a, b):
    return a[0] + b[0], a[1] + b[1]


def isub(a, b):
    return a[0] - b[1], a[1] - b[0]


def iscale(a: tuple[int, int], num: int, den: int) -> tuple[int, int]:
    """Interval times the rational num/den (num may be negative, den > 0)."""
    ps = [a[0] * num, a[1] * num]
    return min(ps) // den, cdiv(max(ps), den)


def iexp(a: tuple[int, int]) -> tuple[int, int]:
    return exp_lower(a[0]), exp_upper(a[1])


def abs_upper(a: tuple[int, int]) -> int:
    return max(abs(a[0]), abs(a[1]))


def divisors_of(primes: list[int], m: int) -> list[int]:
    """Squarefree divisors of D that divide m: products of subsets of the primes dividing m."""
    ds = [1]
    for p in primes:
        if m % p == 0:
            ds = ds + [d * p for d in ds]
    return sorted(ds)


# ---- the operations -------------------------------------------------------------------------------

class Params:
    def __init__(self, args: dict):
        self.t_num = int(args["t_num"])
        self.t_den = int(args["t_den"])
        if not (0 < self.t_num and 2 * self.t_num <= self.t_den):
            raise ValueError("t must be a rational in (0, 1/2]")
        self.scale = int(args.get("scale", K))
        if not (1 <= self.scale <= K):
            raise ValueError(f"scale must be between 1 and {K}")
        self.sigma_num = int(args.get("sigma_num", 0))
        self.sigma_den = int(args.get("sigma_den", 1))
        self.y_num = int(args.get("y_num", 0))
        self.y_den = int(args.get("y_den", 1))
        self.n_minus = int(args.get("n_minus", 1))
        self.n_plus = int(args.get("n_plus", self.n_minus))
        mask = int(args.get("primes", 0))
        self.primes = [p for i, p in enumerate(PRIMES) if mask >> i & 1]
        self.c_num = int(args.get("c_num", 1))
        self.c_den = int(args.get("c_den", 1))
        self.n0 = int(args.get("n0", 0))
        self.width = int(args.get("width", 1))
        if self.sigma_den <= 0 or self.y_den <= 0 or self.c_den <= 0 or self.t_den <= 0:
            raise ValueError("denominators must be positive")
        if self.y_num < 0 or self.y_num > self.y_den:
            raise ValueError("need 0 <= y <= 1")
        if self.n_minus < 1 or self.width < 1:
            raise ValueError("n_minus and width must be positive")
        if self.n_plus < self.n_minus:
            raise ValueError("need n_minus <= n_plus")
        self.D = 1
        for p in self.primes:
            self.D *= p

    def out(self, v: int) -> int:
        """Round a scale-S upper bound up to the requested output scale."""
        o = cdiv(v, 1 << (K - self.scale))
        if not (0 <= o < 1 << 64):
            raise ValueError("value does not fit in 64 bits")
        return o

    # exponent of g: (t/4) ln^2 n - sigma ln n, as an interval
    def g_exponent(self, lnn: tuple[int, int]) -> tuple[int, int]:
        sq = imul(lnn, lnn)
        a = iscale(sq, self.t_num, 4 * self.t_den)
        b = iscale(lnn, self.sigma_num, self.sigma_den)
        return isub(a, b)

    def g_upper(self, n: int) -> int:
        return exp_upper(self.g_exponent(ln_bounds(n))[1])

    def b_interval(self, p: int) -> tuple[int, int]:
        """b_p = exp((t/4) ln^2 p)."""
        lnp = ln_bounds(p)
        return iexp(iscale(imul(lnp, lnp), self.t_num, 4 * self.t_den))

    def lambda_interval(self, d: int) -> tuple[int, int]:
        acc = (S, S)
        for p in self.primes:
            if d % p == 0:
                b = self.b_interval(p)
                acc = imul(acc, (-b[1], -b[0]))
        return acc

    def rho_interval(self, n: int, d: int) -> tuple[int, int]:
        """rho_d(n) = exp(-(t/4) ln d (2 ln n - ln d)), which is b_{n/d} / b_n when d | n and
        decreases in n for every d >= 1."""
        lnn = ln_bounds(n)
        lnd = ln_bounds(d)
        delta = imul(lnd, isub((2 * lnn[0], 2 * lnn[1]), lnd))
        return iexp(iscale(delta, -self.t_num, 4 * self.t_den))

    def pi_interval(self, n: int, d: int) -> tuple[int, int]:
        """pi_d(n) = min(1, n / (d N_-))^y, increasing in n."""
        base = isub(isub(ln_bounds(n), ln_bounds(d)), ln_bounds(self.n_minus))
        clamped = (min(base[0], 0), min(base[1], 0))
        return iexp(iscale(clamped, self.y_num, self.y_den))

    def wc_interval(self) -> tuple[int, int]:
        """C N_-^(-y)."""
        lnN = ln_bounds(self.n_minus)
        return iscale(iexp(iscale((-lnN[1], -lnN[0]), self.y_num, self.y_den)), self.c_num, self.c_den)

    def r_upper(self) -> int:
        a = self.wc_interval()[0]
        if a < 0:
            a = 0
        return cdiv((S - a) * S, S + a) if a < S else 0

    def coefficient_upper(self, divs: list[int], rho: dict, pi: dict, lo: int, hi: int) -> int:
        """Upper bound on max(|beta - alpha|, r |beta + alpha|) over every n in [lo, hi] and every
        cutoff N in [N_-, N_+], from enclosures of rho_d and pi_d valid on [lo, hi].

        The polynomial is truncated at N, so only the divisors d with n / d <= N contribute to
        beta_n and alpha_n: the set {d : d >= n / N}, an upper set of the divisors. As n and N
        range, the threshold n / N ranges over [lo / N_+, hi / N_-]; every upper set whose cut
        lies in that range can occur, and the bound is the largest over them. Below N_- the whole
        set is the only one; beyond D N_+ the empty set is, and the bound is 0."""
        best = 0
        for j in range(len(divs) + 1):
            below = divs[j - 1] if j > 0 else 0
            above_ok = j == len(divs) or lo <= divs[j] * self.n_plus
            if not (above_ok and below * self.n_minus < hi):
                continue
            best = max(best, self.coefficient_of(divs[j:], rho, pi))
        return best

    def coefficient_of(self, divs: list[int], rho: dict, pi: dict) -> int:
        """max(|beta - alpha|, r |beta + alpha|) over the given divisor set."""
        beta = (0, 0)
        alpha = (0, 0)
        for d in divs:
            lam = self.lambda_interval(d)
            beta = iadd(beta, imul(lam, rho[d]))
            alpha = iadd(alpha, imul(imul(lam, pi[d]), rho[d]))
        alpha = iscale(alpha, self.c_num, self.c_den)
        diff = abs_upper(isub(beta, alpha))
        summ = abs_upper(iadd(beta, alpha))
        return max(diff, cdiv(self.r_upper() * summ, S))

    def term_upper(self, n: int) -> int:
        divs = divisors_of(self.primes, n)
        rho = {d: self.rho_interval(n, d) for d in divs}
        pi = {d: self.pi_interval(n, d) for d in divs}
        coeff = self.coefficient_upper(divs, rho, pi, n, n)
        return cdiv(coeff * self.g_upper(n), S)

    def block_upper(self, k: int) -> int:
        a = self.n0 + k * self.width
        b = a + self.width
        if a < 1:
            raise ValueError("blocks must start at 1 or above")
        g_max = max(self.g_upper(a), self.g_upper(b - 1))
        # rho_d decreases in n and pi_d increases, so the block ends enclose both for every n in
        # the block, whatever its residue class.
        def rho(d):
            return self.rho_interval(b - 1, d)[0], self.rho_interval(a, d)[1]

        def pi(d):
            return self.pi_interval(a, d)[0], self.pi_interval(b - 1, d)[1]

        # The residue class c mod D fixes the divisors of gcd(n, D), so the coefficient bound is
        # one per gcd class; count the members of each class.
        counts = {}
        for c in range(self.D):
            cnt = (b + self.D - 1 - c) // self.D - (a + self.D - 1 - c) // self.D
            e = gcd(c, self.D)
            counts[e] = counts.get(e, 0) + cnt
        total = 0
        for e in sorted(counts):
            if counts[e] == 0:
                continue
            divs = divisors_of(self.primes, e)
            coeff = self.coefficient_upper(divs, {d: rho(d) for d in divs}, {d: pi(d) for d in divs}, a, b - 1)
            total += counts[e] * cdiv(coeff * g_max, S)
        return total

    def sigma_lower(self, n: int) -> int:
        """(1+y)/2 + (t/4) ln(N^2 - 1) - t / (144 (N^2 - 1)^2), rounded down, for N >= 2."""
        if n < 2:
            raise ValueError("sigma_lower needs a cutoff of at least 2")
        m = n * n - 1
        half = (S * (self.y_den + self.y_num)) // (2 * self.y_den)
        main = (self.t_num * ln_bounds(m)[0]) // (4 * self.t_den)
        corr = cdiv(S * self.t_num, self.t_den * 144 * m * m)
        return half + main - corr


def numbers(family: Family, prefix: int | None):
    ms = list(itertools.islice(rt.iter_members(family), prefix))
    if rt.prime(family) != NATURALS:
        raise ValueError("heat_dirichlet operations need natural numbers, not elements of a field")
    if any(len(m) != 1 or len(m[0]) != 1 for m in ms):
        raise ValueError("heat_dirichlet operations need 1 x 1 members")
    return ms, [m[0][0] for m in ms]


def run(op: str, family: Family, reduction: str = "all", prefix: int | None = None, **args):
    op = op.removeprefix("heat_dirichlet.")
    ms, ns = numbers(family, prefix)
    P = Params(args)
    if op == "weight_upper":
        if any(n < 1 for n in ns):
            raise ValueError("weight_upper needs members of at least 1")
        values = [P.out(P.g_upper(n)) for n in ns]
    elif op == "mollified_term_upper":
        if any(n < 1 for n in ns):
            raise ValueError("mollified_term_upper needs members of at least 1")
        values = [P.out(P.term_upper(n)) for n in ns]
    elif op == "block_term_upper":
        values = [P.out(P.block_upper(k)) for k in ns]
    elif op == "sigma_lower":
        values = [max(0, P.sigma_lower(n)) >> (K - P.scale) for n in ns]
    else:
        raise ValueError(f"unknown operation {op}")
    return rt.reduce_int(reduction, values, ms, NATURALS)
