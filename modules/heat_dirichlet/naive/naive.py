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

`phase_bound` is the phase-aware bound. Write every n = m k with m 7-smooth (m = 2^a 3^b 5^c 7^e)
and k coprime to 210, and let theta = (theta_2, theta_3, theta_5, theta_7) be the phases of
2^(-iT), ..., 7^(-iT), shared by every term, and psi the phase of the partner sum. With a
general mollifier E(s) = sum_d mu_d d^(-s) on 7-smooth d, the mollified polynomial is
sum_k k^(-sigma) b_k e^(i phi_k) T_k(theta, psi) with independent rough phases phi_k, where

    T_k(theta, psi) = sum_m [cS_{m,k} + e^(i psi) cA_{m,k}] e^(i theta(m)),   theta(m) = sum_p v_p(m) theta_p,
    cS_{m,k} = sum_{d | m, mk/d <= N} mu_d b_{m/d} (m/d)^(eps_k) m^(-sigma),    eps_k = (t/2) ln k,
    cA_{m,k} = C sum_{d | m, mk/d <= N} mu_d (mk/(d N_-))^y b_{m/d} (m/d)^(eps_k) m^(-sigma),

so that dist(E f, (-inf, 0]) >= F(theta, psi) - |E| Z with

    F(theta, psi) = dist(T_1(theta, psi), (-inf, 0]) - sum_{k > 1} b_k k^(-sigma) |T_k(theta, psi)|.

The rough k are grouped into bins [K_i, K_{i+1}) given by the caller; a bin's coefficients are
enclosed over its k (eps_k and the partner weight are monotone, a term whose cutoff condition
mk/d <= N depends on k or N is hulled with 0), and the bin costs W_i max |T| with W_i the sum
of b_k k^(-sigma) over its rough k. A member is a box of the grid g2 x g3 x g5 x g7 x gpsi on
the torus; the value is a lower bound on F over the box, from the centre value, the gradient,
and the second-order remainder (|e^(ix) - 1 - ix| <= x^2/2), plus `offset`, at scale 2^scale.
sigma ranges over [sigma_num, sigma_hi_num] / sigma_den across the cell.
"""
from __future__ import annotations

import itertools
import sys
from math import gcd, isqrt
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
# pi at scale 2^48, rounded down and up.
PI_L = 884279719003555
PI_U = 884279719003556
TRIG_TERMS = 13  # alternating series on [0, pi/2]; the first omitted term is below 2^-70


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


# ---- phase_bound ---------------------------------------------------------------------------------

def sqrt_lower(a: int) -> int:
    """Lower bound on sqrt(a / S) at scale S, for a >= 0."""
    return isqrt(a * S)


def sqrt_upper(a: int) -> int:
    r = isqrt(a * S)
    return r if r * r == a * S else r + 1


def isq(a: tuple[int, int]) -> tuple[int, int]:
    """The square of an interval, tight at 0."""
    lo, hi = a
    top = max(lo * lo, hi * hi)
    bottom = 0 if lo <= 0 <= hi else min(lo * lo, hi * hi)
    return bottom // S, cdiv(top, S)


def abs_lower(a: tuple[int, int]) -> int:
    lo, hi = a
    return 0 if lo <= 0 <= hi else min(abs(lo), abs(hi))


def powfact(x: int, n: int, lower: bool) -> int:
    """x^n / n! at scale S for x >= 0, rounded one way."""
    term = S
    for i in range(1, n + 1):
        term = (term * x) // (S * i) if lower else cdiv(term * x, S * i)
    return term


def cos_bounds(x: int) -> tuple[int, int]:
    """(lower, upper) bounds on cos(x / S) for 0 <= x <= pi/2: the alternating series with each
    term rounded the safe way, and the first omitted term as the tail."""
    lo = 0
    hi = 0
    for k in range(TRIG_TERMS):
        if k % 2 == 0:
            lo += powfact(x, 2 * k, True)
            hi += powfact(x, 2 * k, False)
        else:
            lo -= powfact(x, 2 * k, False)
            hi -= powfact(x, 2 * k, True)
    tail = powfact(x, 2 * TRIG_TERMS, False)
    return lo - tail, hi + tail


def sin_bounds(x: int) -> tuple[int, int]:
    lo = 0
    hi = 0
    for k in range(TRIG_TERMS):
        if k % 2 == 0:
            lo += powfact(x, 2 * k + 1, True)
            hi += powfact(x, 2 * k + 1, False)
        else:
            lo -= powfact(x, 2 * k + 1, False)
            hi -= powfact(x, 2 * k + 1, True)
    tail = powfact(x, 2 * TRIG_TERMS + 1, False)
    return lo - tail, hi + tail


def unit_circle(M: int, J: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Enclosures of cos and sin of 2 pi J / M, for M a multiple of 4 and 0 <= J < M: reduce to
    the first quadrant, where cos decreases and sin increases, then rotate."""
    quarter = M // 4
    q, j = divmod(J, quarter)
    x_lo = (2 * PI_L * j) // M
    x_hi = cdiv(2 * PI_U * j, M)
    c = (cos_bounds(x_hi)[0], cos_bounds(x_lo)[1])
    s_ = (sin_bounds(x_lo)[0], sin_bounds(x_hi)[1])
    neg = lambda a: (-a[1], -a[0])  # noqa: E731
    if q == 0:
        return c, s_
    if q == 1:
        return neg(s_), c
    if q == 2:
        return neg(c), neg(s_)
    return s_, neg(c)


def smooth_numbers(limit: int) -> list[tuple[int, tuple[int, int, int, int]]]:
    """Every 2^a 3^b 5^c 7^e <= limit with its exponent vector, in increasing order."""
    out = []
    a = 1
    while a <= limit:
        b = a
        while b <= limit:
            c = b
            while c <= limit:
                e = c
                while e <= limit:
                    out.append(e)
                    e *= 7
                c *= 5
            b *= 3
        a *= 2
    vectors = []
    for m in sorted(out):
        v = []
        r = m
        for p in PRIMES:
            k = 0
            while r % p == 0:
                r //= p
                k += 1
            v.append(k)
        vectors.append((m, tuple(v)))
    return vectors


def is_rough(k: int) -> bool:
    return gcd(k, 210) == 1


def read_rows(arg) -> list[list[int]]:
    """A `vectors` argument as a list of rows, whether it came as one matrix or a batch of rows."""
    if hasattr(arg, "value"):
        arg = arg.value()
    if hasattr(arg, "tolist"):
        mats = arg.tolist()
        return [row for mat in mats for row in mat]
    return [list(r) for r in arg]


class PhaseParams(Params):
    """The phase-aware bound's request: the cell, the grid, the mollifier and the bins, with
    every member-independent quantity computed once."""

    def __init__(self, args: dict):
        super().__init__(args)
        self.sigma_hi_num = int(args.get("sigma_hi_num", self.sigma_num))
        if self.sigma_hi_num < self.sigma_num:
            raise ValueError("need sigma_num <= sigma_hi_num")
        self.g = [int(args.get(name, 1)) for name in ("g2", "g3", "g5", "g7", "gpsi")]
        if any(x < 1 for x in self.g):
            raise ValueError("grid sizes must be positive")
        self.offset = int(args.get("offset", 0))
        if self.offset < 0:
            raise ValueError("offset must be non-negative")
        self.mollifier = []
        seen = set()
        for row in read_rows(args["mollifier"]):
            if len(row) != 4:
                raise ValueError("mollifier rows are (d, sign, num, den)")
            d, sign, num, den = (int(x) for x in row)
            r = d
            for p in PRIMES:
                while r % p == 0:
                    r //= p
            if d < 1 or r != 1 or d > self.n_plus:
                raise ValueError("mollifier support must be 7-smooth and at most n_plus")
            if d in seen:
                raise ValueError("mollifier support has a repeated d")
            if sign not in (0, 1) or den < 1:
                raise ValueError("mollifier sign must be 0 or 1 and den positive")
            seen.add(d)
            self.mollifier.append((d, -num if sign else num, den))
        rows = read_rows(args["bins"])
        if len(rows) != 1:
            raise ValueError("bins is one row of boundaries")
        self.bins = [int(x) for x in rows[0]]
        if len(self.bins) < 2 or self.bins[0] != 2 or self.bins[-1] <= self.n_plus \
                or any(a >= b for a, b in zip(self.bins, self.bins[1:])):
            raise ValueError("bins must be increasing, start at 2 and end above n_plus")
        self.sigma_iv = ((S * self.sigma_num) // self.sigma_den, cdiv(S * self.sigma_hi_num, self.sigma_den))
        lcm = 1
        for x in self.g:
            lcm = lcm * x // gcd(lcm, x)
        self.M = 4 * lcm
        self.h = [cdiv(PI_U, x) for x in self.g]   # half-widths of a box, in radians at scale S
        # the mollified polynomial reaches d N for d in the support: n / d <= N, not n <= N
        self.smooth = smooth_numbers(max(d for d, _, _ in self.mollifier) * self.n_plus)
        self.circle = [unit_circle(self.M, J) for J in range(self.M)]
        self.lnN = ln_bounds(self.n_minus)
        # the head and the bins: (W, Q, terms) with terms a list of (index into smooth, cS, cA)
        self.head = self.polynomial(1, 1, True)
        self.tail = []
        for lo, hi in zip(self.bins, self.bins[1:]):
            ks = [k for k in range(lo, min(hi, self.n_plus + 1)) if is_rough(k)]
            if not ks:
                continue
            W = sum(self.g_upper(k) for k in ks)
            self.tail.append((W,) + self.polynomial(ks[0], ks[-1], False)[1:])

    def polynomial(self, ka: int, kb: int, head: bool):
        """The coefficient enclosures of T_k over rough k in [ka, kb] and cutoffs in the cell, the
        second-order constant Q, and W = 1 (the head's weight is not used)."""
        Lk = (0, 0) if head else (ln_bounds(ka)[0], ln_bounds(kb)[1])
        terms = []
        mass = 0
        Q = 0
        for idx, (m, v) in enumerate(self.smooth):
            Lm = ln_bounds(m)
            cS = (0, 0)
            cA = (0, 0)
            for d, num, den in self.mollifier:
                if m % d:
                    continue
                j = m // d
                if j * ka > self.n_plus:
                    continue
                certain = j * kb <= self.n_minus
                Lj = ln_bounds(j)
                e_main = isub(iadd(iscale(imul(Lj, Lj), self.t_num, 4 * self.t_den),
                                   iscale(imul(Lj, Lk), self.t_num, 2 * self.t_den)),
                              imul(Lm, self.sigma_iv))
                e_part = iadd(e_main, iscale(isub(iadd(Lj, Lk), self.lnN), self.y_num, self.y_den))
                main = iscale(iexp(e_main), num, den)
                part = iscale(iscale(iexp(e_part), num, den), self.c_num, self.c_den)
                if not certain:
                    main = (min(main[0], 0), max(main[1], 0))
                    part = (min(part[0], 0), max(part[1], 0))
                cS = iadd(cS, main)
                cA = iadd(cA, part)
            if cS == (0, 0) and cA == (0, 0):
                continue
            # The arc a term sweeps over a box: half-angle vh (radians at scale S) and X (table
            # steps). Up to one radian the term is expanded to first order around the centre
            # and its second-order remainder |e^(ix) - 1 - ix| <= x^2/2 goes into Q; beyond that
            # the arc is enclosed by its rectangle instead, with no remainder.
            vh = sum(vp * hp for vp, hp in zip(v, self.h[:4]))
            X = sum(vp * (self.M // (2 * gx)) for vp, gx in zip(v, self.g[:4]))
            aS = abs_upper(cS)
            aA = abs_upper(cA)
            mass += aS + aA
            if vh <= S:
                Q += cdiv(aS * cdiv(vh * vh, S), S)
            if vh + self.h[4] <= S:
                Q += cdiv(aA * cdiv((vh + self.h[4]) ** 2, S), S)
            terms.append((idx, cS, cA, vh, X))
        if mass > 16 * S:
            raise ValueError("polynomial mass exceeds 16")
        return 1, cdiv(Q, 2), terms

    def arc(self, J: int, X: int) -> tuple[tuple[int, int], tuple[int, int]]:
        """Enclosures of cos and sin over the arc of table steps [J - X, J + X]: an extremum lies
        inside when the arc holds a step congruent to it, otherwise the ends are extreme."""
        M = self.M
        if 2 * X >= M:
            return (-S, S), (-S, S)
        a, b = J - X, J + X

        def contains(r: int) -> bool:
            return (b - r) // M >= cdiv(a - r, M)

        ca, sa = self.circle[a % M]
        cb, sb = self.circle[b % M]
        cos_hi = S if contains(0) else max(ca[1], cb[1])
        cos_lo = -S if contains(M // 2) else min(ca[0], cb[0])
        sin_hi = S if contains(M // 4) else max(sa[1], sb[1])
        sin_lo = -S if contains(3 * M // 4) else min(sa[0], sb[0])
        return (cos_lo, cos_hi), (sin_lo, sin_hi)

    def box_of(self, member: list[int]) -> list[int]:
        if len(member) == 1:
            i = member[0]
            js = []
            for gx in reversed(self.g):
                i, r = divmod(i, gx)
                js.append(r)
            if i:
                raise ValueError("box index beyond the grid")
            return js[::-1]
        if len(member) != 5:
            raise ValueError("a box is one index or five")
        if any(not (0 <= j < gx) for j, gx in zip(member, self.g)):
            raise ValueError("box index beyond the grid")
        return list(member)

    def evaluate(self, poly, js: list[int]):
        """Centre value T_c and gradient G (five complex intervals) of a polynomial on the box."""
        _, _, terms = poly
        steps = [(2 * j + 1) * (self.M // (2 * gx)) for j, gx in zip(js, self.g)]
        Tre, Tim = (0, 0), (0, 0)
        G = [[(0, 0), (0, 0)] for _ in range(5)]
        for idx, cS, cA, vh, X in terms:
            _, v = self.smooth[idx]
            J = sum(vp * sp for vp, sp in zip(v, steps[:4])) % self.M
            for c, JJ, XX, taylor, part in ((cS, J, X, vh <= S, False),
                                            (cA, (J + steps[4]) % self.M, X + self.M // (2 * self.g[4]),
                                             vh + self.h[4] <= S, True)):
                if c == (0, 0):
                    continue
                if not taylor:
                    cs, sn = self.arc(JJ, XX)
                    Tre = iadd(Tre, imul(c, cs))
                    Tim = iadd(Tim, imul(c, sn))
                    continue
                cs, sn = self.circle[JJ]
                wre, wim = imul(c, cs), imul(c, sn)
                Tre = iadd(Tre, wre)
                Tim = iadd(Tim, wim)
                # i w = (-Im w, Re w), times the exponent
                for p in range(4):
                    if v[p]:
                        G[p][0] = iadd(G[p][0], (-v[p] * wim[1], -v[p] * wim[0]))
                        G[p][1] = iadd(G[p][1], (v[p] * wre[0], v[p] * wre[1]))
                if part:
                    G[4][0] = iadd(G[4][0], (-wim[1], -wim[0]))
                    G[4][1] = iadd(G[4][1], wre)
        return (Tre, Tim), G

    def mod_upper(self, z) -> int:
        ar, ai = abs_upper(z[0]), abs_upper(z[1])
        return sqrt_upper(cdiv(ar * ar, S) + cdiv(ai * ai, S))

    def bounds(self, poly, js):
        """(modulus upper bound, dist lower bound) of the polynomial over the box."""
        _, Q, _ = poly
        (Tre, Tim), G = self.evaluate(poly, js)
        A = iadd(isq(Tre), isq(Tim))
        B = sum(cdiv(h * abs_upper(iadd(imul(Tre, g[0]), imul(Tim, g[1]))), S) for h, g in zip(self.h, G))
        Cc = sum(cdiv(h * self.mod_upper(g), S) for h, g in zip(self.h, G))
        upper = sqrt_upper(A[1] + 2 * B + cdiv(Cc * Cc, S)) + Q
        lower_mod = sqrt_lower(max(0, A[0] - 2 * B)) - Q
        re_margin = Tre[0] - sum(cdiv(h * abs_upper(g[0]), S) for h, g in zip(self.h, G)) - Q
        im_lower = abs_lower(Tim) - sum(cdiv(h * abs_upper(g[1]), S) for h, g in zip(self.h, G)) - Q
        dist = max(0, im_lower, lower_mod if re_margin >= 0 else 0)
        return upper, dist

    def phase_bound(self, member: list[int]) -> int:
        js = self.box_of(member)
        F = self.bounds(self.head, js)[1]
        for poly in self.tail:
            F -= cdiv(poly[0] * self.bounds(poly, js)[0], S)
        v = (F + self.offset * S) >> (K - self.scale)
        if v < 0:
            return 0
        if v >= 1 << 64:
            raise ValueError("value does not fit in 64 bits")
        return v


def numbers(family: Family, prefix: int | None, boxes: bool = False):
    ms = list(itertools.islice(rt.iter_members(family), prefix))
    if rt.prime(family) != NATURALS:
        raise ValueError("heat_dirichlet operations need natural numbers, not elements of a field")
    if boxes:
        if any(len(m) != 1 or len(m[0]) not in (1, 5) for m in ms):
            raise ValueError("phase_bound needs 1 x 1 or 1 x 5 members")
        return ms, [list(m[0]) for m in ms]
    if any(len(m) != 1 or len(m[0]) != 1 for m in ms):
        raise ValueError("heat_dirichlet operations need 1 x 1 members")
    return ms, [m[0][0] for m in ms]


def run(op: str, family: Family, reduction: str = "all", prefix: int | None = None, **args):
    op = op.removeprefix("heat_dirichlet.")
    if op == "phase_bound":
        ms, boxes = numbers(family, prefix, boxes=True)
        P = PhaseParams(args)
        return rt.reduce_int(reduction, [P.phase_bound(b) for b in boxes], ms, NATURALS)
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
