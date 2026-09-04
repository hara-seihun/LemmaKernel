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

The rough k are grouped into bins [K_i, K_{i+1}) given by the caller; a bin costs W_i max |T|
with W_i the sum of b_k k^(-sigma) over its rough k. Its coefficients are taken at the bin centre
k_c = isqrt(K_a K_b) and expanded in eps_k about eps_c to Taylor order `order`: (m/d)^(eps_k) =
(m/d)^(eps_c) sum_r (delta ln(m/d))^r / r! + remainder, so |T_k| <= sum_r delta^r / r! |T^(r)|
with T^(r) the polynomial whose terms carry (ln j)^r and delta the largest |eps_k - eps_c| over
the bin; the remainder, (delta ln j)^(order+1) / (order+1)! e^(delta ln j) times each term's
size, is a loss. A term whose cutoff condition mk/d <= N depends on k in the bin or on N is
hulled with 0. Terms with m > m0 are not evaluated: their l1 mass is a loss too, as is any tail
polynomial whose weighted low mass is at most `prune` (scale 2^48). A member is a theta box of
the grid g2 x g3 x g5 x g7 on the torus; psi is sampled at npsi boxes inside the operation and
the value is the minimum over them, a lower bound on F over the box from the centre value, the
gradient, and the second-order remainder (|e^(ix) - 1 - ix| <= x^2/2) of every polynomial, minus
the losses, plus `offset`, at scale 2^scale. sigma ranges over [sigma_num, sigma_hi_num] /
sigma_den across the cell.
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
    """The phase-aware bound's request: the cell, the theta grid and the psi samples, the mollifier,
    the bins of rough k, the low/high split at m0, the Taylor order in eps and the pruning
    threshold, with every member-independent quantity computed once.

    A component is one polynomial to evaluate on a box: its weight W (at scale S), its terms over
    the low smooth m <= m0 (index, cS, cA, vh, X) and its second-order constant Q. Everything not
    evaluated (high terms m > m0, Taylor remainders, pruned components) is an l1 mass folded into
    `loss`, subtracted from every box."""

    def __init__(self, args: dict):
        super().__init__(args)
        self.sigma_hi_num = int(args.get("sigma_hi_num", self.sigma_num))
        if self.sigma_hi_num < self.sigma_num:
            raise ValueError("need sigma_num <= sigma_hi_num")
        self.g = [int(args.get(name, 1)) for name in ("g2", "g3", "g5", "g7")]
        self.npsi = int(args.get("npsi", 1))
        if any(x < 1 for x in self.g) or self.npsi < 1:
            raise ValueError("grid sizes and npsi must be positive")
        self.m0 = int(args.get("m0", self.n_plus))
        if self.m0 < 1:
            raise ValueError("m0 must be positive")
        self.order = int(args.get("order", 2))
        if not (0 <= self.order <= 3):
            raise ValueError("order must be 0, 1, 2 or 3")
        self.prune = int(args.get("prune", 0))
        if self.prune < 0:
            raise ValueError("prune must be non-negative")
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
        for x in self.g + [self.npsi]:
            lcm = lcm * x // gcd(lcm, x)
        self.M = 4 * lcm
        self.h = [cdiv(PI_U, x) for x in self.g + [self.npsi]]   # half-widths, radians at scale S
        # the mollified polynomial reaches d N for d in the support: n / d <= N, not n <= N
        self.smooth = smooth_numbers(max(d for d, _, _ in self.mollifier) * self.n_plus)
        self.circle = [unit_circle(self.M, J) for J in range(self.M)]
        self.lnN = ln_bounds(self.n_minus)
        self.loss = 0
        self.components = []
        self.head = self.polynomial(1, S, 1, 0, True)
        self.components.append(self.head)
        for lo, hi in zip(self.bins, self.bins[1:]):
            ks = [k for k in range(lo, min(hi, self.n_plus + 1)) if is_rough(k)]
            if not ks:
                continue
            W = sum(self.g_upper(k) for k in ks)
            ka, kb = ks[0], ks[-1]
            kc = isqrt(ka * kb)
            eps = lambda k: iscale(ln_bounds(k), self.t_num, 2 * self.t_den)  # noqa: E731
            ec, ea, eb = eps(kc), eps(ka), eps(kb)
            delta = max(ec[1] - ea[0], eb[1] - ec[0], 0)   # |eps_k - eps_c| over the bin, at scale S
            # the r-th component carries (delta ln j)^r / r! on every term; the remainder past
            # `order` is a loss, charged in the r = 0 pass
            for r in range(self.order + 1):
                self.components.append(self.polynomial(kc, W, ka, r, False, delta, kb))
        if self.loss > 16 * S:
            raise ValueError("loss exceeds 16")

    def polynomial(self, kc: int, W: int, ka: int, r: int, head: bool, delta=0, kb=None):
        """The component with weight W: coefficient enclosures of sum_m [cS + e^(i psi) cA] e^(i theta(m))
        at the bin centre kc, every term multiplied by (delta ln j)^r / r!, over the cell's
        cutoffs; terms with m > m0 and (in the r = 0 pass) the Taylor remainder go to the loss.
        Returns (W, Q, terms) with terms over low m only, or None when pruned."""
        kb = kc if kb is None else kb
        Lk = (0, 0) if head else ln_bounds(kc)
        terms = []
        mass_low = 0
        mass_high = 0
        rem = 0
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
                for q in range(1, r + 1):
                    main = iscale(imul(main, Lj), delta, S * q)
                    part = iscale(imul(part, Lj), delta, S * q)
                if not certain:
                    main = (min(main[0], 0), max(main[1], 0))
                    part = (min(part[0], 0), max(part[1], 0))
                if r == 0 and not head and j > 1:
                    # the Taylor remainder in eps: (delta ln j)^(order+1) / (order+1)! e^(delta ln j)
                    # times the term's size
                    x = cdiv(delta * Lj[1], S)
                    factor = cdiv(powfact(x, self.order + 1, False) * exp_upper(x), S)
                    rem += cdiv((abs_upper(main) + abs_upper(part)) * factor, S)
                cS = iadd(cS, main)
                cA = iadd(cA, part)
            if cS == (0, 0) and cA == (0, 0):
                continue
            aS = abs_upper(cS)
            aA = abs_upper(cA)
            if m > self.m0:
                mass_high += aS + aA
                continue
            vh = sum(vp * hp for vp, hp in zip(v, self.h[:4]))
            X = sum(vp * (self.M // (2 * gx)) for vp, gx in zip(v, self.g))
            mass_low += aS + aA
            if vh <= S:
                Q += cdiv(aS * cdiv(vh * vh, S), S)
            if vh + self.h[4] <= S:
                Q += cdiv(aA * cdiv((vh + self.h[4]) ** 2, S), S)
            terms.append((idx, cS, cA, vh, X))
        if mass_low + mass_high > 16 * S:
            raise ValueError("polynomial mass exceeds 16")
        self.loss += cdiv(W * (mass_high + rem), S)
        if not head and cdiv(W * mass_low, S) <= self.prune:
            self.loss += cdiv(W * mass_low, S)
            return None
        return W, cdiv(Q, 2), terms, mass_low

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
        if len(member) != 4:
            raise ValueError("a box is one index or four")
        if any(not (0 <= j < gx) for j, gx in zip(member, self.g)):
            raise ValueError("box index beyond the grid")
        return list(member)

    def evaluate(self, poly, js: list[int]):
        """The S and A parts of a component on the theta box: centre values and the four theta
        gradients of each, as complex intervals; the arc-enclosed terms (sweep beyond a radian)
        carry no gradient."""
        _, _, terms, _ = poly
        steps = [(2 * j + 1) * (self.M // (2 * gx)) for j, gx in zip(js, self.g)]
        parts = []
        for which in (0, 1):
            Tre, Tim = (0, 0), (0, 0)
            G = [[(0, 0), (0, 0)] for _ in range(4)]
            for idx, cS, cA, vh, X in terms:
                _, v = self.smooth[idx]
                c = cA if which else cS
                if c == (0, 0):
                    continue
                J = sum(vp * sp for vp, sp in zip(v, steps)) % self.M
                taylor = vh <= S if which == 0 else vh + self.h[4] <= S
                if not taylor:
                    XX = X if which == 0 else X + self.M // (2 * self.npsi)
                    cs, sn = self.arc(J, XX)
                    Tre = iadd(Tre, imul(c, cs))
                    Tim = iadd(Tim, imul(c, sn))
                    continue
                cs, sn = self.circle[J]
                wre, wim = imul(c, cs), imul(c, sn)
                Tre = iadd(Tre, wre)
                Tim = iadd(Tim, wim)
                for p in range(4):
                    if v[p]:
                        G[p][0] = iadd(G[p][0], (-v[p] * wim[1], -v[p] * wim[0]))
                        G[p][1] = iadd(G[p][1], (v[p] * wre[0], v[p] * wre[1]))
            parts.append(((Tre, Tim), G))
        return parts

    def mod_upper(self, z) -> int:
        ar, ai = abs_upper(z[0]), abs_upper(z[1])
        return sqrt_upper(cdiv(ar * ar, S) + cdiv(ai * ai, S))

    @staticmethod
    def rotate(z, cs, sn):
        """z e^(i psi) for the complex interval z and the enclosure (cs, sn) of e^(i psi)."""
        re, im = z
        return isub(imul(re, cs), imul(im, sn)), iadd(imul(re, sn), imul(im, cs))

    def bounds(self, poly, parts, jpsi: int):
        """(modulus upper bound, dist lower bound) of the component over the theta box and the
        psi box jpsi, from the S and A parts."""
        _, Q, _, mass = poly
        (Sc, GS), (Ac, GA) = parts
        cs, sn = self.circle[(2 * jpsi + 1) * (self.M // (2 * self.npsi))]
        Ar = self.rotate(Ac, cs, sn)
        Tre, Tim = iadd(Sc[0], Ar[0]), iadd(Sc[1], Ar[1])
        G = []
        for p in range(4):
            gr = self.rotate(GA[p], cs, sn)
            G.append([iadd(GS[p][0], gr[0]), iadd(GS[p][1], gr[1])])
        # i A e^(i psi) = (-Im, Re) of the rotated partner
        G.append([(-Ar[1][1], -Ar[1][0]), Ar[0]])
        A = iadd(isq(Tre), isq(Tim))
        B = sum(cdiv(h * abs_upper(iadd(imul(Tre, g[0]), imul(Tim, g[1]))), S) for h, g in zip(self.h, G))
        Cc = sum(cdiv(h * self.mod_upper(g), S) for h, g in zip(self.h, G))
        upper = min(sqrt_upper(A[1] + 2 * B + cdiv(Cc * Cc, S)) + Q, mass)   # |T| never exceeds the l1 mass
        lower_mod = sqrt_lower(max(0, A[0] - 2 * B)) - Q
        re_margin = Tre[0] - sum(cdiv(h * abs_upper(g[0]), S) for h, g in zip(self.h, G)) - Q
        im_lower = abs_lower(Tim) - sum(cdiv(h * abs_upper(g[1]), S) for h, g in zip(self.h, G)) - Q
        dist = max(0, im_lower, lower_mod if re_margin >= 0 else 0)
        return upper, dist

    def tail_upper(self, poly, parts, jpsi: int) -> int:
        """Upper bound on |T| of a tail component over the theta box and the psi box jpsi, with
        the first-order term by the triangle inequality |G_p| <= |G^S_p| + |G^A_p| (so only the
        centre value depends on psi): |T(c)| + sum_p h_p (|G^S_p| + |G^A_p|) + h_psi |A(c)| + Q,
        capped by the l1 mass."""
        _, Q, _, mass = poly
        (Sc, GS), (Ac, GA) = parts
        cs, sn = self.circle[(2 * jpsi + 1) * (self.M // (2 * self.npsi))]
        Ar = self.rotate(Ac, cs, sn)
        T = (iadd(Sc[0], Ar[0]), iadd(Sc[1], Ar[1]))
        first = sum(cdiv(h * (self.mod_upper(gs) + self.mod_upper(ga)), S) for h, gs, ga in zip(self.h, GS, GA))
        first += cdiv(self.h[4] * self.mod_upper(Ac), S)
        return min(self.mod_upper(T) + first + Q, mass)

    def phase_bound(self, member: list[int]) -> int:
        js = self.box_of(member)
        evaluated = [(poly, self.evaluate(poly, js)) for poly in self.components if poly is not None]
        best = None
        for jpsi in range(self.npsi):
            F = self.bounds(evaluated[0][0], evaluated[0][1], jpsi)[1]
            for poly, parts in evaluated[1:]:
                F -= cdiv(poly[0] * self.tail_upper(poly, parts, jpsi), S)
            best = F if best is None else min(best, F)
        v = (best - self.loss + self.offset * S) >> (K - self.scale)
        if v < 0:
            return 0
        if v >= 1 << 64:
            raise ValueError("value does not fit in 64 bits")
        return v


# ---- barrier_lower --------------------------------------------------------------------------------
#
# A lower bound on |f_t(x + iy)| (or on Re f_t) over a box in (x, y, t), for the barrier of the
# de Bruijn-Newman programme, where x is of order 10^13 and the cutoff N of order 10^6. With
# s = (1 + y - ix)/2, tau = x/2 and A = (1/2) ln(x / 4 pi), the model's quantities are
#
#     alpha(s)  = A - i pi/4 + e_1,                                  |e_1| <= 2.01 / tau,
#     kappa     = (t/2)(alpha(1 - conj s) - alpha(conj s)),          |kappa| <= t y / (2 (x - 6)),
#     log gamma = -y A + i [ (x/2)(2A - 1) - pi/4 + (pi t/4) A ] + eps  (mod 2 pi i),
#                                                                    |eps| <= (2 + t (4 + 4.2 A)) / x,
#
# (canon/barrier/design.md of the programme proves the three), and
#
#     f_t = sum_{n <= N} exp(c_1 L + c_2 L^2) + gamma sum_{n <= N} exp(d_1 L + d_2 L^2),   L = ln n,
#     c_1 = -s - (t/2) alpha(s),   d_1 = y - conj(s) - (t/2) conj(alpha(s)) - kappa,   c_2 = d_2 = t/4.
#
# The moment method: with a centre (x_c, y_c, t_c), every term is w_n exp(dc_1 L + dc_2 L^2) with
# w_n the term at the centre and dc the displacement of the exponent. Group n into dyadic ranges r
# with centre L_r, L = L_r + l, |l| <= l_max; then exp(dc_1 L + dc_2 L^2) = P_r exp(a l + b l^2)
# with P_r = exp(dc_1 L_r + dc_2 L_r^2), a = dc_1 + 2 dc_2 L_r, b = dc_2, and
#
#     sum_{n in r} w_n exp(...) = P_r sum_{j < J} q_j m_{r,j} + P_r * remainder,
#     m_{r,j} = sum_{n in r} w_n l^j,     q_j = Taylor coefficients of exp(a l + b l^2),
#     |remainder| <= (sum_{n in r} |w_n|) sum_{k >= ceil(J/2)} U^k / k!,    U = |a| l_max + |b| l_max^2.
#
# The moments are complex intervals computed once per request; a box costs the ranges times J.
# Phases: (x_c/2) ln n needs ln n to about 2^-90, so ln n for n <= N is carried at scale 2^112 by
# the recurrence ln n = ln(n-1) + 2 artanh(1/(2n-1)), whose series terms are integer divisions,
# and (x/2)(ln(x/4 pi) - 1) at the grid abscissae uses ln at 2^112 by the mantissa series. Both
# are reduced modulo 2 pi in exact integers and only then brought to scale 2^48.

KH = 112
SH = 1 << KH
TWOPI_HL = 32624163332060752803334972325496544      # floor(2 pi 2^112); 2 pi 2^112 < TWOPI_HL + 1
LN2_HL = 3599025928123676973540407451845618         # floor(ln 2 * 2^112)
LN2_HU = LN2_HL + 1
LN4PI_HL = 13141829246414126302627206044224549      # floor(ln(4 pi) * 2^112)
LN4PI_HU = LN4PI_HL + 1
LNH_TERMS = 36   # atanh series at 2^112; z < 1/3 so the tail is below 3^-73 / 73
LN4PI_L = LN4PI_HL >> (KH - K)
LN4PI_U = cdiv(LN4PI_HU, 1 << (KH - K))
ABS_CAP = 1 << 60


def ln_high(n: int) -> tuple[int, int]:
    """(lower, upper) bounds on ln n at scale SH, for n >= 1, by the mantissa series."""
    e = n.bit_length() - 1
    m = (n << KH) >> e if e <= KH else n >> (e - KH)
    m_hi = m + (1 if e > KH and (n & ((1 << (e - KH)) - 1)) else 0)
    return 2 * _lnh_mantissa(m, True) + e * LN2_HL, 2 * _lnh_mantissa(m_hi, False) + e * LN2_HU + 1


def _lnh_mantissa(m: int, lower: bool) -> int:
    num = (m - SH) * SH
    den = m + SH
    z = num // den if lower else cdiv(num, den)
    z2 = (z * z) // SH if lower else cdiv(z * z, SH)
    acc = 0
    power = z
    for k in range(LNH_TERMS):
        acc += power // (2 * k + 1) if lower else cdiv(power, 2 * k + 1)
        power = (power * z2) // SH if lower else cdiv(power * z2, SH)
    return acc


def artanh_step(n: int) -> tuple[int, int]:
    """(lower, upper) bounds on artanh(1/(2n-1)) at scale SH, n >= 2: p_k = floor(z^(2k+1) SH)
    by successive division, terms p_k/(2k+1) rounded down and (p_k+1)/(2k+1) rounded up, until
    p_k = 0, when the tail is below 2 ulps."""
    d = 2 * n - 1
    d2 = d * d
    p = SH // d
    lo = 0
    hi = 0
    k = 0
    while p > 0:
        lo += p // (2 * k + 1)
        hi += cdiv(p + 1, 2 * k + 1)
        p //= d2
        k += 1
    return lo, hi + 2


def phase_of(num: int, lo_h: int, hi_h: int, den: int) -> tuple[int, int]:
    """The angle num * [lo_h, hi_h] / den (the bracket at scale SH) modulo 2 pi, at scale S: the
    lower end reduced exactly modulo TWOPI_HL, the width kept, and the quotient's worth of the
    2 pi approximation error added on each side."""
    lo = (num * lo_h) // den
    hi = cdiv(num * hi_h, den)
    r = lo % TWOPI_HL
    qb = hi // TWOPI_HL + 1
    return (r - qb) >> (KH - K), cdiv(r + (hi - lo) + qb, 1 << (KH - K))


def reduce_2pi(a: tuple[int, int]) -> tuple[int, int]:
    """The interval shifted by a multiple of 2 pi so that its lower end is in [0, 2 pi)."""
    q = a[0] // (2 * PI_U)
    if q >= 0:
        return max(0, a[0] - q * 2 * PI_U), a[1] - q * 2 * PI_L
    return max(0, a[0] + (-q) * 2 * PI_L), a[1] + (-q) * 2 * PI_U


def cis(a: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    """Enclosures of cos and sin over an interval of angles of width below pi/2: reduce modulo
    2 pi, then to the quadrant, where cos decreases and sin increases (or, when the reduced
    interval straddles pi/2, sin is at most 1), then rotate."""
    u = reduce_2pi(a)
    qq = (2 * u[0]) // PI_U
    v_lo = max(0, u[0] - cdiv(qq * PI_U, 2))
    v_hi = u[1] - (qq * PI_L) // 2
    if v_hi > PI_U or v_lo > v_hi:
        raise ValueError("angle interval too wide")
    c = (cos_bounds(v_hi)[0], cos_bounds(v_lo)[1])
    if v_hi <= PI_L // 2:
        s_ = (sin_bounds(v_lo)[0], sin_bounds(v_hi)[1])
    else:
        s_ = (min(sin_bounds(v_lo)[0], sin_bounds(v_hi)[0]), S)
    neg = lambda z: (-z[1], -z[0])  # noqa: E731
    qq %= 4
    if qq == 0:
        return c, s_
    if qq == 1:
        return neg(s_), c
    if qq == 2:
        return neg(c), neg(s_)
    return s_, neg(c)


def rat_iv(num: int, den: int) -> tuple[int, int]:
    """The rational num/den (den > 0) as an interval at scale S."""
    return (num * S) // den, cdiv(num * S, den)


def cmul(a, b):
    """Product of complex intervals ((re), (im))."""
    return isub(imul(a[0], b[0]), imul(a[1], b[1])), iadd(imul(a[0], b[1]), imul(a[1], b[0]))


def cadd(a, b):
    return iadd(a[0], b[0]), iadd(a[1], b[1])


def cscale(a, p):
    """Complex interval times a real interval."""
    return imul(a[0], p), imul(a[1], p)


def cmod_upper(a) -> int:
    return abs_upper(a[0]) + abs_upper(a[1])


def rad(r: int) -> tuple[int, int]:
    return -r, r


class BarrierParams:
    """The request: the box family [xa, xb] x [ya, yb] x [ta, tb] (one row of nine naturals:
    numerators and denominators for x, y, t), its grid gx x gy x gt, the cutoff N, the number of
    moments J, whether the real part or the modulus is bounded, the offset and the scale; then
    the setup, everything a box does not depend on."""

    def __init__(self, args: dict):
        rows = read_rows(args["box"])
        if len(rows) != 1 or len(rows[0]) != 9:
            raise ValueError("box must be one row of nine naturals")
        (self.xa, self.xb, self.xden, self.ya, self.yb, self.yden, self.ta, self.tb, self.tden) = rows[0]
        self.gx, self.gy, self.gt = int(args["gx"]), int(args["gy"]), int(args["gt"])
        self.N = int(args["n"])
        self.J = int(args["jmax"])
        self.real = int(args["real"])
        self.offset = int(args["offset"])
        self.scale = int(args["scale"])
        if not (self.xden >= 1 and self.yden >= 1 and self.tden >= 1):
            raise ValueError("denominators must be positive")
        if not (self.xa <= self.xb and self.ya <= self.yb and self.ta <= self.tb):
            raise ValueError("box ends reversed")
        if self.xa < 200 * self.xden:
            raise ValueError("x below 200, outside the model's region")
        if self.yb > self.yden or 2 * self.tb > self.tden:
            raise ValueError("y above 1 or t above 1/2")
        if self.xa + self.xb >= 1 << 60:
            raise ValueError("x too large")
        if not (self.gx >= 1 and self.gy >= 1 and self.gt >= 1 and self.N >= 1):
            raise ValueError("grid and cutoff must be positive")
        if not (2 <= self.J <= 24):
            raise ValueError("jmax must be in [2, 24]")
        if self.real not in (0, 1):
            raise ValueError("real must be 0 or 1")
        if not (0 <= self.scale <= K):
            raise ValueError("scale must be in [0, 48]")
        self.setup()

    # rationals of the family: centres and grid points
    def x_at(self, i: int) -> tuple[int, int]:
        return self.xa * self.gx + i * (self.xb - self.xa), self.xden * self.gx

    def y_at(self, j: int) -> tuple[int, int]:
        return self.ya * self.gy + j * (self.yb - self.ya), self.yden * self.gy

    def t_at(self, k: int) -> tuple[int, int]:
        return self.ta * self.gt + k * (self.tb - self.ta), self.tden * self.gt

    def setup(self):
        xc_num, xc_den = self.xa + self.xb, 2 * self.xden
        tc_num, tc_den = self.ta + self.tb, 2 * self.tden
        yc_num, yc_den = self.ya + self.yb, 2 * self.yden
        self.Tc = rat_iv(tc_num, tc_den)
        # A_c = (ln(xc) - ln 4 pi) / 2
        ln_n, ln_d = ln_bounds(xc_num), ln_bounds(xc_den)
        self.Ac = ((ln_n[0] - ln_d[1] - LN4PI_U) // 2, cdiv(ln_n[1] - ln_d[0] - LN4PI_L, 2))
        sigma_c = rat_iv(2 * self.yden + yc_num, 4 * self.yden)          # (1 + y_c) / 2
        half_ym1 = rat_iv(yc_num - 2 * self.yden, 4 * self.yden)        # (y_c - 1) / 2
        rho1 = cdiv(201 * tc_num * self.xden * S, 100 * tc_den * xc_num)   # 2.01 t_c / x_c
        rhok = cdiv(tc_num * yc_num * self.xden * S, 2 * tc_den * yc_den * (xc_num - 12 * self.xden))
        TA2 = iscale(imul(self.Tc, self.Ac), 1, 2)
        re_c1 = iadd(isub(ineg_iv(sigma_c), TA2), rad(rho1))
        re_d1 = iadd(isub(half_ym1, TA2), rad(rho1 + rhok))
        c2 = iscale(self.Tc, 1, 4)
        TP8 = iscale(imul(self.Tc, (PI_L, PI_U)), 1, 8)
        # ln n at 2^112 by the recurrence, then at 2^48
        lnh = [(0, 0)] * (self.N + 1)
        lo_h, hi_h = 0, 0
        for n in range(2, self.N + 1):
            s_lo, s_hi = artanh_step(n)
            lo_h += 2 * s_lo
            hi_h += 2 * s_hi
            lnh[n] = (lo_h, hi_h)
        L48 = [(lo >> (KH - K), cdiv(hi, 1 << (KH - K))) for lo, hi in lnh]
        # dyadic ranges [a, b], centre L_r = (ln_lo a + ln_hi b) / 2 exactly, l_max
        self.ranges = []
        a = 1
        while a <= self.N:
            self.ranges.append((a, min(2 * a - 1, self.N)))
            a *= 2
        self.Lr = [(L48[a][0] + L48[b][1]) // 2 for a, b in self.ranges]
        self.lmax = [max(Lr - L48[a][0], L48[b][1] - Lr, 0) for (a, b), Lr in zip(self.ranges, self.Lr)]
        # the centre weights and the moments
        J = self.J
        zero = ((0, 0), (0, 0))
        self.mm = []
        self.mp = []
        self.mass_m = []
        self.mass_p = []
        for (a, b), Lr in zip(self.ranges, self.Lr):
            mm = [zero] * J
            mp = [zero] * J
            mass_m = 0
            mass_p = 0
            for n in range(a, b + 1):
                L = L48[n]
                phi = phase_of(xc_num, lnh[n][0], lnh[n][1], 2 * xc_den)
                tp = imul(TP8, L)
                rr = cdiv(rho1 * L[1], S)
                rk = cdiv((rho1 + rhok) * L[1], S)
                L2 = imul(c2, isq(L))
                E_m = iadd(imul(re_c1, L), L2)
                E_p = iadd(imul(re_d1, L), L2)
                if E_m[1] > EXP_LIMIT or E_p[1] > EXP_LIMIT:
                    raise ValueError("centre weight exceeds the exponent limit")
                g_m = iexp(E_m)
                g_p = iexp(E_p)
                c_m, s_m = cis(iadd(iadd(phi, tp), rad(rr)))
                c_p, s_p = cis(iadd(ineg_iv(iadd(phi, tp)), rad(rk)))
                w_m = (imul(g_m, c_m), imul(g_m, s_m))
                w_p = (imul(g_p, c_p), imul(g_p, s_p))
                ell = isub(L, (Lr, Lr))
                power = (S, S)
                for j in range(J):
                    mm[j] = cadd(mm[j], cscale(w_m, power))
                    mp[j] = cadd(mp[j], cscale(w_p, power))
                    power = imul(power, ell)
                mass_m += g_m[1]
                mass_p += g_p[1]
            self.mm.append(mm)
            self.mp.append(mp)
            self.mass_m.append(mass_m)
            self.mass_p.append(mass_p)
        # grid abscissae: ln(x_i / 4 pi) at 2^112 and Phi_i = (x_i / 2)(ln(x_i / 4 pi) - 1)
        self.ell_h = []
        self.Phi = []
        for i in range(self.gx + 1):
            num, den = self.x_at(i)
            ln_n, ln_d = ln_high(num), ln_high(den)
            ell = (ln_n[0] - ln_d[1] - LN4PI_HU, ln_n[1] - ln_d[0] - LN4PI_HL)
            self.ell_h.append(ell)
            self.Phi.append(((num * (ell[0] - SH)) // (2 * den), cdiv(num * (ell[1] - SH), 2 * den)))

    def box_of(self, member: list[int]) -> tuple[int, int, int]:
        if len(member) == 1:
            m = member[0]
            if m >= self.gx * self.gy * self.gt:
                raise ValueError("box index beyond the grid")
            return m // (self.gy * self.gt), (m // self.gt) % self.gy, m % self.gt
        if len(member) == 3:
            i, j, k = member
            if not (i < self.gx and j < self.gy and k < self.gt):
                raise ValueError("box beyond the grid")
            return i, j, k
        raise ValueError("barrier_lower needs 1 x 1 or 1 x 3 members")

    def range_sum(self, dc1, dc2, moments, mass):
        """sum_r P_r sum_j q_j m_{r,j} over the ranges as a complex interval, and the remainder
        loss."""
        J = self.J
        K0 = (J + 1) // 2
        total = ((0, 0), (0, 0))
        loss = 0
        for r, (Lr, lmax) in enumerate(zip(self.Lr, self.lmax)):
            Lr_iv = (Lr, Lr)
            E_re = iadd(imul(dc1[0], Lr_iv), imul(dc2, isq(Lr_iv)))
            E_im = imul(dc1[1], Lr_iv)
            if E_re[1] > EXP_LIMIT:
                raise ValueError("box too far from the centre: exponent above 7")
            g = iexp(E_re)
            c, s_ = cis(E_im)
            P = (imul(g, c), imul(g, s_))
            a = (iadd(dc1[0], iscale(imul(dc2, Lr_iv), 2, 1)), dc1[1])
            apow = [((S, S), (0, 0))]
            for k in range(1, J):
                z = cmul(apow[-1], a)
                apow.append((iscale(z[0], 1, k), iscale(z[1], 1, k)))
            bpow = [(S, S)]
            for m in range(1, J // 2 + 1):
                bpow.append(iscale(imul(bpow[-1], dc2), 1, m))
            Sr = ((0, 0), (0, 0))
            for j in range(J):
                q = ((0, 0), (0, 0))
                for m in range(j // 2 + 1):
                    q = cadd(q, cscale(apow[j - 2 * m], bpow[m]))
                Sr = cadd(Sr, cmul(q, moments[r][j]))
            total = cadd(total, cmul(P, Sr))
            a_hi = cmod_upper(a)
            U = cdiv(a_hi * lmax, S) + cdiv(abs_upper(dc2) * cdiv(lmax * lmax, S), S)
            if U >= K0 * S:
                raise ValueError("Taylor remainder does not converge: box too large")
            rem = cdiv(powfact(U, K0, False) * (K0 + 1) * S, (K0 + 1) * S - U)
            loss += cdiv(cmod_upper(P) * cdiv(mass[r] * rem, S), S)
        return total, loss

    def barrier_lower(self, member: list[int]) -> int:
        i, j, k = self.box_of(member)
        xc_num, xc_den = self.xa + self.xb, 2 * self.xden
        tc_num, tc_den = self.ta + self.tb, 2 * self.tden
        yc_num, yc_den = self.ya + self.yb, 2 * self.yden
        # displacements and ranges as intervals
        x0, xd = self.x_at(i)
        x1, _ = self.x_at(i + 1)
        DX = ((2 * x0 - xc_num * self.gx) * S // (2 * xd), cdiv((2 * x1 - xc_num * self.gx) * S, 2 * xd))
        y0, yd = self.y_at(j)
        y1, _ = self.y_at(j + 1)
        DY = ((2 * y0 - yc_num * self.gy) * S // (2 * yd), cdiv((2 * y1 - yc_num * self.gy) * S, 2 * yd))
        Y = (y0 * S // yd, cdiv(y1 * S, yd))
        t0, td = self.t_at(k)
        t1, _ = self.t_at(k + 1)
        DT = ((2 * t0 - tc_num * self.gt) * S // (2 * td), cdiv((2 * t1 - tc_num * self.gt) * S, 2 * td))
        T = (t0 * S // td, cdiv(t1 * S, td))
        ell = (self.ell_h[i][0] >> (KH - K), cdiv(self.ell_h[i + 1][1], 1 << (KH - K)))
        A = (ell[0] // 2, cdiv(ell[1], 2))
        # error radii: 2.01 (t_hi + t_c) / x_lo and (t_hi y_hi + t_c y_c) / (2 (x_lo - 6))
        tsum_num, tsum_den = 2 * t1 * self.tden + tc_num * td, 2 * td * self.tden   # t_hi + t_c
        rho = cdiv(201 * tsum_num * self.xden * S, 100 * tsum_den * self.xa)
        ty_num = 4 * t1 * y1 * tc_den * yc_den + tc_num * yc_num * 4 * td * yd
        ty_den = 4 * td * yd * tc_den * yc_den
        rhok = cdiv(ty_num * self.xden * S, ty_den * 2 * (self.xa - 6 * self.xden))
        # the exponent displacements
        mid = iscale(iadd(imul(DT, A), imul(self.Tc, isub(A, self.Ac))), 1, 2)
        dtp8 = iscale(imul(DT, (PI_L, PI_U)), 1, 8)
        dc1 = (iadd(isub(ineg_iv(iscale(DY, 1, 2)), mid), rad(rho)), iadd(iadd(iscale(DX, 1, 2), dtp8), rad(rho)))
        dd1 = (iadd(isub(iscale(DY, 1, 2), mid), rad(rho + rhok)),
               iadd(ineg_iv(iadd(iscale(DX, 1, 2), dtp8)), rad(rho + rhok)))
        dc2 = iscale(DT, 1, 4)
        main, loss_m = self.range_sum(dc1, dc2, self.mm, self.mass_m)
        part, loss_p = self.range_sum(dd1, dc2, self.mp, self.mass_p)
        # gamma over the box
        eps = cdiv((2 * S + cdiv(T[1] * (4 * S + cdiv(42 * A[1], 10)), S)) * self.xden, self.xa)
        re_g = iadd(ineg_iv(iscale(imul(Y, ell), 1, 2)), rad(eps))
        Phi = phase_of(1, self.Phi[i][0], self.Phi[i + 1][1], 1)
        im_g = iadd(iadd(iadd(Phi, (-cdiv(PI_U, 4), -(PI_L // 4))), iscale(imul(imul((PI_L, PI_U), T), ell), 1, 8)), rad(eps))
        if re_g[1] > EXP_LIMIT:
            raise ValueError("gamma exceeds the exponent limit")
        g = iexp(re_g)
        c, s_ = cis(im_g)
        G = (imul(g, c), imul(g, s_))
        f = cadd(main, cmul(G, part))
        loss = loss_m + cdiv(cmod_upper(G) * loss_p, S)
        if self.real:
            value = f[0][0] - loss
        else:
            re_lo = min(abs_lower(f[0]), ABS_CAP)
            im_lo = min(abs_lower(f[1]), ABS_CAP)
            value = sqrt_lower((re_lo * re_lo + im_lo * im_lo) // S) - loss
        v = (value + self.offset * S) >> (K - self.scale)
        if v < 0:
            return 0
        if v >= 1 << 64:
            raise ValueError("value does not fit in 64 bits")
        return v


def ineg_iv(a):
    return -a[1], -a[0]


def numbers(family: Family, prefix: int | None, boxes: bool = False, widths=(1, 4)):
    ms = list(itertools.islice(rt.iter_members(family), prefix))
    if rt.prime(family) != NATURALS:
        raise ValueError("heat_dirichlet operations need natural numbers, not elements of a field")
    if boxes:
        if any(len(m) != 1 or len(m[0]) not in widths for m in ms):
            raise ValueError(f"boxes need 1 x 1 or 1 x {widths[1]} members")
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
    if op == "barrier_lower":
        ms, boxes = numbers(family, prefix, boxes=True, widths=(1, 3))
        P = BarrierParams(args)
        return rt.reduce_int(reduction, [P.barrier_lower(b) for b in boxes], ms, NATURALS)
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
