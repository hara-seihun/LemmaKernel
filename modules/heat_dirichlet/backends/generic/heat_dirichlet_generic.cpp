/* heat_dirichlet generic backend: rigorous fixed-point upper bounds on the summands of
 * heat-weighted Dirichlet polynomials.
 *
 * Every real number is a 128-bit integer at scale 2^48, and every rounding is a floor or a ceiling
 * chosen so the result is an upper bound of the real quantity. There is no floating point. The
 * algorithm is the naive Python and the Lean reference step for step, so that all three return the
 * same integer; anything "faster but slightly different" would break the oracle. The speed comes
 * from the block operation, which bounds a whole run of consecutive members with a handful of
 * evaluations, not from the arithmetic.
 *
 * Widths: a scale-48 value below 2^15 (the `exp` limit of 7 guarantees exp values below 1097)
 * occupies under 2^60, so products of two such fit a signed 128-bit word with room for the small
 * integer multipliers (divisor counts, block counts) that follow. `phase_bound` keeps every
 * polynomial's coefficient mass below 16 for the same reason.
 *
 * `phase_bound` (the naive module's docstring has the mathematics): the request builds the
 * 7-smooth numbers, a cosine table on the grid, and for the head and every bin of rough k the
 * coefficient enclosures; a member is a box of the grid, and its value comes from the centre
 * value, the gradient and the second-order remainder of each polynomial over the box, with terms
 * whose arc exceeds a radian enclosed by their rectangle instead.
 */
#include "../../../../runtime/src/reduce.hpp"

#include <algorithm>
#include <array>
#include <numeric>
#include <optional>
#include <set>

namespace lk::heat_dirichlet {
namespace {

using R = Result<std::shared_ptr<Object>>;
using I = __int128;
constexpr int INVALID = 1;
constexpr int INTERNAL = 4;

constexpr int K = 48;
constexpr I S = (I)1 << K;
constexpr int LN_TERMS = 30;
constexpr int EXP_TERMS = 24;
constexpr I LN2_L = 195103586505167;
constexpr I LN2_U = 195103586505168;
constexpr I EXP_LIMIT = 7 * S;
constexpr uint64_t PRIMES[4] = {2, 3, 5, 7};
constexpr I PI_L = 884279719003555;
constexpr I PI_U = 884279719003556;
constexpr int TRIG_TERMS = 13;

enum class Op { WeightUpper, MollifiedTermUpper, BlockTermUpper, SigmaLower, PhaseBound };

Result<Op> parse_op(const std::string &name) {
    static const std::map<std::string, Op> names{
        {"weight_upper", Op::WeightUpper}, {"mollified_term_upper", Op::MollifiedTermUpper},
        {"block_term_upper", Op::BlockTermUpper}, {"sigma_lower", Op::SigmaLower},
        {"phase_bound", Op::PhaseBound}};
    auto it = names.find(name);
    if (it == names.end()) return Result<Op>::failure(INTERNAL, "unknown heat_dirichlet operation " + name);
    return Result<Op>::success(it->second);
}

/* Floor and ceiling division for b > 0 (a of either sign), as Python's `//` and `-(-a // b)`. */
inline I fdiv(I a, I b) {
    I q = a / b;
    return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q;
}
inline I cdiv(I a, I b) { return -fdiv(-a, b); }
inline I iabs(I a) { return a < 0 ? -a : a; }

struct Iv {
    I lo, hi;
};

/* ---- ln and exp with directed rounding ---- */

int bit_length(unsigned __int128 n) {
    int b = 0;
    while (n) { ++b; n >>= 1; }
    return b;
}

I ln_mantissa(I m, bool lower) {
    I num = (m - S) * S, den = m + S;
    I z = lower ? fdiv(num, den) : cdiv(num, den);
    I z2 = lower ? fdiv(z * z, S) : cdiv(z * z, S);
    I acc = 0, power = z;
    for (int k = 0; k < LN_TERMS; ++k) {
        acc += lower ? fdiv(power, 2 * k + 1) : cdiv(power, 2 * k + 1);
        power = lower ? fdiv(power * z2, S) : cdiv(power * z2, S);
    }
    return acc;
}

/* Bounds on ln n at scale S, n >= 1 (n below 2^80 keeps n << K inside 128 bits). */
Iv ln_bounds(unsigned __int128 n) {
    int e = bit_length(n) - 1;
    I m, m_hi;
    if (e <= K) {
        m = (I)(n << K) >> e;
        m_hi = m;
    } else {
        m = (I)(n >> (e - K));
        m_hi = m + ((n & (((unsigned __int128)1 << (e - K)) - 1)) ? 1 : 0);
    }
    return {2 * ln_mantissa(m, true) + e * LN2_L, 2 * ln_mantissa(m_hi, false) + e * LN2_U + 1};
}

std::optional<I> exp_upper(I x) {
    if (x > EXP_LIMIT) return std::nullopt;
    I ell = x >= 0 ? LN2_L : LN2_U;
    I q = fdiv(x, ell), r = x - q * ell;
    I acc = S, term = S;
    for (int k = 1; k <= EXP_TERMS; ++k) {
        term = cdiv(term * r, S * k);
        acc += term;
    }
    acc += 1;
    if (q >= 0) return acc << (int)q;
    return cdiv(acc, (I)1 << (int)(-q));
}

std::optional<I> exp_lower(I x) {
    if (x > EXP_LIMIT) return std::nullopt;
    I ell = x >= 0 ? LN2_U : LN2_L;
    I q = fdiv(x, ell), r = x - q * ell;
    I acc = S, term = S;
    for (int k = 1; k <= EXP_TERMS; ++k) {
        term = fdiv(term * r, S * k);
        acc += term;
    }
    if (q >= 0) return acc << (int)q;
    return acc >> (int)(-q);
}

/* ---- signed intervals ---- */

Iv imul(Iv a, Iv b) {
    I p[4] = {a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi};
    I lo = p[0], hi = p[0];
    for (I v : p) { lo = std::min(lo, v); hi = std::max(hi, v); }
    return {fdiv(lo, S), cdiv(hi, S)};
}
Iv iadd(Iv a, Iv b) { return {a.lo + b.lo, a.hi + b.hi}; }
Iv isub(Iv a, Iv b) { return {a.lo - b.hi, a.hi - b.lo}; }
Iv iscale(Iv a, I num, I den) {
    I p0 = a.lo * num, p1 = a.hi * num;
    return {fdiv(std::min(p0, p1), den), cdiv(std::max(p0, p1), den)};
}
std::optional<Iv> iexp(Iv a) {
    auto lo = exp_lower(a.lo), hi = exp_upper(a.hi);
    if (!lo || !hi) return std::nullopt;
    return Iv{*lo, *hi};
}
I abs_upper(Iv a) { return std::max(iabs(a.lo), iabs(a.hi)); }

/* ---- the parameters and the operations ---- */

struct Params {
    I t_num, t_den, sigma_num, sigma_den, y_num, y_den, c_num, c_den;
    uint64_t n_minus, n_plus, n0, width, scale;
    std::vector<uint64_t> primes;
    uint64_t D = 1;
    /* Fixed per request: lambda_d and ln d (index = subset mask over `primes`), r, ln N_-. */
    std::vector<Iv> lambda_by_mask, lnd_by_mask;
    std::vector<uint64_t> d_by_mask;
    I r_upper = 0;
    Iv lnN{0, 0}; /* bounds on ln N_- */

    uint64_t arg(const Request &req, const char *name, uint64_t dflt) const {
        auto it = req.int_args.find(name);
        return it == req.int_args.end() ? dflt : it->second;
    }

    Status init(const Request &req) {
        t_num = arg(req, "t_num", 0);
        t_den = arg(req, "t_den", 1);
        scale = arg(req, "scale", K);
        sigma_num = arg(req, "sigma_num", 0);
        sigma_den = arg(req, "sigma_den", 1);
        y_num = arg(req, "y_num", 0);
        y_den = arg(req, "y_den", 1);
        n_minus = arg(req, "n_minus", 1);
        n_plus = arg(req, "n_plus", n_minus);
        uint64_t mask = arg(req, "primes", 0);
        c_num = arg(req, "c_num", 1);
        c_den = arg(req, "c_den", 1);
        n0 = arg(req, "n0", 0);
        width = arg(req, "width", 1);
        if (t_den <= 0 || sigma_den <= 0 || y_den <= 0 || c_den <= 0) return fail(INVALID, "denominators must be positive");
        if (!(0 < t_num && 2 * t_num <= t_den)) return fail(INVALID, "t must be a rational in (0, 1/2]");
        if (scale < 1 || scale > (uint64_t)K) return fail(INVALID, "scale must be between 1 and 48");
        if (y_num > y_den) return fail(INVALID, "need 0 <= y <= 1");
        if (n_minus < 1 || width < 1) return fail(INVALID, "n_minus and width must be positive");
        if (n_plus < n_minus) return fail(INVALID, "need n_minus <= n_plus");
        for (int i = 0; i < 4; ++i)
            if (mask >> i & 1) primes.push_back(PRIMES[i]);
        for (uint64_t p : primes) D *= p;
        return ok();
    }

    Iv g_exponent(Iv lnn) const {
        Iv sq = imul(lnn, lnn);
        return isub(iscale(sq, t_num, 4 * t_den), iscale(lnn, sigma_num, sigma_den));
    }
    std::optional<I> g_upper(uint64_t n) const { return exp_upper(g_exponent(ln_bounds(n)).hi); }

    std::optional<Iv> b_interval(uint64_t p) const {
        Iv lnp = ln_bounds(p);
        return iexp(iscale(imul(lnp, lnp), t_num, 4 * t_den));
    }
    /* rho_d(n) = exp(-(t/4) ln d (2 ln n - ln d)): b_{n/d} / b_n when d | n, decreasing in n.
     * `lnn` and `lnd` are the bounds on ln n and ln d. */
    std::optional<Iv> rho_interval(Iv lnn, Iv lnd) const {
        Iv delta = imul(lnd, isub({2 * lnn.lo, 2 * lnn.hi}, lnd));
        return iexp(iscale(delta, -t_num, 4 * t_den));
    }
    /* pi_d(n) = min(1, n / (d N_-))^y from bounds on ln n and ln d; increasing in n. */
    std::optional<Iv> pi_interval(Iv lnn, Iv lnd) const {
        Iv base = isub(isub(lnn, lnd), lnN);
        Iv clamped{std::min<I>(base.lo, 0), std::min<I>(base.hi, 0)};
        return iexp(iscale(clamped, y_num, y_den));
    }

    /* The quantities that do not depend on the member: computed once, in the same rounding as the
     * per-member formulas of the naive implementation. */
    Status precompute() {
        size_t subsets = (size_t)1 << primes.size();
        lambda_by_mask.resize(subsets);
        lnd_by_mask.resize(subsets);
        lnN = ln_bounds(n_minus);
        d_by_mask.resize(subsets);
        for (size_t mask = 0; mask < subsets; ++mask) {
            uint64_t d = 1;
            Iv lam{S, S};
            for (size_t i = 0; i < primes.size(); ++i) {
                if (!(mask >> i & 1)) continue;
                d *= primes[i];
                auto b = b_interval(primes[i]);
                if (!b) return fail(INVALID, "exponent exceeds 7");
                lam = imul(lam, {-b->hi, -b->lo});
            }
            d_by_mask[mask] = d;
            lambda_by_mask[mask] = lam;
            lnd_by_mask[mask] = ln_bounds(d);
        }
        auto w = iexp(iscale({-lnN.hi, -lnN.lo}, y_num, y_den));
        if (!w) return fail(INVALID, "exponent exceeds 7");
        Iv wc = iscale(*w, c_num, c_den);
        I a = wc.lo < 0 ? 0 : wc.lo;
        r_upper = a < S ? cdiv((S - a) * S, S + a) : 0;
        return ok();
    }

    /* Divisor subsets of the chosen primes dividing m, as masks in increasing order of d. */
    std::vector<size_t> divisor_masks(uint64_t m) const {
        std::vector<std::pair<uint64_t, size_t>> ds;
        for (size_t mask = 0; mask < d_by_mask.size(); ++mask)
            if (m % d_by_mask[mask] == 0) ds.push_back({d_by_mask[mask], mask});
        std::sort(ds.begin(), ds.end());
        std::vector<size_t> out;
        for (auto &pr : ds) out.push_back(pr.second);
        return out;
    }

    /* Upper bound on max(|beta - alpha|, r |beta + alpha|) over n in [lo, hi] and cutoffs N in
     * [N_-, N_+]. The polynomial is truncated at N, so only divisors with n / d <= N contribute:
     * an upper set {d >= n / N} of the divisors (in increasing order). The threshold ranges over
     * [lo / N_+, hi / N_-]; every upper set whose cut lies there can occur, and the bound is the
     * largest over them. Beyond D N_+ only the empty set is left and the bound is 0. */
    I coefficient_upper(const std::vector<size_t> &masks, const std::vector<Iv> &rho, const std::vector<Iv> &pi,
                        uint64_t lo, uint64_t hi) const {
        I best = 0;
        for (size_t j = 0; j <= masks.size(); ++j) {
            unsigned __int128 below = j > 0 ? d_by_mask[masks[j - 1]] : 0;
            bool above_ok = j == masks.size() || (unsigned __int128)lo <= (unsigned __int128)d_by_mask[masks[j]] * n_plus;
            if (!(above_ok && below * n_minus < hi)) continue;
            std::vector<size_t> sub(masks.begin() + j, masks.end());
            std::vector<Iv> rsub(rho.begin() + j, rho.end()), psub(pi.begin() + j, pi.end());
            best = std::max(best, coefficient_of(sub, rsub, psub));
        }
        return best;
    }

    /* max(|beta - alpha|, r |beta + alpha|) over the given divisor set. */
    I coefficient_of(const std::vector<size_t> &masks, const std::vector<Iv> &rho, const std::vector<Iv> &pi) const {
        Iv beta{0, 0}, alpha{0, 0};
        for (size_t i = 0; i < masks.size(); ++i) {
            Iv lam = lambda_by_mask[masks[i]];
            beta = iadd(beta, imul(lam, rho[i]));
            alpha = iadd(alpha, imul(imul(lam, pi[i]), rho[i]));
        }
        alpha = iscale(alpha, c_num, c_den);
        I diff = abs_upper(isub(beta, alpha)), summ = abs_upper(iadd(beta, alpha));
        return std::max(diff, cdiv(r_upper * summ, S));
    }

    std::optional<I> term_upper(uint64_t n) const {
        auto masks = divisor_masks(n);
        Iv lnn = ln_bounds(n);
        std::vector<Iv> rho, pi;
        for (size_t mask : masks) {
            auto r = rho_interval(lnn, lnd_by_mask[mask]), q = pi_interval(lnn, lnd_by_mask[mask]);
            if (!r || !q) return std::nullopt;
            rho.push_back(*r);
            pi.push_back(*q);
        }
        auto g = exp_upper(g_exponent(lnn).hi);
        if (!g) return std::nullopt;
        return cdiv(coefficient_upper(masks, rho, pi, n, n) * *g, S);
    }

    std::optional<I> block_upper(uint64_t k) const {
        uint64_t a = n0 + k * width, b = a + width;
        if (a < 1) return std::nullopt;
        Iv lna = ln_bounds(a), lnb = ln_bounds(b - 1);
        auto ga = exp_upper(g_exponent(lna).hi), gb = exp_upper(g_exponent(lnb).hi);
        if (!ga || !gb) return std::nullopt;
        I g_max = std::max(*ga, *gb);
        /* rho_d decreases in n and pi_d increases: the block ends enclose both for every member,
         * whatever its residue class. */
        std::vector<Iv> rho_by_mask(d_by_mask.size()), pi_by_mask(d_by_mask.size());
        for (size_t mask = 0; mask < d_by_mask.size(); ++mask) {
            auto lo = rho_interval(lnb, lnd_by_mask[mask]), hi = rho_interval(lna, lnd_by_mask[mask]);
            auto plo = pi_interval(lna, lnd_by_mask[mask]), phi = pi_interval(lnb, lnd_by_mask[mask]);
            if (!lo || !hi || !plo || !phi) return std::nullopt;
            rho_by_mask[mask] = {lo->lo, hi->hi};
            pi_by_mask[mask] = {plo->lo, phi->hi};
        }
        /* The class c mod D fixes the divisors of gcd(n, D): one coefficient bound per gcd. */
        std::vector<I> count_by_gcd(d_by_mask.size(), 0);
        for (uint64_t c = 0; c < D; ++c) {
            I cnt = (I)((b + D - 1 - c) / D) - (I)((a + D - 1 - c) / D);
            uint64_t e = std::gcd(c, D);
            size_t emask = 0;
            for (size_t i = 0; i < primes.size(); ++i)
                if (e % primes[i] == 0) emask |= (size_t)1 << i;
            count_by_gcd[emask] += cnt;
        }
        I total = 0;
        for (size_t emask = 0; emask < d_by_mask.size(); ++emask) {
            if (count_by_gcd[emask] == 0) continue;
            auto masks = divisor_masks(d_by_mask[emask]);
            std::vector<Iv> rho, pi;
            for (size_t mask : masks) {
                rho.push_back(rho_by_mask[mask]);
                pi.push_back(pi_by_mask[mask]);
            }
            total += count_by_gcd[emask] * cdiv(coefficient_upper(masks, rho, pi, a, b - 1) * g_max, S);
        }
        return total;
    }

    I sigma_lower(uint64_t n) const {
        unsigned __int128 m = (unsigned __int128)n * n - 1;
        I half = fdiv(S * (y_den + y_num), 2 * y_den);
        I main = fdiv(t_num * ln_bounds(m).lo, 4 * t_den);
        I corr = cdiv(S * t_num, t_den * 144 * (I)m * (I)m);
        return half + main - corr;
    }

    std::optional<uint64_t> out(std::optional<I> v) const {
        if (!v) return std::nullopt;
        I o = cdiv(*v, (I)1 << (int)(K - scale));
        if (o < 0 || o > (I)UINT64_MAX) return std::nullopt;
        return (uint64_t)o;
    }
};

/* ---- phase_bound ---- */

unsigned __int128 isqrt128(unsigned __int128 n) {
    if (n < 2) return n;
    unsigned __int128 x = (unsigned __int128)1 << ((bit_length(n) + 1) / 2);
    for (;;) {
        unsigned __int128 y = (x + n / x) / 2;
        if (y >= x) return x;
        x = y;
    }
}
/* Bounds on sqrt(a / S) at scale S, a >= 0. */
I sqrt_lower(I a) { return (I)isqrt128((unsigned __int128)(a * S)); }
I sqrt_upper(I a) {
    unsigned __int128 n = (unsigned __int128)(a * S), r = isqrt128(n);
    return (I)(r * r == n ? r : r + 1);
}
Iv isq(Iv a) {
    I top = std::max(a.lo * a.lo, a.hi * a.hi);
    I bottom = (a.lo <= 0 && 0 <= a.hi) ? 0 : std::min(a.lo * a.lo, a.hi * a.hi);
    return {fdiv(bottom, S), cdiv(top, S)};
}
I abs_lower(Iv a) { return (a.lo <= 0 && 0 <= a.hi) ? 0 : std::min(iabs(a.lo), iabs(a.hi)); }

/* x^n / n! at scale S for x >= 0, rounded one way. */
I powfact(I x, int n, bool lower) {
    I term = S;
    for (int i = 1; i <= n; ++i) term = lower ? fdiv(term * x, S * i) : cdiv(term * x, S * i);
    return term;
}
/* Bounds on cos and sin of x / S for 0 <= x <= pi/2: alternating series, each term rounded the
 * safe way, the first omitted term as the tail. */
Iv cos_bounds(I x) {
    I lo = 0, hi = 0;
    for (int k = 0; k < TRIG_TERMS; ++k) {
        if (k % 2 == 0) { lo += powfact(x, 2 * k, true); hi += powfact(x, 2 * k, false); }
        else { lo -= powfact(x, 2 * k, false); hi -= powfact(x, 2 * k, true); }
    }
    I tail = powfact(x, 2 * TRIG_TERMS, false);
    return {lo - tail, hi + tail};
}
Iv sin_bounds(I x) {
    I lo = 0, hi = 0;
    for (int k = 0; k < TRIG_TERMS; ++k) {
        if (k % 2 == 0) { lo += powfact(x, 2 * k + 1, true); hi += powfact(x, 2 * k + 1, false); }
        else { lo -= powfact(x, 2 * k + 1, false); hi -= powfact(x, 2 * k + 1, true); }
    }
    I tail = powfact(x, 2 * TRIG_TERMS + 1, false);
    return {lo - tail, hi + tail};
}
struct Cv {
    Iv re, im;
};
Iv ineg(Iv a) { return {-a.hi, -a.lo}; }
/* Enclosures of cos and sin of 2 pi J / M, M a multiple of 4, 0 <= J < M: the first quadrant,
 * where cos decreases and sin increases, then a rotation. */
Cv unit_circle(uint64_t M, uint64_t J) {
    uint64_t quarter = M / 4, q = J / quarter, j = J % quarter;
    I x_lo = fdiv(2 * PI_L * (I)j, (I)M), x_hi = cdiv(2 * PI_U * (I)j, (I)M);
    Iv c{cos_bounds(x_hi).lo, cos_bounds(x_lo).hi};
    Iv s{sin_bounds(x_lo).lo, sin_bounds(x_hi).hi};
    switch (q) {
    case 0: return {c, s};
    case 1: return {ineg(s), c};
    case 2: return {ineg(c), ineg(s)};
    default: return {s, ineg(c)};
    }
}

struct Smooth {
    uint64_t m;
    int v[4];
};
struct Term {
    uint32_t idx;
    Iv cS, cA;
    I vh;      /* half-angle of the arc over a box, radians at scale S */
    int64_t X; /* the same in table steps */
};
struct Poly {
    I W = S, Q = 0;
    std::vector<Term> terms;
};

struct PhaseParams : Params {
    I sigma_hi_num = 0;
    uint64_t g[5] = {1, 1, 1, 1, 1};
    uint64_t offset = 0;
    struct Mu { uint64_t d; I num, den; };
    std::vector<Mu> mollifier;
    std::vector<uint64_t> bins;
    Iv sigma_iv{0, 0};
    uint64_t M = 4;
    I h[5] = {0, 0, 0, 0, 0};
    std::vector<Smooth> smooth;
    std::vector<Cv> circle;
    Poly head;
    std::vector<Poly> tail;

    static Result<std::vector<std::vector<uint64_t>>> rows_of(const Request &req, const char *name) {
        using RR = Result<std::vector<std::vector<uint64_t>>>;
        auto it = req.handle_args.find(name);
        if (it == req.handle_args.end() || !it->second->matrix)
            return RR::failure(INVALID, std::string("phase_bound needs an lk.naturals argument `") + name + "`");
        const Matrix &m = *it->second->matrix;
        if (m.p != NATURALS) return RR::failure(INVALID, std::string("`") + name + "` must be lk.naturals");
        std::vector<std::vector<uint64_t>> rows;
        for (uint64_t r = 0; r < m.count * m.rows; ++r)
            rows.push_back(std::vector<uint64_t>(m.entries.begin() + r * m.cols, m.entries.begin() + (r + 1) * m.cols));
        return RR::success(rows);
    }

    Status init_phase(const Request &req) {
        auto st = init(req);
        if (!st.ok) return st;
        sigma_hi_num = arg(req, "sigma_hi_num", sigma_num);
        if (sigma_hi_num < sigma_num) return fail(INVALID, "need sigma_num <= sigma_hi_num");
        const char *names[5] = {"g2", "g3", "g5", "g7", "gpsi"};
        for (int i = 0; i < 5; ++i) {
            g[i] = arg(req, names[i], 1);
            if (g[i] < 1 || g[i] > (1u << 20)) return fail(INVALID, "grid sizes must be between 1 and 2^20");
        }
        offset = arg(req, "offset", 0);
        auto moll = rows_of(req, "mollifier");
        if (!moll.ok) return fail(moll.error.status, moll.error.message);
        std::set<uint64_t> seen;
        for (const auto &row : moll.value) {
            if (row.size() != 4) return fail(INVALID, "mollifier rows are (d, sign, num, den)");
            uint64_t d = row[0], sign = row[1], num = row[2], den = row[3];
            uint64_t r = d;
            for (uint64_t p : PRIMES)
                while (r && r % p == 0) r /= p;
            if (d < 1 || r != 1 || d > n_plus) return fail(INVALID, "mollifier support must be 7-smooth and at most n_plus");
            if (seen.count(d)) return fail(INVALID, "mollifier support has a repeated d");
            if (sign > 1 || den < 1) return fail(INVALID, "mollifier sign must be 0 or 1 and den positive");
            seen.insert(d);
            mollifier.push_back({d, sign ? -(I)num : (I)num, (I)den});
        }
        if (mollifier.empty()) return fail(INVALID, "mollifier must have at least one row");
        auto b = rows_of(req, "bins");
        if (!b.ok) return fail(b.error.status, b.error.message);
        if (b.value.size() != 1) return fail(INVALID, "bins is one row of boundaries");
        bins = b.value[0];
        bool increasing = true;
        for (size_t i = 1; i < bins.size(); ++i) increasing = increasing && bins[i - 1] < bins[i];
        if (bins.size() < 2 || bins.front() != 2 || bins.back() <= n_plus || !increasing)
            return fail(INVALID, "bins must be increasing, start at 2 and end above n_plus");
        return ok();
    }

    static void smooth_up_to(uint64_t limit, std::vector<Smooth> &out) {
        for (unsigned __int128 a = 1; a <= limit; a *= 2)
            for (unsigned __int128 b = a; b <= limit; b *= 3)
                for (unsigned __int128 c = b; c <= limit; c *= 5)
                    for (unsigned __int128 e = c; e <= limit; e *= 7) {
                        Smooth s{(uint64_t)e, {0, 0, 0, 0}};
                        uint64_t r = s.m;
                        for (int i = 0; i < 4; ++i)
                            while (r % PRIMES[i] == 0) { r /= PRIMES[i]; ++s.v[i]; }
                        out.push_back(s);
                    }
        std::sort(out.begin(), out.end(), [](const Smooth &x, const Smooth &y) { return x.m < y.m; });
    }
    static bool is_rough(uint64_t k) { return std::gcd(k, (uint64_t)210) == 1; }

    Status precompute_phase() {
        sigma_iv = {fdiv(S * sigma_num, sigma_den), cdiv(S * sigma_hi_num, sigma_den)};
        uint64_t lcm = 1;
        for (uint64_t x : g) lcm = std::lcm(lcm, x);
        if (lcm > (1u << 24)) return fail(INVALID, "grid too fine: lcm of sizes above 2^24");
        M = 4 * lcm;
        for (int i = 0; i < 5; ++i) h[i] = cdiv(PI_U, (I)g[i]);
        uint64_t dmax = 1;
        for (auto &mu : mollifier) dmax = std::max(dmax, mu.d);
        smooth_up_to(dmax * n_plus, smooth);
        circle.resize(M);
        for (uint64_t J = 0; J < M; ++J) circle[J] = unit_circle(M, J);
        lnN = ln_bounds(n_minus);
        auto hd = polynomial(1, 1, true);
        if (!hd) return fail(INVALID, "exponent exceeds 7 or polynomial mass exceeds 16");
        head = *hd;
        for (size_t i = 0; i + 1 < bins.size(); ++i) {
            uint64_t lo = bins[i], hi = std::min<uint64_t>(bins[i + 1], n_plus + 1);
            uint64_t ka = 0, kb = 0;
            I W = 0;
            for (uint64_t k = lo; k < hi; ++k) {
                if (!is_rough(k)) continue;
                if (!ka) ka = k;
                kb = k;
                auto gk = g_upper(k);
                if (!gk) return fail(INVALID, "exponent exceeds 7");
                W += *gk;
            }
            if (!ka) continue;
            auto poly = polynomial(ka, kb, false);
            if (!poly) return fail(INVALID, "exponent exceeds 7 or polynomial mass exceeds 16");
            poly->W = W;
            tail.push_back(std::move(*poly));
        }
        return ok();
    }

    /* Coefficient enclosures of T_k over rough k in [ka, kb] and cutoffs in the cell, with the
     * second-order constant Q. */
    std::optional<Poly> polynomial(uint64_t ka, uint64_t kb, bool is_head) const {
        Poly poly;
        Iv Lk = is_head ? Iv{0, 0} : Iv{ln_bounds(ka).lo, ln_bounds(kb).hi};
        I mass = 0, Q = 0;
        for (size_t idx = 0; idx < smooth.size(); ++idx) {
            const Smooth &sm = smooth[idx];
            Iv Lm = ln_bounds(sm.m);
            Iv cS{0, 0}, cA{0, 0};
            for (const auto &mu : mollifier) {
                if (sm.m % mu.d) continue;
                uint64_t j = sm.m / mu.d;
                if ((unsigned __int128)j * ka > n_plus) continue;
                bool certain = (unsigned __int128)j * kb <= n_minus;
                Iv Lj = ln_bounds(j);
                Iv e_main = isub(iadd(iscale(imul(Lj, Lj), t_num, 4 * t_den), iscale(imul(Lj, Lk), t_num, 2 * t_den)),
                                 imul(Lm, sigma_iv));
                Iv e_part = iadd(e_main, iscale(isub(iadd(Lj, Lk), lnN), y_num, y_den));
                auto em = iexp(e_main), ep = iexp(e_part);
                if (!em || !ep) return std::nullopt;
                Iv main = iscale(*em, mu.num, mu.den);
                Iv part = iscale(iscale(*ep, mu.num, mu.den), c_num, c_den);
                if (!certain) {
                    main = {std::min<I>(main.lo, 0), std::max<I>(main.hi, 0)};
                    part = {std::min<I>(part.lo, 0), std::max<I>(part.hi, 0)};
                }
                cS = iadd(cS, main);
                cA = iadd(cA, part);
            }
            bool zeroS = cS.lo == 0 && cS.hi == 0, zeroA = cA.lo == 0 && cA.hi == 0;
            if (zeroS && zeroA) continue;
            I vh = 0;
            int64_t X = 0;
            for (int p = 0; p < 4; ++p) {
                vh += sm.v[p] * h[p];
                X += sm.v[p] * (int64_t)(M / (2 * g[p]));
            }
            I aS = abs_upper(cS), aA = abs_upper(cA);
            mass += aS + aA;
            if (vh <= S) Q += cdiv(aS * cdiv(vh * vh, S), S);
            if (vh + h[4] <= S) Q += cdiv(aA * cdiv((vh + h[4]) * (vh + h[4]), S), S);
            poly.terms.push_back({(uint32_t)idx, cS, cA, vh, X});
        }
        if (mass > 16 * S) return std::nullopt;
        poly.Q = cdiv(Q, 2);
        return poly;
    }

    /* Enclosures of cos and sin over the arc of table steps [J - X, J + X]. */
    Cv arc(int64_t J, int64_t X) const {
        int64_t Mi = (int64_t)M;
        if (2 * X >= Mi) return {{-S, S}, {-S, S}};
        int64_t a = J - X, b = J + X;
        auto contains = [&](int64_t r) { return fdiv(b - r, Mi) >= cdiv(a - r, Mi); };
        auto mod = [&](int64_t z) { return (uint64_t)(((z % Mi) + Mi) % Mi); };
        Cv ea = circle[mod(a)], eb = circle[mod(b)];
        I cos_hi = contains(0) ? S : std::max(ea.re.hi, eb.re.hi);
        I cos_lo = contains(Mi / 2) ? -S : std::min(ea.re.lo, eb.re.lo);
        I sin_hi = contains(Mi / 4) ? S : std::max(ea.im.hi, eb.im.hi);
        I sin_lo = contains(3 * Mi / 4) ? -S : std::min(ea.im.lo, eb.im.lo);
        return {{cos_lo, cos_hi}, {sin_lo, sin_hi}};
    }

    std::optional<std::array<uint64_t, 5>> box_of(const Family &family, uint64_t i) const {
        std::array<uint64_t, 5> js{};
        if (family.kind == Family::Kind::Range || family.data->cols == 1) {
            uint64_t idx = family.kind == Family::Kind::Range ? family.a + i : (uint64_t)family.data->entries[i];
            for (int p = 4; p >= 0; --p) { js[p] = idx % g[p]; idx /= g[p]; }
            if (idx) return std::nullopt;
            return js;
        }
        for (int p = 0; p < 5; ++p) {
            js[p] = family.data->entries[i * 5 + p];
            if (js[p] >= g[p]) return std::nullopt;
        }
        return js;
    }

    /* Centre value and gradient (five complex intervals) of a polynomial on a box. */
    void evaluate(const Poly &poly, const std::array<uint64_t, 5> &js, Cv &T, Cv G[5]) const {
        int64_t steps[5];
        for (int p = 0; p < 5; ++p) steps[p] = (int64_t)((2 * js[p] + 1) * (M / (2 * g[p])));
        int64_t Mi = (int64_t)M;
        T = {{0, 0}, {0, 0}};
        for (int p = 0; p < 5; ++p) G[p] = {{0, 0}, {0, 0}};
        for (const Term &term : poly.terms) {
            const Smooth &sm = smooth[term.idx];
            int64_t J = 0;
            for (int p = 0; p < 4; ++p) J += sm.v[p] * steps[p];
            J %= Mi;
            for (int part = 0; part < 2; ++part) {
                const Iv &c = part ? term.cA : term.cS;
                if (c.lo == 0 && c.hi == 0) continue;
                int64_t JJ = part ? (J + steps[4]) % Mi : J;
                bool taylor = part ? term.vh + h[4] <= S : term.vh <= S;
                if (!taylor) {
                    Cv e = arc(JJ, part ? term.X + (int64_t)(M / (2 * g[4])) : term.X);
                    T.re = iadd(T.re, imul(c, e.re));
                    T.im = iadd(T.im, imul(c, e.im));
                    continue;
                }
                const Cv &e = circle[(uint64_t)JJ];
                Iv wre = imul(c, e.re), wim = imul(c, e.im);
                T.re = iadd(T.re, wre);
                T.im = iadd(T.im, wim);
                for (int p = 0; p < 4; ++p) {
                    if (!sm.v[p]) continue;
                    I v = sm.v[p];
                    G[p].re = iadd(G[p].re, {-v * wim.hi, -v * wim.lo});
                    G[p].im = iadd(G[p].im, {v * wre.lo, v * wre.hi});
                }
                if (part) {
                    G[4].re = iadd(G[4].re, {-wim.hi, -wim.lo});
                    G[4].im = iadd(G[4].im, wre);
                }
            }
        }
    }

    I mod_upper(const Cv &z) const {
        I ar = abs_upper(z.re), ai = abs_upper(z.im);
        return sqrt_upper(cdiv(ar * ar, S) + cdiv(ai * ai, S));
    }

    /* (modulus upper bound, dist lower bound) of a polynomial over a box. */
    std::pair<I, I> bounds(const Poly &poly, const std::array<uint64_t, 5> &js) const {
        Cv T, G[5];
        evaluate(poly, js, T, G);
        Iv A = iadd(isq(T.re), isq(T.im));
        I B = 0, Cc = 0, re_var = 0, im_var = 0;
        for (int p = 0; p < 5; ++p) {
            B += cdiv(h[p] * abs_upper(iadd(imul(T.re, G[p].re), imul(T.im, G[p].im))), S);
            Cc += cdiv(h[p] * mod_upper(G[p]), S);
            re_var += cdiv(h[p] * abs_upper(G[p].re), S);
            im_var += cdiv(h[p] * abs_upper(G[p].im), S);
        }
        I upper = sqrt_upper(A.hi + 2 * B + cdiv(Cc * Cc, S)) + poly.Q;
        I lower_mod = sqrt_lower(std::max<I>(0, A.lo - 2 * B)) - poly.Q;
        I re_margin = T.re.lo - re_var - poly.Q;
        I im_lower = abs_lower(T.im) - im_var - poly.Q;
        I dist = std::max<I>(0, std::max<I>(im_lower, re_margin >= 0 ? lower_mod : 0));
        return {upper, dist};
    }

    std::optional<uint64_t> phase_bound(const std::array<uint64_t, 5> &js) const {
        I F = bounds(head, js).second;
        for (const Poly &poly : tail) F -= cdiv(poly.W * bounds(poly, js).first, S);
        I v = (F + (I)offset * S) >> (int)(K - scale);
        if (v < 0) return 0;
        if (v > (I)UINT64_MAX) return std::nullopt;
        return (uint64_t)v;
    }
};

Status validate(const Family &family, bool boxes) {
    if (family.prime() != NATURALS)
        return fail(INVALID, "heat_dirichlet operations need members that are natural numbers (lk.naturals), not elements of a field");
    if (boxes) {
        if (family.rows() != 1 || (family.cols() != 1 && family.cols() != 5))
            return fail(INVALID, "phase_bound needs 1 x 1 or 1 x 5 members (a box index or its five grid indices)");
        return ok();
    }
    if (family.rows() != 1 || family.cols() != 1)
        return fail(INVALID, "heat_dirichlet operations need 1 x 1 members, one natural number each; these are " +
                                 std::to_string(family.rows()) + " x " + std::to_string(family.cols()));
    return ok();
}

inline uint64_t member(const Family &family, uint64_t i) {
    return family.kind == Family::Kind::Range ? family.a + i : (uint64_t)family.data->entries[i];
}

R run(const Request &req) {
    auto parsed = parse_op(req.op);
    if (!parsed.ok) return R::failure(parsed.error.status, parsed.error.message);
    Op op = parsed.value;
    auto valid = validate(*req.family, op == Op::PhaseBound);
    if (!valid.ok) return R::failure(valid.error.status, valid.error.message);
    Params P;
    PhaseParams PP;
    Status st;
    if (op == Op::PhaseBound) {
        st = PP.init_phase(req);
        if (!st.ok) return R::failure(st.error.status, st.error.message);
        st = PP.precompute_phase();
        if (!st.ok) return R::failure(st.error.status, st.error.message);
    } else {
        st = P.init(req);
        if (!st.ok) return R::failure(st.error.status, st.error.message);
    }
    if (op == Op::MollifiedTermUpper || op == Op::BlockTermUpper) {
        st = P.precompute();
        if (!st.ok) return R::failure(st.error.status, st.error.message);
    }
    auto size_r = req.family->size();
    if (!size_r.ok) return R::failure(size_r.error.status, size_r.error.message);
    uint64_t size = size_r.value;
    if (op != Op::BlockTermUpper && op != Op::PhaseBound)
        for (uint64_t i = 0; i < size; ++i)
            if (member(*req.family, i) < (op == Op::SigmaLower ? 2u : 1u))
                return R::failure(INVALID, op == Op::SigmaLower ? "sigma_lower needs a cutoff of at least 2"
                                                                 : std::string(req.op) + " needs members of at least 1");

    Reduction reduction = parse_reduction(req.reduction);
    Shared shared;
    auto prepared = prepare_all(reduction, size, shared);
    if (!prepared.ok) return R::failure(prepared.error.status, prepared.error.message);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, size ? size : 1));
    std::vector<Accumulator> accumulators;
    for (uint32_t t = 0; t < threads; ++t) accumulators.emplace_back(reduction, &shared);

    auto statuses = parallel_ranges(size, threads, [&](uint32_t thread, uint64_t begin, uint64_t end) -> Status {
        Accumulator &acc = accumulators[thread];
        for (uint64_t i = begin; i < end; ++i) {
            if (op == Op::PhaseBound) {
                auto js = PP.box_of(*req.family, i);
                if (!js) return fail(INVALID, "box index beyond the grid at member " + std::to_string(i));
                auto v = PP.phase_bound(*js);
                if (!v) return fail(INVALID, "value does not fit in 64 bits at member " + std::to_string(i));
                acc.integer(i, *v);
                continue;
            }
            uint64_t n = member(*req.family, i);
            std::optional<uint64_t> v;
            switch (op) {
            case Op::WeightUpper: v = P.out(P.g_upper(n)); break;
            case Op::MollifiedTermUpper: v = P.out(P.term_upper(n)); break;
            case Op::BlockTermUpper: v = P.out(P.block_upper(n)); break;
            case Op::SigmaLower: {
                I s = P.sigma_lower(n);
                v = (uint64_t)((s < 0 ? 0 : s) >> (int)(K - P.scale));
                break;
            }
            case Op::PhaseBound: break; /* handled above */
            }
            if (!v) return fail(INVALID, "exponent exceeds 7 or value does not fit in 64 bits at member " + std::to_string(n));
            acc.integer(i, *v);
        }
        return ok();
    });
    for (const auto &status : statuses)
        if (!status.ok) return R::failure(status.error.status, status.error.message);
    return assemble(req, reduction, accumulators, shared);
}

BackendRegistration registration{Backend{
    "heat_dirichlet", "generic",
    [] { return true; },
    [](const Request &) { return true; },
    run,
    0}};

} // namespace
} // namespace lk::heat_dirichlet
