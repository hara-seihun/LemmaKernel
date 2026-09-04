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
 * integer multipliers (divisor counts, block counts) that follow.
 */
#include "../../../../runtime/src/reduce.hpp"

#include <algorithm>
#include <numeric>
#include <optional>

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

enum class Op { WeightUpper, MollifiedTermUpper, BlockTermUpper, SigmaLower };

Result<Op> parse_op(const std::string &name) {
    static const std::map<std::string, Op> names{
        {"weight_upper", Op::WeightUpper}, {"mollified_term_upper", Op::MollifiedTermUpper},
        {"block_term_upper", Op::BlockTermUpper}, {"sigma_lower", Op::SigmaLower}};
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

Status validate(const Family &family) {
    if (family.prime() != NATURALS)
        return fail(INVALID, "heat_dirichlet operations need members that are natural numbers (lk.naturals), not elements of a field");
    if (family.rows() != 1 || family.cols() != 1)
        return fail(INVALID, "heat_dirichlet operations need 1 x 1 members, one natural number each; these are " +
                                 std::to_string(family.rows()) + " x " + std::to_string(family.cols()));
    return ok();
}

inline uint64_t member(const Family &family, uint64_t i) {
    return family.kind == Family::Kind::Range ? family.a + i : (uint64_t)family.data->entries[i];
}

R run(const Request &req) {
    auto valid = validate(*req.family);
    if (!valid.ok) return R::failure(valid.error.status, valid.error.message);
    auto parsed = parse_op(req.op);
    if (!parsed.ok) return R::failure(parsed.error.status, parsed.error.message);
    Op op = parsed.value;
    Params P;
    auto st = P.init(req);
    if (!st.ok) return R::failure(st.error.status, st.error.message);
    if (op == Op::MollifiedTermUpper || op == Op::BlockTermUpper) {
        st = P.precompute();
        if (!st.ok) return R::failure(st.error.status, st.error.message);
    }
    auto size_r = req.family->size();
    if (!size_r.ok) return R::failure(size_r.error.status, size_r.error.message);
    uint64_t size = size_r.value;
    if (op != Op::BlockTermUpper)
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
