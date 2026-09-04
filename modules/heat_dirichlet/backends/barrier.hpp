/* heat_dirichlet.barrier_lower: a lower bound on |f_t(x + iy)| (or on Re f_t) over a box in
 * (x, y, t) for the barrier of the de Bruijn-Newman programme, by the moment method. The naive
 * module's docstring has the mathematics; this is the same algorithm step for step at scale 2^48,
 * with the two high-precision phases, (x_c/2) ln n and (x/2)(ln(x/4 pi) - 1), at scale 2^112 in
 * exact wide integers and reduced modulo 2 pi before they are brought down to 2^48.
 *
 * The setup (ln n by the recurrence, the centre weights, the moments of every dyadic range and
 * the phases at the grid abscissae) is done once per request; the moments are integer sums, so
 * splitting them across threads changes nothing. */
#pragma once
#include "heat_dirichlet_common.hpp"

namespace lk::heat_dirichlet::detail {

using U = unsigned __int128;

constexpr int KH = 112;
constexpr U SH = (U)1 << KH;
/* floor(2 pi 2^112), floor(ln 2 2^112), floor(ln(4 pi) 2^112); the true values lie in [c, c + 1). */
constexpr U TWOPI_HL = ((U)0x6487ed5110b46ULL << 64) | 0x11a62633145c06e0ULL;
constexpr U LN2_HL = ((U)0xb17217f7d1cfULL << 64) | 0x79abc9e3b39803f2ULL;
constexpr U LN4PI_HL = ((U)0x287f1347e1dbaULL << 64) | 0xc414f35cd39a8c25ULL;
constexpr int LNH_TERMS = 36;
constexpr int LN_CHUNK = 16;
constexpr I ABS_CAP = (I)1 << 60;

/* ---- 256-bit unsigned helpers, only for the two phases ---- */

struct U256 {
    U hi, lo;
};
inline U256 add256(U256 a, U256 b) {
    U lo = a.lo + b.lo;
    return {a.hi + b.hi + (lo < a.lo ? 1 : 0), lo};
}
inline U256 shl256(U256 a, int k) {
    if (k == 0) return a;
    if (k >= 128) return {a.lo << (k - 128), 0};
    return {(a.hi << k) | (a.lo >> (128 - k)), a.lo << k};
}
inline U256 mul_u128(U a, U b) {
    uint64_t a0 = (uint64_t)a, a1 = (uint64_t)(a >> 64), b0 = (uint64_t)b, b1 = (uint64_t)(b >> 64);
    U p00 = (U)a0 * b0, p01 = (U)a0 * b1, p10 = (U)a1 * b0, p11 = (U)a1 * b1;
    U256 r{p11, p00};
    r = add256(r, shl256({0, p01}, 64));
    r = add256(r, shl256({0, p10}, 64));
    return r;
}
inline bool is_zero(U256 a) { return a.hi == 0 && a.lo == 0; }
/* a >> k, and whether any shifted-out bit was set. */
inline U256 shr256(U256 a, int k, bool *inexact) {
    U256 r;
    if (k >= 128) {
        *inexact = a.lo != 0 || (k > 128 && (a.hi & (((U)1 << (k - 128)) - 1)) != 0);
        r = {0, k == 128 ? a.hi : a.hi >> (k - 128)};
    } else {
        *inexact = (a.lo & (((U)1 << k) - 1)) != 0;
        r = {a.hi >> k, (a.lo >> k) | (a.hi << (128 - k))};
    }
    return r;
}
/* a / m and a % m for m < 2^127, bit by bit. */
inline void divmod256(U256 a, U m, U256 *q, U *r) {
    U rem = 0;
    U256 quo{0, 0};
    for (int bit = 255; bit >= 0; --bit) {
        U word = bit >= 128 ? a.hi : a.lo;
        int b = bit & 127;
        rem = (rem << 1) | ((word >> b) & 1);
        if (rem >= m) {
            rem -= m;
            if (bit >= 128) quo.hi |= (U)1 << b; else quo.lo |= (U)1 << b;
        }
    }
    *q = quo;
    *r = rem;
}
inline U256 sub256(U256 a, U256 b) {
    U lo = a.lo - b.lo;
    return {a.hi - b.hi - (a.lo < b.lo ? 1 : 0), lo};
}
inline U256 inc256(U256 a) { return add256(a, {0, 1}); }

/* ---- ln at scale 2^112 ---- */

/* Bounds on ln n at scale SH for 1 <= n < 2^64, by the mantissa series; q.lo/q.hi are the two
 * rounding directions. */
inline U lnh_mantissa(U m, bool lower) {
    U256 num = mul_u128(m - SH, SH);
    U den = m + SH;
    U256 zq;
    U rem;
    divmod256(num, den, &zq, &rem);
    U z = zq.lo;
    if (!lower && rem) ++z;
    bool inexact;
    U z2 = shr256(mul_u128(z, z), KH, &inexact).lo;
    if (!lower && inexact) ++z2;
    U acc = 0, power = z;
    for (int k = 0; k < LNH_TERMS; ++k) {
        U term = power / (2 * k + 1);
        if (!lower && power % (2 * k + 1)) ++term;
        acc += term;
        power = shr256(mul_u128(power, z2), KH, &inexact).lo;
        if (!lower && inexact) ++power;
    }
    return acc;
}
inline std::pair<I, I> ln_high(uint64_t n) {
    int e = bit_length(n) - 1;
    U m = (U)n << (KH - e);
    return {(I)(2 * lnh_mantissa(m, true)) + (I)e * (I)LN2_HL, (I)(2 * lnh_mantissa(m, false)) + (I)e * (I)(LN2_HL + 1) + 1};
}
/* Bounds on artanh(1/(2n - 1)) at scale SH, n >= 2: p_k = floor(z^(2k+1) SH) by successive
 * division, terms rounded down and (p_k + 1)/(2k + 1) rounded up, until p_k = 0, then two ulps of
 * tail. */
inline std::pair<I, I> artanh_step(uint64_t n) {
    U d = 2 * (U)n - 1, d2 = d * d;
    U p = SH / d, lo = 0, hi = 0;
    for (int k = 0; p > 0; ++k) {
        lo += p / (2 * k + 1);
        hi += (p + 1 + 2 * k) / (2 * k + 1);
        p /= d2;
    }
    return {(I)lo, (I)hi + 2};
}
/* The angle num [lo_h, hi_h] / den (the bracket at scale SH, num < 2^60, den < 2^64) modulo 2 pi
 * at scale S: the lower end reduced exactly modulo TWOPI_HL, the width kept, and the quotient's
 * worth of the 2 pi approximation error on each side. */
/* a / den for a below 2^192 and den below 2^64, limb by limb; the remainder too. */
inline U256 div256_u64(U256 a, uint64_t den, uint64_t *rem) {
    uint64_t l2 = (uint64_t)a.hi, l1 = (uint64_t)(a.lo >> 64), l0 = (uint64_t)a.lo;
    U q2 = (U)l2 / den, r = (U)l2 % den;
    U q1 = ((r << 64) | l1) / den; r = ((r << 64) | l1) % den;
    U q0 = ((r << 64) | l0) / den; r = ((r << 64) | l0) % den;
    *rem = (uint64_t)r;
    return {q2, (q1 << 64) | q0};
}
/* a / m and a % m for a below 2^190 and m near 2^115 (the 2 pi reduction): a long double
 * estimate of the quotient, then the exact remainder and a correction. */
inline void divmod256_est(U256 a, U m, U *q, U *r) {
    long double fa = (long double)a.hi * 340282366920938463463374607431768211456.0L + (long double)a.lo;
    long double fq = fa / (long double)m;
    U qe = fq < 1 ? 0 : (U)fq;
    U256 prod = mul_u128(qe, m);
    while (prod.hi > a.hi || (prod.hi == a.hi && prod.lo > a.lo)) { --qe; prod = sub256(prod, {0, m}); }
    U256 rem = sub256(a, prod);
    while (rem.hi > 0 || rem.lo >= m) { ++qe; rem = sub256(rem, {0, m}); }
    *q = qe;
    *r = rem.lo;
}
inline Iv phase_of(uint64_t num, I lo_h, I hi_h, uint64_t den) {
    uint64_t rem;
    U256 lo = div256_u64(mul_u128((U)num, (U)lo_h), den, &rem);
    U256 hi = div256_u64(mul_u128((U)num, (U)hi_h), den, &rem);
    if (rem) hi = inc256(hi);
    U q, r0, qh;
    divmod256_est(lo, TWOPI_HL, &q, &r0);
    divmod256_est(hi, TWOPI_HL, &q, &qh);
    I r = (I)r0, qb = (I)q + 1, w = (I)sub256(hi, lo).lo;
    return {fdiv(r - qb, (I)1 << (KH - K)), cdiv(r + w + qb, (I)1 << (KH - K))};
}
/* The same for the gamma phase, whose ends are given at scale SH as 256-bit values. */
inline Iv phase_of_h(U256 lo_h, U256 hi_h) {
    U q, r0, qh;
    divmod256_est(lo_h, TWOPI_HL, &q, &r0);
    divmod256_est(hi_h, TWOPI_HL, &q, &qh);
    I r = (I)r0, qb = (I)q + 1, w = (I)sub256(hi_h, lo_h).lo;
    return {fdiv(r - qb, (I)1 << (KH - K)), cdiv(r + w + qb, (I)1 << (KH - K))};
}

/* ---- angles and complex intervals at scale S ---- */

/* cos_bounds and sin_bounds of the common header in one pass: the same floor and ceiling
 * recurrences for x^i / i!, each power taken once. */
inline void cos_sin_bounds(I x, Iv *c, Iv *s) {
    I tl = S, tu = S; /* x^i / i! rounded down and up, i = 0 */
    I clo = 0, chi = 0, slo = 0, shi = 0;
    for (int i = 0; i <= 2 * TRIG_TERMS + 1; ++i) {
        if (i > 0) { tl = fdiv(tl * x, S * i); tu = cdiv(tu * x, S * i); }
        if (i % 2 == 0) {
            int k = i / 2;
            if (k < TRIG_TERMS) { if (k % 2 == 0) { clo += tl; chi += tu; } else { clo -= tu; chi -= tl; } }
            else { clo -= tu; chi += tu; } /* the cosine tail, i = 2 TRIG_TERMS */
        } else {
            int k = (i - 1) / 2;
            if (k < TRIG_TERMS) { if (k % 2 == 0) { slo += tl; shi += tu; } else { slo -= tu; shi -= tl; } }
            else { slo -= tu; shi += tu; } /* the sine tail, i = 2 TRIG_TERMS + 1 */
        }
    }
    *c = {clo, chi};
    *s = {slo, shi};
}

inline Iv reduce_2pi(Iv a) {
    I q = fdiv(a.lo, 2 * PI_U);
    if (q >= 0) return {std::max<I>(0, a.lo - q * 2 * PI_U), a.hi - q * 2 * PI_L};
    return {std::max<I>(0, a.lo + (-q) * 2 * PI_L), a.hi + (-q) * 2 * PI_U};
}
/* cos and sin over an interval of angles of width below pi/2, or nullopt when it is wider. */
inline std::optional<Cv> cis(Iv a) {
    Iv u = reduce_2pi(a);
    I qq = fdiv(2 * u.lo, PI_U);
    I v_lo = std::max<I>(0, u.lo - cdiv(qq * PI_U, 2)), v_hi = u.hi - fdiv(qq * PI_L, 2);
    if (v_hi > PI_U || v_lo > v_hi) return std::nullopt;
    Iv c_lo, s_lo, c_hi, s_hi;
    cos_sin_bounds(v_lo, &c_lo, &s_lo);
    cos_sin_bounds(v_hi, &c_hi, &s_hi);
    Iv c{c_hi.lo, c_lo.hi};
    Iv s;
    if (v_hi <= fdiv(PI_L, 2)) s = {s_lo.lo, s_hi.hi};
    else s = {std::min(s_lo.lo, s_hi.lo), S};
    switch ((int)(((qq % 4) + 4) % 4)) {
    case 0: return Cv{c, s};
    case 1: return Cv{ineg(s), c};
    case 2: return Cv{ineg(c), ineg(s)};
    default: return Cv{s, ineg(c)};
    }
}
inline Cv cmul(Cv a, Cv b) { return {isub(imul(a.re, b.re), imul(a.im, b.im)), iadd(imul(a.re, b.im), imul(a.im, b.re))}; }
inline Cv cadd(Cv a, Cv b) { return {iadd(a.re, b.re), iadd(a.im, b.im)}; }
inline Cv cscale(Cv a, Iv p) { return {imul(a.re, p), imul(a.im, p)}; }
inline I cmod_upper(Cv a) { return abs_upper(a.re) + abs_upper(a.im); }
inline Iv rad(I r) { return {-r, r}; }
inline Iv rat_iv(I num, I den) { return {fdiv(num * S, den), cdiv(num * S, den)}; }

struct BarrierParams {
    I xa, xb, xden, ya, yb, yden, ta, tb, tden;
    uint64_t gx, gy, gt, N, J, scale;
    int real;
    I offset;
    /* setup */
    Iv Tc, Ac;
    std::vector<std::pair<uint64_t, uint64_t>> ranges;
    std::vector<I> Lr, lmax, mass_m, mass_p;
    std::vector<std::vector<Cv>> mm, mp; /* [range][j] */
    std::vector<std::pair<I, I>> ell_h;
    std::vector<std::pair<U256, U256>> Phi;

    uint64_t arg(const Request &req, const char *name, uint64_t dflt) const {
        auto it = req.int_args.find(name);
        return it == req.int_args.end() ? dflt : it->second;
    }

    Status init(const Request &req) {
        auto it = req.handle_args.find("box");
        if (it == req.handle_args.end() || !it->second->matrix) return fail(INVALID, "barrier_lower needs an lk.naturals argument `box`");
        const Matrix &m = *it->second->matrix;
        if (m.p != NATURALS || m.count * m.rows != 1 || m.cols != 11) return fail(INVALID, "box must be one row of eleven naturals");
        /* lk.naturals entries are below 2^32; the x numerators come as two words, high first */
        xa = ((I)m.entries[0] << 32) + (I)m.entries[1]; xb = ((I)m.entries[2] << 32) + (I)m.entries[3]; xden = m.entries[4];
        ya = m.entries[5]; yb = m.entries[6]; yden = m.entries[7];
        ta = m.entries[8]; tb = m.entries[9]; tden = m.entries[10];
        gx = arg(req, "gx", 1); gy = arg(req, "gy", 1); gt = arg(req, "gt", 1);
        N = arg(req, "n", 1);
        J = arg(req, "jmax", 14);
        real = (int)arg(req, "real", 0);
        offset = (I)arg(req, "offset", 0);
        scale = arg(req, "scale", K);
        if (xden < 1 || yden < 1 || tden < 1) return fail(INVALID, "denominators must be positive");
        if (xa > xb || ya > yb || ta > tb) return fail(INVALID, "box ends reversed");
        if (xa < 200 * xden) return fail(INVALID, "x below 200, outside the model's region");
        if (yb > yden || 2 * tb > tden) return fail(INVALID, "y above 1 or t above 1/2");
        if (xa + xb >= ((I)1 << 60)) return fail(INVALID, "x too large");
        if (gx < 1 || gy < 1 || gt < 1 || N < 1) return fail(INVALID, "grid and cutoff must be positive");
        if (gx > (1u << 20) || gy > (1u << 20) || gt > (1u << 20)) return fail(INVALID, "grid sizes must be at most 2^20");
        if (J < 2 || J > 24) return fail(INVALID, "jmax must be in [2, 24]");
        if (real != 0 && real != 1) return fail(INVALID, "real must be 0 or 1");
        if (scale > (uint64_t)K) return fail(INVALID, "scale must be in [0, 48]");
        if ((I)xden * (I)gx >= ((I)1 << 63) || (I)xb * (I)gx >= ((I)1 << 63)) return fail(INVALID, "x grid too fine for 64-bit denominators");
        return ok();
    }

    std::pair<I, I> x_at(uint64_t i) const { return {xa * (I)gx + (I)i * (xb - xa), xden * (I)gx}; }
    std::pair<I, I> y_at(uint64_t j) const { return {ya * (I)gy + (I)j * (yb - ya), yden * (I)gy}; }
    std::pair<I, I> t_at(uint64_t k) const { return {ta * (I)gt + (I)k * (tb - ta), tden * (I)gt}; }

    Status setup(uint32_t threads) {
        I xc_num = xa + xb, xc_den = 2 * xden, tc_num = ta + tb, tc_den = 2 * tden, yc_num = ya + yb, yc_den = 2 * yden;
        Tc = rat_iv(tc_num, tc_den);
        Iv ln_n = ln_bounds((U)xc_num), ln_d = ln_bounds((U)xc_den);
        I ln4pi_l = fdiv((I)LN4PI_HL, (I)1 << (KH - K)), ln4pi_u = cdiv((I)LN4PI_HL + 1, (I)1 << (KH - K));
        Ac = {fdiv(ln_n.lo - ln_d.hi - ln4pi_u, 2), cdiv(ln_n.hi - ln_d.lo - ln4pi_l, 2)};
        Iv sigma_c = rat_iv(2 * yden + yc_num, 4 * yden), half_ym1 = rat_iv(yc_num - 2 * yden, 4 * yden);
        I rho1 = cdiv(201 * tc_num * xden * S, 100 * tc_den * xc_num);
        I rhok = cdiv(tc_num * yc_num * xden * S, 2 * tc_den * yc_den * (xc_num - 12 * xden));
        Iv TA2 = iscale(imul(Tc, Ac), 1, 2);
        Iv re_c1 = iadd(isub(ineg(sigma_c), TA2), rad(rho1));
        Iv re_d1 = iadd(isub(half_ym1, TA2), rad(rho1 + rhok));
        Iv c2 = iscale(Tc, 1, 4);
        Iv TP8 = iscale(imul(Tc, {PI_L, PI_U}), 1, 8);
        /* ln n at 2^112: the series at every chunk start c = max(1, 2^16 floor(n / 2^16)), the
         * recurrence within the chunk; the chunking is part of the definition, so the thread
         * count does not enter the integers. */
        std::vector<std::pair<I, I>> lnh(N + 1, {0, 0});
        uint64_t chunks = (N >> LN_CHUNK) + 1;
        parallel_ranges(chunks, threads, [&](uint32_t, uint64_t begin, uint64_t end) -> Status {
            for (uint64_t c = begin; c < end; ++c) {
                uint64_t start = std::max<uint64_t>(1, c << LN_CHUNK), stop = std::min(N, ((c + 1) << LN_CHUNK) - 1);
                if (start > N) continue;
                auto base = ln_high(start);
                I lo_h = base.first, hi_h = base.second;
                lnh[start] = {lo_h, hi_h};
                for (uint64_t n = start + 1; n <= stop; ++n) {
                    auto st = artanh_step(n);
                    lo_h += 2 * st.first;
                    hi_h += 2 * st.second;
                    lnh[n] = {lo_h, hi_h};
                }
            }
            return ok();
        });
        std::vector<Iv> L48(N + 1);
        for (uint64_t n = 1; n <= N; ++n) L48[n] = {fdiv(lnh[n].first, (I)1 << (KH - K)), cdiv(lnh[n].second, (I)1 << (KH - K))};
        ranges.clear();
        for (uint64_t a = 1; a <= N; a *= 2) ranges.push_back({a, std::min(2 * a - 1, N)});
        size_t R = ranges.size();
        Lr.assign(R, 0); lmax.assign(R, 0);
        for (size_t r = 0; r < R; ++r) {
            auto [a, b] = ranges[r];
            Lr[r] = fdiv(L48[a].lo + L48[b].hi, 2);
            lmax[r] = std::max<I>(std::max<I>(Lr[r] - L48[a].lo, L48[b].hi - Lr[r]), 0);
        }
        /* the centre weights and the moments, n split across threads, partial sums added */
        std::vector<std::vector<std::vector<Cv>>> pm(threads), pp(threads);
        std::vector<std::vector<I>> pmass_m(threads), pmass_p(threads);
        for (uint32_t t = 0; t < threads; ++t) {
            pm[t].assign(R, std::vector<Cv>(J, Cv{{0, 0}, {0, 0}}));
            pp[t].assign(R, std::vector<Cv>(J, Cv{{0, 0}, {0, 0}}));
            pmass_m[t].assign(R, 0);
            pmass_p[t].assign(R, 0);
        }
        auto statuses = parallel_ranges(N, threads, [&](uint32_t t, uint64_t begin, uint64_t end) -> Status {
            for (uint64_t idx = begin; idx < end; ++idx) {
                uint64_t n = idx + 1;
                size_t r = bit_length(n) - 1;
                Iv L = L48[n];
                Iv phi = phase_of((uint64_t)xc_num, lnh[n].first, lnh[n].second, (uint64_t)(2 * xc_den));
                Iv tp = imul(TP8, L);
                I rr = cdiv(rho1 * L.hi, S), rk = cdiv((rho1 + rhok) * L.hi, S);
                Iv L2 = imul(c2, isq(L));
                Iv E_m = iadd(imul(re_c1, L), L2), E_p = iadd(imul(re_d1, L), L2);
                auto g_m = iexp(E_m), g_p = iexp(E_p);
                if (!g_m || !g_p) return fail(INVALID, "centre weight exceeds the exponent limit");
                auto cm = cis(iadd(iadd(phi, tp), rad(rr)));
                auto cp = cis(iadd(ineg(iadd(phi, tp)), rad(rk)));
                if (!cm || !cp) return fail(INVALID, "phase interval too wide");
                Cv w_m{imul(*g_m, cm->re), imul(*g_m, cm->im)};
                Cv w_p{imul(*g_p, cp->re), imul(*g_p, cp->im)};
                Iv ell = isub(L, {Lr[r], Lr[r]});
                Iv power{S, S};
                for (uint64_t j = 0; j < J; ++j) {
                    pm[t][r][j] = cadd(pm[t][r][j], cscale(w_m, power));
                    pp[t][r][j] = cadd(pp[t][r][j], cscale(w_p, power));
                    power = imul(power, ell);
                }
                pmass_m[t][r] += g_m->hi;
                pmass_p[t][r] += g_p->hi;
            }
            return ok();
        });
        for (const auto &st : statuses)
            if (!st.ok) return st;
        mm.assign(R, std::vector<Cv>(J, Cv{{0, 0}, {0, 0}}));
        mp.assign(R, std::vector<Cv>(J, Cv{{0, 0}, {0, 0}}));
        mass_m.assign(R, 0); mass_p.assign(R, 0);
        for (uint32_t t = 0; t < threads; ++t)
            for (size_t r = 0; r < R; ++r) {
                for (uint64_t j = 0; j < J; ++j) {
                    mm[r][j] = cadd(mm[r][j], pm[t][r][j]);
                    mp[r][j] = cadd(mp[r][j], pp[t][r][j]);
                }
                mass_m[r] += pmass_m[t][r];
                mass_p[r] += pmass_p[t][r];
            }
        /* grid abscissae */
        ell_h.assign(gx + 1, {0, 0}); Phi.assign(gx + 1, {U256{0, 0}, U256{0, 0}});
        auto gst = parallel_ranges(gx + 1, threads, [&](uint32_t, uint64_t begin, uint64_t end) -> Status {
            for (uint64_t i = begin; i < end; ++i) {
                auto [num, den] = x_at(i);
                auto ln_n = ln_high((uint64_t)num), ln_d = ln_high((uint64_t)den);
                std::pair<I, I> ell{ln_n.first - ln_d.second - ((I)LN4PI_HL + 1), ln_n.second - ln_d.first - (I)LN4PI_HL};
                ell_h[i] = ell;
                /* Phi = num (ell - SH) / (2 den): num < 2^60, ell - SH < 2^116 */
                U256 plo = mul_u128((U)num, (U)(ell.first - (I)SH)), phi = mul_u128((U)num, (U)(ell.second - (I)SH));
                uint64_t rem;
                U256 qlo = div256_u64(plo, (uint64_t)(2 * den), &rem);
                U256 qhi = div256_u64(phi, (uint64_t)(2 * den), &rem);
                if (rem) qhi = inc256(qhi);
                Phi[i] = {qlo, qhi};
            }
            return ok();
        });
        for (const auto &st : gst)
            if (!st.ok) return st;
        return ok();
    }

    std::optional<std::array<uint64_t, 3>> box_of(const Family &family, uint64_t i) const {
        if (family.kind == Family::Kind::Range || family.cols() == 1) {
            uint64_t m = family.kind == Family::Kind::Range ? family.a + i : (uint64_t)family.data->entries[i];
            if (m >= gx * gy * gt) return std::nullopt;
            return std::array<uint64_t, 3>{m / (gy * gt), (m / gt) % gy, m % gt};
        }
        const auto &e = family.data->entries;
        std::array<uint64_t, 3> js{(uint64_t)e[3 * i], (uint64_t)e[3 * i + 1], (uint64_t)e[3 * i + 2]};
        if (js[0] >= gx || js[1] >= gy || js[2] >= gt) return std::nullopt;
        return js;
    }

    /* sum_r P_r sum_j q_j m_{r,j} and the remainder loss; nullopt when the box is too large. */
    std::optional<std::pair<Cv, I>> range_sum(Cv dc1, Iv dc2, const std::vector<std::vector<Cv>> &moments,
                                              const std::vector<I> &mass, std::vector<Cv> &apow, std::vector<Iv> &bpow) const {
        uint64_t K0 = (J + 1) / 2;
        Cv total{{0, 0}, {0, 0}};
        I loss = 0;
        apow.resize(J);
        bpow.resize(J / 2 + 1);
        for (size_t r = 0; r < ranges.size(); ++r) {
            Iv Lr_iv{Lr[r], Lr[r]};
            Iv E_re = iadd(imul(dc1.re, Lr_iv), imul(dc2, isq(Lr_iv)));
            Iv E_im = imul(dc1.im, Lr_iv);
            auto g = iexp(E_re);
            if (!g) return std::nullopt;
            auto cs = cis(E_im);
            if (!cs) return std::nullopt;
            Cv P{imul(*g, cs->re), imul(*g, cs->im)};
            Cv a{iadd(dc1.re, iscale(imul(dc2, Lr_iv), 2, 1)), dc1.im};
            apow[0] = Cv{{S, S}, {0, 0}};
            for (uint64_t k = 1; k < J; ++k) {
                Cv z = cmul(apow[k - 1], a);
                apow[k] = Cv{iscale(z.re, 1, (I)k), iscale(z.im, 1, (I)k)};
            }
            bpow[0] = Iv{S, S};
            for (uint64_t m = 1; m <= J / 2; ++m) bpow[m] = iscale(imul(bpow[m - 1], dc2), 1, (I)m);
            Cv Sr{{0, 0}, {0, 0}};
            for (uint64_t j = 0; j < J; ++j) {
                Cv q{{0, 0}, {0, 0}};
                for (uint64_t m = 0; m <= j / 2; ++m) q = cadd(q, cscale(apow[j - 2 * m], bpow[m]));
                Sr = cadd(Sr, cmul(q, moments[r][j]));
            }
            total = cadd(total, cmul(P, Sr));
            I a_hi = cmod_upper(a);
            I lm = lmax[r];
            I Ub = cdiv(a_hi * lm, S) + cdiv(abs_upper(dc2) * cdiv(lm * lm, S), S);
            if (Ub >= (I)K0 * S) return std::nullopt;
            I rem = cdiv(powfact(Ub, (int)K0, false) * ((I)K0 + 1) * S, ((I)K0 + 1) * S - Ub);
            loss += cdiv(cmod_upper(P) * cdiv(mass[r] * rem, S), S);
        }
        return std::make_pair(total, loss);
    }

    std::optional<uint64_t> barrier_lower(std::array<uint64_t, 3> js, std::vector<Cv> &apow, std::vector<Iv> &bpow) const {
        uint64_t i = js[0], j = js[1], k = js[2];
        I xc_num = xa + xb, tc_num = ta + tb, tc_den = 2 * tden, yc_num = ya + yb, yc_den = 2 * yden;
        auto [x0, xd] = x_at(i); auto [x1, xd1] = x_at(i + 1); (void)xd1;
        Iv DX{fdiv((2 * x0 - xc_num * (I)gx) * S, 2 * xd), cdiv((2 * x1 - xc_num * (I)gx) * S, 2 * xd)};
        auto [y0, yd] = y_at(j); auto [y1, yd1] = y_at(j + 1); (void)yd1;
        Iv DY{fdiv((2 * y0 - yc_num * (I)gy) * S, 2 * yd), cdiv((2 * y1 - yc_num * (I)gy) * S, 2 * yd)};
        Iv Y{fdiv(y0 * S, yd), cdiv(y1 * S, yd)};
        auto [t0, td] = t_at(k); auto [t1, td1] = t_at(k + 1); (void)td1;
        Iv DT{fdiv((2 * t0 - tc_num * (I)gt) * S, 2 * td), cdiv((2 * t1 - tc_num * (I)gt) * S, 2 * td)};
        Iv T{fdiv(t0 * S, td), cdiv(t1 * S, td)};
        Iv ell{fdiv(ell_h[i].first, (I)1 << (KH - K)), cdiv(ell_h[i + 1].second, (I)1 << (KH - K))};
        Iv A{fdiv(ell.lo, 2), cdiv(ell.hi, 2)};
        I tsum_num = 2 * t1 * tden + tc_num * td, tsum_den = 2 * td * tden;
        I rho = cdiv(201 * tsum_num * xden * S, 100 * tsum_den * xa);
        I ty_num = 4 * t1 * y1 * tc_den * yc_den + tc_num * yc_num * 4 * td * yd;
        I ty_den = 4 * td * yd * tc_den * yc_den;
        I rhok = cdiv(ty_num * xden * S, ty_den * 2 * (xa - 6 * xden));
        Iv mid = iscale(iadd(imul(DT, A), imul(Tc, isub(A, Ac))), 1, 2);
        Iv dtp8 = iscale(imul(DT, {PI_L, PI_U}), 1, 8);
        Cv dc1{iadd(isub(ineg(iscale(DY, 1, 2)), mid), rad(rho)), iadd(iadd(iscale(DX, 1, 2), dtp8), rad(rho))};
        Cv dd1{iadd(isub(iscale(DY, 1, 2), mid), rad(rho + rhok)), iadd(ineg(iadd(iscale(DX, 1, 2), dtp8)), rad(rho + rhok))};
        Iv dc2 = iscale(DT, 1, 4);
        auto main = range_sum(dc1, dc2, mm, mass_m, apow, bpow);
        if (!main) return std::nullopt;
        auto part = range_sum(dd1, dc2, mp, mass_p, apow, bpow);
        if (!part) return std::nullopt;
        I eps = cdiv((2 * S + cdiv(T.hi * (4 * S + cdiv(42 * A.hi, 10)), S)) * xden, xa);
        Iv re_g = iadd(ineg(iscale(imul(Y, ell), 1, 2)), rad(eps));
        Iv Ph = phase_of_h(Phi[i].first, Phi[i + 1].second);
        Iv im_g = iadd(iadd(iadd(Ph, Iv{-cdiv(PI_U, 4), -fdiv(PI_L, 4)}), iscale(imul(imul({PI_L, PI_U}, T), ell), 1, 8)), rad(eps));
        auto g = iexp(re_g);
        if (!g) return std::nullopt;
        auto cs = cis(im_g);
        if (!cs) return std::nullopt;
        Cv G{imul(*g, cs->re), imul(*g, cs->im)};
        Cv f = cadd(main->first, cmul(G, part->first));
        I loss = main->second + cdiv(cmod_upper(G) * part->second, S);
        I value;
        if (real) value = f.re.lo - loss;
        else {
            I re_lo = std::min(abs_lower(f.re), ABS_CAP), im_lo = std::min(abs_lower(f.im), ABS_CAP);
            value = sqrt_lower(fdiv(re_lo * re_lo + im_lo * im_lo, S)) - loss;
        }
        I v = fdiv(value + offset * S, (I)1 << (int)(K - scale));
        if (v < 0) return 0;
        if (v >= ((I)1 << 64)) return std::nullopt;
        return (uint64_t)v;
    }
};

} // namespace lk::heat_dirichlet::detail
