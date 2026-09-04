/* heat_dirichlet phase_bound on the GPU: one thread per theta box, the psi loop inside.
 *
 * This is the box computation of heat_dirichlet_common.hpp (evaluate, bounds, tail_first,
 * tail_upper, phase_bound) step for step at scale 2^48: sums in 64-bit words, products in 128-bit
 * words, every rounding a floor or a ceiling by shifts, square roots exact by integer correction
 * of a double guess. The device therefore returns the same integers as the CPU backend, the naive
 * module and the Lean reference. Built by hipcc into liblemmakernel_hip.so; the hip backend in
 * liblemmakernel loads it at run time. */
#include "phase_gpu_abi.hpp"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>

namespace {

using namespace lk::heat_dirichlet::gpu;
using I = __int128;
using U = unsigned __int128;
constexpr int K = 48;
constexpr I S = (I)1 << K;

struct Iv {
    int64_t lo, hi;
};
struct Cv {
    Iv re, im;
};

__device__ __forceinline__ I fdivS(I a) { return a >> K; }
__device__ __forceinline__ I cdivS(I a) { return -((-a) >> K); }
__device__ __forceinline__ I mul(int64_t a, int64_t b) { return (I)a * (I)b; }
__device__ __forceinline__ int64_t iabs64(int64_t a) { return a < 0 ? -a : a; }

/* A 64 x 64 -> 128 product as (hi, lo), and its floor and ceiling divided by S when the quotient
 * fits 64 bits (every product of the box computation except the head's h-weighted sums, which
 * use I): the quotient is hi << 16 | lo >> 48. */
struct P128 {
    uint64_t lo;
    int64_t hi;
};
__device__ __forceinline__ P128 mul64(int64_t a, int64_t b) { return {(uint64_t)a * (uint64_t)b, __mul64hi(a, b)}; }
__device__ __forceinline__ int64_t floorS(P128 p) { return (int64_t)(((uint64_t)p.hi << 16) | (p.lo >> K)); }
__device__ __forceinline__ int64_t ceilS(P128 p) { return floorS(p) + (int64_t)((p.lo & (((uint64_t)1 << K) - 1)) != 0); }
__device__ __forceinline__ bool less(P128 a, P128 b) { return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo); }

/* [floor(min / S), ceil(max / S)] over the end-point products, by the sign cases. The first
 * operand is the one that is uniform across the wave wherever there is one (a coefficient, the
 * psi rotation), so that its sign case is a scalar branch; the second operand's sign only picks
 * end points by selects. With a >= 0 the extremes are (b.lo >= 0 ? a.lo : a.hi) b.lo and
 * (b.hi <= 0 ? a.lo : a.hi) b.hi; with a <= 0 they are (b.hi > 0 ? a.lo : a.hi) b.hi and
 * (b.lo < 0 ? a.lo : a.hi) b.lo; a straddling 0 takes all four products. */
__device__ __forceinline__ Iv imul(Iv a, Iv b) {
    P128 lo, hi;
    if (a.lo >= 0) {
        lo = mul64(b.lo >= 0 ? a.lo : a.hi, b.lo);
        hi = mul64(b.hi <= 0 ? a.lo : a.hi, b.hi);
    } else if (a.hi <= 0) {
        lo = mul64(b.hi > 0 ? a.lo : a.hi, b.hi);
        hi = mul64(b.lo < 0 ? a.lo : a.hi, b.lo);
    } else {
        P128 p = mul64(a.lo, b.lo), q = mul64(a.lo, b.hi), r = mul64(a.hi, b.lo), t = mul64(a.hi, b.hi);
        P128 lo1 = less(p, q) ? p : q, hi1 = less(p, q) ? q : p;
        P128 lo2 = less(r, t) ? r : t, hi2 = less(r, t) ? t : r;
        lo = less(lo1, lo2) ? lo1 : lo2;
        hi = less(hi1, hi2) ? hi2 : hi1;
    }
    return {floorS(lo), ceilS(hi)};
}
__device__ __forceinline__ Iv iadd(Iv a, Iv b) { return {a.lo + b.lo, a.hi + b.hi}; }
__device__ __forceinline__ Iv isub(Iv a, Iv b) { return {a.lo - b.hi, a.hi - b.lo}; }
__device__ __forceinline__ int64_t abs_upper(Iv a) { int64_t x = iabs64(a.lo), y = iabs64(a.hi); return x > y ? x : y; }
__device__ __forceinline__ int64_t abs_lower(Iv a) {
    if (a.lo <= 0 && 0 <= a.hi) return 0;
    int64_t x = iabs64(a.lo), y = iabs64(a.hi);
    return x < y ? x : y;
}
/* The square of an interval, tight at 0, as a 128-bit interval at scale S. */
__device__ __forceinline__ void isq(Iv a, I &lo, I &hi) {
    I p = mul(a.lo, a.lo), q = mul(a.hi, a.hi);
    I top = p > q ? p : q;
    I bottom = (a.lo <= 0 && 0 <= a.hi) ? 0 : (p < q ? p : q);
    lo = fdivS(bottom);
    hi = cdivS(top);
}

/* floor(sqrt(n)) for n < 2^128: a double guess, then exact correction. */
__device__ __noinline__ uint64_t isqrt128(U n) {
    if (n < 2) return (uint64_t)n;
    double d = (double)(uint64_t)(n >> 64) * 18446744073709551616.0 + (double)(uint64_t)n;
    uint64_t r = (uint64_t)sqrt(d);
    while (r > 0 && (U)r * r > n) --r;
    while ((U)(r + 1) * (r + 1) <= n) ++r;
    return r;
}
__device__ __forceinline__ int64_t sqrt_lower(I a) { return (int64_t)isqrt128((U)(a * S)); }
__device__ __forceinline__ int64_t sqrt_upper(I a) {
    U n = (U)(a * S);
    uint64_t r = isqrt128(n);
    return (int64_t)((U)r * r == n ? r : r + 1);
}
__device__ __forceinline__ int64_t mod_upper(Cv z) {
    int64_t ar = abs_upper(z.re), ai = abs_upper(z.im);
    return sqrt_upper(cdivS(mul(ar, ar)) + cdivS(mul(ai, ai)));
}
/* The same for |z| below 2^55 (centre values: below 16 at scale S), where ar^2 / S fits 64 bits
 * and ceil(sqrt(a S)) = ceil(2^24 sqrt(a)) comes from a double guess corrected against a S in
 * (hi, lo) form. */
__device__ __forceinline__ int64_t mod_upper_small(Cv z) {
    int64_t ar = abs_upper(z.re), ai = abs_upper(z.im);
    uint64_t a = (uint64_t)ceilS(mul64(ar, ar)) + (uint64_t)ceilS(mul64(ai, ai));
    if (a == 0) return 0;
    uint64_t r = (uint64_t)(sqrt((double)a) * 16777216.0);
    /* n = a << 48 as (hi, lo); want the least r with r^2 >= n */
    uint64_t n_hi = a >> 16, n_lo = a << K;
    auto ge = [&](uint64_t x) { /* x^2 >= n */
        uint64_t p_lo = x * x, p_hi = __umul64hi(x, x);
        return p_hi > n_hi || (p_hi == n_hi && p_lo >= n_lo);
    };
    while (r > 0 && ge(r - 1)) --r;
    while (!ge(r)) ++r;
    return (int64_t)r;
}
__device__ __forceinline__ Cv rotate(Cv z, Iv cs, Iv sn) {
    return {isub(imul(cs, z.re), imul(sn, z.im)), iadd(imul(sn, z.re), imul(cs, z.im))};
}

__device__ __forceinline__ int64_t fdiv64(int64_t a, int64_t b) {
    int64_t q = a / b;
    return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q;
}
__device__ __forceinline__ int64_t cdiv64(int64_t a, int64_t b) { return -fdiv64(-a, b); }

/* Enclosures of cos and sin over the arc of table steps [J - X, J + X]. */
__device__ __noinline__ Cv arc(const Circle *circle, int64_t M, int64_t J, int64_t X) {
    if (2 * X >= M) return {{-(int64_t)S, (int64_t)S}, {-(int64_t)S, (int64_t)S}};
    int64_t a = J - X, b = J + X;
    auto contains = [&](int64_t r) { return fdiv64(b - r, M) >= cdiv64(a - r, M); };
    auto mod = [&](int64_t z) { return (uint64_t)(((z % M) + M) % M); };
    Circle ea = circle[mod(a)], eb = circle[mod(b)];
    int64_t cos_hi = contains(0) ? (int64_t)S : (ea.c_hi > eb.c_hi ? ea.c_hi : eb.c_hi);
    int64_t cos_lo = contains(M / 2) ? -(int64_t)S : (ea.c_lo < eb.c_lo ? ea.c_lo : eb.c_lo);
    int64_t sin_hi = contains(M / 4) ? (int64_t)S : (ea.s_hi > eb.s_hi ? ea.s_hi : eb.s_hi);
    int64_t sin_lo = contains(3 * M / 4) ? -(int64_t)S : (ea.s_lo < eb.s_lo ? ea.s_lo : eb.s_lo);
    return {{cos_lo, cos_hi}, {sin_lo, sin_hi}};
}

struct Parts {
    Cv S, A, GS[4], GA[4];
};

/* J mod M for 0 <= J < 2^63 and M < 2^32 without a division: a multiply-high quotient guess
 * (never above the true quotient) and a correction. */
__device__ __forceinline__ uint32_t mod_small(uint64_t J, uint32_t M, uint64_t invM) {
    uint64_t q = (uint64_t)(((U)J * invM) >> 64);
    uint64_t r = J - q * M;
    while (r >= M) r -= M;
    return (uint32_t)r;
}

/* One term's part (S or A) into its centre value T and gradients G. */
__device__ __forceinline__ void accumulate(const Params &P, const Circle *circle, int64_t M, const Term &term,
                                           Iv c, bool taylor, int64_t X, uint32_t J, Cv &T, Cv (&G)[4]) {
    if (!taylor) {
        Cv e = arc(circle, M, J, X);
        T.re = iadd(T.re, imul(c, e.re));
        T.im = iadd(T.im, imul(c, e.im));
        return;
    }
    Circle e = circle[J];
    Iv wre = imul(c, {e.c_lo, e.c_hi}), wim = imul(c, {e.s_lo, e.s_hi});
    T.re = iadd(T.re, wre);
    T.im = iadd(T.im, wim);
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        int64_t v = term.v[p];
        if (!v) continue;
        G[p].re.lo -= v * wim.hi;
        G[p].re.hi -= v * wim.lo;
        G[p].im.lo += v * wre.lo;
        G[p].im.hi += v * wre.hi;
    }
}

/* The S and A parts of a component on the theta box. */
__device__ __forceinline__ void evaluate(const Params &P, const Component &comp, const Term *terms, const Circle *circle,
                                         const int64_t steps[4], Parts &out) {
    int64_t M = (int64_t)P.M;
    const uint32_t M32 = (uint32_t)P.M;
    const uint64_t invM = P.invM;
    const int64_t XA = (int64_t)(P.M / (2 * P.npsi));
    Cv S{{0, 0}, {0, 0}}, A{{0, 0}, {0, 0}};
    Cv GS[4], GA[4];
#pragma unroll
    for (int p = 0; p < 4; ++p) GS[p] = GA[p] = {{0, 0}, {0, 0}};
    for (uint32_t t = comp.begin; t < comp.begin + comp.count; ++t) {
        Term term = terms[t];
        int64_t Jfull = 0;
#pragma unroll
        for (int p = 0; p < 4; ++p) Jfull += (int64_t)term.v[p] * steps[p];
        uint32_t J = mod_small((uint64_t)Jfull, M32, invM);
        if (term.cS_lo != 0 || term.cS_hi != 0)
            accumulate(P, circle, M, term, {term.cS_lo, term.cS_hi}, term.taylorS, term.X, J, S, GS);
        if (term.cA_lo != 0 || term.cA_hi != 0)
            accumulate(P, circle, M, term, {term.cA_lo, term.cA_hi}, term.taylorA, (int64_t)term.X + XA, J, A, GA);
    }
    out.S = S;
    out.A = A;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        out.GS[p] = GS[p];
        out.GA[p] = GA[p];
    }
}

/* (upper bound on |T|, lower bound on dist(T, (-inf, 0])) of the head over the theta box and psi
 * box jpsi, precisely. */
__device__ void bounds(const Params &P, const Component &comp, const Parts &parts, Iv cs, Iv sn, int64_t &dist) {
    Cv Ar = rotate(parts.A, cs, sn);
    Cv T{iadd(parts.S.re, Ar.re), iadd(parts.S.im, Ar.im)};
    Cv G[5];
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        Cv gr = rotate(parts.GA[p], cs, sn);
        G[p] = {iadd(parts.GS[p].re, gr.re), iadd(parts.GS[p].im, gr.im)};
    }
    G[4] = {{-Ar.im.hi, -Ar.im.lo}, Ar.re};
    I Alo, Ahi, Blo, Bhi;
    isq(T.re, Alo, Ahi);
    isq(T.im, Blo, Bhi);
    Alo += Blo;
    Ahi += Bhi;
    I B = 0, re_var = 0, im_var = 0;
#pragma unroll
    for (int p = 0; p < 5; ++p) {
        Iv x = iadd(imul(T.re, G[p].re), imul(T.im, G[p].im));
        B += cdivS(mul(P.h[p], abs_upper(x)));
        re_var += cdivS(mul(P.h[p], abs_upper(G[p].re)));
        im_var += cdivS(mul(P.h[p], abs_upper(G[p].im)));
    }
    I lower_mod = (I)sqrt_lower((Alo - 2 * B) > 0 ? (Alo - 2 * B) : 0) - comp.Q;
    I re_margin = (I)T.re.lo - re_var - comp.Q;
    I im_lower = (I)abs_lower(T.im) - im_var - comp.Q;
    I d = im_lower;
    I e = re_margin >= 0 ? lower_mod : 0;
    if (e > d) d = e;
    if (d < 0) d = 0;
    dist = (int64_t)d;
}

/* The psi-independent part of a tail component's upper bound. */
__device__ int64_t tail_first(const Params &P, const Component &comp, const Parts &parts) {
    I first = 0;
#pragma unroll
    for (int p = 0; p < 4; ++p) first += cdivS(mul(P.h[p], mod_upper(parts.GS[p]) + mod_upper(parts.GA[p])));
    first += cdivS(mul(P.h[4], mod_upper(parts.A)));
    return (int64_t)(first + comp.Q);
}

__global__ void phase_kernel(const Params P, const Component *comps, const Term *terms, const Circle *circle,
                             const uint32_t *boxes, uint64_t nboxes, uint64_t *out) {
    extern __shared__ int64_t F[]; /* [npsi][blockDim.x] */
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    bool active = i < nboxes;
    int64_t steps[4] = {0, 0, 0, 0};
    if (active)
#pragma unroll
        for (int p = 0; p < 4; ++p) steps[p] = (int64_t)(2 * boxes[4 * i + p] + 1) * (int64_t)(P.M / (2 * P.g[p]));
    const unsigned lane = threadIdx.x, width = blockDim.x;
    Parts parts;
    /* the head: dist per psi */
    if (active) {
        evaluate(P, comps[0], terms, circle, steps, parts);
        for (uint32_t j = 0; j < P.npsi; ++j) {
            Circle e = circle[(2 * (uint64_t)j + 1) * (P.M / (2 * P.npsi))];
            int64_t dist;
            bounds(P, comps[0], parts, {e.c_lo, e.c_hi}, {e.s_lo, e.s_hi}, dist);
            F[j * width + lane] = dist;
        }
        for (uint32_t k = 1; k < P.ncomp; ++k) {
            const Component comp = comps[k];
            evaluate(P, comp, terms, circle, steps, parts);
            int64_t first = tail_first(P, comp, parts);
            for (uint32_t j = 0; j < P.npsi; ++j) {
                Circle e = circle[(2 * (uint64_t)j + 1) * (P.M / (2 * P.npsi))];
                Cv Ar = rotate(parts.A, {e.c_lo, e.c_hi}, {e.s_lo, e.s_hi});
                Cv T{iadd(parts.S.re, Ar.re), iadd(parts.S.im, Ar.im)};
                int64_t upper = mod_upper_small(T) + first;
                if (upper > comp.mass) upper = comp.mass;
                F[j * width + lane] -= (int64_t)cdivS(mul(comp.W, upper));
            }
        }
        int64_t best = F[lane];
        for (uint32_t j = 1; j < P.npsi; ++j) {
            int64_t f = F[j * width + lane];
            if (f < best) best = f;
        }
        I v = ((I)best - (I)P.loss + (I)P.offset * S) >> P.shift;
        if (v < 0) out[i] = 0;
        else if (v > (I)UINT64_MAX - 1) out[i] = UINT64_MAX;
        else out[i] = (uint64_t)v;
    }
}

int fail(char *err, size_t errlen, const char *what, hipError_t e) {
    snprintf(err, errlen, "%s: %s", what, hipGetErrorString(e));
    return 1;
}

} // namespace

extern "C" int lk_heat_dirichlet_phase_available() {
    int n = 0;
    if (hipGetDeviceCount(&n) != hipSuccess) return 0;
    return n > 0;
}

extern "C" int lk_heat_dirichlet_phase_gpu(const Params *P, const Component *comps, const Term *terms,
                                           const Circle *circle, const uint32_t *boxes, uint64_t nboxes,
                                           uint64_t *out, char *err, size_t errlen) {
    if (!nboxes) return 0;
    Component *d_comps = nullptr;
    Term *d_terms = nullptr;
    Circle *d_circle = nullptr;
    uint32_t *d_boxes = nullptr;
    uint64_t *d_out = nullptr;
    hipError_t e;
    auto cleanup = [&] {
        if (d_comps) (void)hipFree(d_comps);
        if (d_terms) (void)hipFree(d_terms);
        if (d_circle) (void)hipFree(d_circle);
        if (d_boxes) (void)hipFree(d_boxes);
        if (d_out) (void)hipFree(d_out);
    };
#define LK_HIP(call, what) \
    if ((e = (call)) != hipSuccess) { cleanup(); return fail(err, errlen, what, e); }
    LK_HIP(hipMalloc(&d_comps, P->ncomp * sizeof(Component)), "hipMalloc components");
    LK_HIP(hipMalloc(&d_terms, (P->nterms ? P->nterms : 1) * sizeof(Term)), "hipMalloc terms");
    LK_HIP(hipMalloc(&d_circle, P->ncircle * sizeof(Circle)), "hipMalloc circle");
    LK_HIP(hipMalloc(&d_boxes, nboxes * 4 * sizeof(uint32_t)), "hipMalloc boxes");
    LK_HIP(hipMalloc(&d_out, nboxes * sizeof(uint64_t)), "hipMalloc out");
    LK_HIP(hipMemcpy(d_comps, comps, P->ncomp * sizeof(Component), hipMemcpyHostToDevice), "copy components");
    if (P->nterms) LK_HIP(hipMemcpy(d_terms, terms, P->nterms * sizeof(Term), hipMemcpyHostToDevice), "copy terms");
    LK_HIP(hipMemcpy(d_circle, circle, P->ncircle * sizeof(Circle), hipMemcpyHostToDevice), "copy circle");
    LK_HIP(hipMemcpy(d_boxes, boxes, nboxes * 4 * sizeof(uint32_t), hipMemcpyHostToDevice), "copy boxes");
    unsigned block = P->npsi > 64 ? 32 : 64;
    size_t shmem = (size_t)block * P->npsi * sizeof(int64_t);
    uint64_t grid = (nboxes + block - 1) / block;
    phase_kernel<<<(unsigned)grid, block, shmem>>>(*P, d_comps, d_terms, d_circle, d_boxes, nboxes, d_out);
    LK_HIP(hipGetLastError(), "launch");
    LK_HIP(hipDeviceSynchronize(), "kernel");
    LK_HIP(hipMemcpy(out, d_out, nboxes * sizeof(uint64_t), hipMemcpyDeviceToHost), "copy out");
#undef LK_HIP
    cleanup();
    return 0;
}
