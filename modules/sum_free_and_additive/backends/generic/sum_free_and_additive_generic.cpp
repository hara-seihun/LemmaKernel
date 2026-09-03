/* Portable additive-combinatorics predicates on finite sets of naturals.
 *
 * Every quantity of a prefix P lives in bitsets over the ambient group (Z/n, or the integers
 * below 2L+1 where L is the largest element): P itself, its reversal, its sumset S = P+P, its
 * difference set D = P-P, and the set F of values that cannot be added without breaking the
 * predicate. Adding x turns each of these into a few shifted ORs, so the cost of a node is a few
 * words per 64 values of the ambient group instead of a loop over the prefix, and the whole
 * candidate set of the next level is read off F at once:
 *
 *   sum-free   F = (P+P) u (P-P) u {v : 2v in P} u {0}
 *   Sidon      F = (P+P-P) u {v : 2v in P+P} u P
 *   AP-free    F = every value completing an L-term progression with L-1 terms of P
 *
 * The family is walked over dictionary indices in canonical order. At a node, the candidates
 * are the allowed indices above the last one; when fewer remain than the subset still needs
 * the node is dead, and at the last level the answer is a popcount. Subtrees never entered are
 * accounted for as index ranges, so completeness is exact. Work is split over depth-two
 * prefixes handed to threads in canonical order, which keeps `first` correct and balances the
 * unequal subtrees of a pruned search.
 *
 * The integer operations have no pruning and are computed at every leaf from the parent's
 * bitsets, so a leaf costs the same few words as a node. The difference-multiplicity operation
 * keeps a count per difference code over the prefix instead.
 *
 * Difference codes match the Lean reference: (x + n - y) % n in Z/n, and x + shift - y over the
 * integers. Only counts of codes escape, so the shift the backend picks (L) need not be the
 * reference's. */
#include "../../../../runtime/src/reduce.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <numeric>

namespace lk::sum_free_and_additive {
namespace {

using R = Result<std::shared_ptr<Object>>;
constexpr int INVALID = 1;
constexpr int INTERNAL = 4;
constexpr uint64_t MAX_SPAN = 1ULL << 22;
constexpr uint64_t MEMORY_BUDGET = 1ULL << 30; /* bytes of per-thread level state, all threads */

enum class Op { IsSumFree, IsSidon, IsApFree, SumsetSize, DifferenceSetSize, SchurTripleCount,
                MaxDifferenceMultiplicity, IsSmallDoubling };

/* `extends_sum_free` is `is_sum_free` of the member together with a fixed `context`. */
Result<Op> parse_op(const std::string &name, bool &with_context) {
    static const std::map<std::string, Op> names{
        {"is_sum_free", Op::IsSumFree}, {"is_sidon", Op::IsSidon}, {"is_ap_free", Op::IsApFree},
        {"extends_sum_free", Op::IsSumFree}, {"extends_sidon", Op::IsSidon}, {"extends_ap_free", Op::IsApFree},
        {"sumset_size", Op::SumsetSize}, {"difference_set_size", Op::DifferenceSetSize},
        {"schur_triple_count", Op::SchurTripleCount},
        {"max_difference_multiplicity", Op::MaxDifferenceMultiplicity},
        {"is_small_doubling", Op::IsSmallDoubling}};
    auto it = names.find(name);
    if (it == names.end()) return Result<Op>::failure(INTERNAL, "unknown sum_free_and_additive operation " + name);
    with_context = name.rfind("extends_", 0) == 0;
    return Result<Op>::success(it->second);
}

/* Which values a prefix forbids for the next element (Forbid), a prefix-size bound (Doubling),
 * or a value at every leaf (Integer). */
enum class Mode { Forbid, Doubling, Integer };
Mode mode_of(Op op) {
    switch (op) {
    case Op::IsSumFree: case Op::IsSidon: case Op::IsApFree: return Mode::Forbid;
    case Op::IsSmallDoubling: return Mode::Doubling;
    default: return Mode::Integer;
    }
}

/* ---- bitsets over the ambient group ------------------------------------------------------- */

/* dst |= src << s, s possibly negative, over W words; bits shifted outside are dropped. */
inline void or_shift(uint64_t *dst, const uint64_t *src, int64_t s, uint64_t W) {
    if (s >= 0) {
        uint64_t q = (uint64_t)s >> 6, r = (uint64_t)s & 63;
        if (q >= W) return;
        if (r == 0) {
            for (uint64_t w = q; w < W; ++w) dst[w] |= src[w - q];
        } else {
            dst[q] |= src[0] << r;
            for (uint64_t w = q + 1; w < W; ++w) dst[w] |= (src[w - q] << r) | (src[w - q - 1] >> (64 - r));
        }
    } else {
        uint64_t t = (uint64_t)(-s), q = t >> 6, r = t & 63;
        if (q >= W) return;
        uint64_t last = W - q;
        if (r == 0) {
            for (uint64_t w = 0; w < last; ++w) dst[w] |= src[w + q];
        } else {
            for (uint64_t w = 0; w + 1 < last; ++w) dst[w] |= (src[w + q] >> r) | (src[w + q + 1] << (64 - r));
            dst[last - 1] |= src[W - 1] >> r;
        }
    }
}

inline uint64_t popcount(const uint64_t *a, uint64_t W) {
    uint64_t c = 0;
    for (uint64_t w = 0; w < W; ++w) c += __builtin_popcountll(a[w]);
    return c;
}
inline uint64_t popcount_and(const uint64_t *a, const uint64_t *b, uint64_t W) {
    uint64_t c = 0;
    for (uint64_t w = 0; w < W; ++w) c += __builtin_popcountll(a[w] & b[w]);
    return c;
}
inline uint64_t popcount_andnot(const uint64_t *a, const uint64_t *b, uint64_t W) { /* |a \ b| */
    uint64_t c = 0;
    for (uint64_t w = 0; w < W; ++w) c += __builtin_popcountll(a[w] & ~b[w]);
    return c;
}
inline bool bit(const uint64_t *a, uint64_t v) { return (a[v >> 6] >> (v & 63)) & 1; }
inline void set(uint64_t *a, uint64_t v) { a[v >> 6] |= 1ULL << (v & 63); }
inline void clear(uint64_t *a, uint64_t v) { a[v >> 6] &= ~(1ULL << (v & 63)); }

/* The ambient group and its encodings. Sums and elements are values below `span`; a difference
 * x - y has a code: (x - y) mod n, or x - y + L over the integers. The reversal of P has bit
 * rev(y) at y in P, so that x - P is one shift of it. */
struct Ambient {
    uint64_t modulus = 0, L = 0, span = 0, W = 0;
    uint64_t mask = ~0ULL; /* of the last word */

    uint64_t add(uint64_t x, uint64_t y) const { return modulus ? (x + y) % modulus : x + y; }
    uint64_t rev(uint64_t y) const { return modulus ? (modulus - y) % modulus : L - y; }
    uint64_t zero_code() const { return modulus ? 0 : L; }
    /* Shifts that place x - P (from the reversal) and P - x (from P) as values or as codes. */
    int64_t x_minus_P_value(uint64_t x) const { return modulus ? (int64_t)x : (int64_t)x - (int64_t)L; }
    int64_t P_minus_x_value(uint64_t x) const { return -(int64_t)x; }
    int64_t x_minus_P_code(uint64_t x) const { return (int64_t)x; }
    int64_t P_minus_x_code(uint64_t x) const { return modulus ? -(int64_t)x : (int64_t)L - (int64_t)x; }
    /* x + D as values, from difference codes. */
    int64_t x_plus_codes(uint64_t x) const { return modulus ? (int64_t)x : (int64_t)x - (int64_t)L; }

    /* dst |= src shifted by s: a plain shift over the integers, a rotation in Z/n. */
    void place(uint64_t *dst, const uint64_t *src, int64_t s) const {
        if (modulus) {
            int64_t n = (int64_t)modulus;
            s %= n;
            if (s < 0) s += n;
            or_shift(dst, src, s, W);
            if (s) or_shift(dst, src, s - n, W);
        } else {
            or_shift(dst, src, s, W);
        }
        dst[W - 1] &= mask;
    }
    void set_value(uint64_t *dst, uint64_t v) const { if (v < span) set(dst, v); }
    /* Every z with 2z = v, into dst. */
    void halves(uint64_t *dst, uint64_t v) const {
        if (!modulus) { if (!(v & 1)) set(dst, v / 2); return; }
        if (modulus & 1) { set(dst, (unsigned __int128)v * ((modulus + 1) / 2) % modulus); return; }
        if (v & 1) return;
        set(dst, v / 2);
        set(dst, v / 2 + modulus / 2);
    }
};

/* Bitset arithmetic for a compile-time word count, on a local array the compiler keeps in
 * registers once the loops unroll. The generic Ambient::place works through memory, and its
 * read-modify-write of a destination that was just copied stalls on store forwarding; these
 * build the new bitset in registers and store it once. */
#define LK_INLINE inline __attribute__((always_inline))
/* Basic-block vectorisation repacks the register array through the stack and reintroduces the
 * store-forwarding stall, so the kernels that use Words are compiled without it. */
#define LK_SCALAR __attribute__((optimize("no-tree-slp-vectorize", "no-tree-vectorize")))
template <int WT> struct Words {
    static LK_INLINE void or_shift(uint64_t *a, const uint64_t *src, int64_t s) {
        if (s >= 0) {
            uint64_t q = (uint64_t)s >> 6, r = (uint64_t)s & 63;
#pragma GCC novector
            for (int w = 0; w < WT; ++w) {
                int64_t i = w - (int64_t)q;
                if (i < 0) continue;
                uint64_t v = src[i] << r;
                if (r && i > 0) v |= src[i - 1] >> (64 - r);
                a[w] |= v;
            }
        } else {
            uint64_t t = (uint64_t)(-s), q = t >> 6, r = t & 63;
#pragma GCC novector
            for (int w = 0; w < WT; ++w) {
                uint64_t i = (uint64_t)w + q;
                if (i >= (uint64_t)WT) continue;
                uint64_t v = src[i] >> r;
                if (r && i + 1 < (uint64_t)WT) v |= src[i + 1] << (64 - r);
                a[w] |= v;
            }
        }
    }
    static LK_INLINE void place(uint64_t *a, const uint64_t *src, int64_t s, const Ambient &amb) {
        if (amb.modulus) {
            int64_t n = (int64_t)amb.modulus;
            s %= n;
            if (s < 0) s += n;
            or_shift(a, src, s);
            if (s) or_shift(a, src, s - n);
        } else {
            or_shift(a, src, s);
        }
    }
    static LK_INLINE void load(uint64_t *a, const uint64_t *src) {
#pragma GCC novector
        for (int w = 0; w < WT; ++w) a[w] = src[w];
    }
    static LK_INLINE void store(uint64_t *dst, const uint64_t *a, const Ambient &amb) {
#pragma GCC novector
        for (int w = 0; w < WT; ++w) dst[w] = a[w];
        dst[WT - 1] &= amb.mask;
    }
    /* Constant word indices only, so that `a` can live in registers. */
    static LK_INLINE void set(uint64_t *a, uint64_t v) {
        uint64_t q = v >> 6, b = 1ULL << (v & 63);
        for (int w = 0; w < WT; ++w) a[w] |= (uint64_t)w == q ? b : 0;
    }
    static LK_INLINE void set_value(uint64_t *a, uint64_t v, const Ambient &amb) { if (v < amb.span) set(a, v); }
};
constexpr int MAX_TEMPLATE_WORDS = 8;

/* ---- a search problem -------------------------------------------------------------------- */

/* Sidon and progression-free sets over the integers are translation invariant and hereditary,
 * so the least span of a t-element set, G(t), bounds every search over an increasing
 * dictionary: an element chosen with r more to come must sit at least G(r+1) below the largest
 * value. G is computed by this same search, one t at a time, and kept for the process. */
/* Operations with a least span per set size. A 2-term progression is any pair, so no set of two
 * or more elements is 2-AP-free and the span table would search forever. */
bool spannable(Op op, uint64_t length) { return op == Op::IsSidon || (op == Op::IsApFree && length >= 3); }

struct Problem {
    Op op;
    Mode mode;
    Ambient amb;
    uint64_t k = 0, m = 0, length = 0, bound_num = 0, bound_den = 1;
    const Entry *dict = nullptr;
    std::vector<int32_t> index_of_value;
    std::vector<std::vector<Index>> cum; /* cum[j][c] = sum_{c' in [j, c)} C(m-1-c', k-1-j) */
    /* Elements every member is taken together with (the extends_* operations), increasing.
     * They sit below every dictionary element when the dictionary is sorted. */
    std::vector<uint64_t> context;
    /* Increasing integer dictionaries only: */
    bool sorted = false;
    bool interval = false; /* consecutive values: index = value - dict[0], so F stands in for the allowed set */
    std::vector<uint64_t> spans;    /* G(t) for t < k; spans[k] is a lower bound on G(k) */
    std::vector<uint32_t> ge;       /* ge[v] = first index whose value is >= v, or m */
    std::vector<int64_t> level_hi;  /* largest index choosable at level j from the span table, -1 if none */
    /* Count only canonical sets (first gap <= last gap), weighting them by their orbit. Needs a
     * dictionary symmetric under v -> dict[0] + dict[m-1] - v and a reflection-invariant op. */
    bool mirror = false;

    uint64_t maxval() const { return dict[m - 1]; }
    int64_t index_at_most(int64_t v) const { /* largest index with value <= v, or -1 */
        if (v < 0) return -1;
        uint64_t vv = std::min<uint64_t>((uint64_t)v, maxval());
        return (int64_t)ge[vv + 1] - 1;
    }
};

void finish_problem(Problem &P) {
    P.mode = mode_of(P.op);
    P.index_of_value.assign(P.amb.span, -1);
    for (uint64_t i = 0; i < P.m; ++i) P.index_of_value[P.dict[i]] = (int32_t)i;
    /* The sorted walk keeps every level above the one before, so the context must lie below the
     * dictionary; otherwise the general update, which forbids in both directions, is used. */
    P.sorted = !P.amb.modulus && P.m > 0 && (P.context.empty() || P.context.back() < P.dict[0]);
    for (uint64_t i = 1; i < P.m && P.sorted; ++i) P.sorted = P.dict[i] > P.dict[i - 1];
    P.interval = P.sorted && P.maxval() - P.dict[0] == P.m - 1;
    if (P.sorted) {
        P.ge.assign(P.maxval() + 2, (uint32_t)P.m);
        for (uint64_t v = 0, i = 0; v <= P.maxval(); ++v) {
            while (i < P.m && P.dict[i] < v) ++i;
            P.ge[v] = (uint32_t)i;
        }
    }
    P.level_hi.assign(P.k + 1, 0);
    for (uint64_t j = 0; j < P.k; ++j) {
        int64_t hi = (int64_t)(P.m - (P.k - j));
        if (P.sorted && !P.spans.empty())
            hi = std::min(hi, P.index_at_most((int64_t)P.maxval() - (int64_t)P.spans[P.k - j]));
        P.level_hi[j] = hi;
    }
}

/* ---- one thread's walk ------------------------------------------------------------------- */

enum Slot { P = 0, PREV, S, D, F, SLOTS };

struct Walker {
    const Problem &prob;
    Op op;
    Mode mode;
    Reduction reduction;
    Ambient amb;
    uint64_t k, m, length, bound_num, bound_den;
    const Entry *dict;
    const std::vector<int32_t> *index_of_value;
    const std::vector<std::vector<Index>> *cum;
    uint64_t W, WA;
    uint32_t slots;
    int64_t nctx;              /* context elements, held at levels -nctx..-1 */
    bool context_alive = true; /* the context itself has the property */
    Accumulator acc;
    std::vector<int64_t> hi_of_level;  /* level_hi tightened by the mirror rule for this unit */
    uint64_t g1 = 0;                   /* first gap of the current unit, mirror mode */

    std::vector<uint64_t> store;    /* (k+1) levels x slots x W */
    std::vector<uint64_t> allowed;  /* (k+1) levels x WA, Forbid mode */
    std::vector<uint64_t> cands;    /* (k+1) levels x WA: candidate scratch per level */
    std::vector<uint64_t> tmp, tmp2;
    std::vector<uint32_t> vals;     /* the prefix */
    std::vector<uint64_t> measure;  /* per level: |S|, |D|, Schur count or max multiplicity */
    std::vector<uint8_t> pending;   /* per level: push() left P, PREV, S, D to finish() */
    std::vector<uint8_t> pending_f; /* per level: push() left F to finish_forbidden() */
    uint64_t &meas(int64_t j) { return measure[j + nctx]; }
    uint8_t &pend(int64_t j) { return pending[j + nctx]; }
    uint8_t &pendf(int64_t j) { return pending_f[j + nctx]; }
    std::vector<uint32_t> mult;     /* per difference code, MaxDifferenceMultiplicity */

    Walker(const Problem &p, Reduction r, Shared *shared)
        : prob(p), op(p.op), mode(p.mode), reduction(r), amb(p.amb), k(p.k), m(p.m), length(p.length),
          bound_num(p.bound_num), bound_den(p.bound_den), dict(p.dict), index_of_value(&p.index_of_value),
          cum(&p.cum), W(p.amb.W), WA((p.m + 63) / 64), slots(SLOTS), nctx((int64_t)p.context.size()),
          acc(r, shared), hi_of_level(p.level_hi) {
        uint64_t levels = k + 1 + nctx;
        store.assign(levels * slots * W, 0);
        cands.assign(levels * WA, 0);
        tmp.assign(W, 0);
        tmp2.assign(W, 0);
        measure.assign(levels, 0);
        pending.assign(levels, 0);
        pending_f.assign(levels, 0);
        if (mode == Mode::Forbid) {
            if (op == Op::IsSumFree) amb.set_value(level(-nctx, F), 0);
            if (!prob.interval) {
                allowed.assign(levels * WA, 0);
                uint64_t *A = level_allowed(-nctx);
                for (uint64_t i = 0; i < m; ++i) set(A, i);
                forbid_new(level(-nctx, F), nullptr, A);
            }
        }
        if (op == Op::MaxDifferenceMultiplicity) mult.assign(amb.span, 0);
        /* The context is pushed once, as the levels below 0; every member extends it. */
        for (int64_t i = 0; i < nctx && context_alive; ++i) {
            int64_t j = i - nctx;
            uint64_t x = p.context[i];
            if (mode == Mode::Forbid && bit(level(j, F), x)) { context_alive = false; break; }
            context_alive = push(j, x);
            if (context_alive) { finish(j + 1); finish_forbidden(j + 1); }
        }
    }

    static uint64_t bytes_per_thread(uint64_t k, uint64_t c, uint64_t W, uint64_t m) {
        return (k + 1 + c) * (SLOTS * W + 2 * ((m + 63) / 64)) * 8 + 2 * W * 8;
    }

    uint64_t *level(int64_t j, Slot s) { return store.data() + ((j + nctx) * slots + s) * W; }
    uint64_t *level_allowed(int64_t j) { return allowed.data() + (j + nctx) * WA; }
    bool allowed_index(uint64_t j, uint64_t c) {
        return prob.interval ? !bit(level(j, F), c + dict[0]) : bit(level_allowed(j), c);
    }

    /* Clear from `A` the indices whose values are set in `now` but not in `before` (all of `now`
     * when `before` is null). */
    void forbid_new(const uint64_t *now, const uint64_t *before, uint64_t *A) {
        const auto &iov = *index_of_value;
        for (uint64_t w = 0; w < W; ++w) {
            uint64_t d = before ? now[w] & ~before[w] : now[w];
            while (d) {
                uint64_t v = w * 64 + __builtin_ctzll(d);
                d &= d - 1;
                int32_t i = iov[v];
                if (i >= 0) clear(A, i);
            }
        }
    }

    /* Level j+1 from level j and the new element x. Returns false when the prefix is dead: it
     * contains a progression, or its sumset already exceeds the doubling bound.
     *
     * Forbid-mode operations compute only F here, which is all a child needs to learn whether it
     * has enough candidates; most children do not. The remaining bitsets of level j+1 are
     * produced by finish() when the level is about to be expanded. */
    bool push(int64_t j, uint64_t x) {
        uint64_t *P0 = level(j, P), *R0 = level(j, PREV), *S0 = level(j, S), *D0 = level(j, D), *F0 = level(j, F);
        uint64_t *P1 = level(j + 1, P), *R1 = level(j + 1, PREV), *S1 = level(j + 1, S), *D1 = level(j + 1, D),
                 *F1 = level(j + 1, F);
        bool alive = true;
        if (mode == Mode::Forbid && prob.sorted) {
            switch (W) {
            case 1: sorted_forbid<1>(j, x); break;
            case 2: sorted_forbid<2>(j, x); break;
            case 3: sorted_forbid<3>(j, x); break;
            case 4: sorted_forbid<4>(j, x); break;
            case 5: sorted_forbid<5>(j, x); break;
            case 6: sorted_forbid<6>(j, x); break;
            case 7: sorted_forbid<7>(j, x); break;
            case 8: sorted_forbid<8>(j, x); break;
            default: sorted_forbid<0>(j, x); break;
            }
            if (op == Op::IsSidon) pendf(j + 1) = true;
            else if (!prob.interval) {
                uint64_t *A0 = level_allowed(j), *A1 = level_allowed(j + 1);
                std::memcpy(A1, A0, WA * 8);
                forbid_new(F1, F0, A1);
            }
            vals.push_back((uint32_t)x);
            return true;
        }
        if (mode != Mode::Forbid) {
            std::memcpy(P1, P0, W * 8);
            set(P1, x);
        }
        switch (op) {
        case Op::IsSumFree:
            switch (W) {
            case 1: sum_free_forbid<1>(j, x); break;
            case 2: sum_free_forbid<2>(j, x); break;
            case 3: sum_free_forbid<3>(j, x); break;
            case 4: sum_free_forbid<4>(j, x); break;
            case 5: sum_free_forbid<5>(j, x); break;
            case 6: sum_free_forbid<6>(j, x); break;
            case 7: sum_free_forbid<7>(j, x); break;
            case 8: sum_free_forbid<8>(j, x); break;
            default:
                std::memcpy(F1, F0, W * 8);
                amb.place(F1, P0, (int64_t)x);                    /* x + P */
                amb.set_value(F1, amb.add(x, x));                 /* 2x */
                amb.place(F1, R0, amb.x_minus_P_value(x));        /* x - P */
                amb.place(F1, P0, amb.P_minus_x_value(x));        /* P - x */
                amb.halves(F1, x);                                /* 2v = x */
            }
            pend(j + 1) = true;
            break;
        case Op::IsSidon:
            switch (W) {
            case 1: sidon_forbid<1>(j, x); break;
            case 2: sidon_forbid<2>(j, x); break;
            case 3: sidon_forbid<3>(j, x); break;
            case 4: sidon_forbid<4>(j, x); break;
            case 5: sidon_forbid<5>(j, x); break;
            case 6: sidon_forbid<6>(j, x); break;
            case 7: sidon_forbid<7>(j, x); break;
            case 8: sidon_forbid<8>(j, x); break;
            default:
                std::memcpy(F1, F0, W * 8);
                amb.place(F1, S0, -(int64_t)x);                   /* S - x */
                amb.place(F1, D0, amb.x_plus_codes(x));           /* x + (P - P) */
                amb.place(F1, R0, amb.x_minus_P_value(x) + (int64_t)x); /* 2x - P */
                for (uint64_t w = 0; w < W; ++w) F1[w] |= P0[w];
                set(F1, x);
                for (uint32_t y : vals) amb.halves(F1, amb.add(x, y));
                amb.halves(F1, amb.add(x, x));
            }
            pend(j + 1) = true;
            break;
        case Op::IsApFree:
            std::memcpy(P1, P0, W * 8);
            set(P1, x);
            std::memcpy(F1, F0, W * 8);
            alive = !progressions_through(x, P1, F1);
            break;
        case Op::IsSmallDoubling:
        case Op::SumsetSize:
            std::memcpy(S1, S0, W * 8);
            amb.place(S1, P0, (int64_t)x);
            amb.set_value(S1, amb.add(x, x));
            meas(j + 1) = meas(j) + popcount_andnot(S1, S0, W);
            if (op == Op::IsSmallDoubling)
                alive = (unsigned __int128)meas(j + 1) * bound_den <= (unsigned __int128)bound_num * k;
            break;
        case Op::DifferenceSetSize:
            std::memcpy(R1, R0, W * 8);
            set(R1, amb.rev(x));
            std::memcpy(D1, D0, W * 8);
            amb.place(D1, R0, amb.x_minus_P_code(x));
            amb.place(D1, P0, amb.P_minus_x_code(x));
            set(D1, amb.zero_code());
            meas(j + 1) = meas(j) + popcount_andnot(D1, D0, W);
            break;
        case Op::SchurTripleCount:
            std::memcpy(R1, R0, W * 8);
            set(R1, amb.rev(x));
            meas(j + 1) = meas(j) + schur_delta(P0, R0, P1, x);
            break;
        case Op::MaxDifferenceMultiplicity: {
            uint64_t best = meas(j);
            for (uint32_t y : vals) {
                best = std::max<uint64_t>(best, ++mult[code(x, y)]);
                best = std::max<uint64_t>(best, ++mult[code(y, x)]);
            }
            meas(j + 1) = best;
            break;
        }
        }
        if (mode == Mode::Forbid && !prob.interval) {
            uint64_t *A0 = level_allowed(j), *A1 = level_allowed(j + 1);
            std::memcpy(A1, A0, WA * 8);
            forbid_new(F1, F0, A1);
        }
        vals.push_back((uint32_t)x);
        return alive;
    }

    /* Increasing integer dictionaries: every later element exceeds x, so only the part of the
     * forbidden set above x matters, and most terms of the general update fall below x.
     *   sum-free  v allowed iff v not in P+P:            F |= x + P1
     *   Sidon     v allowed iff v not in P + D+:         D+ |= x - P0 (from the reversal), F |= x + D+
     *   AP-free   v is the last term of a progression whose other terms lie in P1 and whose
     *             largest known term is x:               F |= 2x - P0 for length 3; a loop otherwise
     * F is complete above x, so a candidate never creates a violation and nothing is pending.
     * WT == 0 is the runtime-width fallback through Ambient::place. */
    template <int WT> LK_SCALAR void sorted_forbid(int64_t j, uint64_t x) {
        const uint64_t *P0 = level(j, P), *R0 = level(j, PREV), *D0 = level(j, D), *F0 = level(j, F);
        uint64_t *P1 = level(j + 1, P), *R1 = level(j + 1, PREV), *D1 = level(j + 1, D), *F1 = level(j + 1, F);
        uint64_t L = amb.L;
        if (op == Op::IsSumFree) {
            std::memcpy(P1, P0, W * 8);
            set(P1, x);
            if constexpr (WT == 0) {
                std::memcpy(F1, F0, W * 8);
                amb.place(F1, P1, (int64_t)x);
            } else {
                uint64_t a[WT];
                Words<WT>::load(a, F0);
                Words<WT>::place(a, P1, (int64_t)x, amb);
                Words<WT>::store(F1, a, amb);
            }
            return;
        }
        if (op == Op::IsSidon) { /* F waits for finish_forbidden(): the gap bound only needs D+ */
            std::memcpy(R1, R0, W * 8);
            set(R1, L - x);
            if constexpr (WT == 0) {
                std::memcpy(D1, D0, W * 8);
                amb.place(D1, R0, (int64_t)x - (int64_t)L); /* x - P0 as values */
            } else {
                uint64_t d[WT];
                Words<WT>::load(d, D0);
                Words<WT>::place(d, R0, (int64_t)x - (int64_t)L, amb);
                Words<WT>::store(D1, d, amb);
            }
            (void)F0; (void)F1; (void)P0; (void)P1;
            return;
        }
        /* AP-free */
        std::memcpy(P1, P0, W * 8);
        set(P1, x);
        std::memcpy(R1, R0, W * 8);
        set(R1, L - x);
        if (length == 3) {
            if constexpr (WT == 0) {
                std::memcpy(F1, F0, W * 8);
                amb.place(F1, R0, 2 * (int64_t)x - (int64_t)L); /* 2x - P0 */
            } else {
                uint64_t a[WT];
                Words<WT>::load(a, F0);
                Words<WT>::place(a, R0, 2 * (int64_t)x - (int64_t)L, amb);
                Words<WT>::store(F1, a, amb);
            }
            return;
        }
        std::memcpy(F1, F0, W * 8);
        if (length == 2) { std::fill(F1, F1 + W, ~0ULL); F1[W - 1] &= amb.mask; return; }
        uint64_t step = length - 2;
        for (uint32_t y : vals) {
            uint64_t gap = x - y;
            if (gap % step) continue;
            uint64_t d = gap / step;
            bool all = true;
            for (uint64_t t = 1; t < step && all; ++t) all = bit(P0, x - t * d);
            if (all) amb.set_value(F1, x + d);
        }
    }

    template <int WT> LK_SCALAR void sum_free_forbid(int64_t j, uint64_t x) {
        const uint64_t *P0 = level(j, P), *R0 = level(j, PREV), *F0 = level(j, F);
        uint64_t a[WT];
        Words<WT>::load(a, F0);
        Words<WT>::place(a, P0, (int64_t)x, amb);                    /* x + P */
        Words<WT>::set_value(a, amb.add(x, x), amb);                 /* 2x */
        Words<WT>::place(a, R0, amb.x_minus_P_value(x), amb);        /* x - P */
        Words<WT>::place(a, P0, amb.P_minus_x_value(x), amb);        /* P - x */
        Words<WT>::store(level(j + 1, F), a, amb);
        amb.halves(level(j + 1, F), x);                              /* 2v = x */
    }

    template <int WT> LK_SCALAR void sidon_forbid(int64_t j, uint64_t x) {
        const uint64_t *P0 = level(j, P), *R0 = level(j, PREV), *S0 = level(j, S), *D0 = level(j, D),
                       *F0 = level(j, F);
        uint64_t a[WT];
        Words<WT>::load(a, F0);
        Words<WT>::place(a, S0, -(int64_t)x, amb);                          /* S - x */
        Words<WT>::place(a, D0, amb.x_plus_codes(x), amb);                  /* x + (P - P) */
        Words<WT>::place(a, R0, amb.x_minus_P_value(x) + (int64_t)x, amb);  /* 2x - P */
#pragma GCC novector
        for (int w = 0; w < WT; ++w) a[w] |= P0[w];
        Words<WT>::set(a, x);
        if (!amb.modulus) { /* 2v = x + y: the halves of x + P, and of 2x */
            for (uint32_t y : vals) if (!((x + y) & 1)) Words<WT>::set(a, (x + y) / 2);
            Words<WT>::set(a, x);
        }
        Words<WT>::store(level(j + 1, F), a, amb);
        if (amb.modulus) {
            uint64_t *F1 = level(j + 1, F);
            for (uint32_t y : vals) amb.halves(F1, amb.add(x, y));
            amb.halves(F1, amb.add(x, x));
        }
    }

    template <int WT> LK_SCALAR void finish_t(int64_t j, uint64_t x) {
        const uint64_t *P0 = level(j - 1, P), *R0 = level(j - 1, PREV), *S0 = level(j - 1, S), *D0 = level(j - 1, D);
        uint64_t a[WT];
        Words<WT>::load(a, P0);
        Words<WT>::set(a, x);
        Words<WT>::store(level(j, P), a, amb);
        Words<WT>::load(a, R0);
        Words<WT>::set(a, amb.rev(x));
        Words<WT>::store(level(j, PREV), a, amb);
        if (op == Op::IsSidon) {
            Words<WT>::load(a, S0);
            Words<WT>::place(a, P0, (int64_t)x, amb);
            Words<WT>::set_value(a, amb.add(x, x), amb);
            Words<WT>::store(level(j, S), a, amb);
            Words<WT>::load(a, D0);
            Words<WT>::place(a, R0, amb.x_minus_P_code(x), amb);
            Words<WT>::place(a, P0, amb.P_minus_x_code(x), amb);
            Words<WT>::set(a, amb.zero_code());
            Words<WT>::store(level(j, D), a, amb);
        }
    }

    /* F of level j for sorted Sidon: F_{j-1} | (x + D+_j) | {x}, and the allowed indices. */
    template <int WT> LK_SCALAR void finish_forbidden_t(int64_t j, uint64_t x) {
        const uint64_t *F0 = level(j - 1, F), *D1 = level(j, D);
        uint64_t *F1 = level(j, F);
        if constexpr (WT == 0) {
            std::memcpy(F1, F0, W * 8);
            amb.place(F1, D1, (int64_t)x);
            set(F1, x);
        } else {
            uint64_t a[WT];
            Words<WT>::load(a, F0);
            Words<WT>::place(a, D1, (int64_t)x, amb);
            Words<WT>::set(a, x);
            Words<WT>::store(F1, a, amb);
        }
    }
    void finish_forbidden(int64_t j) {
        if (j + nctx == 0 || !pendf(j)) return;
        pendf(j) = false;
        uint64_t x = vals[j + nctx - 1];
        switch (W) {
        case 1: finish_forbidden_t<1>(j, x); break;
        case 2: finish_forbidden_t<2>(j, x); break;
        case 3: finish_forbidden_t<3>(j, x); break;
        case 4: finish_forbidden_t<4>(j, x); break;
        case 5: finish_forbidden_t<5>(j, x); break;
        case 6: finish_forbidden_t<6>(j, x); break;
        case 7: finish_forbidden_t<7>(j, x); break;
        case 8: finish_forbidden_t<8>(j, x); break;
        default: finish_forbidden_t<0>(j, x); break;
        }
        if (!prob.interval) {
            uint64_t *A0 = level_allowed(j - 1), *A1 = level_allowed(j);
            std::memcpy(A1, A0, WA * 8);
            forbid_new(level(j, F), level(j - 1, F), A1);
        }
    }

    /* The bitsets of level j that push() left out, from level j-1 and the element it added. */
    void finish(int64_t j) {
        if (j + nctx == 0 || !pend(j)) return;
        pend(j) = false;
        uint64_t x = vals[j + nctx - 1];
        switch (W) {
        case 1: finish_t<1>(j, x); return;
        case 2: finish_t<2>(j, x); return;
        case 3: finish_t<3>(j, x); return;
        case 4: finish_t<4>(j, x); return;
        case 5: finish_t<5>(j, x); return;
        case 6: finish_t<6>(j, x); return;
        case 7: finish_t<7>(j, x); return;
        case 8: finish_t<8>(j, x); return;
        default: break;
        }
        uint64_t *P0 = level(j - 1, P), *R0 = level(j - 1, PREV), *S0 = level(j - 1, S), *D0 = level(j - 1, D);
        uint64_t *P1 = level(j, P), *R1 = level(j, PREV), *S1 = level(j, S), *D1 = level(j, D);
        std::memcpy(P1, P0, W * 8);
        set(P1, x);
        std::memcpy(R1, R0, W * 8);
        set(R1, amb.rev(x));
        if (op == Op::IsSidon) {
            std::memcpy(S1, S0, W * 8);
            amb.place(S1, P0, (int64_t)x);
            amb.set_value(S1, amb.add(x, x));
            std::memcpy(D1, D0, W * 8);
            amb.place(D1, R0, amb.x_minus_P_code(x));
            amb.place(D1, P0, amb.P_minus_x_code(x));
            set(D1, amb.zero_code());
        }
    }

    void pop() {
        uint32_t x = vals.back();
        vals.pop_back();
        if (op == Op::MaxDifferenceMultiplicity)
            for (uint32_t y : vals) { --mult[code(x, y)]; --mult[code(y, x)]; }
    }

    uint64_t code(uint64_t x, uint64_t y) const {
        return amb.modulus ? (x + amb.modulus - y) % amb.modulus : x + amb.L - y;
    }

    /* New ordered pairs with a sum in P u {x}: (x,y), (y,x), (x,x), and pairs of P summing to x. */
    uint64_t schur_delta(const uint64_t *P0, const uint64_t *R0, const uint64_t *P1, uint64_t x) {
        std::fill(tmp.begin(), tmp.end(), 0);
        amb.place(tmp.data(), P0, (int64_t)x);              /* x + P */
        uint64_t d = 2 * popcount_and(tmp.data(), P1, W);
        uint64_t twice = amb.add(x, x);
        if (twice < amb.span && bit(P1, twice)) d += 1;
        std::fill(tmp.begin(), tmp.end(), 0);
        amb.place(tmp.data(), R0, amb.x_minus_P_value(x));  /* x - P */
        d += popcount_and(tmp.data(), P0, W);
        return d;
    }

    /* Progressions of `length` terms through x with every other term in P1 = P u {x}: a
     * progression with one distinct missing value forbids that value; one with none means the
     * prefix already contains a progression. Each such progression contains some y of P at
     * offset s from x, 0 < s < length, so its difference solves s*d = x - y; in Z/n it may
     * instead repeat x itself at offset s, so s*d = 0 with d != 0. Two elements always form a
     * progression of length 2. Together these make F complete: an element outside F never
     * creates a progression. */
    bool progressions_through(uint64_t x, const uint64_t *P1, uint64_t *F1) {
        if (length == 2) {
            std::fill(F1, F1 + W, ~0ULL);
            F1[W - 1] &= amb.mask;
            return !vals.empty();
        }
        bool contained = false;
        auto examine = [&](uint64_t d) {
            /* terms x + t*d for t in [-(length-1), length-1] */
            uint64_t n = amb.modulus;
            int64_t Lm = (int64_t)length - 1;
            for (int64_t t0 = -Lm; t0 <= 0; ++t0) {
                int64_t missing = INT64_MIN;
                bool two = false;
                for (int64_t t = t0; t < t0 + (int64_t)length && !two; ++t) {
                    int64_t v;
                    if (n) v = (int64_t)((x + (uint64_t)((t % (int64_t)n + (int64_t)n) % (int64_t)n) * d) % n);
                    else v = (int64_t)x + t * (int64_t)d;
                    bool in = v >= 0 && (uint64_t)v < amb.span && bit(P1, (uint64_t)v);
                    if (in) continue;
                    if (missing == INT64_MIN) missing = v;
                    else if (missing != v) two = true;
                }
                if (two) continue;
                if (missing == INT64_MIN) { contained = true; return; }
                if (missing >= 0 && (uint64_t)missing < amb.span) set(F1, (uint64_t)missing);
            }
        };
        if (amb.modulus)
            for (uint64_t s = 2; s < length && !contained; ++s) {
                uint64_t g = std::gcd(s, amb.modulus), n2 = amb.modulus / g;
                for (uint64_t t = 1; t < g && !contained; ++t) examine(t * n2);
            }
        for (uint32_t y : vals) {
            for (uint64_t s = 1; s < length && !contained; ++s) {
                if (!amb.modulus) {
                    uint64_t gap = x > y ? x - y : y - x;
                    if (gap % s == 0 && gap / s > 0) examine(gap / s);
                } else {
                    uint64_t n = amb.modulus, rhs = (x + n - y) % n, g = std::gcd(s, n);
                    if (rhs % g) continue;
                    uint64_t n2 = n / g, s2 = (s / g) % n2, r2 = (rhs / g) % n2;
                    /* inverse of s2 modulo n2 by extended Euclid */
                    int64_t a = (int64_t)s2, b = (int64_t)n2, u0 = 1, u1 = 0;
                    while (b) { int64_t q = a / b; a -= q * b; std::swap(a, b); u0 -= q * u1; std::swap(u0, u1); }
                    uint64_t inv = n2 == 1 ? 0 : (uint64_t)((u0 % (int64_t)n2 + (int64_t)n2) % (int64_t)n2);
                    uint64_t d0 = n2 == 1 ? 0 : (unsigned __int128)r2 * inv % n2;
                    for (uint64_t t = 0; t < g && !contained; ++t) {
                        uint64_t d = (d0 + t * n2) % n;
                        if (d) examine(d);
                    }
                }
            }
            if (contained) break;
        }
        return contained;
    }

    /* ---- the walk over dictionary indices ---- */

    Index child_first(uint64_t j, uint64_t prev_plus_1, Index base, uint64_t c) const {
        return base + (*cum)[j][c] - (*cum)[j][prev_plus_1];
    }
    Index child_size(uint64_t j, uint64_t c) const { return (*cum)[j][c + 1] - (*cum)[j][c]; }

    /* Candidates at level j: allowed (Forbid) or every index, in [lo, hi]. */
    LK_SCALAR uint64_t candidates_into(uint64_t j, uint64_t lo, uint64_t hi, uint64_t *out) {
        if (hi < lo) { std::fill(out, out + WA, 0); return 0; }
        const uint64_t *src = mode == Mode::Forbid && !prob.interval ? level_allowed(j) : nullptr;
        const uint64_t *Fj = level(j, F);
        bool from_F = mode == Mode::Forbid && prob.interval; /* out = ~F >> dict[0] */
        uint64_t q = dict[0] >> 6, r = dict[0] & 63;
        uint64_t wlo = lo >> 6, whi = hi >> 6, total = 0;
        for (uint64_t w = wlo; w <= whi; ++w) {
            uint64_t v;
            if (from_F) {
                uint64_t a = w + q < W ? Fj[w + q] : 0, b = w + q + 1 < W ? Fj[w + q + 1] : 0;
                v = ~(r ? (a >> r) | (b << (64 - r)) : a);
            } else v = src ? src[w] : ~0ULL;
            if (w == wlo) v &= ~0ULL << (lo & 63);
            if (w == whi && (hi & 63) != 63) v &= (1ULL << ((hi & 63) + 1)) - 1;
            out[w] = v;
            total += __builtin_popcountll(v);
        }
        for (uint64_t w = 0; w < wlo; ++w) out[w] = 0;
        for (uint64_t w = whi + 1; w < WA; ++w) out[w] = 0;
        return total;
    }

    /* |cand ∩ (c, hi] \ (x + D+)| in index space, x = dict[c], for an interval dictionary. */
    LK_SCALAR uint64_t child_candidate_bound(const uint64_t *cand, uint64_t c, uint64_t hi, uint64_t x) {
        const uint64_t *Dp = level((int64_t)vals.size() - nctx, D);
        uint64_t s = x - dict[0], q = s >> 6, r = s & 63;
        uint64_t lo = c + 1, wlo = lo >> 6, whi = hi >> 6, n = 0;
        for (uint64_t w = wlo; w <= whi; ++w) {
            uint64_t sh = 0;
            if (w >= q) {
                sh = Dp[w - q] << r;
                if (r && w > q) sh |= Dp[w - q - 1] >> (64 - r);
            }
            uint64_t v = cand[w] & ~sh;
            if (w == wlo) v &= ~0ULL << (lo & 63);
            if (w == whi && (hi & 63) != 63) v &= (1ULL << ((hi & 63) + 1)) - 1;
            n += __builtin_popcountll(v);
        }
        return n;
    }

    /* The sums of the r and r-1 smallest positive integers absent from the difference set. */
    void unused_difference_sums(const uint64_t *Dp, uint64_t r, uint64_t &sum_all, uint64_t &sum_rest) {
        sum_all = sum_rest = 0;
        uint64_t taken = 0;
        for (uint64_t w = 0; w < W && taken < r; ++w) {
            uint64_t free = ~Dp[w];
            if (w == 0) free &= ~1ULL; /* 0 is not a gap */
            while (free && taken < r) {
                uint64_t d = w * 64 + __builtin_ctzll(free);
                free &= free - 1;
                ++taken;
                sum_all += d;
                if (taken < r) sum_rest += d;
            }
        }
    }

    /* The subtree of a prefix of j >= 1 elements ending at index `prev`, leaves [base, base+size).
     * The size is read off the table rather than passed: a fifth argument would go on the stack
     * as two 8-byte stores read back as one 16-byte load, a store-forwarding stall per node. */
    void walk(uint64_t j, uint64_t prev, Index base) {
        Index size = child_size(j - 1, prev);
        uint64_t need = k - j;
        uint64_t lo = prev + 1;
        int64_t hi = hi_of_level[j]; /* children; `total` counts every candidate above prev */
        if (op == Op::IsSidon && prob.sorted && need >= 2) {
            /* The `need` gaps still to be laid are distinct positive differences not yet in D+, so
             * they sum to at least the `need` smallest unused ones; the next element must leave
             * room for the other need-1 of them below the largest value. */
            uint64_t sum_all, sum_rest;
            unused_difference_sums(level(j, D), need, sum_all, sum_rest);
            if (vals.back() + sum_all > prob.maxval()) { acc.booleans(base, size, false); return; }
            hi = std::min(hi, prob.index_at_most((int64_t)prob.maxval() - (int64_t)sum_rest));
        }
        if ((int64_t)lo > hi) { acc.booleans(base, size, false); return; }
        finish_forbidden(j);
        uint64_t *cand = cands.data() + (j + nctx) * WA;
        uint64_t total = candidates_into(j, lo, m - 1, cand);
        if (total < need) { acc.booleans(base, size, false); return; }
        Index decided = 0;
        if (need == 1) {
            if (mode != Mode::Forbid) leaves_valued(j, lo, m, base);
            else if (prob.mirror) leaves_mirror(lo, base, size, cand);
            else leaves_forbid(lo, base, size, cand, total);
            return;
        }
        finish(j);
        /* Sidon over an interval: before pushing x, bound the child's candidates from above by
         * the candidates above x outside x + D+ (the child also removes 2x - P). Most children
         * die of too few candidates, and this costs a shifted AND rather than a push. */
        bool prefilter = op == Op::IsSidon && prob.interval && need >= 3;
        uint64_t seen = 0;
        for (uint64_t w = 0; w < WA; ++w) {
            uint64_t bits = cand[w];
            while (bits) {
                uint64_t c = w * 64 + __builtin_ctzll(bits);
                bits &= bits - 1;
                ++seen;
                if ((int64_t)c > hi || total - seen < need - 1) { w = WA; break; } /* too few candidates after c */
                Index first = child_first(j, lo, base, c), below = child_size(j, c);
                if (acc.exhausted(first)) { w = WA; break; }
                if (j == 1 && prob.mirror && !mirror_bounds(vals[nctx], dict[c])) {
                    acc.booleans(first, below, false);
                    decided += below;
                    continue;
                }
                if (prefilter && child_candidate_bound(cand, c, m - 1, dict[c]) < need - 1) {
                    acc.booleans(first, below, false);
                    decided += below;
                    continue;
                }
                bool alive = push(j, dict[c]);
                if (alive) walk(j + 1, c, first);
                else acc.booleans(first, below, false);
                pop();
                decided += below;
            }
        }
        if (decided < size) acc.booleans(base, size - decided, false);
    }

    /* Leaves under the mirror rule: a leaf value v closes the set with last gap v - a, where a is
     * the previous element. Sets with last gap above the first gap stand for themselves and
     * their reflections; equal gaps stand for themselves; smaller gaps are reflections of sets
     * counted elsewhere. */
    void leaves_mirror(uint64_t lo, Index base, Index size, const uint64_t *cand) {
        uint64_t a = vals.back();
        uint64_t threshold = a + g1;
        uint64_t start = threshold <= prob.maxval() ? prob.ge[threshold] : m;
        uint64_t twice = 0, once = 0;
        for (uint64_t w = start / 64; w < WA && start < m; ++w) {
            uint64_t bits = cand[w];
            if (w == start / 64) bits &= ~0ULL << (start & 63);
            if (!bits) continue;
            if (reduction == Reduction::First) {
                uint64_t c = w * 64 + __builtin_ctzll(bits);
                Index index = base + (c - lo);
                acc.booleans(base, index - base, false);
                acc.booleans(index, 1, true);
                return;
            }
            twice += __builtin_popcountll(bits);
        }
        if (reduction == Reduction::First) { acc.booleans(base, size, false); return; }
        if (start < m && dict[start] == threshold && bit(cand, start)) { once = 1; twice -= 1; }
        acc.count += 2 * twice + once;
        acc.visited += size;
    }

    void leaves_forbid(uint64_t lo, Index base, Index size, const uint64_t *cand, uint64_t total) {
        switch (reduction) {
        case Reduction::Count:
            acc.count += total;
            acc.visited += size;
            return;
        case Reduction::First: {
            for (uint64_t w = 0; w < WA; ++w)
                if (cand[w]) {
                    uint64_t c = w * 64 + __builtin_ctzll(cand[w]);
                    Index index = base + (c - lo);
                    acc.booleans(base, index - base, false);
                    acc.booleans(index, 1, true);
                    return;
                }
            acc.booleans(base, size, false);
            return;
        }
        default: {
            Index decided = 0;
            for (uint64_t w = 0; w < WA; ++w) {
                uint64_t bits = cand[w];
                while (bits) {
                    uint64_t c = w * 64 + __builtin_ctzll(bits);
                    bits &= bits - 1;
                    acc.booleans(base + (c - lo), 1, true);
                    ++decided;
                }
            }
            acc.booleans(base, size - decided, false);
        }
        }
    }

    /* Doubling and integer operations at the last level: every index in [lo, end) is a leaf. */
    void leaves_valued(uint64_t j, uint64_t lo, uint64_t end, Index base) {
        uint64_t *P0 = level(j, P), *R0 = level(j, PREV), *S0 = level(j, S), *D0 = level(j, D);
        for (uint64_t c = lo; c < end; ++c) {
            uint64_t x = dict[c];
            Index index = base + (c - lo);
            if (acc.exhausted(index)) return;
            switch (op) {
            case Op::IsSmallDoubling:
            case Op::SumsetSize: {
                std::fill(tmp.begin(), tmp.end(), 0);
                amb.place(tmp.data(), P0, (int64_t)x);
                amb.set_value(tmp.data(), amb.add(x, x));
                uint64_t v = meas(j) + popcount_andnot(tmp.data(), S0, W);
                if (op == Op::SumsetSize) acc.integer(index, v);
                else acc.boolean(index, (unsigned __int128)v * bound_den <= (unsigned __int128)bound_num * k);
                break;
            }
            case Op::DifferenceSetSize: {
                std::fill(tmp.begin(), tmp.end(), 0);
                amb.place(tmp.data(), R0, amb.x_minus_P_code(x));
                amb.place(tmp.data(), P0, amb.P_minus_x_code(x));
                set(tmp.data(), amb.zero_code());
                acc.integer(index, meas(j) + popcount_andnot(tmp.data(), D0, W));
                break;
            }
            case Op::SchurTripleCount: {
                std::memcpy(tmp2.data(), P0, W * 8);
                set(tmp2.data(), x);
                acc.integer(index, meas(j) + schur_delta(P0, R0, tmp2.data(), x));
                break;
            }
            case Op::MaxDifferenceMultiplicity: {
                uint64_t best = meas(j);
                for (uint32_t y : vals) {
                    best = std::max<uint64_t>(best, ++mult[code(x, y)]);
                    best = std::max<uint64_t>(best, ++mult[code(y, x)]);
                }
                for (uint32_t y : vals) { --mult[code(x, y)]; --mult[code(y, x)]; }
                acc.integer(index, best);
                break;
            }
            default: break;
            }
        }
    }

    /* One unit of work: the subtree under the index prefix idx[0..d), d <= max(1, k-2) or d == k
     * == 2, with leaves [first, first+size). Units arrive in canonical order, so the levels
     * shared with the previous unit's prefix are kept rather than rebuilt. */
    std::vector<uint32_t> cur;  /* the index prefix currently pushed */
    uint64_t dead_level = UINT64_MAX; /* first level of `cur` that is dead, if any */

    void unit(const uint32_t *idx, uint64_t d, Index first, Index size) {
        if (acc.exhausted(first)) return;
        if (!context_alive) { acc.booleans(first, size, false); return; }
        uint64_t common = 0;
        while (common < d && common < cur.size() && cur[common] == idx[common]) ++common;
        while (cur.size() > common) { cur.pop_back(); pop(); }
        if (dead_level >= common) dead_level = UINT64_MAX;
        for (uint64_t j = common; j < d && dead_level == UINT64_MAX; ++j) {
            uint64_t c = idx[j];
            bool ok_here = (int64_t)c <= prob.level_hi[j] && (mode != Mode::Forbid || allowed_index(j, c));
            if (ok_here && j == 1 && prob.mirror) ok_here = mirror_bounds(dict[idx[0]], dict[c]);
            if (!ok_here) { dead_level = j; break; }
            if (!push(j, dict[c])) dead_level = j;
            else { finish(j + 1); finish_forbidden(j + 1); }
            cur.push_back((uint32_t)c);
        }
        if (dead_level != UINT64_MAX) { acc.booleans(first, size, false); return; }
        if (d == k) { /* k <= 2: the unit is a leaf */
            if (mode == Mode::Integer) acc.integer(first, meas(k));
            else acc.booleans(first, 1, true);
        } else if (k == 1) {
            if (mode == Mode::Forbid) acc.booleans(first, 1, allowed_index(0, idx[0]));
            else leaves_valued(0, idx[0], idx[0] + 1, first);
        } else walk(d, idx[d - 1], first);
    }

    /* Under the mirror rule the last gap is at least g1 = a1 - a0, so the elements chosen at
     * level j with r >= 1 still to come after them span at least max(G(r+1), G(r) + g1) up to
     * the largest value. Returns false when a1 itself is already too high. */
    bool mirror_bounds(uint64_t a0, uint64_t a1) {
        g1 = a1 - a0;
        const auto &G = prob.spans;
        int64_t top = (int64_t)prob.maxval();
        for (uint64_t j = 2; j + 1 < k; ++j) {
            uint64_t r = k - j - 1;
            uint64_t span = std::max(G[r + 1], G[r] + g1);
            hi_of_level[j] = std::min(prob.level_hi[j], prob.index_at_most(top - (int64_t)span));
        }
        uint64_t r = k - 2; /* after a1 */
        return (int64_t)a1 + (int64_t)std::max(G[r + 1], G[r] + g1) <= top;
    }

    /* An explicit member: its rows in order. */
    void member(uint64_t index, const Entry *rows, uint64_t count) {
        while ((int64_t)vals.size() > nctx) pop();
        cur.clear();
        dead_level = UINT64_MAX;
        bool alive = context_alive;
        uint64_t j = 0;
        for (; j < count && alive; ++j) {
            uint64_t x = rows[j];
            if (mode == Mode::Forbid && bit(level(j, F), x)) { alive = false; break; }
            alive = push(j, x);
            if (alive) finish(j + 1);
        }
        if (mode == Mode::Integer) acc.integer(index, meas(j));
        else acc.booleans(index, 1, alive);
        while ((int64_t)vals.size() > nctx) pop();
    }
};

/* ---- validation and the driver ----------------------------------------------------------- */

const char *set_name(const Family &fam) {
    return fam.kind == Family::Kind::Explicit ? "member" : "dictionary";
}

Status scan(const Family &fam, uint64_t modulus, uint64_t &largest) {
    const Matrix &data = *fam.data;
    if (data.cols != 1)
        return fail(INVALID, std::string(set_name(fam)) + " must have one column: one element per row");
    for (uint64_t i = 0; i < data.count * data.rows; ++i) {
        uint64_t v = data.entries[i];
        if (modulus && v >= modulus)
            return fail(INVALID, std::string(set_name(fam)) + " has an element that is not below the modulus");
        largest = std::max(largest, v);
    }
    return ok();
}

Status check_distinct(const Family &fam, uint64_t span) {
    const Matrix &data = *fam.data;
    uint64_t sets = fam.kind == Family::Kind::Explicit ? data.count : 1;
    uint64_t per_set = fam.kind == Family::Kind::Explicit ? data.rows : data.count;
    std::vector<uint8_t> seen(span, 0);
    for (uint64_t s = 0; s < sets; ++s) {
        const Entry *first = data.entries.data() + s * per_set;
        bool repeated = false;
        for (uint64_t i = 0; i < per_set && !repeated; ++i) repeated = seen[first[i]]++;
        for (uint64_t i = 0; i < per_set; ++i) seen[first[i]] = 0;
        if (repeated) return fail(INVALID, std::string(set_name(fam)) + " contains a duplicate element");
    }
    return ok();
}

/* Saturates at INDEX_MAX: public families fit by construction, and the span table's internal
 * searches only need indices for `first` ordering, where saturation is harmless. */
Index binom_fits(uint64_t n, uint64_t r) {
    if (r > n) return 0;
    if (r > n - r) r = n - r;
    Index v = 1;
    for (uint64_t i = 1; i <= r; ++i) {
        Index f = n - r + i;
        if (v > INDEX_MAX / f) return INDEX_MAX;
        v = v * f / i;
    }
    return v;
}

Ambient make_ambient(uint64_t modulus, uint64_t largest) {
    Ambient amb;
    amb.modulus = modulus;
    amb.L = modulus ? 0 : largest;
    amb.span = modulus ? modulus : 2 * largest + 1;
    amb.W = (amb.span + 63) / 64;
    amb.mask = (amb.span & 63) ? (1ULL << (amb.span & 63)) - 1 : ~0ULL;
    return amb;
}

/* cum[j][c] for c in [j, m]: leaves under indices j..c-1 at depth j; cum[j][m] = C(m-j, k-j). */
void fill_cum(Problem &P) {
    P.cum.assign(P.k, {});
    for (uint64_t j = 0; j < P.k; ++j) {
        P.cum[j].assign(P.m + 1, 0);
        for (uint64_t c = j; c < P.m; ++c) P.cum[j][c + 1] = P.cum[j][c] + binom_fits(P.m - 1 - c, P.k - 1 - j);
    }
}

uint32_t thread_budget(const Problem &P, uint32_t requested) {
    uint64_t per_thread = Walker::bytes_per_thread(P.k, P.context.size(), P.amb.W, P.m);
    return std::max<uint32_t>(1, std::min<uint64_t>(requested, MEMORY_BUDGET / per_thread));
}

/* Walk every k-subset of the dictionary, in canonical order, over `threads` threads. */
std::vector<Accumulator> search(const Problem &P, Reduction reduction, uint32_t threads, Shared &shared) {
    uint64_t k = P.k, m = P.m;
    if (k == 0) {
        Accumulator acc(reduction, &shared);
        if (P.mode == Mode::Integer) acc.integer(0, 0);
        else acc.booleans(0, 1, true);
        return {acc};
    }
    /* Units are index prefixes of depth d in canonical order. Deeper prefixes balance the
     * unequal subtrees of a pruned search across threads; the depth is the largest whose
     * prefixes, filtered by the span bounds, fit the cap. When k == 2 a unit is a leaf. */
    uint64_t d = k <= 2 ? k : std::min<uint64_t>(k - 2, 6);
    uint64_t cap = std::max<uint64_t>(1ULL << 16, 4096ULL * threads);
    std::vector<uint32_t> prefixes; /* d indices per unit */
    std::vector<Index> starts;      /* first leaf index per unit */
    for (;; --d) {
        prefixes.clear();
        starts.clear();
        std::vector<uint32_t> idx(d);
        uint64_t units = 0;
        bool over = false;
        /* DFS over prefixes with c_j > c_{j-1}, c_j <= level_hi[j] */
        auto rec = [&](auto &self, uint64_t j, uint64_t prev, Index base) -> void {
            if (over) return;
            if (j == d) {
                if (++units > cap) { over = true; return; }
                prefixes.insert(prefixes.end(), idx.begin(), idx.end());
                starts.push_back(base);
                return;
            }
            uint64_t lo = j == 0 ? 0 : prev + 1;
            for (uint64_t c = lo; (int64_t)c <= P.level_hi[j] && !over; ++c) {
                idx[j] = (uint32_t)c;
                self(self, j + 1, c, base + P.cum[j][c] - P.cum[j][lo]);
            }
        };
        rec(rec, 0, 0, 0);
        if (!over) break;
        if (d <= 2) { prefixes.clear(); starts.clear(); break; }
    }
    bool materialised = !prefixes.empty() || (int64_t)0 > P.level_hi[0];
    uint64_t units = starts.size();
    /* Over the cap at depth 2: (c1, c2) units computed on the fly. */
    uint64_t c1_max = m - k;
    std::vector<uint64_t> row_start;
    if (!materialised) {
        if (k == 1) units = m;
        else {
            row_start.assign(c1_max + 2, 0);
            for (uint64_t c1 = 0; c1 <= c1_max; ++c1) row_start[c1 + 1] = row_start[c1] + (m - k + 1 - c1);
            units = row_start[c1_max + 1];
        }
        d = std::min<uint64_t>(k, 2);
    }
    threads = std::max<uint32_t>(1, std::min<uint64_t>(threads, units ? units : 1));
    std::vector<Walker> walkers;
    walkers.reserve(threads);
    for (uint32_t t = 0; t < threads; ++t) walkers.emplace_back(P, reduction, &shared);
    if (materialised) { /* prefixes outside the span bounds were never listed: they are all false */
        Index listed = 0;
        for (uint64_t u = 0; u < units; ++u) {
            uint32_t last = prefixes[u * d + d - 1];
            listed += P.cum[d - 1][last + 1] - P.cum[d - 1][last];
        }
        if (listed < P.cum[0][m]) walkers[0].acc.booleans(0, P.cum[0][m] - listed, false);
    }
    parallel_ranges(units, threads, [&](uint32_t t, uint64_t begin, uint64_t end) {
        Walker &w = walkers[t];
        uint32_t two[2];
        for (uint64_t u = begin; u < end; ++u) {
            if (materialised) {
                const uint32_t *idx = prefixes.data() + u * d;
                Index size = P.cum[d - 1][idx[d - 1] + 1] - P.cum[d - 1][idx[d - 1]];
                w.unit(idx, d, starts[u], size);
                continue;
            }
            if (k == 1) { two[0] = (uint32_t)u; w.unit(two, 1, u, 1); continue; }
            uint64_t c1 = (uint64_t)(std::upper_bound(row_start.begin(), row_start.end(), u) - row_start.begin()) - 1;
            uint64_t c2 = c1 + 1 + (u - row_start[c1]);
            Index first = P.cum[0][c1] + P.cum[1][c2] - P.cum[1][c1 + 1];
            Index below = P.cum[1][c2 + 1] - P.cum[1][c2];
            two[0] = (uint32_t)c1; two[1] = (uint32_t)c2;
            w.unit(two, 2, first, below);
        }
        return ok();
    });
    std::vector<Accumulator> accs;
    for (auto &w : walkers) accs.push_back(w.acc);
    return accs;
}

/* ---- the span table ---------------------------------------------------------------------- */

std::mutex span_mutex;
std::map<std::pair<int, uint64_t>, std::vector<uint64_t>> span_cache; /* (op, length) -> G(0..t) */

/* Is there a t-element set with the property inside [0, L]? Uses G(1..t-1) and the mirror rule. */
bool exists_in(Op op, uint64_t length, uint64_t t, uint64_t L, const std::vector<uint64_t> &G, uint32_t threads) {
    std::vector<Entry> dict(L + 1);
    std::iota(dict.begin(), dict.end(), 0);
    Problem P;
    P.op = op; P.length = length; P.k = t; P.m = L + 1; P.dict = dict.data();
    P.amb = make_ambient(0, L);
    P.spans = G;
    /* The ladder asks about L only after every shorter span failed, so a t-element set in
     * [0, L] has span exactly L: the bound pins its first element to 0 and its last to L. */
    P.spans.resize(t + 1, L);
    P.mirror = t >= 3;
    finish_problem(P);
    fill_cum(P);
    Shared shared;
    search(P, Reduction::First, thread_budget(P, threads), shared);
    return shared.best.load() != INDEX_MAX;
}

/* Least spans settled in the literature, used as the head of the span table so that a search
 * at the frontier does not first redo every rung below it. Each entry is a theorem someone
 * else proved by exhaustive computation; the pruning built on it is exactly as sound as the
 * entry, so the sources are named and the module's invariants recompute the head of each
 * table with this backend's own search.
 *
 * Sidon: OEIS A003022, the lengths of the optimal Golomb rulers with 2..28 marks; 26-28 are
 * distributed.net's OGR-26 (2009), OGR-27 (2014) and OGR-28 (2022).
 * 3-AP-free: derived from OEIS A003002 (r_3(n), the largest 3-AP-free subset of [1, n]),
 * whose b-file (Cariboni, 2024) reaches n = 211 with r_3(211) = 43; G(t) is the least n with
 * r_3(n) >= t, less one, so the table gives G(1..43) and says G(44) >= 211. Gasarch, Glenn and
 * Kruskal (arXiv:2501.01634) confirm r_3(n) for n <= 186 independently. */
struct KnownSpans {
    std::vector<uint64_t> exact; /* G(0..T) */
    uint64_t next_lower;         /* G(T+1) >= next_lower */
};
/* LK_SUM_FREE_SPANS=search makes the ladder ignore the tables, so the module's invariants can
 * recompute their heads with this backend alone. */
bool use_known_spans() {
    static const bool use = [] { const char *e = getenv("LK_SUM_FREE_SPANS"); return !(e && std::string(e) == "search"); }();
    return use;
}
KnownSpans known_spans(Op op, uint64_t length) {
    if (!use_known_spans()) return {{0, 0}, 1};
    if (op == Op::IsSidon)
        return {{0, 0, 1, 3, 6, 11, 17, 25, 34, 44, 55, 72, 85, 106, 127, 151, 177, 199, 216, 246, 283, 333,
                 356, 372, 425, 480, 492, 553, 585}, 586};
    if (op == Op::IsApFree && length == 3)
        return {{0, 0, 1, 3, 4, 8, 10, 12, 13, 19, 23, 25, 29, 31, 35, 39, 40, 50, 53, 57, 62, 70, 73, 81, 83,
                 91, 94, 99, 103, 110, 113, 120, 121, 136, 144, 149, 156, 162, 164, 168, 173, 193, 203, 208}, 211};
    return {{0, 0}, 1};
}

/* The best lower bound on G(t) from G(0..t-1) and the known tables. */
uint64_t span_lower(Op op, uint64_t length, uint64_t t, const std::vector<uint64_t> &G) {
    uint64_t L = G[t - 1] + 1;
    if (op == Op::IsSidon) L = std::max(L, t * (t - 1) / 2); /* t-1 distinct positive gaps */
    KnownSpans known = known_spans(op, length);
    if (t < known.exact.size()) L = std::max(L, known.exact[t]);
    else if (t == known.exact.size()) L = std::max(L, known.next_lower);
    return L;
}

/* G(0..tmax), exact. Beyond the known table, G(t) is found by asking for a t-element set in
 * [0, L] for increasing L starting from the best lower bound; the failed searches are
 * exhaustive but heavily pruned by the entries below. */
std::vector<uint64_t> span_table(Op op, uint64_t length, uint64_t tmax, uint32_t threads) {
    std::lock_guard<std::mutex> lock(span_mutex);
    auto &G = span_cache[{(int)op, length}];
    if (G.empty()) G = known_spans(op, length).exact;
    while (G.size() <= tmax) {
        uint64_t t = G.size();
        uint64_t L = span_lower(op, length, t, G);
        while (!exists_in(op, length, t, L, G, threads)) ++L;
        G.push_back(L);
    }
    return std::vector<uint64_t>(G.begin(), G.begin() + tmax + 1);
}

uint64_t modulus_of(const Request &req) { return req.int_args.at("modulus"); }

R run(const Request &req) {
    const Family &fam = *req.family;
    if (fam.kind != Family::Kind::Explicit && fam.kind != Family::Kind::Subsets &&
        fam.kind != Family::Kind::SubsetsOf)
        return R::failure(INVALID, "sum_free_and_additive is defined on explicit, subsets and subsets_of families only");
    if (fam.prime() != NATURALS)
        return R::failure(INVALID, "sum_free_and_additive members must be lk.naturals");
    bool with_context = false;
    auto parsed = parse_op(req.op, with_context);
    if (!parsed.ok) return R::failure(parsed.error.status, parsed.error.message);
    Op op = parsed.value;
    std::vector<uint64_t> context;
    if (with_context) {
        auto it = req.handle_args.find("context");
        if (it == req.handle_args.end() || !it->second->matrix)
            return R::failure(INVALID, "`context` must be an lk.naturals vector");
        const Matrix &cm = *it->second->matrix;
        if (cm.p != NATURALS || cm.count != 1 || cm.rows != 1)
            return R::failure(INVALID, "`context` must be one row of natural numbers");
        context.assign(cm.entries.begin(), cm.entries.end());
        std::sort(context.begin(), context.end());
        for (size_t i = 0; i < context.size(); ++i) {
            if (modulus_of(req) && context[i] >= modulus_of(req))
                return R::failure(INVALID, "`context` has an element that is not below the modulus");
            if (i && context[i] == context[i - 1]) return R::failure(INVALID, "`context` contains a duplicate element");
        }
    }

    uint64_t modulus = modulus_of(req);
    uint64_t length = req.int_args.count("length") ? req.int_args.at("length") : 0;
    uint64_t bound_num = req.int_args.count("bound_num") ? req.int_args.at("bound_num") : 0;
    uint64_t bound_den = req.int_args.count("bound_den") ? req.int_args.at("bound_den") : 1;
    if (op == Op::IsApFree && length < 2)
        return R::failure(INVALID, "is_ap_free needs a progression length of at least 2");
    if (op == Op::IsSmallDoubling && bound_den < 1)
        return R::failure(INVALID, "is_small_doubling needs bound_den >= 1");

    uint64_t largest = context.empty() ? 0 : context.back();
    auto scanned = scan(fam, modulus, largest);
    if (!scanned.ok) return R::failure(scanned.error.status, scanned.error.message);
    Problem P;
    P.op = op; P.length = length; P.bound_num = bound_num; P.bound_den = bound_den;
    P.context = context;
    P.amb = make_ambient(modulus, largest);
    if (P.amb.span > MAX_SPAN)
        return R::failure(INVALID, "the generic backend needs a modulus below 2^22, or elements below 2^21 when the modulus is 0");
    auto distinct = check_distinct(fam, P.amb.span);
    if (!distinct.ok) return R::failure(distinct.error.status, distinct.error.message);
    if (!context.empty()) {
        std::vector<uint8_t> in_context(P.amb.span, 0);
        for (uint64_t v : context) in_context[v] = 1;
        const Matrix &data = *fam.data;
        for (uint64_t i = 0; i < data.count * data.rows; ++i)
            if (in_context[data.entries[i]])
                return R::failure(INVALID, std::string("the ") + set_name(fam) + " shares an element with `context`");
    }

    auto size_r = fam.size();
    if (!size_r.ok) return R::failure(size_r.error.status, size_r.error.message);
    Index size = size_r.value;
    Reduction reduction = parse_reduction(req.reduction);
    Shared shared;
    auto prepared = prepare_all(reduction, size, shared);
    if (!prepared.ok) return R::failure(prepared.error.status, prepared.error.message);

    bool is_explicit = fam.kind == Family::Kind::Explicit;
    P.k = is_explicit ? fam.data->rows : fam.k;
    P.m = is_explicit ? 0 : fam.data->count;
    P.dict = fam.data->entries.data();
    finish_problem(P);
    if (Walker::bytes_per_thread(P.k, P.context.size(), P.amb.W, P.m) > MEMORY_BUDGET)
        return R::failure(INVALID, "the generic backend needs (subset size) x (ambient group size) below 2^33 bits");
    uint32_t threads = thread_budget(P, req.threads);

    if (is_explicit) {
        uint64_t count = fam.data->count;
        threads = std::max<uint32_t>(1, std::min<uint64_t>(threads, count));
        std::vector<Walker> walkers;
        walkers.reserve(threads);
        for (uint32_t t = 0; t < threads; ++t) walkers.emplace_back(P, reduction, &shared);
        parallel_ranges(count, threads, [&](uint32_t t, uint64_t begin, uint64_t end) {
            for (uint64_t i = begin; i < end; ++i) {
                if (walkers[t].acc.exhausted(i)) break;
                walkers[t].member(i, fam.data->at(i), P.k);
            }
            return ok();
        });
        std::vector<Accumulator> accs;
        for (auto &w : walkers) accs.push_back(w.acc);
        return assemble(req, reduction, accs, shared);
    }

    if (P.sorted && spannable(op, length) && P.k >= 2) {
        P.spans = span_table(op, length, P.k - 1, threads);
        P.spans.push_back(span_lower(op, length, P.k, P.spans));
        bool symmetric = P.k >= 3 && reduction == Reduction::Count;
        for (uint64_t i = 0; i < P.m && symmetric; ++i)
            symmetric = P.dict[i] + P.dict[P.m - 1 - i] == P.dict[0] + P.dict[P.m - 1];
        P.mirror = symmetric && context.empty();
        finish_problem(P);
    }
    fill_cum(P);
    auto accs = search(P, reduction, threads, shared);
    return assemble(req, reduction, accs, shared);
}

BackendRegistration registration{Backend{
    "sum_free_and_additive", "generic",
    [] { return true; },
    [](const Request &req) {
        return req.family->kind == Family::Kind::Explicit || req.family->kind == Family::Kind::Subsets ||
               req.family->kind == Family::Kind::SubsetsOf;
    },
    run,
    0,
    /* big_families: indices are 128-bit throughout; a pruned search decides C(150, 15)
       members without visiting them. */
    true}};

} // namespace
} // namespace lk::sum_free_and_additive
