/* simplicial_complexes generic backend: portable C++.
 *
 * A member is a 0/1 matrix over F_2, so a row is a set of vertices and fits in a 64-bit mask.
 * The complex of a member is built once per request and every operation reads it:
 *
 *   nonfaces = 0  the rows generate: walk the submasks of each row (s = (s - 1) & g), then sort
 *                 and deduplicate;
 *   nonfaces = 1  the rows are forbidden: grow sets vertex by vertex in increasing order and
 *                 never leave the complex, so the cost is proportional to the answer.
 *
 * Faces are then sorted by (number of vertices, mask), which puts each dimension in a
 * contiguous block whose masks increase, so a face lookup is a binary search inside a block and
 * a boundary matrix is one pass over a block. Betti numbers eliminate those boundary matrices:
 * over F_2 as bitsets (64 columns per word), over any other prime densely with gfp::Field.
 * Shellability searches subsets of the facets rather than their orders, because whether a facet
 * may come next depends only on the set already placed; the search memoises on that subset.
 *
 * Costs, so the next backend knows what to beat: one complex per member (O(faces * rows)), and
 * for betti two eliminations of dense f_d x f_{d-1} matrices. Nothing is shared between members
 * that share a prefix, which is the obvious next move for `subsets` families.
 */
#include "../../../../runtime/src/reduce.hpp"
#include "../../../gfp/backends/generic/field.hpp"

#include <bit>

namespace lk::simplicial_complexes {
namespace {

using R = Result<std::shared_ptr<Object>>;
constexpr int INVALID = 1;

constexpr uint64_t MAX_VERTICES = 64;
constexpr uint64_t MAX_FACES = 1ULL << 22;
constexpr uint64_t MAX_BOUNDARY_ENTRIES = 1ULL << 24;
constexpr uint64_t MAX_FACETS = 20;

enum class Op { FCount, Faces, Euler, Betti, Shellable };

/* One thread's complex and its scratch space. */
struct Worker {
    const Family &fam;
    uint64_t n;         /* vertices = columns of a member */
    int nonfaces;
    Matrix member;
    std::vector<uint64_t> gens;      /* row masks */
    std::vector<uint64_t> faces;     /* nonempty faces, by (popcount, mask) */
    std::vector<uint64_t> block;     /* block[k] = first face with k vertices; size n + 2 */
    std::vector<uint64_t> facets;
    std::vector<uint64_t> follow;    /* shelling: follow[i * t + j] = facets that let j follow i */
    std::vector<uint8_t> seen;       /* shelling: subsets of the facets already refuted */
    std::vector<Entry> dense;        /* boundary matrix over F_p */
    std::vector<uint64_t> bits;      /* boundary matrix over F_2 */

    Worker(const Family &f, int nf) : fam(f), n(f.cols()), nonfaces(nf) {}

    uint64_t dim_count(uint64_t k) const { /* faces with k vertices */
        return k + 1 < block.size() ? block[k + 1] - block[k] : 0;
    }

    bool is_face(uint64_t m) const {
        uint64_t k = (uint64_t)std::popcount(m);
        if (k + 1 >= block.size()) return false;
        return std::binary_search(faces.begin() + block[k], faces.begin() + block[k + 1], m);
    }

    /* Grow `set` by vertices from `v` upwards, keeping every set that contains no forbidden row. */
    bool grow(uint64_t set, uint64_t v) {
        for (; v < n; ++v) {
            uint64_t next = set | (1ULL << v);
            bool allowed = true;
            for (uint64_t g : gens)
                if ((g & ~next) == 0) { allowed = false; break; }
            if (!allowed) continue;
            faces.push_back(next);
            if (faces.size() > MAX_FACES) return false;
            if (!grow(next, v + 1)) return false;
        }
        return true;
    }

    Status build(uint64_t index) {
        auto st = fam.member_into(index, member);
        if (!st.ok) return st;
        gens.clear();
        for (uint64_t r = 0; r < member.rows; ++r) {
            uint64_t mask = 0;
            for (uint64_t c = 0; c < member.cols; ++c)
                if (member.entries[r * member.cols + c]) mask |= 1ULL << c;
            gens.push_back(mask);
        }
        faces.clear();
        if (nonfaces == 0) {
            unsigned __int128 total = 0;
            for (uint64_t g : gens) total += ((unsigned __int128)1 << std::popcount(g)) - 1;
            if (total > MAX_FACES)
                return fail(INVALID, "member " + std::to_string(index) + " generates more than " +
                                         std::to_string(MAX_FACES) + " faces");
            for (uint64_t g : gens)
                for (uint64_t s = g; s; s = (s - 1) & g) faces.push_back(s);
            std::sort(faces.begin(), faces.end());
            faces.erase(std::unique(faces.begin(), faces.end()), faces.end());
        } else {
            bool empty = false;
            for (uint64_t g : gens)
                if (g == 0) empty = true; /* the empty set is forbidden: no set at all survives */
            if (!empty && !grow(0, 0))
                return fail(INVALID, "member " + std::to_string(index) + " has more than " +
                                         std::to_string(MAX_FACES) + " faces");
        }
        std::sort(faces.begin(), faces.end(), [](uint64_t a, uint64_t b) {
            int pa = std::popcount(a), pb = std::popcount(b);
            return pa != pb ? pa < pb : a < b;
        });
        block.assign(n + 2, faces.size());
        block[0] = 0;
        for (uint64_t i = faces.size(); i-- > 0;) block[std::popcount(faces[i])] = i;
        for (uint64_t k = block.size() - 1; k-- > 0;)
            if (block[k] > block[k + 1]) block[k] = block[k + 1];
        return ok();
    }

    /* sum_i (-1)^i f_i modulo p, in [0, p). */
    uint64_t euler(uint64_t p) const {
        uint64_t even = 0, odd = 0;
        for (uint64_t k = 1; k <= n; ++k) (k % 2 == 1 ? even : odd) += dim_count(k);
        return (even % p + (p - odd % p)) % p;
    }

    /* Rank of the matrix of boundary_d: rows are the faces with d + 1 vertices, columns the
     * faces with d, and the entry is (-1)^j when the row's j-th vertex is dropped. */
    Result<uint64_t> boundary_rank(uint64_t d, const gfp::Field &field) {
        if (d == 0) return Result<uint64_t>::success(0);
        uint64_t rows = dim_count(d + 1), cols = dim_count(d);
        if (rows == 0 || cols == 0) return Result<uint64_t>::success(0);
        if ((unsigned __int128)rows * cols > MAX_BOUNDARY_ENTRIES)
            return Result<uint64_t>::failure(INVALID, "boundary matrix of " + std::to_string(rows) + " x " +
                                                          std::to_string(cols) + " entries is too large");
        const uint64_t *lower = faces.data() + block[d];
        const uint64_t *upper = faces.data() + block[d + 1];
        if (field.p == 2) {
            uint64_t words = (cols + 63) / 64;
            bits.assign(rows * words, 0);
            for (uint64_t i = 0; i < rows; ++i) {
                uint64_t F = upper[i], rest = F;
                while (rest) {
                    uint64_t v = (uint64_t)std::countr_zero(rest);
                    rest &= rest - 1;
                    uint64_t j = (uint64_t)(std::lower_bound(lower, lower + cols, F ^ (1ULL << v)) - lower);
                    bits[i * words + j / 64] |= 1ULL << (j % 64);
                }
            }
            return Result<uint64_t>::success(bitset_rank(rows, words));
        }
        dense.assign(rows * cols, 0);
        for (uint64_t i = 0; i < rows; ++i) {
            uint64_t F = upper[i], rest = F, sign = 0;
            while (rest) {
                uint64_t v = (uint64_t)std::countr_zero(rest);
                rest &= rest - 1;
                uint64_t j = (uint64_t)(std::lower_bound(lower, lower + cols, F ^ (1ULL << v)) - lower);
                dense[i * cols + j] = (Entry)(sign % 2 == 0 ? 1 : field.p - 1);
                ++sign;
            }
        }
        return Result<uint64_t>::success(dense_rank(rows, cols, field));
    }

    uint64_t bitset_rank(uint64_t rows, uint64_t words) {
        uint64_t rank = 0;
        for (uint64_t i = 0; i < rows; ++i) {
            uint64_t *row = bits.data() + i * words;
            for (uint64_t r = 0; r < rank; ++r) {
                const uint64_t *piv = bits.data() + r * words;
                uint64_t lead = pivot_of(piv, words);
                if (row[lead / 64] >> (lead % 64) & 1)
                    for (uint64_t w = 0; w < words; ++w) row[w] ^= piv[w];
            }
            bool zero = true;
            for (uint64_t w = 0; w < words; ++w) zero &= row[w] == 0;
            if (zero) continue;
            if (i != rank) std::swap_ranges(row, row + words, bits.data() + rank * words);
            ++rank;
        }
        return rank;
    }

    static uint64_t pivot_of(const uint64_t *row, uint64_t words) {
        for (uint64_t w = 0; w < words; ++w)
            if (row[w]) return w * 64 + (uint64_t)std::countr_zero(row[w]);
        return words * 64;
    }

    uint64_t dense_rank(uint64_t rows, uint64_t cols, const gfp::Field &field) {
        uint64_t r = 0;
        for (uint64_t c = 0; c < cols && r < rows; ++c) {
            uint64_t pivot = rows;
            for (uint64_t i = r; i < rows; ++i)
                if (dense[i * cols + c]) { pivot = i; break; }
            if (pivot == rows) continue;
            if (pivot != r)
                std::swap_ranges(dense.begin() + pivot * cols, dense.begin() + (pivot + 1) * cols,
                                 dense.begin() + r * cols);
            Entry *prow = dense.data() + r * cols;
            if (prow[c] != 1) field.scale(prow, field.inverse(prow[c]), cols);
            for (uint64_t i = r + 1; i < rows; ++i) {
                Entry f = dense[i * cols + c];
                if (f) field.subtract_multiple(dense.data() + i * cols, prow, f, cols);
            }
            ++r;
        }
        return r;
    }

    /* The maximal faces. */
    void collect_facets() {
        facets.clear();
        for (uint64_t F : faces) {
            bool maximal = true;
            for (uint64_t v = 0; v < n && maximal; ++v)
                if (!(F >> v & 1) && is_face(F | (1ULL << v))) maximal = false;
            if (maximal) facets.push_back(F);
        }
    }

    /* Bjorner-Wachs shellability, searching subsets of the facets. */
    Result<bool> shellable() {
        collect_facets();
        uint64_t t = facets.size();
        if (t > MAX_FACETS)
            return Result<bool>::failure(INVALID, "is_shellable: the complex has " + std::to_string(t) +
                                                      " facets, more than " + std::to_string(MAX_FACETS));
        if (t <= 1) return Result<bool>::success(true);
        follow.assign(t * t, 0);
        for (uint64_t i = 0; i < t; ++i)
            for (uint64_t j = 0; j < t; ++j) {
                uint64_t meet = facets[i] & facets[j], allowed = 0;
                for (uint64_t k = 0; k < t; ++k)
                    if ((meet & ~facets[k]) == 0 && std::popcount(facets[j] & ~facets[k]) == 1)
                        allowed |= 1ULL << k;
                follow[i * t + j] = allowed;
            }
        seen.assign((size_t)1 << t, 0);
        return Result<bool>::success(search(0, t, (1ULL << t) - 1));
    }

    bool search(uint64_t chosen, uint64_t t, uint64_t full) {
        if (chosen == full) return true;
        if (seen[chosen]) return false;
        seen[chosen] = 1;
        for (uint64_t j = 0; j < t; ++j) {
            if (chosen >> j & 1) continue;
            bool allowed = true;
            for (uint64_t i = 0; i < t && allowed; ++i)
                if ((chosen >> i & 1) && (follow[i * t + j] & chosen) == 0) allowed = false;
            if (allowed && search(chosen | (1ULL << j), t, full)) return true;
        }
        return false;
    }
};

R run(const Request &req) {
    const Family &fam = *req.family;
    if (fam.prime() != 2)
        return R::failure(INVALID, "simplicial_complexes: members must be 0/1 matrices over F_2");
    if (fam.cols() > MAX_VERTICES)
        return R::failure(INVALID, "simplicial_complexes: members may have at most " +
                                       std::to_string(MAX_VERTICES) + " columns (vertices)");
    uint64_t nonfaces = req.int_args.at("nonfaces");
    if (nonfaces > 1)
        return R::failure(INVALID, "nonfaces must be 0 or 1: 0 reads the rows as faces, 1 as forbidden sets");

    Op op;
    if (req.op == "f_count") op = Op::FCount;
    else if (req.op == "faces") op = Op::Faces;
    else if (req.op == "euler_characteristic") op = Op::Euler;
    else if (req.op == "betti") op = Op::Betti;
    else if (req.op == "is_shellable") op = Op::Shellable;
    else return R::failure(4, "unknown simplicial_complexes operation " + req.op);

    uint64_t p = 2, dim = 0;
    if (op == Op::Euler || op == Op::Betti) {
        p = req.int_args.at("p");
        if (p < 2 || p >= (1ULL << 32) || !is_prime(p))
            return R::failure(INVALID, "p = " + std::to_string(p) + " is not a prime below 2^32");
    }
    if (op == Op::FCount || op == Op::Betti) dim = req.int_args.at("dim");

    auto size_r = fam.size();
    if (!size_r.ok) return R::failure(size_r.error.status, size_r.error.message);
    uint64_t size = size_r.value;
    Reduction reduction = parse_reduction(req.reduction);
    Shared shared;
    auto st = prepare_all(reduction, size, shared);
    if (!st.ok) return R::failure(st.error.status, st.error.message);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, size ? size : 1));
    gfp::Field field(p);
    std::vector<Worker> workers;
    std::vector<Accumulator> accs;
    workers.reserve(threads);
    accs.reserve(threads);
    for (uint32_t t = 0; t < threads; ++t) {
        workers.emplace_back(fam, (int)nonfaces);
        accs.emplace_back(reduction, &shared);
    }
    auto statuses = parallel_ranges(size, threads, [&](uint32_t t, uint64_t begin, uint64_t end) -> Status {
        Worker &w = workers[t];
        Accumulator &acc = accs[t];
        for (uint64_t i = begin; i < end; ++i) {
            if (acc.exhausted(i)) break;
            auto built = w.build(i);
            if (!built.ok) return built;
            switch (op) {
            case Op::Faces: acc.integer(i, w.faces.size()); break;
            case Op::FCount: acc.integer(i, dim < w.n ? w.dim_count(dim + 1) : 0); break;
            case Op::Euler: acc.integer(i, w.euler(p)); break;
            case Op::Betti: {
                if (dim >= w.n) { acc.integer(i, 0); break; }
                auto low = w.boundary_rank(dim, field);
                if (!low.ok) return fail(low.error.status, low.error.message);
                auto high = w.boundary_rank(dim + 1, field);
                if (!high.ok) return fail(high.error.status, high.error.message);
                acc.integer(i, w.dim_count(dim + 1) - low.value - high.value);
                break;
            }
            case Op::Shellable: {
                auto s = w.shellable();
                if (!s.ok) return fail(s.error.status, s.error.message);
                acc.boolean(i, s.value);
                break;
            }
            }
        }
        return ok();
    });
    for (const auto &s : statuses)
        if (!s.ok) return R::failure(s.error.status, s.error.message);
    return assemble(req, reduction, accs, shared);
}

BackendRegistration registration{Backend{
    "simplicial_complexes", "generic",
    [] { return true; },
    [](const Request &) { return true; },
    run,
    0}};

} // namespace
} // namespace lk::simplicial_complexes
