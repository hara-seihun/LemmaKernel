/* gfp generic backend: portable C++ for any prime p < 2^32.
 *
 * The walk operations come from ../walk.hpp driven by EchelonBasis (field.hpp): members that
 * share a prefix of rows share the elimination of that prefix, and reductions that cannot see
 * individual members prune subtrees. The explicit-only operations (solve, inverse, rref_witness)
 * are classic Gauss-Jordan per member, threaded over the batch. */
#include "field.hpp"

namespace lk::gfp {
namespace {

/* Classic Gauss-Jordan on an augmented matrix [A | B], pivoting on the first nonzero row.
 * Returns the rank; A's part is in rref and the same operations were applied to B. */
uint64_t gauss_jordan(const Field &f, Entry *a, uint64_t rows, uint64_t cols_a, uint64_t cols_total, std::vector<uint32_t> &pivots) {
    uint64_t r = 0;
    pivots.clear();
    std::vector<Entry> tmp(cols_total);
    for (uint64_t c = 0; c < cols_a && r < rows; ++c) {
        uint64_t pr = rows;
        for (uint64_t i = r; i < rows; ++i)
            if (a[i * cols_total + c]) { pr = i; break; }
        if (pr == rows) continue;
        if (pr != r) std::swap_ranges(a + pr * cols_total, a + (pr + 1) * cols_total, a + r * cols_total);
        Entry *prow = a + r * cols_total;
        if (prow[c] != 1) f.scale(prow, f.inverse(prow[c]), cols_total);
        for (uint64_t i = 0; i < rows; ++i) {
            if (i == r) continue;
            Entry cc = a[i * cols_total + c];
            if (cc) f.subtract_multiple(a + i * cols_total, prow, cc, cols_total);
        }
        pivots.push_back((uint32_t)c);
        ++r;
    }
    return r;
}

template <class Fn> void parallel_members(uint64_t count, uint32_t threads, Fn fn) {
    threads = std::max<uint32_t>(1, std::min<uint64_t>(threads, count));
    std::atomic<uint64_t> next{0};
    auto work = [&] {
        for (;;) {
            uint64_t i = next.fetch_add(1);
            if (i >= count) break;
            fn(i);
        }
    };
    if (threads == 1) work();
    else {
        std::vector<std::thread> pool;
        for (uint32_t t = 0; t < threads; ++t) pool.emplace_back(work);
        for (auto &th : pool) th.join();
    }
}

Result<std::shared_ptr<Object>> run_explicit(const Request &req) {
    using R = Result<std::shared_ptr<Object>>;
    const Matrix &a = *req.family->data;
    Field f(a.p);
    auto o = std::make_shared<Object>();
    if (req.op == "rref_witness") {
        auto w = std::make_shared<Witness>();
        w->p = a.p; w->count = a.count; w->rows = a.rows; w->cols = a.cols;
        w->r.assign(a.count * a.rows * a.cols, 0);
        w->t.assign(a.count * a.rows * a.rows, 0);
        uint64_t tot = a.cols + a.rows;
        parallel_members(a.count, req.threads, [&](uint64_t i) {
            std::vector<Entry> aug(a.rows * tot, 0);
            for (uint64_t r = 0; r < a.rows; ++r) {
                std::copy(a.at(i) + r * a.cols, a.at(i) + (r + 1) * a.cols, aug.begin() + r * tot);
                aug[r * tot + a.cols + r] = 1;
            }
            std::vector<uint32_t> piv;
            gauss_jordan(f, aug.data(), a.rows, a.cols, tot, piv);
            for (uint64_t r = 0; r < a.rows; ++r) {
                std::copy(aug.begin() + r * tot, aug.begin() + r * tot + a.cols, w->r.begin() + (i * a.rows + r) * a.cols);
                std::copy(aug.begin() + r * tot + a.cols, aug.begin() + (r + 1) * tot, w->t.begin() + (i * a.rows + r) * a.rows);
            }
        });
        o->kind = "gfp.witness";
        o->witness = w;
        return R::success(o);
    }
    if (req.op == "inverse") {
        if (a.rows != a.cols) return R::failure(INVALID, "inverse needs square members");
        auto inv = std::make_shared<Inverses>();
        inv->p = a.p; inv->count = a.count; inv->n = a.rows;
        inv->invertible.assign(a.count, 0);
        inv->entries.assign(a.count * a.rows * a.rows, 0);
        uint64_t n = a.rows, tot = 2 * n;
        parallel_members(a.count, req.threads, [&](uint64_t i) {
            std::vector<Entry> aug(n * tot, 0);
            for (uint64_t r = 0; r < n; ++r) {
                std::copy(a.at(i) + r * n, a.at(i) + (r + 1) * n, aug.begin() + r * tot);
                aug[r * tot + n + r] = 1;
            }
            std::vector<uint32_t> piv;
            uint64_t rank = gauss_jordan(f, aug.data(), n, n, tot, piv);
            if (rank == n) {
                inv->invertible[i] = 1;
                for (uint64_t r = 0; r < n; ++r)
                    std::copy(aug.begin() + r * tot + n, aug.begin() + (r + 1) * tot, inv->entries.begin() + (i * n + r) * n);
            }
        });
        o->kind = "gfp.inverses";
        o->inverses = inv;
        return R::success(o);
    }
    if (req.op == "solve") {
        auto it = req.handle_args.find("rhs");
        if (it == req.handle_args.end() || !it->second->matrix) return R::failure(INVALID, "solve needs a gfp.matrix argument `rhs`");
        const Matrix &b = *it->second->matrix;
        if (b.p != a.p || b.count != a.count || b.rows != 1 || b.cols != a.rows)
            return R::failure(INVALID, "rhs must be a batch of count 1 x rows vectors over the same prime");
        auto sol = std::make_shared<Solutions>();
        sol->p = a.p; sol->count = a.count; sol->length = a.cols;
        sol->solvable.assign(a.count, 0);
        sol->entries.assign(a.count * a.cols, 0);
        uint64_t tot = a.cols + 1;
        parallel_members(a.count, req.threads, [&](uint64_t i) {
            std::vector<Entry> aug(a.rows * tot, 0);
            for (uint64_t r = 0; r < a.rows; ++r) {
                std::copy(a.at(i) + r * a.cols, a.at(i) + (r + 1) * a.cols, aug.begin() + r * tot);
                aug[r * tot + a.cols] = b.entries[i * a.rows + r];
            }
            std::vector<uint32_t> piv;
            uint64_t rank = gauss_jordan(f, aug.data(), a.rows, a.cols, tot, piv);
            bool consistent = true;
            for (uint64_t r = rank; r < a.rows; ++r)
                if (aug[r * tot + a.cols]) consistent = false;
            if (!consistent) return;
            sol->solvable[i] = 1;
            for (uint64_t r = 0; r < rank; ++r) sol->entries[i * a.cols + piv[r]] = aug[r * tot + a.cols];
        });
        o->kind = "gfp.solutions";
        o->solutions = sol;
        return R::success(o);
    }
    return R::failure(INTERNAL, "unknown explicit operation " + req.op);
}

Result<std::shared_ptr<Object>> run(const Request &req) {
    Query q;
    if (!parse_query(req.op, q)) return run_explicit(req);
    return run_walk<EchelonBasis>(req, q, parse_reduction(req.reduction),
                                  [](uint64_t p, uint64_t cols) { return EchelonBasis(p, cols); });
}

BackendRegistration registration{Backend{
    "gfp", "generic",
    [] { return true; },
    [](const Request &req) { return is_prime(req.family->prime()) && req.family->prime() < (1ULL << 32); },
    run,
    0}};

} // namespace
} // namespace lk::gfp
