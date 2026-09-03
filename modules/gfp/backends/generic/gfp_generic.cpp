/* gfp generic backend: portable C++ for any prime p < 2^32.
 *
 * Families are walked depth-first (runtime/src/family.cpp). This backend keeps one echelon basis
 * per walk and extends it by one reduced row per push, so members sharing a prefix share the
 * elimination of that prefix. Reductions that cannot see individual members let it prune:
 * full_row_rank skips a subtree as soon as a dependent row is pushed, full_col_rank and in_span
 * take a whole subtree once the prefix already decides the answer. Threads split the top-level
 * branches; results merge in canonical order. */
#include "../../../../runtime/src/registry.hpp"

#include <algorithm>
#include <atomic>
#include <thread>

namespace lk {
namespace {

constexpr int INVALID = 1;
constexpr int INTERNAL = 4;

struct Field {
    uint64_t p;
    uint64_t barrett; /* floor(2^64 / p) */
    std::vector<Entry> inv_table;

    explicit Field(uint64_t prime) : p(prime) {
        barrett = p == 1 ? 0 : (uint64_t)(((unsigned __int128)1 << 64) / p);
        if (p <= (1u << 16)) {
            inv_table.assign(p, 0);
            for (uint64_t a = 1; a < p; ++a) inv_table[a] = (Entry)pow(a, p - 2);
        }
    }
    uint64_t reduce(uint64_t x) const {
        uint64_t q = (uint64_t)(((unsigned __int128)x * barrett) >> 64);
        uint64_t r = x - q * p;
        return r >= p ? r - p : r;
    }
    uint64_t pow(uint64_t a, uint64_t e) const {
        uint64_t r = 1;
        a %= p;
        while (e) {
            if (e & 1) r = (uint64_t)((unsigned __int128)r * a % p);
            a = (uint64_t)((unsigned __int128)a * a % p);
            e >>= 1;
        }
        return r;
    }
    Entry inverse(Entry a) const { return inv_table.empty() ? (Entry)pow(a, p - 2) : inv_table[a]; }

    /* row -= c * other, in place */
    void subtract_multiple(Entry *row, const Entry *other, Entry c, uint64_t cols) const {
        uint64_t m = p - c;
        for (uint64_t j = 0; j < cols; ++j) row[j] = (Entry)reduce((uint64_t)row[j] + m * other[j]);
    }
    void scale(Entry *row, Entry c, uint64_t cols) const {
        for (uint64_t j = 0; j < cols; ++j) row[j] = (Entry)reduce((uint64_t)row[j] * c);
    }
};

/* Echelon basis kept in insertion order: row i is zero at the pivots of rows before it. */
struct EchelonBasis {
    const Field &f;
    uint64_t cols;
    std::vector<Entry> rows;
    std::vector<uint32_t> pivots;

    EchelonBasis(const Field &field, uint64_t c) : f(field), cols(c) {}
    uint64_t rank() const { return pivots.size(); }
    Entry *row(uint64_t i) { return rows.data() + i * cols; }
    const Entry *row(uint64_t i) const { return rows.data() + i * cols; }

    void reduce_into(Entry *v) const {
        for (uint64_t i = 0; i < pivots.size(); ++i) {
            Entry c = v[pivots[i]];
            if (c) f.subtract_multiple(v, row(i), c, cols);
        }
    }
    /* Reduce v against the basis; append it if it is independent. Returns whether it was added. */
    bool add(const Entry *v, std::vector<Entry> &scratch) {
        scratch.assign(v, v + cols);
        reduce_into(scratch.data());
        uint64_t lead = cols;
        for (uint64_t j = 0; j < cols; ++j)
            if (scratch[j]) { lead = j; break; }
        if (lead == cols) return false;
        if (scratch[lead] != 1) f.scale(scratch.data(), f.inverse(scratch[lead]), cols);
        rows.insert(rows.end(), scratch.begin(), scratch.end());
        pivots.push_back((uint32_t)lead);
        return true;
    }
    void remove_last() {
        rows.resize(rows.size() - cols);
        pivots.pop_back();
    }
    /* Reduced row echelon form of the row space: rows sorted by pivot with every other pivot
     * column cleared. `out` receives rank*cols entries; `piv` the sorted pivots. */
    void rref(std::vector<Entry> &out, std::vector<uint32_t> &piv) const {
        uint64_t r = rank();
        out = rows;
        piv = pivots;
        for (uint64_t i = 1; i < r; ++i)
            for (uint64_t e = 0; e < i; ++e) {
                Entry c = out[e * cols + piv[i]];
                if (c) f.subtract_multiple(out.data() + e * cols, out.data() + i * cols, c, cols);
            }
        std::vector<uint64_t> order(r);
        for (uint64_t i = 0; i < r; ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](uint64_t a, uint64_t b) { return piv[a] < piv[b]; });
        std::vector<Entry> sorted(r * cols);
        std::vector<uint32_t> sp(r);
        for (uint64_t i = 0; i < r; ++i) {
            std::copy(out.begin() + order[i] * cols, out.begin() + (order[i] + 1) * cols, sorted.begin() + i * cols);
            sp[i] = piv[order[i]];
        }
        out.swap(sorted);
        piv.swap(sp);
    }
};

enum class Query { Rank, Nullity, FullRowRank, FullColRank, InSpan, Rref, Nullspace };
enum class Reduction { All, Count, Histogram, Hits };

struct Outputs {
    /* `all` */
    std::vector<uint64_t> integers;
    std::vector<Entry> matrices;                   /* rref: size*rows*cols */
    std::vector<std::vector<Entry>> ragged;        /* nullspace: per index */
};

struct Walker : Family::Visitor {
    const Field &f;
    Query query;
    Reduction reduction;
    uint64_t cols, member_rows;
    bool prune;
    EchelonBasis basis;
    std::vector<uint8_t> added;
    std::vector<Entry> scratch;
    std::vector<Entry> target_stack; /* (depth+1) * cols, reduced target per level */
    Outputs *out;
    uint64_t visited = 0, count = 0;
    std::vector<uint64_t> hist;
    std::vector<uint64_t> hit_indices;
    std::vector<Entry> rref_buf;
    std::vector<uint32_t> piv_buf;

    Walker(const Field &field, Query q, Reduction r, uint64_t c, uint64_t rows, const Entry *target, Outputs *o)
        : f(field), query(q), reduction(r), cols(c), member_rows(rows), prune(r != Reduction::All), basis(field, c), out(o) {
        if (target) target_stack.assign(target, target + cols);
    }

    uint64_t depth() const { return added.size(); }

    Step push(const Entry *row) override {
        bool independent = basis.add(row, scratch);
        added.push_back(independent);
        if (query == Query::InSpan) {
            const Entry *prev = target_stack.data() + (depth() - 1) * cols;
            target_stack.insert(target_stack.end(), prev, prev + cols);
            if (independent) {
                Entry *cur = target_stack.data() + depth() * cols;
                Entry c = cur[basis.pivots.back()];
                if (c) f.subtract_multiple(cur, basis.row(basis.rank() - 1), c, cols);
            }
        }
        if (!prune) return Step::Descend;
        switch (query) {
        case Query::FullRowRank:
            return independent ? Step::Descend : Step::Skip;
        case Query::FullColRank: {
            uint64_t r = basis.rank();
            if (r == cols) return Step::TakeAll;
            if (r + (member_rows - depth()) < cols) return Step::Skip;
            return Step::Descend;
        }
        case Query::InSpan: {
            const Entry *cur = target_stack.data() + depth() * cols;
            bool zero = std::all_of(cur, cur + cols, [](Entry e) { return e == 0; });
            return zero ? Step::TakeAll : Step::Descend;
        }
        default:
            return Step::Descend;
        }
    }

    void pop() override {
        if (added.back()) basis.remove_last();
        added.pop_back();
        if (query == Query::InSpan) target_stack.resize(target_stack.size() - cols);
    }

    bool boolean_value() const {
        switch (query) {
        case Query::FullRowRank: return basis.rank() == member_rows;
        case Query::FullColRank: return basis.rank() == cols;
        case Query::InSpan: {
            const Entry *cur = target_stack.data() + depth() * cols;
            return std::all_of(cur, cur + cols, [](Entry e) { return e == 0; });
        }
        default: return false;
        }
    }

    void record_true(uint64_t first, uint64_t n) {
        switch (reduction) {
        case Reduction::Count: count += n; break;
        case Reduction::Hits:
            for (uint64_t i = 0; i < n; ++i) hit_indices.push_back(first + i);
            break;
        case Reduction::All:
            for (uint64_t i = 0; i < n; ++i) out->integers[first + i] = 1;
            break;
        case Reduction::Histogram: break;
        }
    }

    void leaf(uint64_t index) override {
        ++visited;
        switch (query) {
        case Query::Rank:
        case Query::Nullity: {
            uint64_t v = query == Query::Rank ? basis.rank() : cols - basis.rank();
            if (reduction == Reduction::Histogram) {
                if (hist.size() <= v) hist.resize(v + 1, 0);
                ++hist[v];
            } else out->integers[index] = v;
            break;
        }
        case Query::FullRowRank:
        case Query::FullColRank:
        case Query::InSpan:
            if (boolean_value()) record_true(index, 1);
            break;
        case Query::Rref: {
            basis.rref(rref_buf, piv_buf);
            Entry *dst = out->matrices.data() + index * member_rows * cols;
            std::fill(dst, dst + member_rows * cols, 0);
            std::copy(rref_buf.begin(), rref_buf.end(), dst);
            break;
        }
        case Query::Nullspace: {
            basis.rref(rref_buf, piv_buf);
            std::vector<Entry> &vecs = out->ragged[index];
            vecs.clear();
            uint64_t r = piv_buf.size();
            uint64_t pi = 0;
            for (uint64_t c = 0; c < cols; ++c) {
                if (pi < r && piv_buf[pi] == c) { ++pi; continue; }
                std::vector<Entry> v(cols, 0);
                v[c] = 1;
                for (uint64_t i = 0; i < r; ++i) {
                    Entry e = rref_buf[i * cols + c];
                    v[piv_buf[i]] = e ? (Entry)(f.p - e) : 0;
                }
                vecs.insert(vecs.end(), v.begin(), v.end());
            }
            break;
        }
        }
    }

    void take_all(uint64_t first, uint64_t n) override {
        visited += n;
        record_true(first, n);
    }
    void skip_all(uint64_t, uint64_t n) override { visited += n; }
};

Result<std::shared_ptr<Object>> run_walk(const Request &req, Query query, Reduction reduction) {
    const Family &fam = *req.family;
    Field field(fam.prime());
    uint64_t cols = fam.cols(), rows = fam.rows();
    auto size_r = fam.size();
    if (!size_r.ok) return Result<std::shared_ptr<Object>>::failure(size_r.error.status, size_r.error.message);
    uint64_t size = size_r.value;
    auto tops_r = fam.top_count();
    if (!tops_r.ok) return Result<std::shared_ptr<Object>>::failure(tops_r.error.status, tops_r.error.message);
    uint64_t tops = tops_r.value;

    const Entry *target = nullptr;
    std::shared_ptr<Matrix> target_m;
    if (query == Query::InSpan) {
        auto it = req.handle_args.find("target");
        if (it == req.handle_args.end() || !it->second->matrix)
            return Result<std::shared_ptr<Object>>::failure(INVALID, "in_span needs a gfp.matrix argument `target`");
        target_m = it->second->matrix;
        if (target_m->count != 1 || target_m->rows != 1 || target_m->cols != cols || target_m->p != field.p)
            return Result<std::shared_ptr<Object>>::failure(INVALID, "target must be a single 1 x cols matrix over the family's prime");
        target = target_m->entries.data();
    }
    Outputs out;
    if (reduction == Reduction::All) {
        if (size > (1ULL << 40)) return Result<std::shared_ptr<Object>>::failure(INVALID, "family too large to materialise");
        if (query == Query::Rref) out.matrices.assign(size * rows * cols, 0);
        else if (query == Query::Nullspace) out.ragged.resize(size);
        else out.integers.assign(size, 0);
    }

    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, tops));
    std::vector<Walker> walkers;
    walkers.reserve(threads);
    for (uint32_t t = 0; t < threads; ++t) walkers.emplace_back(field, query, reduction, cols, rows, target, &out);
    std::atomic<uint64_t> next_branch{0};
    std::vector<Status> statuses(threads, ok());
    uint64_t chunk = std::max<uint64_t>(1, tops / (threads * 16));
    auto work = [&](uint32_t t) {
        for (;;) {
            uint64_t begin = next_branch.fetch_add(chunk);
            if (begin >= tops) break;
            uint64_t end = std::min(tops, begin + chunk);
            Status st = fam.enumerate(walkers[t], begin, end);
            if (!st.ok) { statuses[t] = st; break; }
        }
    };
    if (threads == 1) work(0);
    else {
        std::vector<std::thread> pool;
        for (uint32_t t = 0; t < threads; ++t) pool.emplace_back(work, t);
        for (auto &th : pool) th.join();
    }
    for (const auto &st : statuses)
        if (!st.ok) return Result<std::shared_ptr<Object>>::failure(st.error.status, st.error.message);

    uint64_t visited = 0, count = 0;
    std::vector<uint64_t> hist, hits;
    for (auto &w : walkers) {
        visited += w.visited;
        count += w.count;
        if (hist.size() < w.hist.size()) hist.resize(w.hist.size(), 0);
        for (size_t i = 0; i < w.hist.size(); ++i) hist[i] += w.hist[i];
        hits.insert(hits.end(), w.hit_indices.begin(), w.hit_indices.end());
    }
    if (visited != size)
        return Result<std::shared_ptr<Object>>::failure(INTERNAL, "enumeration visited " + std::to_string(visited) + " members of " + std::to_string(size));

    auto o = std::make_shared<Object>();
    switch (reduction) {
    case Reduction::Count:
        o->kind = "count";
        o->count = std::make_shared<Count>(Count{count, visited, size});
        break;
    case Reduction::Histogram:
        o->kind = "histogram";
        o->histogram = std::make_shared<Histogram>(Histogram{visited, size, hist});
        break;
    case Reduction::Hits: {
        std::sort(hits.begin(), hits.end());
        uint64_t limit = req.int_args.count("limit") ? req.int_args.at("limit") : 0;
        auto h = std::make_shared<Hits>();
        h->p = field.p; h->rows = rows; h->cols = cols; h->total = hits.size(); h->visited = visited; h->family_size = size;
        h->indices = hits;
        uint64_t mat = std::min<uint64_t>(limit, hits.size());
        for (uint64_t i = 0; i < mat; ++i) {
            auto m = fam.member(hits[i]);
            if (!m.ok) return Result<std::shared_ptr<Object>>::failure(m.error.status, m.error.message);
            h->members.insert(h->members.end(), m.value.entries.begin(), m.value.entries.end());
        }
        o->kind = "hits";
        o->hits = h;
        break;
    }
    case Reduction::All:
        if (query == Query::Rref) {
            o->kind = "gfp.matrix";
            o->matrix = std::make_shared<Matrix>(Matrix{field.p, size, rows, cols, std::move(out.matrices)});
        } else if (query == Query::Nullspace) {
            auto b = std::make_shared<Basis>();
            b->p = field.p; b->count = size; b->cols = cols;
            b->offsets.push_back(0);
            for (auto &v : out.ragged) {
                b->entries.insert(b->entries.end(), v.begin(), v.end());
                b->offsets.push_back(b->offsets.back() + v.size() / cols);
            }
            o->kind = "gfp.basis";
            o->basis = b;
        } else {
            o->kind = "integers";
            o->integers = std::make_shared<Integers>(Integers{std::move(out.integers)});
        }
        break;
    }
    return Result<std::shared_ptr<Object>>::success(o);
}

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
    static const std::map<std::string, Query> queries{
        {"rank", Query::Rank}, {"nullity", Query::Nullity}, {"full_row_rank", Query::FullRowRank},
        {"full_col_rank", Query::FullColRank}, {"in_span", Query::InSpan}, {"rref", Query::Rref},
        {"nullspace", Query::Nullspace}};
    static const std::map<std::string, Reduction> reductions{
        {"all", Reduction::All}, {"count", Reduction::Count}, {"histogram", Reduction::Histogram}, {"hits", Reduction::Hits}};
    auto q = queries.find(req.op);
    if (q == queries.end()) return run_explicit(req);
    return run_walk(req, q->second, reductions.at(req.reduction));
}

BackendRegistration registration{Backend{
    "gfp", "generic",
    [] { return true; },
    [](const Request &req) { return req.family->prime() < (1ULL << 32); },
    run,
    0}};

} // namespace
} // namespace lk
