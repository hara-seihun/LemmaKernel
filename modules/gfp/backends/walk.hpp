/* Shared enumeration driver for gfp backends.
 *
 * A gfp backend that handles the family-walk operations (rank, nullity, full_row_rank,
 * full_col_rank, in_span, rref, nullspace) only has to supply a Basis type; everything else
 * (walking the family tree, pruning, threads, reductions, result objects, the completeness check)
 * lives here. The generic backend's EchelonBasis in generic/field.hpp is the reference Basis; a
 * new backend copies its shape and changes the arithmetic, the row layout, or both.
 *
 * Basis interface:
 *
 *   Basis(const Basis &);                     copyable; each thread gets its own copy
 *   uint64_t rank() const;                    rows currently held
 *   bool add(const Entry *row);               reduce `row` (cols entries, each < p) against the
 *                                             basis and append it if it is independent;
 *                                             return whether it was appended
 *   void remove_last();                       undo the most recent successful add
 *   using Target = ...;                       a row kept reduced against the basis (for in_span)
 *   Target pack(const Entry *row) const;
 *   void reduce_by_last(Target &t) const;     reduce t by the row the last add() appended
 *   bool is_zero(const Target &t) const;
 *   void rref(std::vector<Entry> &out, std::vector<uint32_t> &pivots) const;
 *                                             rows sorted by pivot, other pivot columns cleared,
 *                                             leading entries 1, unpacked as rank*cols entries
 *
 * Every add() sees rows in the order the family pushes them; the basis only ever grows by the
 * last row and shrinks by the last row, so a stack of "was this push appended" is enough to
 * keep it in step with the walk. */
#pragma once
#include "../../../runtime/src/registry.hpp"

#include <algorithm>
#include <atomic>
#include <thread>

namespace lk::gfp {

constexpr int INVALID = 1;
constexpr int INTERNAL = 4;

enum class Query { Rank, Nullity, FullRowRank, FullColRank, InSpan, Rref, Nullspace };
enum class Reduction { All, Count, Histogram, Hits };

/* Names from the manifest to enums; false if `op` is not a walk operation. */
inline bool parse_query(const std::string &op, Query &q) {
    static const std::map<std::string, Query> table{
        {"rank", Query::Rank}, {"nullity", Query::Nullity}, {"full_row_rank", Query::FullRowRank},
        {"full_col_rank", Query::FullColRank}, {"in_span", Query::InSpan}, {"rref", Query::Rref},
        {"nullspace", Query::Nullspace}};
    auto it = table.find(op);
    if (it == table.end()) return false;
    q = it->second;
    return true;
}

inline Reduction parse_reduction(const std::string &r) {
    static const std::map<std::string, Reduction> table{
        {"all", Reduction::All}, {"count", Reduction::Count}, {"histogram", Reduction::Histogram}, {"hits", Reduction::Hits}};
    return table.at(r);
}

struct Outputs {
    std::vector<uint64_t> integers;         /* `all` of an integer or boolean operation */
    std::vector<Entry> matrices;            /* `all` of rref: size*rows*cols */
    std::vector<std::vector<Entry>> ragged; /* `all` of nullspace: per member */
};

template <class Basis> struct Walker : Family::Visitor {
    Query query;
    Reduction reduction;
    uint64_t p, cols, member_rows;
    bool prune;
    Basis basis;
    std::vector<uint8_t> added;
    std::vector<typename Basis::Target> target_stack; /* one per depth, level 0 = the target itself */
    Outputs *out;
    uint64_t visited = 0, count = 0;
    std::vector<uint64_t> hist;
    std::vector<uint64_t> hit_indices;
    std::vector<Entry> rref_buf;
    std::vector<uint32_t> piv_buf;

    Walker(Basis b, Query q, Reduction r, uint64_t prime, uint64_t c, uint64_t rows, const Entry *target, Outputs *o)
        : query(q), reduction(r), p(prime), cols(c), member_rows(rows), prune(r != Reduction::All), basis(std::move(b)), out(o) {
        if (target) target_stack.push_back(basis.pack(target));
    }

    uint64_t depth() const { return added.size(); }

    Step push(const Entry *row) override {
        bool independent = basis.add(row);
        added.push_back(independent);
        if (query == Query::InSpan) {
            target_stack.push_back(target_stack.back());
            if (independent) basis.reduce_by_last(target_stack.back());
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
        case Query::InSpan:
            return basis.is_zero(target_stack.back()) ? Step::TakeAll : Step::Descend;
        default:
            return Step::Descend;
        }
    }

    void pop() override {
        if (added.back()) basis.remove_last();
        added.pop_back();
        if (query == Query::InSpan) target_stack.pop_back();
    }

    bool boolean_value() const {
        switch (query) {
        case Query::FullRowRank: return basis.rank() == member_rows;
        case Query::FullColRank: return basis.rank() == cols;
        case Query::InSpan: return basis.is_zero(target_stack.back());
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
            uint64_t r = piv_buf.size(), pi = 0;
            for (uint64_t c = 0; c < cols; ++c) {
                if (pi < r && piv_buf[pi] == c) { ++pi; continue; }
                std::vector<Entry> v(cols, 0);
                v[c] = 1;
                for (uint64_t i = 0; i < r; ++i) {
                    Entry e = rref_buf[i * cols + c];
                    v[piv_buf[i]] = e ? (Entry)(p - e) : 0;
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

/* Run a walk operation. `make_basis(p, cols)` builds a fresh Basis for one thread. */
template <class Basis, class MakeBasis>
Result<std::shared_ptr<Object>> run_walk(const Request &req, Query query, Reduction reduction, MakeBasis make_basis) {
    using R = Result<std::shared_ptr<Object>>;
    const Family &fam = *req.family;
    uint64_t p = fam.prime(), cols = fam.cols(), rows = fam.rows();
    auto size_r = fam.size();
    if (!size_r.ok) return R::failure(size_r.error.status, size_r.error.message);
    uint64_t size = size_r.value;
    auto tops_r = fam.top_count();
    if (!tops_r.ok) return R::failure(tops_r.error.status, tops_r.error.message);
    uint64_t tops = tops_r.value;

    const Entry *target = nullptr;
    std::shared_ptr<Matrix> target_m;
    if (query == Query::InSpan) {
        auto it = req.handle_args.find("target");
        if (it == req.handle_args.end() || !it->second->matrix)
            return R::failure(INVALID, "in_span needs a gfp.matrix argument `target`");
        target_m = it->second->matrix;
        if (target_m->count != 1 || target_m->rows != 1 || target_m->cols != cols || target_m->p != p)
            return R::failure(INVALID, "target must be a single 1 x cols matrix over the family's prime");
        target = target_m->entries.data();
    }
    Outputs out;
    if (reduction == Reduction::All) {
        if (size > (1ULL << 40)) return R::failure(INVALID, "family too large to materialise");
        if (query == Query::Rref) out.matrices.assign(size * rows * cols, 0);
        else if (query == Query::Nullspace) out.ragged.resize(size);
        else out.integers.assign(size, 0);
    }

    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, tops));
    std::vector<Walker<Basis>> walkers;
    walkers.reserve(threads);
    for (uint32_t t = 0; t < threads; ++t)
        walkers.emplace_back(make_basis(p, cols), query, reduction, p, cols, rows, target, &out);
    std::atomic<uint64_t> next_branch{0};
    std::vector<Status> statuses(threads, ok());
    uint64_t chunk = std::max<uint64_t>(1, tops / (threads * 16));
    auto work = [&](uint32_t t) {
        for (;;) {
            uint64_t begin = next_branch.fetch_add(chunk);
            if (begin >= tops) break;
            Status st = fam.enumerate(walkers[t], begin, std::min(tops, begin + chunk));
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
        if (!st.ok) return R::failure(st.error.status, st.error.message);

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
        return R::failure(INTERNAL, "enumeration visited " + std::to_string(visited) + " members of " + std::to_string(size));

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
        h->p = p; h->rows = rows; h->cols = cols; h->total = hits.size(); h->visited = visited; h->family_size = size;
        h->indices = hits;
        uint64_t mat = std::min<uint64_t>(limit, hits.size());
        for (uint64_t i = 0; i < mat; ++i) {
            auto m = fam.member(hits[i]);
            if (!m.ok) return R::failure(m.error.status, m.error.message);
            h->members.insert(h->members.end(), m.value.entries.begin(), m.value.entries.end());
        }
        o->kind = "hits";
        o->hits = h;
        break;
    }
    case Reduction::All:
        if (query == Query::Rref) {
            o->kind = "gfp.matrix";
            o->matrix = std::make_shared<Matrix>(Matrix{p, size, rows, cols, std::move(out.matrices)});
        } else if (query == Query::Nullspace) {
            auto b = std::make_shared<lk::Basis>();
            b->p = p; b->count = size; b->cols = cols;
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
    return R::success(o);
}

} // namespace lk::gfp
