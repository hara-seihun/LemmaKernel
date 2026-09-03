/* Reductions, shared by every backend that produces one integer or boolean value per member.
 *
 * A backend keeps one Accumulator per thread, records values or hits into it, and calls
 * assemble() once at the end. assemble() merges the accumulators, checks that every member was
 * accounted for, sorts hit indices, materialises the first `limit` hits, and builds the result
 * object for the reduction named in the request. */
#pragma once
#include "registry.hpp"

#include <algorithm>
#include <atomic>
#include <thread>

namespace lk {

enum class Reduction { All, Count, Histogram, Hits };

inline Reduction parse_reduction(const std::string &r) {
    static const std::map<std::string, Reduction> table{
        {"all", Reduction::All}, {"count", Reduction::Count}, {"histogram", Reduction::Histogram}, {"hits", Reduction::Hits}};
    return table.at(r);
}

struct Accumulator {
    Reduction reduction;
    std::vector<uint64_t> *all; /* `all`: shared output, one slot per member index */
    uint64_t visited = 0, count = 0;
    std::vector<uint64_t> hist;
    std::vector<uint64_t> hit_indices;

    Accumulator(Reduction r, std::vector<uint64_t> *all_out) : reduction(r), all(all_out) {}

    void integer(uint64_t index, uint64_t v) {
        ++visited;
        if (reduction == Reduction::Histogram) {
            if (hist.size() <= v) hist.resize(v + 1, 0);
            ++hist[v];
        } else (*all)[index] = v;
    }
    /* `n` consecutive members starting at `first` all have boolean value `value`. */
    void booleans(uint64_t first, uint64_t n, bool value) {
        visited += n;
        if (!value) return;
        switch (reduction) {
        case Reduction::Count: count += n; break;
        case Reduction::Hits: for (uint64_t i = 0; i < n; ++i) hit_indices.push_back(first + i); break;
        case Reduction::All: for (uint64_t i = 0; i < n; ++i) (*all)[first + i] = 1; break;
        case Reduction::Histogram: break;
        }
    }
    void boolean(uint64_t index, bool value) { booleans(index, 1, value); }
};

/* Size the shared `all` output for a family, or refuse if it cannot be materialised. */
inline Status prepare_all(Reduction r, uint64_t size, std::vector<uint64_t> &all) {
    if (r != Reduction::All) return ok();
    if (size > (1ULL << 40)) return fail(1, "family too large to materialise");
    all.assign(size, 0);
    return ok();
}

inline Result<std::shared_ptr<Object>> assemble(const Request &req, Reduction reduction, std::vector<Accumulator> &accs,
                                                std::vector<uint64_t> &&all) {
    using R = Result<std::shared_ptr<Object>>;
    const Family &fam = *req.family;
    auto size_r = fam.size();
    if (!size_r.ok) return R::failure(size_r.error.status, size_r.error.message);
    uint64_t size = size_r.value;
    uint64_t visited = 0, count = 0;
    std::vector<uint64_t> hist, hits;
    for (auto &a : accs) {
        visited += a.visited;
        count += a.count;
        if (hist.size() < a.hist.size()) hist.resize(a.hist.size(), 0);
        for (size_t i = 0; i < a.hist.size(); ++i) hist[i] += a.hist[i];
        hits.insert(hits.end(), a.hit_indices.begin(), a.hit_indices.end());
    }
    if (visited != size)
        return R::failure(4, "enumeration visited " + std::to_string(visited) + " members of " + std::to_string(size));
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
        h->p = fam.prime(); h->rows = fam.rows(); h->cols = fam.cols();
        h->total = hits.size(); h->visited = visited; h->family_size = size;
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
        o->kind = "integers";
        o->integers = std::make_shared<Integers>(Integers{std::move(all)});
        break;
    }
    return R::success(o);
}

/* Run `fn(thread, begin, end)` over [0, total) in chunks on `threads` threads. */
template <class Fn> std::vector<Status> parallel_ranges(uint64_t total, uint32_t threads, Fn fn) {
    threads = std::max<uint32_t>(1, std::min<uint64_t>(threads, total ? total : 1));
    std::vector<Status> statuses(threads, ok());
    std::atomic<uint64_t> next{0};
    uint64_t chunk = std::max<uint64_t>(1, total / (threads * 16));
    auto work = [&](uint32_t t) {
        for (;;) {
            uint64_t begin = next.fetch_add(chunk);
            if (begin >= total) break;
            Status st = fn(t, begin, std::min(total, begin + chunk));
            if (!st.ok) { statuses[t] = st; break; }
        }
    };
    if (threads == 1) work(0);
    else {
        std::vector<std::thread> pool;
        for (uint32_t t = 0; t < threads; ++t) pool.emplace_back(work, t);
        for (auto &th : pool) th.join();
    }
    return statuses;
}

} // namespace lk
