/* Reductions, shared by every backend that produces one integer or boolean value per member.
 *
 * A backend keeps one Accumulator per thread over one Shared, records values or hits into it,
 * and calls assemble() once at the end. assemble() merges the accumulators, checks that every
 * member was accounted for, sorts hit indices, materialises what the reduction returns, and
 * builds the result object for the reduction named in the request.
 *
 * `first` stops early: once some thread has found a true member, every subtree whose indices
 * all lie above it is abandoned. Backends that walk a family through Family::Visitor get this
 * by asking `acc.exhausted(first)` in push(); backends that loop over indices ask it per index.
 * Abandoned members are not counted as visited; assemble() checks that everything below the hit
 * was decided, and the result reports `visited` as index + 1 so that it does not depend on how
 * many members other threads happened to decide first. */
#pragma once
#include "registry.hpp"

#include <algorithm>
#include <atomic>
#include <thread>

namespace lk {

enum class Reduction { All, Count, Histogram, Hits, First, Sum, Max, Min };

inline Reduction parse_reduction(const std::string &r) {
    static const std::map<std::string, Reduction> table{
        {"all", Reduction::All}, {"count", Reduction::Count}, {"histogram", Reduction::Histogram},
        {"hits", Reduction::Hits}, {"first", Reduction::First}, {"sum", Reduction::Sum},
        {"max", Reduction::Max}, {"min", Reduction::Min}};
    return table.at(r);
}

/* State every thread's accumulator shares: the `all` output and the best `first` index. */
struct Shared {
    std::vector<uint64_t> all; /* `all`: one slot per member index */
    std::atomic<uint64_t> best{UINT64_MAX};
};

struct Accumulator {
    Reduction reduction;
    Shared *shared;
    uint64_t visited = 0, count = 0;
    unsigned __int128 sum = 0;
    std::vector<uint64_t> hist;
    std::vector<uint64_t> hit_indices;
    bool has_extreme = false;
    uint64_t extreme_value = 0, extreme_index = 0;

    Accumulator(Reduction r, Shared *s) : reduction(r), shared(s) {}

    /* `first` only: the subtree starting at `first` cannot improve on a hit already found. */
    bool exhausted(uint64_t first) const {
        return reduction == Reduction::First && first >= shared->best.load(std::memory_order_relaxed);
    }

    void integer(uint64_t index, uint64_t v) {
        ++visited;
        switch (reduction) {
        case Reduction::Histogram:
            if (hist.size() <= v) hist.resize(v + 1, 0);
            ++hist[v];
            break;
        case Reduction::Sum: sum += v; break;
        case Reduction::Max:
        case Reduction::Min: {
            bool better = !has_extreme || (reduction == Reduction::Max ? v > extreme_value : v < extreme_value) ||
                          (v == extreme_value && index < extreme_index);
            if (better) { has_extreme = true; extreme_value = v; extreme_index = index; }
            break;
        }
        default: shared->all[index] = v; break;
        }
    }
    /* `n` consecutive members starting at `first` all have boolean value `value`. */
    void booleans(uint64_t first, uint64_t n, bool value) {
        if (reduction == Reduction::First) {
            if (exhausted(first)) return; /* abandoned, not decided */
            visited += n;
            if (!value) return;
            uint64_t cur = shared->best.load(std::memory_order_relaxed);
            while (first < cur && !shared->best.compare_exchange_weak(cur, first, std::memory_order_relaxed)) {}
            return;
        }
        visited += n;
        if (!value) return;
        switch (reduction) {
        case Reduction::Count: count += n; break;
        case Reduction::Hits: for (uint64_t i = 0; i < n; ++i) hit_indices.push_back(first + i); break;
        case Reduction::All: for (uint64_t i = 0; i < n; ++i) shared->all[first + i] = 1; break;
        default: break;
        }
    }
    void boolean(uint64_t index, bool value) { booleans(index, 1, value); }
};

/* Size the shared `all` output for a family, or refuse if it cannot be materialised. */
inline Status prepare_all(Reduction r, uint64_t size, Shared &shared) {
    if (r != Reduction::All) return ok();
    if (size > (1ULL << 40)) return fail(1, "family too large to materialise");
    shared.all.assign(size, 0);
    return ok();
}

inline Result<std::shared_ptr<Object>> assemble(const Request &req, Reduction reduction, std::vector<Accumulator> &accs,
                                                Shared &shared) {
    using R = Result<std::shared_ptr<Object>>;
    const Family &fam = *req.family;
    auto size_r = fam.size();
    if (!size_r.ok) return R::failure(size_r.error.status, size_r.error.message);
    uint64_t size = size_r.value;
    uint64_t visited = 0, count = 0;
    unsigned __int128 sum = 0;
    std::vector<uint64_t> hist, hits;
    bool has_extreme = false;
    uint64_t extreme_value = 0, extreme_index = 0;
    for (auto &a : accs) {
        visited += a.visited;
        count += a.count;
        sum += a.sum;
        if (hist.size() < a.hist.size()) hist.resize(a.hist.size(), 0);
        for (size_t i = 0; i < a.hist.size(); ++i) hist[i] += a.hist[i];
        hits.insert(hits.end(), a.hit_indices.begin(), a.hit_indices.end());
        if (a.has_extreme) {
            bool better = !has_extreme || (reduction == Reduction::Max ? a.extreme_value > extreme_value : a.extreme_value < extreme_value) ||
                          (a.extreme_value == extreme_value && a.extreme_index < extreme_index);
            if (better) { has_extreme = true; extreme_value = a.extreme_value; extreme_index = a.extreme_index; }
        }
    }
    uint64_t best = shared.best.load();
    bool complete = reduction == Reduction::First ? (best == UINT64_MAX ? visited == size : visited >= best + 1) : visited == size;
    if (!complete)
        return R::failure(4, "enumeration visited " + std::to_string(visited) + " members of " + std::to_string(size));
    auto materialise = [&](uint64_t index, std::vector<Entry> &into) -> Status {
        auto m = fam.member(index);
        if (!m.ok) return fail(m.error.status, m.error.message);
        into = std::move(m.value.entries);
        return ok();
    };
    auto o = std::make_shared<Object>();
    switch (reduction) {
    case Reduction::Count:
        o->kind = "count";
        o->count = std::make_shared<Count>(Count{count, visited, size});
        break;
    case Reduction::Sum:
        if (sum > UINT64_MAX) return R::failure(1, "sum does not fit in 64 bits");
        o->kind = "count";
        o->count = std::make_shared<Count>(Count{(uint64_t)sum, visited, size});
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
    case Reduction::First: {
        auto f = std::make_shared<First>();
        f->p = fam.prime(); f->rows = fam.rows(); f->cols = fam.cols();
        f->visited = best != UINT64_MAX ? best + 1 : size; f->family_size = size;
        if (best != UINT64_MAX) {
            f->found = 1; f->index = best;
            auto st = materialise(best, f->member);
            if (!st.ok) return R::failure(st.error.status, st.error.message);
        }
        o->kind = "first";
        o->first = f;
        break;
    }
    case Reduction::Max:
    case Reduction::Min: {
        if (!has_extreme) return R::failure(1, "max/min of an empty family");
        auto e = std::make_shared<Extremum>();
        e->p = fam.prime(); e->rows = fam.rows(); e->cols = fam.cols();
        e->value = extreme_value; e->index = extreme_index; e->visited = visited; e->family_size = size;
        auto st = materialise(extreme_index, e->member);
        if (!st.ok) return R::failure(st.error.status, st.error.message);
        o->kind = "extremum";
        o->extremum = e;
        break;
    }
    case Reduction::All:
        o->kind = "integers";
        o->integers = std::make_shared<Integers>(Integers{std::move(shared.all)});
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
