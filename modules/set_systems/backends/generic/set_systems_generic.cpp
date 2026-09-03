/* Portable backend for set_systems.
 *
 * The outer subsets_of family is walked depth first. Each pushed row is one binary incidence
 * word, so hereditary predicate failures discard the whole subtree below that prefix. Degrees
 * and the lower shadow are updated on push and undone on pop. */
#include "../../../../runtime/src/reduce.hpp"

#include <bit>
#include <unordered_map>

namespace lk::set_systems {
namespace {

using R = Result<std::shared_ptr<Object>>;
constexpr int INVALID = 1;
constexpr int INTERNAL = 4;

enum class Op { IsIntersecting, IsAntichain, IsSunflowerFree, MaxDegree, ShadowSize, IsEkrExtremal, IsSpernerExtremal };

uint64_t binomial(uint64_t n, uint64_t k) {
    if (k > n) return 0;
    k = std::min(k, n - k);
    unsigned __int128 value = 1;
    for (uint64_t i = 1; i <= k; ++i) value = value * (n - k + i) / i;
    return value > UINT64_MAX ? UINT64_MAX : (uint64_t)value;
}

struct Walker : Family::Visitor {
    Op op;
    Reduction reduction;
    uint64_t n, system_size, sunflower_k;
    Shared *shared;
    Accumulator acc;
    std::vector<uint64_t> sets;
    std::vector<uint64_t> degrees;
    std::unordered_map<uint64_t, uint64_t> shadow_counts;
    uint64_t ekr_rank = 0;
    bool ekr_shape = false;

    Walker(Op operation, Reduction red, uint64_t points, uint64_t size, uint64_t petals, Shared *state)
        : op(operation), reduction(red), n(points), system_size(size), sunflower_k(petals), shared(state), acc(red, state),
          degrees(points, 0) {}

    static bool intersects(uint64_t a, uint64_t b) { return (a & b) != 0; }
    static bool comparable(uint64_t a, uint64_t b) { return (a & b) == a || (a & b) == b; }

    bool pairwise_intersects_newest() const {
        uint64_t newest = sets.back();
        for (size_t i = 0; i + 1 < sets.size(); ++i)
            if (!intersects(sets[i], newest)) return false;
        return true;
    }

    bool incomparable_newest() const {
        uint64_t newest = sets.back();
        for (size_t i = 0; i + 1 < sets.size(); ++i)
            if (comparable(sets[i], newest)) return false;
        return true;
    }

    bool sunflower_with_newest(const std::vector<size_t> &prior) const {
        auto petal = [&](size_t i) { return i < prior.size() ? sets[prior[i]] : sets.back(); };
        size_t count = prior.size() + 1;
        uint64_t core = petal(0) & petal(1);
        for (size_t i = 0; i < count; ++i)
            for (size_t j = i + 1; j < count; ++j)
                if ((petal(i) & petal(j)) != core) return false;
        return true;
    }

    bool choose_sunflower(size_t start, size_t need, std::vector<size_t> &chosen) const {
        if (need == 0) return sunflower_with_newest(chosen);
        size_t prior_count = sets.size() - 1;
        if (prior_count - start < need) return false;
        for (size_t i = start; i + need <= prior_count; ++i) {
            chosen.push_back(i);
            if (choose_sunflower(i + 1, need - 1, chosen)) return true;
            chosen.pop_back();
        }
        return false;
    }

    bool creates_sunflower() const {
        if (sunflower_k > sets.size()) return false;
        std::vector<size_t> chosen;
        chosen.reserve((size_t)sunflower_k - 1);
        return choose_sunflower(0, (size_t)sunflower_k - 1, chosen);
    }

    void add_degree(uint64_t mask) {
        for (uint64_t i = 0; i < n; ++i)
            if (mask & (1ULL << i)) ++degrees[i];
    }

    void remove_degree(uint64_t mask) {
        for (uint64_t i = 0; i < n; ++i)
            if (mask & (1ULL << i)) --degrees[i];
    }

    void add_shadow(uint64_t mask) {
        for (uint64_t i = 0; i < n; ++i)
            if (mask & (1ULL << i)) ++shadow_counts[mask & ~(1ULL << i)];
    }

    void remove_shadow(uint64_t mask) {
        for (uint64_t i = 0; i < n; ++i) {
            if (!(mask & (1ULL << i))) continue;
            uint64_t shadow = mask & ~(1ULL << i);
            auto it = shadow_counts.find(shadow);
            if (--it->second == 0) shadow_counts.erase(it);
        }
    }

    Step push(const Entry *row, Index first, Index) override {
        uint64_t mask = 0;
        for (uint64_t i = 0; i < n; ++i)
            if (row[i] == 1) mask |= 1ULL << i;
        sets.push_back(mask);
        if (op == Op::MaxDegree) add_degree(mask);
        if (op == Op::ShadowSize) add_shadow(mask);
        if (acc.exhausted(first)) return Step::Skip;

        bool good = true;
        switch (op) {
        case Op::IsIntersecting:
            good = pairwise_intersects_newest();
            break;
        case Op::IsAntichain:
            good = incomparable_newest();
            break;
        case Op::IsSunflowerFree:
            good = !creates_sunflower();
            break;
        case Op::IsEkrExtremal: {
            uint64_t rank = std::popcount(mask);
            if (sets.size() == 1) {
                ekr_rank = rank;
                ekr_shape = rank > 0 && 2 * rank <= n && system_size == binomial(n - 1, rank - 1);
            }
            good = ekr_shape && rank == ekr_rank && pairwise_intersects_newest();
            break;
        }
        case Op::IsSpernerExtremal:
            good = system_size == binomial(n, n / 2) && incomparable_newest();
            break;
        case Op::MaxDegree:
        case Op::ShadowSize:
            break;
        }
        return good ? Step::Descend : Step::Skip;
    }

    void pop() override {
        uint64_t mask = sets.back();
        if (op == Op::MaxDegree) remove_degree(mask);
        if (op == Op::ShadowSize) remove_shadow(mask);
        sets.pop_back();
    }

    void leaf(Index index) override {
        switch (op) {
        case Op::MaxDegree:
            acc.integer(index, *std::max_element(degrees.begin(), degrees.end()));
            break;
        case Op::ShadowSize:
            acc.integer(index, shadow_counts.size());
            break;
        default:
            acc.boolean(index, true);
            break;
        }
    }

    void take_all(Index first, Index count) override { acc.booleans(first, count, true); }
    void skip_all(Index first, Index count) override { acc.booleans(first, count, false); }
};

Result<Op> parse_op(const std::string &name) {
    if (name == "is_intersecting") return Result<Op>::success(Op::IsIntersecting);
    if (name == "is_antichain") return Result<Op>::success(Op::IsAntichain);
    if (name == "is_sunflower_free") return Result<Op>::success(Op::IsSunflowerFree);
    if (name == "max_degree") return Result<Op>::success(Op::MaxDegree);
    if (name == "shadow_size") return Result<Op>::success(Op::ShadowSize);
    if (name == "is_ekr_extremal") return Result<Op>::success(Op::IsEkrExtremal);
    if (name == "is_sperner_extremal") return Result<Op>::success(Op::IsSpernerExtremal);
    return Result<Op>::failure(INTERNAL, "unknown set_systems operation " + name);
}

R run(const Request &req) {
    const Family &family = *req.family;
    if (family.kind != Family::Kind::SubsetsOf || !family.child || family.child->kind != Family::Kind::Words ||
        family.child->p != 2)
        return R::failure(INVALID, "set_systems needs a family of the form subsets_of(words(2,n),m)");

    auto parsed = parse_op(req.op);
    if (!parsed.ok) return R::failure(parsed.error.status, parsed.error.message);
    uint64_t sunflower_k = 0;
    if (parsed.value == Op::IsSunflowerFree) {
        auto it = req.int_args.find("k");
        if (it == req.int_args.end()) return R::failure(INVALID, "is_sunflower_free needs integer argument `k`");
        sunflower_k = it->second;
        if (sunflower_k < 2) return R::failure(INVALID, "is_sunflower_free: k must be at least 2");
    }

    auto size_result = family.size();
    if (!size_result.ok) return R::failure(size_result.error.status, size_result.error.message);
    auto tops_result = family.top_count();
    if (!tops_result.ok) return R::failure(tops_result.error.status, tops_result.error.message);
    uint64_t size = size_result.value;
    uint64_t tops = tops_result.value;
    Reduction reduction = parse_reduction(req.reduction);
    Shared shared;
    auto prepared = prepare_all(reduction, size, shared);
    if (!prepared.ok) return R::failure(prepared.error.status, prepared.error.message);

    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, tops));
    std::vector<Walker> walkers;
    walkers.reserve(threads);
    for (uint32_t t = 0; t < threads; ++t)
        walkers.emplace_back(parsed.value, reduction, family.child->n, family.k, sunflower_k, &shared);
    auto statuses = parallel_ranges(tops, threads, [&](uint32_t t, uint64_t begin, uint64_t end) {
        return family.enumerate(walkers[t], begin, end);
    });
    for (const auto &status : statuses)
        if (!status.ok) return R::failure(status.error.status, status.error.message);

    std::vector<Accumulator> accumulators;
    accumulators.reserve(threads);
    for (auto &walker : walkers) accumulators.push_back(std::move(walker.acc));
    return assemble(req, reduction, accumulators, shared);
}

BackendRegistration registration{Backend{
    "set_systems", "generic",
    [] { return true; },
    [](const Request &) { return true; },
    run,
    0}};

} // namespace
} // namespace lk::set_systems
