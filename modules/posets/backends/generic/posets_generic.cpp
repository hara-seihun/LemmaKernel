#include "../../../../runtime/src/reduce.hpp"

#include <cmath>
#include <functional>
#include <limits>
#include <numeric>

namespace lk::posets {
namespace {

using R = Result<std::shared_ptr<Object>>;
constexpr int INVALID = 1;
constexpr int INTERNAL = 4;
constexpr unsigned __int128 OVERFLOW = (unsigned __int128)UINT64_MAX + 1;

struct Relation {
    uint64_t n = 0;
    std::vector<uint8_t> le;
    bool operator()(uint64_t i, uint64_t j) const { return le[i * n + j] != 0; }
};

Result<Relation> explicit_relation(const Matrix &m) {
    if (m.rows != m.cols) return Result<Relation>::failure(INVALID, "explicit member is not a square 0/1 partial order relation");
    Relation r{m.rows, std::vector<uint8_t>(m.rows * m.rows)};
    for (uint64_t i = 0; i < r.n; ++i)
        for (uint64_t j = 0; j < r.n; ++j) {
            Entry x = m.entries[i * r.n + j];
            if (x > 1) return Result<Relation>::failure(INVALID, "explicit member is not a square 0/1 partial order relation");
            r.le[i * r.n + j] = static_cast<uint8_t>(x);
        }
    for (uint64_t i = 0; i < r.n; ++i)
        if (!r(i, i)) return Result<Relation>::failure(INVALID, "relation is not a partial order: it is not reflexive");
    for (uint64_t i = 0; i < r.n; ++i)
        for (uint64_t j = 0; j < r.n; ++j)
            if (i != j && r(i, j) && r(j, i))
                return Result<Relation>::failure(INVALID, "relation is not a partial order: it is not antisymmetric");
    for (uint64_t i = 0; i < r.n; ++i)
        for (uint64_t j = 0; j < r.n; ++j) {
            if (!r(i, j)) continue;
            for (uint64_t k = 0; k < r.n; ++k)
                if (r(j, k) && !r(i, k))
                    return Result<Relation>::failure(INVALID, "relation is not a partial order: it is not transitive");
        }
    return Result<Relation>::success(std::move(r));
}

Result<Relation> subset_relation(const Matrix &m) {
    Relation r{m.rows, std::vector<uint8_t>(m.rows * m.rows)};
    for (uint64_t i = 0; i < r.n; ++i)
        for (uint64_t j = i + 1; j < r.n; ++j) {
            bool same = true;
            for (uint64_t c = 0; c < m.cols; ++c)
                same &= (m.entries[i * m.cols + c] != 0) == (m.entries[j * m.cols + c] != 0);
            if (same) return Result<Relation>::failure(INVALID, "subset member has duplicate supports");
        }
    for (uint64_t i = 0; i < r.n; ++i)
        for (uint64_t j = 0; j < r.n; ++j) {
            bool included = true;
            for (uint64_t c = 0; c < m.cols; ++c)
                if (m.entries[i * m.cols + c] != 0 && m.entries[j * m.cols + c] == 0) { included = false; break; }
            r.le[i * r.n + j] = static_cast<uint8_t>(included);
        }
    return Result<Relation>::success(std::move(r));
}

Result<Relation> divisor_relation(const Matrix &m) {
    uint64_t x = m.entries[0];
    if (x == 0) return Result<Relation>::failure(INVALID, "divisor posets need a positive integer");
    std::vector<uint64_t> low, high;
    for (uint64_t d = 1; d <= x / d; ++d) {
        if (x % d) continue;
        low.push_back(d);
        if (d != x / d) high.push_back(x / d);
    }
    low.insert(low.end(), high.rbegin(), high.rend());
    Relation r{low.size(), std::vector<uint8_t>(low.size() * low.size())};
    for (uint64_t i = 0; i < r.n; ++i)
        for (uint64_t j = 0; j < r.n; ++j) r.le[i * r.n + j] = static_cast<uint8_t>(low[j] % low[i] == 0);
    return Result<Relation>::success(std::move(r));
}

Result<Relation> relation_at(const Family &family, uint64_t index) {
    auto member = family.member(index);
    if (!member.ok) return Result<Relation>::failure(member.error.status, member.error.message);
    switch (family.kind) {
    case Family::Kind::Explicit: return explicit_relation(member.value);
    case Family::Kind::Subsets:
    case Family::Kind::SubsetsOf: return subset_relation(member.value);
    case Family::Kind::Range: return divisor_relation(member.value);
    default: return Result<Relation>::failure(INVALID, "family does not present finite posets");
    }
}

std::vector<uint64_t> predecessor_masks(const Relation &r) {
    std::vector<uint64_t> pred(r.n, 0);
    for (uint64_t x = 0; x < r.n; ++x)
        for (uint64_t y = 0; y < r.n; ++y)
            if (x != y && r(y, x)) pred[x] |= 1ULL << y;
    return pred;
}

Result<uint64_t> linear_extensions(const Relation &r) {
    if (r.n > 24) return Result<uint64_t>::failure(INVALID, "linear_extension_count accepts posets with at most 24 elements");
    auto pred = predecessor_masks(r);
    uint64_t states = 1ULL << r.n;
    std::vector<uint64_t> dp(states, 0);
    dp[0] = 1;
    for (uint64_t chosen = 0; chosen < states; ++chosen) {
        if (!dp[chosen]) continue;
        for (uint64_t x = 0; x < r.n; ++x) {
            uint64_t bit = 1ULL << x;
            if ((chosen & bit) || (pred[x] & ~chosen)) continue;
            unsigned __int128 next = (unsigned __int128)dp[chosen | bit] + dp[chosen];
            if (next > UINT64_MAX) return Result<uint64_t>::failure(INVALID, "linear extension count does not fit in 64 bits");
            dp[chosen | bit] = static_cast<uint64_t>(next);
        }
    }
    return Result<uint64_t>::success(dp.back());
}

uint64_t height(const Relation &r) {
    std::vector<uint64_t> order(r.n), down(r.n), dp(r.n, 1);
    std::iota(order.begin(), order.end(), 0);
    for (uint64_t x = 0; x < r.n; ++x)
        for (uint64_t y = 0; y < r.n; ++y) down[x] += x != y && r(y, x);
    std::sort(order.begin(), order.end(), [&](uint64_t a, uint64_t b) { return down[a] != down[b] ? down[a] < down[b] : a < b; });
    uint64_t answer = 0;
    for (uint64_t x : order) {
        for (uint64_t y : order)
            if (x != y && r(y, x)) dp[x] = std::max(dp[x], dp[y] + 1);
        answer = std::max(answer, dp[x]);
    }
    return answer;
}

bool augment(const Relation &r, uint64_t x, std::vector<int64_t> &matched, std::vector<uint8_t> &seen) {
    for (uint64_t y = 0; y < r.n; ++y) {
        if (x == y || !r(x, y) || seen[y]) continue;
        seen[y] = 1;
        if (matched[y] < 0 || augment(r, static_cast<uint64_t>(matched[y]), matched, seen)) {
            matched[y] = static_cast<int64_t>(x);
            return true;
        }
    }
    return false;
}

uint64_t width(const Relation &r) {
    std::vector<int64_t> matched(r.n, -1);
    uint64_t matching = 0;
    for (uint64_t x = 0; x < r.n; ++x) {
        std::vector<uint8_t> seen(r.n, 0);
        matching += augment(r, x, matched, seen);
    }
    return r.n - matching;
}

int64_t bound(const Relation &r, uint64_t a, uint64_t b, bool upper) {
    for (uint64_t x = 0; x < r.n; ++x) {
        bool candidate = upper ? r(a, x) && r(b, x) : r(x, a) && r(x, b);
        if (!candidate) continue;
        bool best = true;
        for (uint64_t y = 0; y < r.n; ++y) {
            bool other = upper ? r(a, y) && r(b, y) : r(y, a) && r(y, b);
            if (other && !(upper ? r(x, y) : r(y, x))) { best = false; break; }
        }
        if (best) return static_cast<int64_t>(x);
    }
    return -1;
}

bool lattice_tables(const Relation &r, std::vector<int64_t> &meet, std::vector<int64_t> &join) {
    meet.resize(r.n * r.n);
    join.resize(r.n * r.n);
    for (uint64_t a = 0; a < r.n; ++a)
        for (uint64_t b = 0; b < r.n; ++b) {
            meet[a * r.n + b] = bound(r, a, b, false);
            join[a * r.n + b] = bound(r, a, b, true);
            if (meet[a * r.n + b] < 0 || join[a * r.n + b] < 0) return false;
        }
    return true;
}

bool is_lattice(const Relation &r) {
    std::vector<int64_t> meet, join;
    return lattice_tables(r, meet, join);
}

bool is_distributive(const Relation &r) {
    std::vector<int64_t> meet, join;
    if (!lattice_tables(r, meet, join)) return false;
    auto m = [&](uint64_t a, uint64_t b) { return static_cast<uint64_t>(meet[a * r.n + b]); };
    auto j = [&](uint64_t a, uint64_t b) { return static_cast<uint64_t>(join[a * r.n + b]); };
    for (uint64_t x = 0; x < r.n; ++x)
        for (uint64_t y = 0; y < r.n; ++y)
            for (uint64_t z = 0; z < r.n; ++z) {
                if (m(x, j(y, z)) != j(m(x, y), m(x, z))) return false;
                if (j(x, m(y, z)) != m(j(x, y), j(x, z))) return false;
            }
    return true;
}

unsigned __int128 choose_capped(uint64_t n, uint64_t k) {
    if (k > n) return 0;
    k = std::min(k, n - k);
    unsigned __int128 c = 1;
    for (uint64_t i = 1; i <= k; ++i) {
        c = c * (n - k + i) / i;
        if (c >= OVERFLOW) return OVERFLOW;
    }
    return c;
}

Result<uint64_t> order_polynomial(const Relation &r, uint64_t t) {
    if (r.n > 24) return Result<uint64_t>::failure(INVALID, "order_polynomial accepts posets with at most 24 elements");
    if (r.n == 0) return Result<uint64_t>::success(1);
    if (t == 0) return Result<uint64_t>::success(0);
    auto strict_pred = predecessor_masks(r);
    std::vector<uint64_t> pred(r.n);
    for (uint64_t x = 0; x < r.n; ++x) pred[x] = strict_pred[x] | (1ULL << x);
    uint64_t states = 1ULL << r.n;
    std::vector<unsigned __int128> dp(states, 0);
    std::vector<uint8_t> ideals(states, 0);
    for (uint64_t mask = 0; mask < states; ++mask) {
        bool ideal = true;
        for (uint64_t x = 0; x < r.n && ideal; ++x)
            if ((mask & (1ULL << x)) && (pred[x] & mask) != pred[x]) ideal = false;
        ideals[mask] = static_cast<uint8_t>(ideal);
        dp[mask] = ideal;
    }
    uint64_t levels = std::min<uint64_t>(r.n, t);
    std::vector<unsigned __int128> surjective(levels + 1, 0);
    for (uint64_t k = 1; k <= levels; ++k) {
        unsigned __int128 omega = dp.back();
        if (omega >= OVERFLOW) return Result<uint64_t>::failure(INVALID, "order polynomial value does not fit in 64 bits");
        unsigned __int128 covered = 0;
        for (uint64_t j = 1; j < k; ++j) covered += choose_capped(k, j) * surjective[j];
        surjective[k] = omega - covered;
        if (k != levels) {
            for (uint64_t bit = 0; bit < r.n; ++bit) {
                uint64_t b = 1ULL << bit;
                for (uint64_t mask = 0; mask < states; ++mask)
                    if (mask & b) dp[mask] = std::min(OVERFLOW, dp[mask] + dp[mask ^ b]);
            }
            for (uint64_t mask = 0; mask < states; ++mask)
                if (!ideals[mask]) dp[mask] = 0;
        }
    }
    unsigned __int128 answer = 0;
    for (uint64_t k = 1; k <= levels; ++k) {
        unsigned __int128 term = choose_capped(t, k);
        if (term >= OVERFLOW || (surjective[k] && term > UINT64_MAX / surjective[k]))
            return Result<uint64_t>::failure(INVALID, "order polynomial value does not fit in 64 bits");
        answer += term * surjective[k];
        if (answer > UINT64_MAX) return Result<uint64_t>::failure(INVALID, "order polynomial value does not fit in 64 bits");
    }
    return Result<uint64_t>::success(static_cast<uint64_t>(answer));
}

Result<std::vector<int64_t>> mobius(const Relation &r) {
    std::vector<uint64_t> order(r.n), down(r.n);
    std::iota(order.begin(), order.end(), 0);
    for (uint64_t x = 0; x < r.n; ++x)
        for (uint64_t y = 0; y < r.n; ++y) down[x] += x != y && r(y, x);
    std::sort(order.begin(), order.end(), [&](uint64_t a, uint64_t b) { return down[a] != down[b] ? down[a] < down[b] : a < b; });
    std::vector<int64_t> out(r.n * r.n, 0);
    for (uint64_t i = 0; i < r.n; ++i) {
        for (uint64_t j : order) {
            if (!r(i, j)) continue;
            if (i == j) { out[i * r.n + j] = 1; continue; }
            __int128 sum = 0;
            for (uint64_t k = 0; k < r.n; ++k)
                if (k != j && r(i, k) && r(k, j)) sum += out[i * r.n + k];
            __int128 value = -sum;
            if (value < INT64_MIN || value > INT64_MAX)
                return Result<std::vector<int64_t>>::failure(INVALID, "Möbius value does not fit in signed 64 bits");
            out[i * r.n + j] = static_cast<int64_t>(value);
        }
    }
    return Result<std::vector<int64_t>>::success(std::move(out));
}

enum class Op { LinearExtensions, IsLattice, IsDistributive, Width, Height, OrderPolynomial };

R run_reduced(const Request &req, Op op) {
    auto size_result = req.family->size();
    if (!size_result.ok) return R::failure(size_result.error.status, size_result.error.message);
    uint64_t size = size_result.value;
    Reduction reduction = parse_reduction(req.reduction);
    Shared shared;
    auto prepared = prepare_all(reduction, size, shared);
    if (!prepared.ok) return R::failure(prepared.error.status, prepared.error.message);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, size ? size : 1));
    std::vector<Accumulator> accumulators;
    for (uint32_t t = 0; t < threads; ++t) accumulators.emplace_back(reduction, &shared);
    auto statuses = parallel_ranges(size, threads, [&](uint32_t thread, uint64_t begin, uint64_t end) -> Status {
        for (uint64_t index = begin; index < end; ++index) {
            if (accumulators[thread].exhausted(index)) break;
            auto rel = relation_at(*req.family, index);
            if (!rel.ok) return fail(rel.error.status, rel.error.message);
            if (op == Op::IsLattice) accumulators[thread].boolean(index, is_lattice(rel.value));
            else if (op == Op::IsDistributive) accumulators[thread].boolean(index, is_distributive(rel.value));
            else {
                Result<uint64_t> value = op == Op::LinearExtensions ? linear_extensions(rel.value) :
                    op == Op::Width ? Result<uint64_t>::success(width(rel.value)) :
                    op == Op::Height ? Result<uint64_t>::success(height(rel.value)) :
                    order_polynomial(rel.value, req.int_args.at("t"));
                if (!value.ok) return fail(value.error.status, value.error.message);
                accumulators[thread].integer(index, value.value);
            }
        }
        return ok();
    });
    for (const auto &status : statuses)
        if (!status.ok) return R::failure(status.error.status, status.error.message);
    return assemble(req, reduction, accumulators, shared);
}

R run_mobius(const Request &req) {
    auto size_result = req.family->size();
    if (!size_result.ok) return R::failure(size_result.error.status, size_result.error.message);
    uint64_t size = size_result.value;
    std::vector<std::vector<int64_t>> matrices(size);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, size ? size : 1));
    auto statuses = parallel_ranges(size, threads, [&](uint32_t, uint64_t begin, uint64_t end) -> Status {
        for (uint64_t index = begin; index < end; ++index) {
            auto rel = relation_at(*req.family, index);
            if (!rel.ok) return fail(rel.error.status, rel.error.message);
            auto values = mobius(rel.value);
            if (!values.ok) return fail(values.error.status, values.error.message);
            matrices[index] = std::move(values.value);
        }
        return ok();
    });
    for (const auto &status : statuses)
        if (!status.ok) return R::failure(status.error.status, status.error.message);
    auto values = std::make_shared<SignedMatrices>();
    values->count = size;
    values->offsets.push_back(0);
    for (auto &matrix : matrices) {
        values->entries.insert(values->entries.end(), matrix.begin(), matrix.end());
        values->offsets.push_back(values->entries.size());
    }
    auto object = std::make_shared<Object>();
    object->kind = "posets.mobius";
    object->signed_matrices = values;
    return R::success(object);
}

R run(const Request &req) {
    if (req.op == "mobius_function") return run_mobius(req);
    if (req.op == "linear_extension_count") return run_reduced(req, Op::LinearExtensions);
    if (req.op == "is_lattice") return run_reduced(req, Op::IsLattice);
    if (req.op == "is_distributive") return run_reduced(req, Op::IsDistributive);
    if (req.op == "width") return run_reduced(req, Op::Width);
    if (req.op == "height") return run_reduced(req, Op::Height);
    if (req.op == "order_polynomial") return run_reduced(req, Op::OrderPolynomial);
    return R::failure(INTERNAL, "unknown posets operation " + req.op);
}

BackendRegistration registration{Backend{
    "posets", "generic",
    [] { return true; },
    [](const Request &) { return true; },
    run,
    0}};

} // namespace
} // namespace lk::posets
