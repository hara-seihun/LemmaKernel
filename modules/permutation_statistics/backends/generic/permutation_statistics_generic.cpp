#include "../../../../runtime/src/reduce.hpp"

#include <limits>

namespace lk::permutation_statistics {
namespace {

using R = Result<std::shared_ptr<Object>>;
constexpr int INVALID = 1;

enum class Op { Inversions, Descents, MajorIndex, CycleType, PatternAvoids, BruhatLeq };

Result<Op> parse_op(const std::string &name) {
    if (name == "inversions") return Result<Op>::success(Op::Inversions);
    if (name == "descents") return Result<Op>::success(Op::Descents);
    if (name == "major_index") return Result<Op>::success(Op::MajorIndex);
    if (name == "cycle_type") return Result<Op>::success(Op::CycleType);
    if (name == "pattern_avoids") return Result<Op>::success(Op::PatternAvoids);
    if (name == "bruhat_leq") return Result<Op>::success(Op::BruhatLeq);
    return Result<Op>::failure(4, "unknown permutation_statistics operation " + name);
}

bool valid_permutation(const Entry *perm, uint64_t n, std::vector<uint8_t> &seen) {
    seen.assign(n, 0);
    for (uint64_t i = 0; i < n; ++i) {
        uint64_t value = perm[i];
        if (value >= n || seen[value]) return false;
        seen[value] = 1;
    }
    return true;
}

uint64_t inversion_count(const Entry *perm, uint64_t n) {
    uint64_t count = 0;
    for (uint64_t i = 0; i < n; ++i)
        for (uint64_t j = i + 1; j < n; ++j) count += perm[i] > perm[j];
    return count;
}

uint64_t descent_count(const Entry *perm, uint64_t n) {
    uint64_t count = 0;
    for (uint64_t i = 1; i < n; ++i) count += perm[i - 1] > perm[i];
    return count;
}

uint64_t major_index(const Entry *perm, uint64_t n) {
    uint64_t value = 0;
    for (uint64_t i = 1; i < n; ++i)
        if (perm[i - 1] > perm[i]) value += i;
    return value;
}

struct PartitionCounter {
    uint64_t n;
    std::vector<uint64_t> counts;
    std::vector<uint8_t> overflow;

    explicit PartitionCounter(uint64_t degree)
        : n(degree), counts((degree + 1) * (degree + 1), 0), overflow(counts.size(), 0) {
        for (uint64_t maximum = 0; maximum <= n; ++maximum) at(0, maximum) = 1;
        for (uint64_t remaining = 1; remaining <= n; ++remaining) {
            for (uint64_t maximum = 1; maximum <= n; ++maximum) {
                uint64_t a = at(remaining, maximum - 1);
                uint64_t b = maximum <= remaining ? at(remaining - maximum, maximum) : 0;
                bool bad = over(remaining, maximum - 1) ||
                           (maximum <= remaining && over(remaining - maximum, maximum));
                unsigned __int128 sum = (unsigned __int128)a + b;
                if (bad || sum > UINT64_MAX) {
                    at(remaining, maximum) = UINT64_MAX;
                    over(remaining, maximum) = 1;
                } else {
                    at(remaining, maximum) = (uint64_t)sum;
                }
            }
        }
    }

    uint64_t &at(uint64_t remaining, uint64_t maximum) {
        return counts[remaining * (n + 1) + maximum];
    }
    uint64_t at(uint64_t remaining, uint64_t maximum) const {
        return counts[remaining * (n + 1) + std::min(maximum, n)];
    }
    uint8_t &over(uint64_t remaining, uint64_t maximum) {
        return overflow[remaining * (n + 1) + maximum];
    }
    bool over(uint64_t remaining, uint64_t maximum) const {
        return overflow[remaining * (n + 1) + std::min(maximum, n)] != 0;
    }
    bool fits() const { return !over(n, n); }

    uint64_t rank(const std::vector<uint64_t> &parts) const {
        uint64_t remaining = n, maximum = n, rank = 0;
        for (uint64_t part : parts) {
            uint64_t top = std::min(remaining, maximum);
            for (uint64_t candidate = top; candidate > part; --candidate)
                rank += at(remaining - candidate, candidate);
            remaining -= part;
            maximum = part;
        }
        return rank;
    }
};

std::vector<uint64_t> cycle_lengths(const Entry *perm, uint64_t n, std::vector<uint8_t> &seen) {
    seen.assign(n, 0);
    std::vector<uint64_t> lengths;
    for (uint64_t start = 0; start < n; ++start) {
        if (seen[start]) continue;
        uint64_t current = start, length = 0;
        do {
            seen[current] = 1;
            current = perm[current];
            ++length;
        } while (current != start);
        lengths.push_back(length);
    }
    std::sort(lengths.begin(), lengths.end(), std::greater<uint64_t>());
    return lengths;
}

bool contains_pattern_search(const Entry *perm, uint64_t n, const Entry *pattern, uint64_t k,
                             uint64_t next, std::vector<Entry> &selected) {
    if (selected.size() == k) return true;
    uint64_t need = k - selected.size();
    if (n - std::min(n, next) < need) return false;
    uint64_t pattern_pos = selected.size();
    for (uint64_t position = next; position + need <= n; ++position) {
        Entry value = perm[position];
        bool compatible = true;
        for (uint64_t j = 0; j < selected.size(); ++j) {
            if ((selected[j] < value) != (pattern[j] < pattern[pattern_pos])) {
                compatible = false;
                break;
            }
        }
        if (!compatible) continue;
        selected.push_back(value);
        if (contains_pattern_search(perm, n, pattern, k, position + 1, selected)) return true;
        selected.pop_back();
    }
    return false;
}

bool avoids_patterns(const Entry *perm, uint64_t n, const Matrix &patterns, std::vector<Entry> &selected) {
    for (uint64_t i = 0; i < patterns.count; ++i) {
        selected.clear();
        if (contains_pattern_search(perm, n, patterns.at(i), patterns.cols, 0, selected)) return false;
    }
    return true;
}

bool bruhat_leq(const Entry *lower, const Entry *upper, uint64_t n,
                std::vector<uint64_t> &lower_counts, std::vector<uint64_t> &upper_counts) {
    lower_counts.assign(n + 1, 0);
    upper_counts.assign(n + 1, 0);
    for (uint64_t prefix = 1; prefix <= n; ++prefix) {
        ++lower_counts[lower[prefix - 1]];
        ++upper_counts[upper[prefix - 1]];
        uint64_t lower_tail = 0, upper_tail = 0;
        for (uint64_t threshold = n + 1; threshold-- > 0;) {
            lower_tail += lower_counts[threshold];
            upper_tail += upper_counts[threshold];
            if (lower_tail > upper_tail) return false;
        }
    }
    return true;
}

R run(const Request &req) {
    auto parsed = parse_op(req.op);
    if (!parsed.ok) return R::failure(parsed.error.status, parsed.error.message);
    Op op = parsed.value;
    const Family &family = *req.family;
    if ((family.kind != Family::Kind::Explicit && family.kind != Family::Kind::GroupElements) ||
        family.prime() != 0 || family.rows() != 1)
        return R::failure(INVALID, "permutation_statistics needs an explicit or group_elements permutation family");

    auto size_result = family.size();
    if (!size_result.ok) return R::failure(size_result.error.status, size_result.error.message);
    uint64_t size = size_result.value, n = family.cols();

    std::shared_ptr<Matrix> patterns;
    const Entry *upper = nullptr;
    if (op == Op::PatternAvoids) {
        auto it = req.handle_args.find("patterns");
        if (it == req.handle_args.end() || !it->second->matrix || it->second->matrix->p != 0)
            return R::failure(INVALID, "patterns must be an orbits.perms batch");
        patterns = it->second->matrix;
    }
    if (op == Op::BruhatLeq) {
        auto it = req.handle_args.find("upper");
        if (it == req.handle_args.end() || !it->second->matrix || it->second->matrix->p != 0)
            return R::failure(INVALID, "upper must be an orbits.perms batch");
        const Matrix &bound = *it->second->matrix;
        if (bound.count != 1) return R::failure(INVALID, "bruhat_leq needs exactly one upper permutation");
        if (bound.cols != n) return R::failure(INVALID, "bruhat_leq upper must have the same degree as the family");
        upper = bound.entries.data();
    }

    std::unique_ptr<PartitionCounter> partitions;
    if (op == Op::CycleType) {
        if (n > 512) return R::failure(INVALID, "cycle_type partition code does not fit in 64 bits");
        partitions = std::make_unique<PartitionCounter>(n);
        if (!partitions->fits())
            return R::failure(INVALID, "cycle_type partition code does not fit in 64 bits");
    }

    Reduction reduction = parse_reduction(req.reduction);
    Shared shared;
    auto prepared = prepare_all(reduction, size, shared);
    if (!prepared.ok) return R::failure(prepared.error.status, prepared.error.message);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, size ? size : 1));
    std::vector<Accumulator> accumulators;
    std::vector<Matrix> members(threads);
    for (uint32_t thread = 0; thread < threads; ++thread) accumulators.emplace_back(reduction, &shared);

    auto statuses = parallel_ranges(size, threads, [&](uint32_t thread, uint64_t begin, uint64_t end) -> Status {
        std::vector<uint8_t> seen;
        std::vector<Entry> selected;
        std::vector<uint64_t> lower_counts, upper_counts;
        for (uint64_t index = begin; index < end; ++index) {
            if ((op == Op::PatternAvoids || op == Op::BruhatLeq) && accumulators[thread].exhausted(index)) break;
            auto status = family.member_into(index, members[thread]);
            if (!status.ok) return status;
            const Entry *perm = members[thread].entries.data();
            if (!valid_permutation(perm, n, seen))
                return fail(INVALID, "permutation family contains an invalid permutation");
            switch (op) {
            case Op::Inversions:
                accumulators[thread].integer(index, inversion_count(perm, n));
                break;
            case Op::Descents:
                accumulators[thread].integer(index, descent_count(perm, n));
                break;
            case Op::MajorIndex:
                accumulators[thread].integer(index, major_index(perm, n));
                break;
            case Op::CycleType:
                accumulators[thread].integer(index, partitions->rank(cycle_lengths(perm, n, seen)));
                break;
            case Op::PatternAvoids:
                accumulators[thread].boolean(index, avoids_patterns(perm, n, *patterns, selected));
                break;
            case Op::BruhatLeq:
                accumulators[thread].boolean(index, bruhat_leq(perm, upper, n, lower_counts, upper_counts));
                break;
            }
        }
        return ok();
    });
    for (const auto &status : statuses)
        if (!status.ok) return R::failure(status.error.status, status.error.message);
    return assemble(req, reduction, accumulators, shared);
}

BackendRegistration registration{Backend{
    "permutation_statistics", "generic",
    [] { return true; },
    [](const Request &) { return true; },
    run,
    0}};

} // namespace
} // namespace lk::permutation_statistics
