/* burnside generic backend: orbit counts from permutation cycle types.
 *
 * The acted-on family is never enumerated. A fixed k-subset is a union of permutation cycles,
 * so its count is a coefficient of product (1 + x^cycle_length). A fixed word is constant on
 * each cycle, so an alphabet of size q gives q^number_of_cycles fixed words. */
#include "../../../../runtime/src/reduce.hpp"

#include <map>

namespace lk::burnside {
namespace {

using R = Result<std::shared_ptr<Object>>;
constexpr int INVALID = 1;

Result<uint64_t> action_degree(const Family &family) {
    if (family.kind == Family::Kind::Subsets || family.kind == Family::Kind::SubsetsOf)
        return Result<uint64_t>::success(family.data->count);
    if (family.kind == Family::Kind::Words)
        return Result<uint64_t>::success(family.n);
    return Result<uint64_t>::failure(INVALID, "burnside operations are defined on subsets, subsets_of, and words families only");
}

Result<std::shared_ptr<Matrix>> permutation_arg(const Request &req, const char *name) {
    auto it = req.handle_args.find(name);
    if (it == req.handle_args.end() || !it->second->matrix || it->second->matrix->p != 0)
        return Result<std::shared_ptr<Matrix>>::failure(INVALID, std::string("`") + name + "` must be a permutation batch");
    return Result<std::shared_ptr<Matrix>>::success(it->second->matrix);
}

std::vector<uint64_t> cycle_type(const Entry *permutation, uint64_t n) {
    std::vector<uint64_t> counts(n, 0);
    std::vector<uint8_t> seen(n, 0);
    for (uint64_t i = 0; i < n; ++i) {
        if (seen[i]) continue;
        uint64_t length = 0;
        for (uint64_t j = i; !seen[j]; j = permutation[j]) {
            seen[j] = 1;
            ++length;
        }
        ++counts[length - 1];
    }
    return counts;
}

Result<uint64_t> fixed_subsets(const Family &family, const std::vector<uint64_t> &counts) {
    uint64_t k = family.k;
    std::vector<uint64_t> poly(k + 1, 0);
    poly[0] = 1;
    for (uint64_t length = 1; length <= counts.size(); ++length) {
        for (uint64_t copy = 0; copy < counts[length - 1]; ++copy) {
            if (length > k) continue;
            for (uint64_t d = k; d >= length; --d) {
                unsigned __int128 sum = (unsigned __int128)poly[d] + poly[d - length];
                if (sum > UINT64_MAX)
                    return Result<uint64_t>::failure(INVALID, "fixed-point count does not fit in 64 bits");
                poly[d] = (uint64_t)sum;
            }
        }
    }
    return Result<uint64_t>::success(poly[k]);
}

Result<uint64_t> fixed_words(const Family &family, const std::vector<uint64_t> &counts) {
    uint64_t cycles = 0;
    for (uint64_t c : counts) cycles += c;
    uint64_t value = 1;
    for (uint64_t i = 0; i < cycles; ++i) {
        unsigned __int128 product = (unsigned __int128)value * family.p;
        if (product > UINT64_MAX)
            return Result<uint64_t>::failure(INVALID, "fixed-point count does not fit in 64 bits");
        value = (uint64_t)product;
    }
    return Result<uint64_t>::success(value);
}

Result<uint64_t> fixed_count(const Family &family, const Entry *permutation, uint64_t n) {
    auto counts = cycle_type(permutation, n);
    if (family.kind == Family::Kind::Words) return fixed_words(family, counts);
    return fixed_subsets(family, counts);
}

R counts_object(uint64_t value) {
    auto object = std::make_shared<Object>();
    object->kind = "burnside.counts";
    object->integers = std::make_shared<Integers>(Integers{{value}});
    return R::success(object);
}

Result<std::vector<Entry>> group_elements(const Request &req, uint64_t degree) {
    auto generators = permutation_arg(req, "group");
    if (!generators.ok) return Result<std::vector<Entry>>::failure(generators.error.status, "`group` must be a permutation group");
    if (generators.value->count == 0)
        return Result<std::vector<Entry>>::failure(INVALID, "group needs at least one generator");
    if (generators.value->cols != degree)
        return Result<std::vector<Entry>>::failure(
            INVALID, "permutations have " + std::to_string(generators.value->cols) +
                         " points but the family has " + std::to_string(degree) + " positions");
    return permutation_closure(*generators.value, 1ULL << 26);
}

R run_fixed_count(const Request &req, uint64_t degree) {
    auto g = permutation_arg(req, "g");
    if (!g.ok) return R::failure(g.error.status, g.error.message);
    if (g.value->count != 1) return R::failure(INVALID, "`g` must be a single permutation");
    if (g.value->cols != degree)
        return R::failure(INVALID, "permutation has " + std::to_string(g.value->cols) +
                                       " points but the family has " + std::to_string(degree) + " positions");
    auto value = fixed_count(*req.family, g.value->entries.data(), degree);
    if (!value.ok) return R::failure(value.error.status, value.error.message);
    return counts_object(value.value);
}

R run_orbit_count(const Request &req, uint64_t degree) {
    auto elements = group_elements(req, degree);
    if (!elements.ok) return R::failure(elements.error.status, elements.error.message);
    uint64_t order = elements.value.size() / degree;
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, order));
    std::vector<unsigned __int128> sums(threads, 0);
    auto statuses = parallel_ranges(order, threads, [&](uint32_t thread, uint64_t begin, uint64_t end) -> Status {
        for (uint64_t i = begin; i < end; ++i) {
            auto fixed = fixed_count(*req.family, elements.value.data() + i * degree, degree);
            if (!fixed.ok) return fail(fixed.error.status, fixed.error.message);
            sums[thread] += fixed.value;
        }
        return ok();
    });
    for (const auto &status : statuses)
        if (!status.ok) return R::failure(status.error.status, status.error.message);
    unsigned __int128 total = 0;
    for (auto sum : sums) total += sum;
    if (total % order != 0) return R::failure(4, "Burnside sum is not divisible by the group order");
    total /= order;
    if (total > UINT64_MAX) return R::failure(INVALID, "orbit count does not fit in 64 bits");
    return counts_object((uint64_t)total);
}

R run_cycle_index(const Request &req, uint64_t degree) {
    auto elements = group_elements(req, degree);
    if (!elements.ok) return R::failure(elements.error.status, elements.error.message);
    uint64_t order = elements.value.size() / degree;
    std::map<std::vector<uint64_t>, uint64_t> terms;
    for (uint64_t i = 0; i < order; ++i)
        ++terms[cycle_type(elements.value.data() + i * degree, degree)];

    auto index = std::make_shared<CycleIndex>();
    index->degree = degree;
    index->denominator = order;
    for (const auto &[cycles, multiplicity] : terms) {
        index->multiplicities.push_back(multiplicity);
        index->cycles.insert(index->cycles.end(), cycles.begin(), cycles.end());
    }
    auto object = std::make_shared<Object>();
    object->kind = "burnside.cycle_index";
    object->cycle_index = index;
    return R::success(object);
}

R run(const Request &req) {
    if (req.reduction != "all") return R::failure(INVALID, req.op + " values only reduce with `all`");
    auto degree = action_degree(*req.family);
    if (!degree.ok) return R::failure(degree.error.status, degree.error.message);
    if (req.op == "fixed_count") return run_fixed_count(req, degree.value);
    if (req.op == "orbit_count") return run_orbit_count(req, degree.value);
    if (req.op == "cycle_index") return run_cycle_index(req, degree.value);
    return R::failure(4, "unknown burnside operation " + req.op);
}

BackendRegistration registration{Backend{
    "burnside", "generic",
    [] { return true; },
    [](const Request &) { return true; },
    run,
    0}};

} // namespace
} // namespace lk::burnside
