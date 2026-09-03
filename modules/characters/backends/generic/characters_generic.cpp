/* Exact ordinary characters of finite abelian permutation groups.
 *
 * The group is closed in the canonical lexicographic order used by group_elements. A finite
 * abelian group is split into cyclic direct factors. Enumerating the dual factors then gives all
 * irreducible characters as exact powers of a primitive root whose order is the group exponent.
 */
#include "../../../../runtime/src/registry.hpp"

#include <algorithm>
#include <numeric>
#include <unordered_map>

namespace lk::characters {
namespace {

using R = Result<std::shared_ptr<Object>>;
constexpr int INVALID = 1;
constexpr uint64_t MAX_ORDER = 4096;

struct Group {
    uint64_t degree = 0;
    uint64_t order = 0;
    uint64_t identity = 0;
    uint64_t exponent = 1;
    std::vector<Entry> elements;
    std::vector<uint32_t> multiplication;
    std::vector<uint32_t> element_orders;

    uint32_t mul(uint64_t a, uint64_t b) const { return multiplication[a * order + b]; }
};

std::string permutation_key(const Entry *permutation, uint64_t degree) {
    return std::string(reinterpret_cast<const char *>(permutation), degree * sizeof(Entry));
}

Result<Group> make_group(const Matrix &generators) {
    using G = Result<Group>;
    auto closure = permutation_closure(generators, MAX_ORDER);
    if (!closure.ok) return G::failure(closure.error.status, closure.error.message);

    Group group;
    group.degree = generators.cols;
    group.elements = std::move(closure.value);
    group.order = group.elements.size() / group.degree;
    std::unordered_map<std::string, uint32_t> index;
    for (uint64_t i = 0; i < group.order; ++i)
        index.emplace(permutation_key(group.elements.data() + i * group.degree, group.degree), static_cast<uint32_t>(i));

    std::vector<Entry> product(group.degree);
    for (uint64_t point = 0; point < group.degree; ++point) product[point] = static_cast<Entry>(point);
    auto identity = index.find(permutation_key(product.data(), group.degree));
    if (identity == index.end()) return G::failure(4, "group closure omitted the identity");
    group.identity = identity->second;

    group.multiplication.resize(group.order * group.order);
    for (uint64_t a = 0; a < group.order; ++a) {
        const Entry *left = group.elements.data() + a * group.degree;
        for (uint64_t b = 0; b < group.order; ++b) {
            const Entry *right = group.elements.data() + b * group.degree;
            for (uint64_t point = 0; point < group.degree; ++point) product[point] = right[left[point]];
            auto found = index.find(permutation_key(product.data(), group.degree));
            if (found == index.end()) return G::failure(4, "group multiplication escaped its closure");
            group.multiplication[a * group.order + b] = found->second;
        }
    }
    for (uint64_t a = 0; a < group.order; ++a)
        for (uint64_t b = a + 1; b < group.order; ++b)
            if (group.mul(a, b) != group.mul(b, a))
                return G::failure(INVALID, "character_table generic currently accepts abelian groups only");

    group.element_orders.resize(group.order);
    for (uint64_t a = 0; a < group.order; ++a) {
        uint32_t power = static_cast<uint32_t>(group.identity);
        uint64_t element_order = 0;
        do {
            power = group.mul(power, a);
            ++element_order;
            if (element_order > group.order) return G::failure(4, "element order does not divide the group order");
        } while (power != group.identity);
        group.element_orders[a] = static_cast<uint32_t>(element_order);
        uint64_t divisor = std::gcd(group.exponent, element_order);
        unsigned __int128 next = static_cast<unsigned __int128>(group.exponent / divisor) * element_order;
        if (next > UINT32_MAX) return G::failure(INVALID, "group exponent does not fit in 32 bits");
        group.exponent = static_cast<uint64_t>(next);
    }
    return G::success(std::move(group));
}

struct Decomposition {
    std::vector<uint32_t> factors;
    std::vector<std::vector<uint32_t>> coordinates;
};

bool decompose_search(const Group &group, const std::vector<uint32_t> &subgroup,
                      const std::vector<std::vector<uint32_t>> &coordinates,
                      std::vector<uint32_t> factors, Decomposition &answer) {
    if (subgroup.size() == group.order) {
        answer = {std::move(factors), coordinates};
        return true;
    }
    std::vector<uint8_t> in_subgroup(group.order, 0);
    for (uint32_t x : subgroup) in_subgroup[x] = 1;
    std::vector<uint32_t> candidates;
    for (uint32_t x = 0; x < group.order; ++x)
        if (!in_subgroup[x]) candidates.push_back(x);
    std::sort(candidates.begin(), candidates.end(), [&](uint32_t a, uint32_t b) {
        if (group.element_orders[a] != group.element_orders[b]) return group.element_orders[a] > group.element_orders[b];
        return a < b;
    });

    for (uint32_t generator : candidates) {
        uint32_t cyclic_order = group.element_orders[generator];
        std::vector<uint32_t> powers(cyclic_order, static_cast<uint32_t>(group.identity));
        for (uint32_t k = 1; k < cyclic_order; ++k) powers[k] = group.mul(powers[k - 1], generator);
        bool disjoint = true;
        for (uint32_t k = 1; k < cyclic_order; ++k)
            if (in_subgroup[powers[k]]) { disjoint = false; break; }
        if (!disjoint) continue;

        std::vector<uint32_t> extended;
        extended.reserve(subgroup.size() * cyclic_order);
        auto next_coordinates = coordinates;
        std::vector<uint8_t> seen(group.order, 0);
        bool direct = true;
        for (uint32_t h : subgroup) {
            for (uint32_t k = 0; k < cyclic_order; ++k) {
                uint32_t x = group.mul(h, powers[k]);
                if (seen[x]) { direct = false; break; }
                seen[x] = 1;
                extended.push_back(x);
                next_coordinates[x] = coordinates[h];
                next_coordinates[x].push_back(k);
            }
            if (!direct) break;
        }
        if (!direct) continue;
        auto next_factors = factors;
        next_factors.push_back(cyclic_order);
        if (decompose_search(group, extended, next_coordinates, std::move(next_factors), answer)) return true;
    }
    return false;
}

Result<Decomposition> decompose(const Group &group) {
    std::vector<std::vector<uint32_t>> coordinates(group.order);
    Decomposition result;
    if (!decompose_search(group, {static_cast<uint32_t>(group.identity)}, coordinates, {}, result))
        return Result<Decomposition>::failure(4, "could not split the finite abelian group into cyclic factors");
    return Result<Decomposition>::success(std::move(result));
}

std::vector<std::vector<Entry>> character_rows(const Group &group, const Decomposition &decomposition) {
    std::vector<std::vector<Entry>> rows;
    std::vector<uint32_t> dual(decomposition.factors.size(), 0);
    bool done = false;
    while (!done) {
        std::vector<Entry> row(group.order, 0);
        for (uint64_t element = 0; element < group.order; ++element) {
            unsigned __int128 exponent = 0;
            for (uint64_t i = 0; i < dual.size(); ++i)
                exponent += static_cast<unsigned __int128>(dual[i]) * decomposition.coordinates[element][i] *
                            (group.exponent / decomposition.factors[i]);
            row[element] = static_cast<Entry>(exponent % group.exponent);
        }
        rows.push_back(std::move(row));
        if (dual.empty()) break;
        for (uint64_t i = dual.size(); i-- > 0;) {
            if (++dual[i] < decomposition.factors[i]) break;
            dual[i] = 0;
            if (i == 0) done = true;
        }
    }
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    return rows;
}

using Computation = std::pair<Group, std::vector<std::vector<Entry>>>;

Result<Computation> compute_generators(const Matrix &generators) {
    using Out = Result<Computation>;
    auto group = make_group(generators);
    if (!group.ok) return Out::failure(group.error.status, group.error.message);
    auto decomposition = decompose(group.value);
    if (!decomposition.ok) return Out::failure(decomposition.error.status, decomposition.error.message);
    auto rows = character_rows(group.value, decomposition.value);
    if (rows.size() != group.value.order)
        return Out::failure(4, "dual-group enumeration produced the wrong number of characters");
    return Out::success({std::move(group.value), std::move(rows)});
}

Result<Computation> compute(const Family &family) {
    if (family.kind != Family::Kind::GroupElements)
        return Result<Computation>::failure(INVALID, "characters operations are defined on group_elements families only");
    return compute_generators(*family.data);
}

R run_table(const Group &group, const std::vector<std::vector<Entry>> &rows) {
    auto table = std::make_shared<CharacterTable>();
    table->order = group.order;
    table->classes = group.order;
    table->conductor = group.exponent;
    for (uint64_t i = 0; i < group.order; ++i) {
        table->representatives.push_back(i);
        table->class_sizes.push_back(1);
        table->degrees.push_back(1);
    }
    for (const auto &row : rows) table->spectra.insert(table->spectra.end(), row.begin(), row.end());
    auto object = std::make_shared<Object>();
    object->kind = "characters.table";
    object->character_table = std::move(table);
    return R::success(std::move(object));
}

R run_indicators(const Group &group, const std::vector<std::vector<Entry>> &rows) {
    auto indicators = std::make_shared<CharacterIndicators>();
    for (const auto &row : rows) {
        bool real = std::all_of(row.begin(), row.end(), [&](Entry exponent) {
            return (2ULL * exponent) % group.exponent == 0;
        });
        indicators->values.push_back(real ? 1 : 0);
    }
    auto object = std::make_shared<Object>();
    object->kind = "characters.indicators";
    object->character_indicators = std::move(indicators);
    return R::success(std::move(object));
}

bool same_restriction(const Computation &ambient, const Computation &subgroup,
                      const std::vector<Entry> &ambient_row, const std::vector<Entry> &subgroup_row) {
    const Group &large = ambient.first;
    const Group &small = subgroup.first;
    if (large.degree != small.degree || large.exponent % small.exponent != 0) return false;
    std::unordered_map<std::string, uint32_t> ambient_index;
    for (uint32_t i = 0; i < large.order; ++i)
        ambient_index.emplace(permutation_key(large.elements.data() + i * large.degree, large.degree), i);
    uint64_t scale = large.exponent / small.exponent;
    for (uint64_t i = 0; i < small.order; ++i) {
        auto found = ambient_index.find(permutation_key(small.elements.data() + i * small.degree, small.degree));
        if (found == ambient_index.end()) return false;
        if (ambient_row[found->second] != subgroup_row[i] * scale % large.exponent) return false;
    }
    return true;
}

R run_multiplicities(const Request &request, const Computation &ambient) {
    auto argument = request.handle_args.find("subgroup");
    if (argument == request.handle_args.end() || !argument->second->matrix || argument->second->matrix->p != 0)
        return R::failure(INVALID, "`subgroup` must be a permutation group");
    auto subgroup = compute_generators(*argument->second->matrix);
    if (!subgroup.ok) return R::failure(subgroup.error.status, subgroup.error.message);
    if (ambient.first.degree != subgroup.value.first.degree)
        return R::failure(INVALID, "subgroup permutations must have the same degree as the ambient group");
    std::unordered_map<std::string, uint32_t> ambient_elements;
    for (uint32_t i = 0; i < ambient.first.order; ++i)
        ambient_elements.emplace(permutation_key(ambient.first.elements.data() + i * ambient.first.degree, ambient.first.degree), i);
    for (uint64_t i = 0; i < subgroup.value.first.order; ++i)
        if (!ambient_elements.count(permutation_key(subgroup.value.first.elements.data() + i * ambient.first.degree, ambient.first.degree)))
            return R::failure(INVALID, "`subgroup` is not contained in the ambient group");

    auto character = request.int_args.find("character");
    if (character == request.int_args.end()) return R::failure(INVALID, "missing `character` row index");
    const auto &source = request.op == "restrict" ? ambient.second : subgroup.value.second;
    const auto &targets = request.op == "restrict" ? subgroup.value.second : ambient.second;
    if (character->second >= source.size()) return R::failure(INVALID, "character index is outside the table");
    std::vector<uint64_t> values;
    values.reserve(targets.size());
    for (const auto &target : targets) {
        bool matches = request.op == "restrict"
            ? same_restriction(ambient, subgroup.value, source[character->second], target)
            : same_restriction(ambient, subgroup.value, target, source[character->second]);
        values.push_back(matches ? 1 : 0);
    }
    auto object = std::make_shared<Object>();
    object->kind = "characters.multiplicities";
    object->integers = std::make_shared<Integers>(Integers{std::move(values)});
    return R::success(std::move(object));
}

R run(const Request &request) {
    if (request.reduction != "all") return R::failure(INVALID, request.op + " values only reduce with `all`");
    auto result = compute(*request.family);
    if (!result.ok) return R::failure(result.error.status, result.error.message);
    if (request.op == "character_table") return run_table(result.value.first, result.value.second);
    if (request.op == "frobenius_schur") return run_indicators(result.value.first, result.value.second);
    if (request.op == "restrict" || request.op == "induce") return run_multiplicities(request, result.value);
    return R::failure(4, "unknown characters operation " + request.op);
}

BackendRegistration registration{Backend{
    "characters", "generic",
    [] { return true; },
    [](const Request &) { return true; },
    run,
    0}};

} // namespace
} // namespace lk::characters
