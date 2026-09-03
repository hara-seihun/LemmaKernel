#include "../../../../runtime/src/graph.hpp"
#include "../../../../runtime/src/group_table.hpp"
#include "../../../../runtime/src/reduce.hpp"

#include <functional>
#include <unordered_map>
#include <unordered_set>

namespace lk::cayley_iso {
namespace {

using R = Result<std::shared_ptr<Object>>;
constexpr int INVALID = 1;
constexpr int INTERNAL = 4;

struct EntryVectorHash {
    size_t operator()(const std::vector<Entry> &values) const {
        uint64_t hash = 1469598103934665603ULL;
        for (Entry value : values) {
            hash ^= value;
            hash *= 1099511628211ULL;
        }
        return (size_t)hash;
    }
};

struct GroupModel {
    const Entry *table;
    uint64_t order;
    uint64_t identity;
    std::vector<uint32_t> inverse;
    std::vector<std::vector<uint32_t>> inverse_atoms;

    Entry mul(uint64_t a, uint64_t b) const { return table[a * order + b]; }

    static Result<GroupModel> build(const Entry *table, uint64_t order) {
        using G = Result<GroupModel>;
        uint64_t identity = order;
        for (uint64_t candidate = 0; candidate < order; ++candidate) {
            bool valid = true;
            for (uint64_t x = 0; x < order; ++x)
                if (table[candidate * order + x] != x || table[x * order + candidate] != x) {
                    valid = false;
                    break;
                }
            if (valid) {
                identity = candidate;
                break;
            }
        }
        if (identity == order) return G::failure(INTERNAL, "validated group table has no identity");

        GroupModel group{table, order, identity, std::vector<uint32_t>(order), {}};
        std::vector<uint8_t> used(order, 0);
        used[identity] = 1;
        for (uint64_t x = 0; x < order; ++x) {
            uint64_t inverse = order;
            for (uint64_t y = 0; y < order; ++y)
                if (group.mul(x, y) == identity && group.mul(y, x) == identity) {
                    inverse = y;
                    break;
                }
            if (inverse == order)
                return G::failure(INTERNAL, "validated group table has an element without an inverse");
            group.inverse[x] = (uint32_t)inverse;
        }
        for (uint32_t x = 0; x < order; ++x) {
            if (used[x]) continue;
            uint32_t y = group.inverse[x];
            used[x] = used[y] = 1;
            if (x == y) group.inverse_atoms.push_back({x});
            else group.inverse_atoms.push_back({std::min(x, y), std::max(x, y)});
        }
        std::sort(group.inverse_atoms.begin(), group.inverse_atoms.end());
        return G::success(std::move(group));
    }
};

std::string canonical_aut_key(const std::vector<uint8_t> &selected,
                              const std::vector<std::vector<Entry>> &automorphisms) {
    std::string best;
    std::string image(selected.size(), '\0');
    for (const auto &automorphism : automorphisms) {
        std::fill(image.begin(), image.end(), '\0');
        for (uint64_t x = 0; x < selected.size(); ++x)
            if (selected[x]) image[automorphism[x]] = '\1';
        if (best.empty() || image < best) best = image;
    }
    return best;
}

std::vector<Entry> canonical_graph_key(const GroupModel &group, const std::vector<uint8_t> &selected) {
    std::vector<Entry> adjacency(group.order * group.order, 0);
    for (uint64_t x = 0; x < group.order; ++x)
        for (uint64_t y = x + 1; y < group.order; ++y) {
            bool edge = selected[group.mul(group.inverse[x], y)] != 0;
            adjacency[x * group.order + y] = adjacency[y * group.order + x] = edge;
        }
    return graph::canonical(adjacency.data(), group.order);
}

struct ClassResult {
    uint64_t aut_classes = 0;
    uint64_t iso_classes = 0;
    bool non_ci = false;
};

Result<ClassResult> classify(const GroupModel &group, uint64_t k, bool need_aut_count,
                             bool need_iso_count, bool need_ci) {
    using C = Result<ClassResult>;
    if (k > group.order - 1) return C::success(ClassResult{});
    uint64_t target = std::min(k, group.order - 1 - k);
    std::vector<std::vector<Entry>> automorphisms;
    if (need_aut_count || need_ci)
        automorphisms = group_table::automorphisms(group.table, group.order);
    if ((need_aut_count || need_ci) && automorphisms.empty())
        return C::failure(INTERNAL, "failed to enumerate the identity automorphism of G");

    std::unordered_set<std::string> aut_classes;
    std::unordered_set<std::vector<Entry>, EntryVectorHash> iso_classes;
    std::unordered_map<std::vector<Entry>, std::string, EntryVectorHash> iso_to_aut;
    std::vector<uint8_t> selected(group.order, 0);

    std::function<bool(uint64_t, uint64_t)> enumerate = [&](uint64_t atom, uint64_t used) {
        if (used > target) return true;
        if (atom == group.inverse_atoms.size()) {
            if (used != target) return true;
            std::string aut_key;
            if (need_aut_count || need_ci) aut_key = canonical_aut_key(selected, automorphisms);
            std::vector<Entry> iso_key;
            if (need_iso_count || need_ci) iso_key = canonical_graph_key(group, selected);
            if (need_aut_count) aut_classes.insert(aut_key);
            if (need_iso_count) iso_classes.insert(iso_key);
            if (need_ci) {
                auto [it, inserted] = iso_to_aut.emplace(std::move(iso_key), aut_key);
                if (!inserted && it->second != aut_key) return false;
            }
            return true;
        }

        const auto &current = group.inverse_atoms[atom];
        if (used + current.size() <= target) {
            for (uint32_t x : current) selected[x] = 1;
            if (!enumerate(atom + 1, used + current.size())) return false;
            for (uint32_t x : current) selected[x] = 0;
        }
        return enumerate(atom + 1, used);
    };

    bool exhausted = enumerate(0, 0);
    ClassResult result;
    result.aut_classes = aut_classes.size();
    result.iso_classes = iso_classes.size();
    result.non_ci = !exhausted;
    return C::success(result);
}

R run(const Request &req) {
    if (req.family->kind != Family::Kind::GroupTables)
        return R::failure(INVALID, "cayley_iso operations need a group_tables family");

    bool need_aut_count = req.op == "aut_class_count";
    bool need_iso_count = req.op == "iso_class_count";
    bool need_ci = req.op == "is_ci" || req.op == "is_non_ci";
    if (!need_aut_count && !need_iso_count && !need_ci)
        return R::failure(INTERNAL, "unknown cayley_iso operation " + req.op);

    const Matrix &tables = *req.family->data;
    uint64_t count = tables.count;
    uint64_t order = tables.rows;
    uint64_t k = req.int_args.at("k");
    Reduction reduction = parse_reduction(req.reduction);
    Shared shared;
    auto prepared = prepare_all(reduction, count, shared);
    if (!prepared.ok) return R::failure(prepared.error.status, prepared.error.message);

    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, count));
    std::vector<Accumulator> accumulators;
    for (uint32_t thread = 0; thread < threads; ++thread)
        accumulators.emplace_back(reduction, &shared);
    auto statuses = parallel_ranges(count, threads, [&](uint32_t thread, uint64_t begin, uint64_t end) -> Status {
        for (uint64_t index = begin; index < end; ++index) {
            if (accumulators[thread].exhausted(index)) break;
            auto group = GroupModel::build(tables.at(index), order);
            if (!group.ok) return fail(group.error.status, group.error.message);
            auto result = classify(group.value, k, need_aut_count, need_iso_count, need_ci);
            if (!result.ok) return fail(result.error.status, result.error.message);
            if (need_aut_count) accumulators[thread].integer(index, result.value.aut_classes);
            else if (need_iso_count) accumulators[thread].integer(index, result.value.iso_classes);
            else {
                bool is_ci = !result.value.non_ci;
                accumulators[thread].boolean(index, req.op == "is_ci" ? is_ci : !is_ci);
            }
        }
        return ok();
    });
    for (const auto &status : statuses)
        if (!status.ok) return R::failure(status.error.status, status.error.message);
    return assemble(req, reduction, accumulators, shared);
}

BackendRegistration registration{Backend{
    "cayley_iso", "generic",
    [] { return true; },
    [](const Request &req) { return req.family->kind == Family::Kind::GroupTables; },
    run,
    0}};

} // namespace
} // namespace lk::cayley_iso
