#pragma once

#include "../../../runtime/src/group_table.hpp"
#include "../../../runtime/src/reduce.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <functional>
#include <optional>
#include <unordered_set>

namespace lk::cayley_iso {

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
    std::vector<uint32_t> atom_of;

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

        GroupModel group{table, order, identity, std::vector<uint32_t>(order), {},
                         std::vector<uint32_t>(order, UINT32_MAX)};
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
        for (uint32_t atom = 0; atom < group.inverse_atoms.size(); ++atom)
            for (uint32_t x : group.inverse_atoms[atom]) group.atom_of[x] = atom;
        return G::success(std::move(group));
    }
};

struct AtomAction {
    std::vector<uint32_t> image;
    std::array<uint64_t, 8 * 256> mask_lookup{};
};

inline Result<std::vector<AtomAction>> atom_actions(
    const GroupModel &group, const std::vector<std::vector<Entry>> &automorphisms) {
    using A = Result<std::vector<AtomAction>>;
    std::vector<AtomAction> actions;
    actions.reserve(automorphisms.size());
    for (const auto &automorphism : automorphisms) {
        AtomAction action;
        action.image.resize(group.inverse_atoms.size());
        for (uint32_t atom = 0; atom < group.inverse_atoms.size(); ++atom) {
            uint32_t image = group.atom_of[automorphism[group.inverse_atoms[atom][0]]];
            if (image == UINT32_MAX)
                return A::failure(INTERNAL, "group automorphism did not preserve inverse atoms");
            for (uint32_t x : group.inverse_atoms[atom])
                if (group.atom_of[automorphism[x]] != image)
                    return A::failure(INTERNAL, "group automorphism split an inverse atom");
            action.image[atom] = image;
        }
        if (group.inverse_atoms.size() <= 64)
            for (uint32_t byte = 0; byte < 8; ++byte)
                for (uint32_t value = 1; value < 256; ++value) {
                    uint32_t bit = (uint32_t)std::countr_zero(value);
                    uint32_t atom = byte * 8 + bit;
                    uint64_t mapped = action.mask_lookup[byte * 256 + (value & (value - 1))];
                    if (atom < group.inverse_atoms.size()) mapped |= uint64_t{1} << action.image[atom];
                    action.mask_lookup[byte * 256 + value] = mapped;
                }
        actions.push_back(std::move(action));
    }
    return A::success(std::move(actions));
}

inline uint64_t permute_mask(uint64_t mask, const AtomAction &action) {
    uint64_t result = 0;
    for (uint32_t byte = 0; byte < 8; ++byte) {
        result |= action.mask_lookup[byte * 256 + (mask & 0xff)];
        mask >>= 8;
    }
    return result;
}

inline bool words_less(const std::vector<uint64_t> &left, const std::vector<uint64_t> &right) {
    for (size_t word = left.size(); word-- > 0;)
        if (left[word] != right[word]) return left[word] < right[word];
    return false;
}

inline std::vector<uint64_t> permute_words(const std::vector<uint64_t> &words,
                                           const AtomAction &action) {
    std::vector<uint64_t> result(words.size(), 0);
    for (uint32_t atom = 0; atom < action.image.size(); ++atom)
        if ((words[atom / 64] >> (atom % 64)) & 1)
            result[action.image[atom] / 64] |= uint64_t{1} << (action.image[atom] % 64);
    return result;
}

inline bool is_orbit_representative(const std::vector<uint64_t> &words,
                                    const std::vector<AtomAction> &actions) {
    if (words.size() == 1) {
        uint64_t mask = words[0];
        for (const auto &action : actions)
            if (permute_mask(mask, action) < mask) return false;
        return true;
    }
    for (const auto &action : actions)
        if (words_less(permute_words(words, action), words)) return false;
    return true;
}

inline std::vector<uint8_t> selected_elements(const GroupModel &group,
                                              const std::vector<uint64_t> &words) {
    std::vector<uint8_t> selected(group.order, 0);
    for (uint32_t atom = 0; atom < group.inverse_atoms.size(); ++atom)
        if ((words[atom / 64] >> (atom % 64)) & 1)
            for (uint32_t x : group.inverse_atoms[atom]) selected[x] = 1;
    return selected;
}

inline Result<std::vector<std::vector<uint64_t>>> orbit_representatives(
    const GroupModel &group, std::optional<uint64_t> fixed_weight) {
    using Q = Result<std::vector<std::vector<uint64_t>>>;
    auto automorphisms = group_table::automorphisms(group.table, group.order);
    if (automorphisms.empty())
        return Q::failure(INTERNAL, "failed to enumerate the identity automorphism of G");
    auto built_actions = atom_actions(group, automorphisms);
    if (!built_actions.ok) return Q::failure(built_actions.error.status, built_actions.error.message);
    const auto &actions = built_actions.value;

    uint64_t maximum = fixed_weight.value_or((group.order - 1) / 2);
    std::vector<std::vector<uint64_t>> representatives;
    std::vector<uint64_t> words((group.inverse_atoms.size() + 63) / 64, 0);
    std::function<void(uint32_t, uint64_t)> visit = [&](uint32_t atom, uint64_t weight) {
        if (weight > maximum) return;
        if (atom == group.inverse_atoms.size()) {
            if ((!fixed_weight || weight == *fixed_weight) && is_orbit_representative(words, actions))
                representatives.push_back(words);
            return;
        }
        visit(atom + 1, weight);
        words[atom / 64] |= uint64_t{1} << (atom % 64);
        visit(atom + 1, weight + group.inverse_atoms[atom].size());
        words[atom / 64] &= ~(uint64_t{1} << (atom % 64));
    };
    visit(0, 0);
    return Q::success(std::move(representatives));
}

struct ClassResult {
    uint64_t aut_classes = 0;
    uint64_t iso_classes = 0;
    bool non_ci = false;
};

template <class Canonicalizer>
Result<ClassResult> classify(const GroupModel &group, std::optional<uint64_t> requested_weight,
                             bool need_iso_count, bool need_ci, uint32_t threads,
                             Canonicalizer canonicalize) {
    using C = Result<ClassResult>;
    if (requested_weight && *requested_weight > group.order - 1)
        return C::success(ClassResult{});
    std::optional<uint64_t> weight = requested_weight;
    if (weight) *weight = std::min(*weight, group.order - 1 - *weight);
    auto found = orbit_representatives(group, weight);
    if (!found.ok) return C::failure(found.error.status, found.error.message);
    auto &representatives = found.value;

    ClassResult result;
    result.aut_classes = representatives.size();
    if (!need_iso_count && !need_ci) return C::success(result);

    std::vector<std::vector<Entry>> keys(representatives.size());
    auto statuses = parallel_ranges(representatives.size(), threads,
        [&](uint32_t, uint64_t begin, uint64_t end) -> Status {
            for (uint64_t index = begin; index < end; ++index) {
                auto key = canonicalize(group, representatives[index]);
                if (!key.ok) return fail(key.error.status, key.error.message);
                keys[index] = std::move(key.value);
            }
            return ok();
        });
    for (const auto &status : statuses)
        if (!status.ok) return C::failure(status.error.status, status.error.message);

    std::unordered_set<std::vector<Entry>, EntryVectorHash> classes;
    classes.reserve(keys.size() * 2 + 1);
    for (auto &key : keys)
        if (!classes.insert(std::move(key)).second) result.non_ci = true;
    result.iso_classes = classes.size();
    return C::success(result);
}

template <class Canonicalizer>
R run_backend(const Request &req, Canonicalizer canonicalize) {
    if (req.family->kind != Family::Kind::GroupTables)
        return R::failure(INVALID, "cayley_iso operations need a group_tables family");

    bool need_aut_count = req.op == "aut_class_count";
    bool need_iso_count = req.op == "iso_class_count";
    bool fixed_ci = req.op == "is_ci" || req.op == "is_non_ci";
    bool whole_ci = req.op == "is_ci_group" || req.op == "is_non_ci_group";
    if (!need_aut_count && !need_iso_count && !fixed_ci && !whole_ci)
        return R::failure(INTERNAL, "unknown cayley_iso operation " + req.op);

    const Matrix &tables = *req.family->data;
    uint64_t count = tables.count;
    uint64_t order = tables.rows;
    std::optional<uint64_t> weight;
    if (!whole_ci) weight = req.int_args.at("k");
    Reduction reduction = parse_reduction(req.reduction);
    Shared shared;
    auto prepared = prepare_all(reduction, count, shared);
    if (!prepared.ok) return R::failure(prepared.error.status, prepared.error.message);

    uint32_t outer_threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, count));
    uint32_t inner_threads = std::max<uint32_t>(1, req.threads / outer_threads);
    std::vector<Accumulator> accumulators;
    for (uint32_t thread = 0; thread < outer_threads; ++thread)
        accumulators.emplace_back(reduction, &shared);
    auto statuses = parallel_ranges(count, outer_threads,
        [&](uint32_t thread, uint64_t begin, uint64_t end) -> Status {
            for (uint64_t index = begin; index < end; ++index) {
                if (accumulators[thread].exhausted(index)) break;
                auto group = GroupModel::build(tables.at(index), order);
                if (!group.ok) return fail(group.error.status, group.error.message);
                auto classified = classify(group.value, weight, need_iso_count,
                                           fixed_ci || whole_ci, inner_threads, canonicalize);
                if (!classified.ok)
                    return fail(classified.error.status, classified.error.message);
                if (need_aut_count)
                    accumulators[thread].integer(index, classified.value.aut_classes);
                else if (need_iso_count)
                    accumulators[thread].integer(index, classified.value.iso_classes);
                else {
                    bool is_ci = !classified.value.non_ci;
                    bool positive = req.op == "is_ci" || req.op == "is_ci_group";
                    accumulators[thread].boolean(index, positive ? is_ci : !is_ci);
                }
            }
            return ok();
        });
    for (const auto &status : statuses)
        if (!status.ok) return R::failure(status.error.status, status.error.message);
    return assemble(req, reduction, accumulators, shared);
}

} // namespace lk::cayley_iso
