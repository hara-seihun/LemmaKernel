/* Portable subgroup enumeration for finite permutation groups.
 *
 * Parent elements are closed and sorted lexicographically. A multiplication table then makes a
 * subgroup an increasing vector of element indices. Beginning with the identity subgroup, every
 * parent element is adjoined to every discovered subgroup. Complete index vectors deduplicate the
 * resulting closures. Conjugation and maximality are computed on that finished subgroup lattice. */
#include "../../../../runtime/src/reduce.hpp"

#include <numeric>
#include <unordered_map>
#include <unordered_set>

namespace lk::subgroups {
namespace {

using R = Result<std::shared_ptr<Object>>;
using Perm = std::vector<Entry>;
using Subgroup = std::vector<uint64_t>;
constexpr int INVALID = 1;
constexpr uint64_t GROUP_LIMIT = 4096;
constexpr uint64_t SUBGROUP_LIMIT = 1ULL << 20;

struct VectorHash {
    template <class T> size_t operator()(const std::vector<T> &xs) const {
        uint64_t h = 1469598103934665603ULL;
        for (T x : xs) {
            uint64_t word = static_cast<uint64_t>(x);
            for (unsigned i = 0; i < sizeof(T); ++i) {
                h ^= static_cast<uint8_t>(word >> (8 * i));
                h *= 1099511628211ULL;
            }
        }
        return static_cast<size_t>(h);
    }
};

Status validate_rows(const Matrix &member) {
    if (member.p != 0 || member.cols == 0)
        return fail(INVALID, "family members must contain permutation rows");
    for (uint64_t r = 0; r < member.rows; ++r) {
        std::vector<uint8_t> seen(member.cols, 0);
        for (uint64_t i = 0; i < member.cols; ++i) {
            Entry x = member.entries[r * member.cols + i];
            if (x >= member.cols || seen[x]++)
                return fail(INVALID, "family members must contain permutation rows of one degree");
        }
    }
    return ok();
}

Status validate_batch(const Matrix &generators) {
    if (generators.p != 0 || generators.rows != 1 || generators.cols == 0)
        return fail(INVALID, "group generators must be a batch of permutation rows");
    for (uint64_t g = 0; g < generators.count; ++g) {
        std::vector<uint8_t> seen(generators.cols, 0);
        for (uint64_t i = 0; i < generators.cols; ++i) {
            Entry x = generators.entries[g * generators.cols + i];
            if (x >= generators.cols || seen[x]++)
                return fail(INVALID, "group generators must contain permutation rows of one degree");
        }
    }
    return ok();
}

Result<Matrix> generators_at(const Family &family, uint64_t index) {
    Matrix member;
    auto status = family.member_into(index, member);
    if (!status.ok) return Result<Matrix>::failure(status.error.status, status.error.message);
    status = validate_rows(member);
    if (!status.ok) return Result<Matrix>::failure(status.error.status, status.error.message);
    Matrix generators;
    generators.p = 0;
    generators.count = member.rows;
    generators.rows = 1;
    generators.cols = member.cols;
    generators.entries = std::move(member.entries);
    return Result<Matrix>::success(std::move(generators));
}

struct FiniteGroup {
    uint64_t degree = 0;
    std::vector<Perm> elements;
    std::unordered_map<Perm, uint64_t, VectorHash> index;
    std::vector<uint32_t> table;
    std::vector<uint32_t> inverses;
    uint32_t identity = 0;

    uint64_t order() const { return elements.size(); }
    uint32_t multiply(uint64_t a, uint64_t b) const { return table[a * order() + b]; }

    static Result<FiniteGroup> build(const Matrix &generators) {
        auto valid = validate_batch(generators);
        if (!valid.ok) return Result<FiniteGroup>::failure(valid.error.status, valid.error.message);
        auto closed = permutation_closure(generators, GROUP_LIMIT);
        if (!closed.ok) return Result<FiniteGroup>::failure(closed.error.status, closed.error.message);

        FiniteGroup group;
        group.degree = generators.cols;
        uint64_t q = closed.value.size() / group.degree;
        group.elements.resize(q, Perm(group.degree));
        for (uint64_t i = 0; i < q; ++i) {
            std::copy(closed.value.begin() + i * group.degree,
                      closed.value.begin() + (i + 1) * group.degree, group.elements[i].begin());
            group.index.emplace(group.elements[i], i);
        }
        Perm id(group.degree);
        std::iota(id.begin(), id.end(), 0);
        auto identity = group.index.find(id);
        if (identity == group.index.end())
            return Result<FiniteGroup>::failure(INVALID, "generated permutation group has no identity");
        group.identity = static_cast<uint32_t>(identity->second);

        if ((unsigned __int128)q * q > SIZE_MAX / sizeof(uint32_t))
            return Result<FiniteGroup>::failure(INVALID, "group multiplication table is too large");
        group.table.resize(q * q);
        Perm product(group.degree);
        for (uint64_t a = 0; a < q; ++a)
            for (uint64_t b = 0; b < q; ++b) {
                for (uint64_t i = 0; i < group.degree; ++i)
                    product[i] = group.elements[b][group.elements[a][i]];
                auto found = group.index.find(product);
                if (found == group.index.end())
                    return Result<FiniteGroup>::failure(INVALID, "permutation closure is not closed");
                group.table[a * q + b] = static_cast<uint32_t>(found->second);
            }

        group.inverses.resize(q);
        for (uint64_t a = 0; a < q; ++a) {
            uint64_t b = 0;
            while (b < q && group.multiply(a, b) != group.identity) ++b;
            if (b == q) return Result<FiniteGroup>::failure(INVALID, "group element has no inverse");
            group.inverses[a] = static_cast<uint32_t>(b);
        }
        return Result<FiniteGroup>::success(std::move(group));
    }

    Subgroup generated(const Subgroup &generators) const {
        std::vector<uint8_t> seen(order(), 0);
        Subgroup queue{identity};
        seen[identity] = 1;
        for (size_t front = 0; front < queue.size(); ++front)
            for (uint64_t generator : generators) {
                uint64_t product = multiply(queue[front], generator);
                if (!seen[product]) {
                    seen[product] = 1;
                    queue.push_back(product);
                }
            }
        std::sort(queue.begin(), queue.end());
        return queue;
    }

    Result<std::vector<Subgroup>> subgroups() const {
        std::vector<Subgroup> found{{identity}};
        std::unordered_set<Subgroup, VectorHash> known;
        known.insert(found.front());
        for (size_t front = 0; front < found.size(); ++front) {
            std::vector<uint8_t> inside(order(), 0);
            for (uint64_t x : found[front]) inside[x] = 1;
            for (uint64_t x = 0; x < order(); ++x) {
                if (inside[x]) continue;
                Subgroup generators = found[front];
                generators.push_back(x);
                Subgroup extension = generated(generators);
                if (known.insert(extension).second) {
                    found.push_back(std::move(extension));
                    if (found.size() > SUBGROUP_LIMIT)
                        return Result<std::vector<Subgroup>>::failure(
                            INVALID, "group has more than " + std::to_string(SUBGROUP_LIMIT) + " subgroups");
                }
            }
        }
        std::sort(found.begin(), found.end());
        return Result<std::vector<Subgroup>>::success(std::move(found));
    }

    Subgroup conjugate(const Subgroup &subgroup, uint64_t g) const {
        Subgroup out;
        out.reserve(subgroup.size());
        for (uint64_t h : subgroup)
            out.push_back(multiply(multiply(inverses[g], h), g));
        std::sort(out.begin(), out.end());
        return out;
    }

    Subgroup canonical_conjugate(const Subgroup &subgroup) const {
        Subgroup least = subgroup;
        for (uint64_t g = 0; g < order(); ++g) {
            Subgroup image = conjugate(subgroup, g);
            if (image < least) least = std::move(image);
        }
        return least;
    }

    std::vector<Subgroup> classes(const std::vector<Subgroup> &all) const {
        std::vector<Subgroup> out;
        for (const auto &subgroup : all)
            if (canonical_conjugate(subgroup) == subgroup) out.push_back(subgroup);
        return out;
    }

    static bool contained(const Subgroup &a, const Subgroup &b) {
        return std::includes(b.begin(), b.end(), a.begin(), a.end());
    }

    std::vector<Subgroup> maximal(const std::vector<Subgroup> &all) const {
        std::vector<Subgroup> out;
        for (const auto &subgroup : all) {
            if (subgroup.size() == order()) continue;
            bool maximal = true;
            for (const auto &larger : all)
                if (subgroup.size() < larger.size() && larger.size() < order() && contained(subgroup, larger)) {
                    maximal = false;
                    break;
                }
            if (maximal && canonical_conjugate(subgroup) == subgroup) out.push_back(subgroup);
        }
        return out;
    }

    Result<bool> normal_subgroup(const Matrix &candidate_generators) const {
        auto closed = permutation_closure(candidate_generators, GROUP_LIMIT);
        if (!closed.ok) return Result<bool>::failure(closed.error.status, closed.error.message);
        Subgroup subgroup;
        uint64_t count = closed.value.size() / degree;
        for (uint64_t i = 0; i < count; ++i) {
            Perm element(closed.value.begin() + i * degree, closed.value.begin() + (i + 1) * degree);
            auto found = index.find(element);
            if (found == index.end()) return Result<bool>::success(false);
            subgroup.push_back(found->second);
        }
        std::sort(subgroup.begin(), subgroup.end());
        for (uint64_t g = 0; g < order(); ++g)
            if (conjugate(subgroup, g) != subgroup) return Result<bool>::success(false);
        return Result<bool>::success(true);
    }
};

Result<FiniteGroup> group_at(const Family &family, uint64_t index) {
    auto generators = generators_at(family, index);
    if (!generators.ok) return Result<FiniteGroup>::failure(generators.error.status, generators.error.message);
    return FiniteGroup::build(generators.value);
}

enum class ListOp { Classes, Maximal };

R run_lists(const Request &req, ListOp op) {
    if (req.reduction != "all")
        return R::failure(INVALID, "subgroup lists only accept the all reduction");
    if (req.family->kind != Family::Kind::Subsets && req.family->kind != Family::Kind::Explicit)
        return R::failure(INVALID, "subgroup enumeration needs subsets or explicit families");
    auto size = req.family->size();
    if (!size.ok) return R::failure(size.error.status, size.error.message);
    std::vector<std::vector<Subgroup>> values(size.value);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, size.value ? size.value : 1));
    auto statuses = parallel_ranges(size.value, threads, [&](uint32_t, uint64_t begin, uint64_t end) -> Status {
        for (uint64_t i = begin; i < end; ++i) {
            auto group = group_at(*req.family, i);
            if (!group.ok) return fail(group.error.status, group.error.message);
            auto all = group.value.subgroups();
            if (!all.ok) return fail(all.error.status, all.error.message);
            values[i] = op == ListOp::Classes ? group.value.classes(all.value) : group.value.maximal(all.value);
        }
        return ok();
    });
    for (const auto &status : statuses)
        if (!status.ok) return R::failure(status.error.status, status.error.message);

    auto lists = std::make_shared<SubgroupLists>();
    lists->count = size.value;
    lists->group_offsets.push_back(0);
    lists->subgroup_offsets.push_back(0);
    for (const auto &per_group : values) {
        lists->group_offsets.push_back(lists->group_offsets.back() + per_group.size());
        for (const auto &subgroup : per_group) {
            lists->elements.insert(lists->elements.end(), subgroup.begin(), subgroup.end());
            lists->subgroup_offsets.push_back(lists->elements.size());
        }
    }
    auto object = std::make_shared<Object>();
    object->kind = "subgroups.lists";
    object->subgroup_lists = std::move(lists);
    return R::success(object);
}

R run_count(const Request &req) {
    if (req.family->kind != Family::Kind::Subsets && req.family->kind != Family::Kind::Explicit)
        return R::failure(INVALID, "subgroup enumeration needs subsets or explicit families");
    auto size = req.family->size();
    if (!size.ok) return R::failure(size.error.status, size.error.message);
    Reduction reduction = parse_reduction(req.reduction);
    Shared shared;
    auto prepared = prepare_all(reduction, size.value, shared);
    if (!prepared.ok) return R::failure(prepared.error.status, prepared.error.message);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, size.value ? size.value : 1));
    std::vector<Accumulator> accs;
    for (uint32_t t = 0; t < threads; ++t) accs.emplace_back(reduction, &shared);
    auto statuses = parallel_ranges(size.value, threads, [&](uint32_t t, uint64_t begin, uint64_t end) -> Status {
        for (uint64_t i = begin; i < end; ++i) {
            if (accs[t].exhausted(i)) break;
            auto group = group_at(*req.family, i);
            if (!group.ok) return fail(group.error.status, group.error.message);
            auto all = group.value.subgroups();
            if (!all.ok) return fail(all.error.status, all.error.message);
            accs[t].integer(i, all.value.size());
        }
        return ok();
    });
    for (const auto &status : statuses)
        if (!status.ok) return R::failure(status.error.status, status.error.message);
    return assemble(req, reduction, accs, shared);
}

R run_normal(const Request &req) {
    if (req.family->kind != Family::Kind::Subsets && req.family->kind != Family::Kind::Explicit)
        return R::failure(INVALID, "is_normal needs subsets or explicit families");
    auto argument = req.handle_args.find("group");
    if (argument == req.handle_args.end() || !argument->second->matrix || argument->second->matrix->p != 0)
        return R::failure(INVALID, "is_normal needs a permutation group argument");
    const Matrix &parent_generators = *argument->second->matrix;
    if (parent_generators.rows != 1)
        return R::failure(INVALID, "is_normal needs a permutation group argument");
    if (parent_generators.cols != req.family->cols())
        return R::failure(INVALID, "candidate and parent must have the same degree");
    auto parent = FiniteGroup::build(parent_generators);
    if (!parent.ok) return R::failure(parent.error.status, parent.error.message);

    auto size = req.family->size();
    if (!size.ok) return R::failure(size.error.status, size.error.message);
    Reduction reduction = parse_reduction(req.reduction);
    Shared shared;
    auto prepared = prepare_all(reduction, size.value, shared);
    if (!prepared.ok) return R::failure(prepared.error.status, prepared.error.message);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, size.value ? size.value : 1));
    std::vector<Accumulator> accs;
    for (uint32_t t = 0; t < threads; ++t) accs.emplace_back(reduction, &shared);
    auto statuses = parallel_ranges(size.value, threads, [&](uint32_t t, uint64_t begin, uint64_t end) -> Status {
        for (uint64_t i = begin; i < end; ++i) {
            if (accs[t].exhausted(i)) break;
            auto candidate = generators_at(*req.family, i);
            if (!candidate.ok) return fail(candidate.error.status, candidate.error.message);
            auto normal = parent.value.normal_subgroup(candidate.value);
            if (!normal.ok) return fail(normal.error.status, normal.error.message);
            accs[t].boolean(i, normal.value);
        }
        return ok();
    });
    for (const auto &status : statuses)
        if (!status.ok) return R::failure(status.error.status, status.error.message);
    return assemble(req, reduction, accs, shared);
}

R run(const Request &req) {
    if (req.op == "subgroup_count") return run_count(req);
    if (req.op == "conjugacy_classes") return run_lists(req, ListOp::Classes);
    if (req.op == "maximal_subgroups") return run_lists(req, ListOp::Maximal);
    if (req.op == "is_normal") return run_normal(req);
    return R::failure(4, "unknown subgroups operation " + req.op);
}

BackendRegistration registration{Backend{
    "subgroups", "generic",
    [] { return true; },
    [](const Request &) { return true; },
    run,
    0}};

} // namespace
} // namespace lk::subgroups
