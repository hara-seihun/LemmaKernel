/* Portable Schreier-Sims backend for permutation groups.
 *
 * A family member is a matrix with p == 0. Its rows are generators on `cols` points. Stabilizer
 * generators come directly from Schreier's lemma; order and membership never enumerate the
 * group. Family members are independent, so reductions split them across runtime threads. */
#include "../../../../runtime/src/reduce.hpp"

#include <numeric>
#include <unordered_map>

namespace lk::perm_groups {
namespace {

using R = Result<std::shared_ptr<Object>>;
using Perm = std::vector<Entry>;
constexpr int INVALID = 1;

Perm identity(uint64_t n) {
    Perm out(n);
    std::iota(out.begin(), out.end(), 0);
    return out;
}

Perm compose(const Perm &g, const Perm &h) {
    Perm out(g.size());
    for (size_t i = 0; i < g.size(); ++i) out[i] = h[g[i]];
    return out;
}

Perm inverse(const Perm &g) {
    Perm out(g.size());
    for (size_t i = 0; i < g.size(); ++i) out[g[i]] = (Entry)i;
    return out;
}

void canonicalize(std::vector<Perm> &gens, const Perm &id) {
    gens.erase(std::remove(gens.begin(), gens.end(), id), gens.end());
    std::sort(gens.begin(), gens.end());
    gens.erase(std::unique(gens.begin(), gens.end()), gens.end());
}

struct Level {
    Entry base;
    std::vector<Perm> gens;
    std::vector<Perm> reps;
    std::vector<Entry> orbit;
};

struct Chain {
    uint64_t n;
    Perm id;
    std::vector<Level> levels;

    Chain(uint64_t degree, std::vector<Perm> gens) : n(degree), id(identity(degree)) {
        canonicalize(gens, id);
        for (uint64_t depth = 0; depth < n && !gens.empty(); ++depth) {
            uint64_t base = n;
            for (uint64_t i = 0; i < n && base == n; ++i)
                for (const auto &g : gens)
                    if (g[i] != i) { base = i; break; }
            if (base == n) break;

            std::vector<uint8_t> seen(n, 0);
            std::vector<Perm> reps(n);
            std::vector<Entry> orbit{(Entry)base};
            seen[base] = 1;
            reps[base] = id;
            for (size_t front = 0; front < orbit.size(); ++front) {
                Entry x = orbit[front];
                for (const auto &s : gens) {
                    Entry y = s[x];
                    if (!seen[y]) {
                        seen[y] = 1;
                        reps[y] = compose(reps[x], s);
                        orbit.push_back(y);
                    }
                }
            }

            std::vector<Perm> inverse_reps(n);
            for (Entry x : orbit) inverse_reps[x] = inverse(reps[x]);
            std::vector<Perm> next;
            next.reserve(orbit.size() * gens.size());
            for (Entry x : orbit)
                for (const auto &s : gens) {
                    Entry y = s[x];
                    Perm schreier = compose(compose(reps[x], s), inverse_reps[y]);
                    if (schreier != id) next.push_back(std::move(schreier));
                }
            canonicalize(next, id);
            levels.push_back(Level{(Entry)base, std::move(gens), std::move(reps), std::move(orbit)});
            gens = std::move(next);
        }
    }

    Result<uint64_t> order() const {
        unsigned __int128 value = 1;
        for (const auto &level : levels) {
            value *= level.orbit.size();
            if (value > UINT64_MAX)
                return Result<uint64_t>::failure(INVALID, "group order does not fit in 64 bits");
        }
        return Result<uint64_t>::success((uint64_t)value);
    }

    bool contains(Perm residue) const {
        for (const auto &level : levels) {
            Entry image = residue[level.base];
            if (level.reps[image].empty()) return false;
            residue = compose(residue, inverse(level.reps[image]));
        }
        return residue == id;
    }

    std::pair<std::vector<Entry>, std::vector<Perm>> bsgs() const {
        std::vector<Entry> base;
        std::vector<Perm> strong;
        for (const auto &level : levels) {
            base.push_back(level.base);
            strong.insert(strong.end(), level.gens.begin(), level.gens.end());
        }
        canonicalize(strong, id);
        return {std::move(base), std::move(strong)};
    }
};

Status validate_generators(const Matrix &member) {
    if (member.p != 0 || member.rows == 0 || member.cols == 0)
        return fail(INVALID, "family members must be nonempty lists of permutation rows");
    for (uint64_t r = 0; r < member.rows; ++r) {
        std::vector<uint8_t> seen(member.cols, 0);
        for (uint64_t i = 0; i < member.cols; ++i) {
            Entry x = member.entries[r * member.cols + i];
            if (x >= member.cols || seen[x])
                return fail(INVALID, "family members must contain permutation rows of one degree");
            seen[x] = 1;
        }
    }
    return ok();
}

std::vector<Perm> generators(const Matrix &member) {
    std::vector<Perm> out(member.rows, Perm(member.cols));
    for (uint64_t r = 0; r < member.rows; ++r)
        std::copy(member.entries.begin() + r * member.cols,
                  member.entries.begin() + (r + 1) * member.cols, out[r].begin());
    return out;
}

std::vector<Entry> point_partition(const std::vector<Perm> &gens, uint64_t n) {
    std::vector<Entry> labels(n, UINT32_MAX);
    std::vector<Entry> queue;
    for (uint64_t start = 0; start < n; ++start) {
        if (labels[start] != UINT32_MAX) continue;
        queue.clear();
        queue.push_back((Entry)start);
        labels[start] = (Entry)start;
        for (size_t front = 0; front < queue.size(); ++front) {
            Entry x = queue[front];
            for (const auto &g : gens) {
                Entry y = g[x];
                if (labels[y] == UINT32_MAX) {
                    labels[y] = (Entry)start;
                    queue.push_back(y);
                }
            }
        }
    }
    return labels;
}

bool is_transitive(const std::vector<Entry> &labels) {
    return std::all_of(labels.begin(), labels.end(), [](Entry x) { return x == 0; });
}

struct DisjointSets {
    std::vector<Entry> parent, size;
    explicit DisjointSets(uint64_t n) : parent(n), size(n, 1) { std::iota(parent.begin(), parent.end(), 0); }
    Entry find(Entry x) {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    }
    bool join(Entry a, Entry b) {
        a = find(a); b = find(b);
        if (a == b) return false;
        if (size[a] < size[b]) std::swap(a, b);
        parent[b] = a; size[a] += size[b];
        return true;
    }
};

/* The least invariant equivalence relation identifying 0 and beta. A transitive action is
 * primitive exactly when this relation is universal for every beta != 0. */
bool is_primitive(const std::vector<Perm> &gens, uint64_t n, const std::vector<Entry> &partition) {
    if (!is_transitive(partition)) return false;
    for (uint64_t beta = 1; beta < n; ++beta) {
        DisjointSets sets(n);
        sets.join(0, (Entry)beta);
        bool changed;
        do {
            changed = false;
            for (const auto &g : gens) {
                std::unordered_map<Entry, Entry> first_image;
                for (uint64_t x = 0; x < n; ++x) {
                    Entry root = sets.find((Entry)x);
                    auto [it, inserted] = first_image.emplace(root, g[x]);
                    if (!inserted) changed |= sets.join(it->second, g[x]);
                }
            }
        } while (changed);
        Entry root = sets.find(0);
        uint64_t block = 0;
        for (uint64_t x = 0; x < n; ++x) block += sets.find((Entry)x) == root;
        if (block != n) return false;
    }
    return true;
}

Result<std::pair<uint64_t, Matrix>> group_at(const Family &family, uint64_t index) {
    Matrix member;
    auto st = family.member_into(index, member);
    if (!st.ok) return Result<std::pair<uint64_t, Matrix>>::failure(st.error.status, st.error.message);
    st = validate_generators(member);
    if (!st.ok) return Result<std::pair<uint64_t, Matrix>>::failure(st.error.status, st.error.message);
    return Result<std::pair<uint64_t, Matrix>>::success({member.cols, std::move(member)});
}

enum class ScalarOp { Order, Contains, Transitive, Primitive };

R run_scalar(const Request &req, ScalarOp op) {
    const Family &family = *req.family;
    auto size_r = family.size();
    if (!size_r.ok) return R::failure(size_r.error.status, size_r.error.message);
    uint64_t size = size_r.value;

    Perm target;
    if (op == ScalarOp::Contains) {
        auto it = req.handle_args.find("target");
        if (it == req.handle_args.end() || !it->second->matrix || it->second->matrix->p != 0 ||
            it->second->matrix->rows != 1 || it->second->matrix->count != 1)
            return R::failure(INVALID, "contains target must contain exactly one permutation");
        target.assign(it->second->matrix->entries.begin(), it->second->matrix->entries.end());
        if (target.size() != family.cols())
            return R::failure(INVALID, "contains target must have the same degree as the groups");
    }

    Reduction reduction = parse_reduction(req.reduction);
    Shared shared;
    auto st = prepare_all(reduction, size, shared);
    if (!st.ok) return R::failure(st.error.status, st.error.message);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, size ? size : 1));
    std::vector<Accumulator> accs;
    for (uint32_t t = 0; t < threads; ++t) accs.emplace_back(reduction, &shared);
    auto statuses = parallel_ranges(size, threads, [&](uint32_t t, uint64_t begin, uint64_t end) -> Status {
        for (uint64_t i = begin; i < end; ++i) {
            if (accs[t].exhausted(i)) break;
            auto member = group_at(family, i);
            if (!member.ok) return fail(member.error.status, member.error.message);
            auto gens = generators(member.value.second);
            if (op == ScalarOp::Order || op == ScalarOp::Contains) {
                Chain chain(member.value.first, gens);
                if (op == ScalarOp::Order) {
                    auto value = chain.order();
                    if (!value.ok) return fail(value.error.status, value.error.message);
                    accs[t].integer(i, value.value);
                } else accs[t].boolean(i, chain.contains(target));
            } else {
                auto partition = point_partition(gens, member.value.first);
                if (op == ScalarOp::Transitive) accs[t].boolean(i, is_transitive(partition));
                else accs[t].boolean(i, is_primitive(gens, member.value.first, partition));
            }
        }
        return ok();
    });
    for (const auto &status : statuses)
        if (!status.ok) return R::failure(status.error.status, status.error.message);
    return assemble(req, reduction, accs, shared);
}

R run_partitions(const Request &req) {
    if (req.reduction != "all") return R::failure(INVALID, "orbit_partition values only reduce with `all`");
    auto size_r = req.family->size();
    if (!size_r.ok) return R::failure(size_r.error.status, size_r.error.message);
    uint64_t count = size_r.value, n = req.family->cols();
    auto out = std::make_shared<Partitions>();
    out->count = count; out->n = n; out->labels.resize(count * n);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, count ? count : 1));
    auto statuses = parallel_ranges(count, threads, [&](uint32_t, uint64_t begin, uint64_t end) -> Status {
        for (uint64_t i = begin; i < end; ++i) {
            auto member = group_at(*req.family, i);
            if (!member.ok) return fail(member.error.status, member.error.message);
            auto labels = point_partition(generators(member.value.second), n);
            std::copy(labels.begin(), labels.end(), out->labels.begin() + i * n);
        }
        return ok();
    });
    for (const auto &status : statuses)
        if (!status.ok) return R::failure(status.error.status, status.error.message);
    auto object = std::make_shared<Object>();
    object->kind = "perm_groups.partition";
    object->partitions = out;
    return R::success(object);
}

R run_bsgs(const Request &req) {
    if (req.reduction != "all") return R::failure(INVALID, "base_and_strong_generators values only reduce with `all`");
    auto size_r = req.family->size();
    if (!size_r.ok) return R::failure(size_r.error.status, size_r.error.message);
    uint64_t count = size_r.value, n = req.family->cols();
    std::vector<std::vector<Entry>> bases(count);
    std::vector<std::vector<Perm>> strong(count);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, count ? count : 1));
    auto statuses = parallel_ranges(count, threads, [&](uint32_t, uint64_t begin, uint64_t end) -> Status {
        for (uint64_t i = begin; i < end; ++i) {
            auto member = group_at(*req.family, i);
            if (!member.ok) return fail(member.error.status, member.error.message);
            Chain chain(n, generators(member.value.second));
            std::tie(bases[i], strong[i]) = chain.bsgs();
        }
        return ok();
    });
    for (const auto &status : statuses)
        if (!status.ok) return R::failure(status.error.status, status.error.message);

    auto out = std::make_shared<Bsgs>();
    out->count = count; out->n = n;
    out->base_offsets.push_back(0);
    out->strong_offsets.push_back(0);
    for (uint64_t i = 0; i < count; ++i) {
        out->bases.insert(out->bases.end(), bases[i].begin(), bases[i].end());
        for (const auto &g : strong[i]) out->strong.insert(out->strong.end(), g.begin(), g.end());
        out->base_offsets.push_back(out->bases.size());
        out->strong_offsets.push_back(out->strong.size() / n);
    }
    auto object = std::make_shared<Object>();
    object->kind = "perm_groups.bsgs";
    object->bsgs = out;
    return R::success(object);
}

R run(const Request &req) {
    if (req.op == "order") return run_scalar(req, ScalarOp::Order);
    if (req.op == "contains") return run_scalar(req, ScalarOp::Contains);
    if (req.op == "is_transitive") return run_scalar(req, ScalarOp::Transitive);
    if (req.op == "is_primitive") return run_scalar(req, ScalarOp::Primitive);
    if (req.op == "orbit_partition") return run_partitions(req);
    if (req.op == "base_and_strong_generators") return run_bsgs(req);
    return R::failure(4, "unknown perm_groups operation " + req.op);
}

BackendRegistration registration{Backend{
    "perm_groups", "generic",
    [] { return true; },
    [](const Request &) { return true; },
    run,
    0}};

} // namespace
} // namespace lk::perm_groups
