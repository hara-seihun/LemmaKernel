/* automorphisms generic backend: portable C++.
 *
 * A partial automorphism is an injective map between two generated subgroups. Assigning one image
 * forces the images of every product with an element already assigned. This turns a search over
 * n! permutations into a search over possible images of a small generating set for most groups.
 * Element order and centralizer size reject impossible images before propagation. */
#include "../../../../runtime/src/reduce.hpp"

#include <unordered_set>

namespace lk::automorphisms {
namespace {

using R = Result<std::shared_ptr<Object>>;
constexpr int INVALID = 1;

struct VectorHash {
    size_t operator()(const std::vector<Entry> &v) const {
        uint64_t h = 1469598103934665603ULL;
        for (Entry x : v) { h ^= x; h *= 1099511628211ULL; }
        return (size_t)h;
    }
};

class AutomorphismSearch {
public:
    AutomorphismSearch(const Entry *table, uint64_t order)
        : t(table), n(order), image(n, -1), preimage(n, -1), element_order(n), centralizer(n) {
        identity = find_identity();
        for (uint64_t a = 0; a < n; ++a) {
            element_order[a] = order_of(a);
            uint64_t c = 0;
            for (uint64_t b = 0; b < n; ++b) c += mul(a, b) == mul(b, a);
            centralizer[a] = c;
        }
    }

    std::vector<std::vector<Entry>> run() {
        bind(identity, identity);
        descend();
        std::sort(found.begin(), found.end());
        return found;
    }

private:
    const Entry *t;
    uint64_t n, identity = 0;
    std::vector<int64_t> image, preimage;
    std::vector<uint64_t> element_order, centralizer;
    std::vector<uint64_t> trail;
    std::vector<std::vector<Entry>> found;

    Entry mul(uint64_t a, uint64_t b) const { return t[a * n + b]; }

    uint64_t find_identity() const {
        for (uint64_t e = 0; e < n; ++e) {
            bool ok = true;
            for (uint64_t x = 0; x < n; ++x)
                if (mul(e, x) != x || mul(x, e) != x) { ok = false; break; }
            if (ok) return e;
        }
        return n;
    }

    uint64_t order_of(uint64_t a) const {
        uint64_t x = identity;
        for (uint64_t k = 1; k <= n; ++k) {
            x = mul(x, a);
            if (x == identity) return k;
        }
        return 0;
    }

    bool compatible(uint64_t a, uint64_t b) const {
        return element_order[a] == element_order[b] && centralizer[a] == centralizer[b];
    }

    bool bind(uint64_t a, uint64_t b) {
        if (image[a] >= 0) return (uint64_t)image[a] == b;
        if (preimage[b] >= 0 || !compatible(a, b)) return false;
        image[a] = (int64_t)b;
        preimage[b] = (int64_t)a;
        trail.push_back(a);

        for (uint64_t x = 0; x < n; ++x) {
            if (image[x] < 0) continue;
            uint64_t y = (uint64_t)image[x];
            if (!bind(mul(a, x), mul(b, y))) return false;
            if (!bind(mul(x, a), mul(y, b))) return false;
        }
        return true;
    }

    void rollback(size_t mark) {
        while (trail.size() > mark) {
            uint64_t a = trail.back();
            trail.pop_back();
            preimage[(uint64_t)image[a]] = -1;
            image[a] = -1;
        }
    }

    void descend() {
        uint64_t source = n;
        size_t best = SIZE_MAX;
        for (uint64_t a = 0; a < n; ++a) {
            if (image[a] >= 0) continue;
            size_t choices = 0;
            for (uint64_t b = 0; b < n; ++b)
                if (preimage[b] < 0 && compatible(a, b)) ++choices;
            if (choices < best) { best = choices; source = a; }
        }
        if (source == n) {
            std::vector<Entry> f(n);
            for (uint64_t a = 0; a < n; ++a) f[a] = (Entry)image[a];
            found.push_back(std::move(f));
            return;
        }
        for (uint64_t target = 0; target < n; ++target) {
            if (preimage[target] >= 0 || !compatible(source, target)) continue;
            size_t mark = trail.size();
            if (bind(source, target)) descend();
            rollback(mark);
        }
    }
};

std::vector<Entry> compose(const std::vector<Entry> &a, const std::vector<Entry> &b) {
    std::vector<Entry> out(a.size());
    for (size_t i = 0; i < a.size(); ++i) out[i] = b[a[i]];
    return out;
}

std::unordered_set<std::vector<Entry>, VectorHash>
generated_group(uint64_t n, const std::vector<std::vector<Entry>> &generators) {
    std::vector<Entry> identity(n);
    for (uint64_t i = 0; i < n; ++i) identity[i] = (Entry)i;
    std::unordered_set<std::vector<Entry>, VectorHash> seen;
    std::vector<std::vector<Entry>> queue{identity};
    seen.insert(identity);
    for (size_t front = 0; front < queue.size(); ++front)
        for (const auto &g : generators) {
            auto h = compose(queue[front], g);
            if (seen.insert(h).second) queue.push_back(std::move(h));
        }
    return seen;
}

std::vector<std::vector<Entry>> canonical_generators(uint64_t n, const std::vector<std::vector<Entry>> &autos) {
    std::vector<std::vector<Entry>> generators;
    auto subgroup = generated_group(n, generators);
    for (const auto &a : autos) {
        if (subgroup.count(a)) continue;
        generators.push_back(a);
        subgroup = generated_group(n, generators);
    }
    return generators;
}

uint64_t center_size(const Entry *table, uint64_t n) {
    uint64_t count = 0;
    for (uint64_t a = 0; a < n; ++a) {
        bool central = true;
        for (uint64_t b = 0; b < n; ++b)
            if (table[a * n + b] != table[b * n + a]) { central = false; break; }
        count += central;
    }
    return count;
}

enum class Op { AutOrder, HolomorphOrder, InnerAutIndex };

R run_integer(const Request &req, Op op) {
    const Matrix &tables = *req.family->data;
    uint64_t count = tables.count, n = tables.rows;
    Reduction reduction = parse_reduction(req.reduction);
    Shared shared;
    auto prepared = prepare_all(reduction, count, shared);
    if (!prepared.ok) return R::failure(prepared.error.status, prepared.error.message);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, count));
    std::vector<Accumulator> accs;
    for (uint32_t i = 0; i < threads; ++i) accs.emplace_back(reduction, &shared);
    auto statuses = parallel_ranges(count, threads, [&](uint32_t thread, uint64_t begin, uint64_t end) -> Status {
        for (uint64_t i = begin; i < end; ++i) {
            if (accs[thread].exhausted(i)) break;
            auto autos = AutomorphismSearch(tables.at(i), n).run();
            unsigned __int128 value = autos.size();
            if (op == Op::HolomorphOrder) value *= n;
            if (op == Op::InnerAutIndex) value = value * center_size(tables.at(i), n) / n;
            if (value > UINT64_MAX) return fail(INVALID, "automorphism answer does not fit in u64");
            accs[thread].integer(i, (uint64_t)value);
        }
        return ok();
    });
    for (const auto &status : statuses)
        if (!status.ok) return R::failure(status.error.status, status.error.message);
    return assemble(req, reduction, accs, shared);
}

R run_generators(const Request &req) {
    if (req.reduction != "all") return R::failure(INVALID, "aut_generators only accepts the all reduction");
    const Matrix &tables = *req.family->data;
    uint64_t count = tables.count, n = tables.rows;
    std::vector<std::vector<std::vector<Entry>>> per_group(count);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, count));
    auto statuses = parallel_ranges(count, threads, [&](uint32_t, uint64_t begin, uint64_t end) -> Status {
        for (uint64_t i = begin; i < end; ++i) {
            auto autos = AutomorphismSearch(tables.at(i), n).run();
            per_group[i] = canonical_generators(n, autos);
        }
        return ok();
    });
    for (const auto &status : statuses)
        if (!status.ok) return R::failure(status.error.status, status.error.message);

    auto result = std::make_shared<PermutationGenerators>();
    result->count = count;
    result->order = n;
    result->offsets.push_back(0);
    for (const auto &generators : per_group) {
        for (const auto &g : generators) result->entries.insert(result->entries.end(), g.begin(), g.end());
        result->offsets.push_back(result->offsets.back() + generators.size());
    }
    auto object = std::make_shared<Object>();
    object->kind = "automorphisms.generators";
    object->permutation_generators = std::move(result);
    return R::success(object);
}

R run(const Request &req) {
    if (req.family->kind != Family::Kind::GroupTables)
        return R::failure(INVALID, "automorphisms operations need a group_tables family");
    if (req.op == "aut_order") return run_integer(req, Op::AutOrder);
    if (req.op == "holomorph_order") return run_integer(req, Op::HolomorphOrder);
    if (req.op == "inner_aut_index") return run_integer(req, Op::InnerAutIndex);
    if (req.op == "aut_generators") return run_generators(req);
    return R::failure(4, "unknown automorphisms operation " + req.op);
}

BackendRegistration registration{Backend{
    "automorphisms", "generic",
    [] { return true; },
    [](const Request &req) { return req.family->kind == Family::Kind::GroupTables; },
    run,
    0}};

} // namespace
} // namespace lk::automorphisms
