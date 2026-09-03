#include "../../../../runtime/src/reduce.hpp"

#include <limits>
#include <unordered_map>

namespace lk::difference_sets {
namespace {

using R = Result<std::shared_ptr<Object>>;
constexpr int INVALID = 1;
constexpr int OUT_OF_MEMORY = 3;
constexpr int INTERNAL = 4;

std::string key(const Entry *g, uint64_t n) {
    return std::string(reinterpret_cast<const char *>(g), n * sizeof(Entry));
}

struct GroupData {
    uint64_t v, n, identity;
    std::vector<Entry> elements;
    std::vector<Entry> inverses;
    std::vector<uint32_t> inverse_index;
    std::unordered_map<std::string, uint32_t> index;

    explicit GroupData(const Matrix &dictionary)
        : v(dictionary.count), n(dictionary.cols), identity(0), elements(dictionary.entries),
          inverses(v * n), inverse_index(v) {
        index.reserve(v * 2);
        for (uint64_t g = 0; g < v; ++g)
            index.emplace(key(elements.data() + g * n, n), static_cast<uint32_t>(g));
        std::vector<Entry> id(n);
        for (uint64_t i = 0; i < n; ++i) id[i] = static_cast<Entry>(i);
        identity = index.at(key(id.data(), n));
        for (uint64_t g = 0; g < v; ++g) {
            const Entry *p = elements.data() + g * n;
            Entry *inv = inverses.data() + g * n;
            for (uint64_t i = 0; i < n; ++i) inv[p[i]] = static_cast<Entry>(i);
            inverse_index[g] = index.at(key(inv, n));
        }
    }

    uint32_t element_index(const Entry *g) const {
        auto it = index.find(key(g, n));
        return it == index.end() ? static_cast<uint32_t>(v) : it->second;
    }

    uint32_t quotient(uint32_t x, uint32_t y, std::vector<Entry> &scratch) const {
        const Entry *a = elements.data() + static_cast<uint64_t>(x) * n;
        const Entry *yi = inverses.data() + static_cast<uint64_t>(y) * n;
        scratch.resize(n);
        for (uint64_t i = 0; i < n; ++i) scratch[i] = a[yi[i]];
        return index.at(key(scratch.data(), n));
    }
};

enum class Op { DifferenceSet, Pds, Relative };

struct Setup {
    const Family *family;
    std::shared_ptr<GroupData> group;
    Op op = Op::DifferenceSet;
    uint64_t k, lambda = 0, mu = 0;
    std::vector<uint8_t> forbidden;
    bool impossible = false;
};

Result<Setup> setup(const Request &req, bool predicate) {
    using S = Result<Setup>;
    const Family &f = *req.family;
    if (f.kind != Family::Kind::SubsetsOf || !f.child || f.child->kind != Family::Kind::GroupElements)
        return S::failure(INVALID, "difference_sets needs subsets_of(group_elements(G), k)");

    Setup s;
    s.family = &f;
    s.k = f.k;
    try {
        s.group = std::make_shared<GroupData>(*f.data);
    } catch (const std::out_of_range &) {
        return S::failure(INTERNAL, "ambient group dictionary is inconsistent");
    }
    if (!predicate) return S::success(std::move(s));

    unsigned __int128 pairs = static_cast<unsigned __int128>(s.k) * (s.k - 1);
    if (req.op == "is_difference_set") {
        s.op = Op::DifferenceSet;
        if (s.group->v <= 1 || pairs % (s.group->v - 1)) s.impossible = true;
        else s.lambda = static_cast<uint64_t>(pairs / (s.group->v - 1));
        return S::success(std::move(s));
    }
    if (req.op == "is_pds") {
        s.op = Op::Pds;
        s.lambda = req.int_args.at("lambda");
        s.mu = req.int_args.at("mu");
        if (s.k >= s.group->v) s.impossible = true;
        else {
            unsigned __int128 target = static_cast<unsigned __int128>(s.k) * s.lambda +
                                       static_cast<unsigned __int128>(s.group->v - 1 - s.k) * s.mu;
            if (pairs != target) s.impossible = true;
        }
        return S::success(std::move(s));
    }
    if (req.op != "is_relative_difference_set")
        return S::failure(INTERNAL, "unknown difference_sets predicate " + req.op);

    s.op = Op::Relative;
    auto it = req.handle_args.find("forbidden");
    if (it == req.handle_args.end() || !it->second->matrix)
        return S::failure(INVALID, "forbidden must generate a subgroup of the ambient group");
    const Matrix &gens = *it->second->matrix;
    if (gens.p != 0 || gens.rows != 1 || gens.count == 0 || gens.cols != s.group->n)
        return S::failure(INVALID, "forbidden must generate a subgroup of the ambient group");
    auto closure = permutation_closure(gens, 1ULL << 26);
    if (!closure.ok) return S::failure(closure.error.status, closure.error.message);
    s.forbidden.assign(s.group->v, 0);
    uint64_t subgroup_order = closure.value.size() / s.group->n;
    for (uint64_t i = 0; i < subgroup_order; ++i) {
        uint32_t g = s.group->element_index(closure.value.data() + i * s.group->n);
        if (g == s.group->v)
            return S::failure(INVALID, "forbidden must generate a subgroup of the ambient group");
        s.forbidden[g] = 1;
    }
    uint64_t outside = s.group->v - subgroup_order;
    if (outside == 0 || pairs % outside) s.impossible = true;
    else s.lambda = static_cast<uint64_t>(pairs / outside);
    return S::success(std::move(s));
}

struct PredicateVisitor : Family::Visitor {
    const Setup &setup;
    Accumulator &acc;
    std::vector<uint32_t> selected;
    std::vector<uint8_t> chosen;
    std::vector<uint64_t> counts;
    std::vector<Entry> scratch;
    struct Frame {
        bool active;
        uint32_t element;
        std::vector<uint32_t> changed;
    };
    std::vector<Frame> frames;
    bool bad_row = false;

    PredicateVisitor(const Setup &s, Accumulator &a)
        : setup(s), acc(a), chosen(s.group->v, 0), counts(s.group->v, 0) {}

    bool possible(uint32_t newest) const {
        const GroupData &g = *setup.group;
        if (setup.op == Op::DifferenceSet) {
            for (uint64_t i = 0; i < g.v; ++i)
                if (i != g.identity && counts[i] > setup.lambda) return false;
            return true;
        }
        if (setup.op == Op::Relative) {
            for (uint64_t i = 0; i < g.v; ++i) {
                if (i == g.identity) continue;
                if (setup.forbidden[i] ? counts[i] != 0 : counts[i] > setup.lambda) return false;
            }
            return true;
        }
        if (chosen[g.identity]) return false;
        for (uint32_t x : selected)
            if (g.inverse_index[x] < newest && !chosen[g.inverse_index[x]]) return false;
        uint64_t undecided_limit = std::max(setup.lambda, setup.mu);
        for (uint64_t i = 0; i < g.v; ++i) {
            if (i == g.identity) continue;
            uint64_t limit = chosen[i] ? setup.lambda : i < newest ? setup.mu : undecided_limit;
            if (counts[i] > limit) return false;
        }
        return true;
    }

    bool final_value() const {
        const GroupData &g = *setup.group;
        if (setup.op == Op::DifferenceSet) {
            for (uint64_t i = 0; i < g.v; ++i)
                if (i != g.identity && counts[i] != setup.lambda) return false;
            return true;
        }
        if (setup.op == Op::Relative) {
            for (uint64_t i = 0; i < g.v; ++i) {
                if (i == g.identity) continue;
                if (counts[i] != (setup.forbidden[i] ? 0 : setup.lambda)) return false;
            }
            return true;
        }
        if (chosen[g.identity]) return false;
        for (uint32_t x : selected)
            if (!chosen[g.inverse_index[x]]) return false;
        for (uint64_t i = 0; i < g.v; ++i) {
            if (i == g.identity) continue;
            if (counts[i] != (chosen[i] ? setup.lambda : setup.mu)) return false;
        }
        return true;
    }

    Step push(const Entry *row, uint64_t first, uint64_t) override {
        if (acc.exhausted(first)) {
            frames.push_back(Frame{false, 0, {}});
            return Step::Skip;
        }
        uint32_t x = setup.group->element_index(row);
        if (x == setup.group->v) {
            bad_row = true;
            frames.push_back(Frame{false, 0, {}});
            return Step::Skip;
        }
        Frame frame{true, x, {}};
        frame.changed.reserve(2 * selected.size());
        for (uint32_t y : selected) {
            uint32_t xy = setup.group->quotient(x, y, scratch);
            uint32_t yx = setup.group->quotient(y, x, scratch);
            ++counts[xy];
            ++counts[yx];
            frame.changed.push_back(xy);
            frame.changed.push_back(yx);
        }
        selected.push_back(x);
        chosen[x] = 1;
        frames.push_back(std::move(frame));
        return possible(x) ? Step::Descend : Step::Skip;
    }

    void pop() override {
        Frame frame = std::move(frames.back());
        frames.pop_back();
        if (!frame.active) return;
        for (uint32_t g : frame.changed) --counts[g];
        chosen[frame.element] = 0;
        selected.pop_back();
    }

    void leaf(uint64_t index) override { acc.boolean(index, final_value()); }
    void take_all(uint64_t first, uint64_t count) override { acc.booleans(first, count, true); }
    void skip_all(uint64_t first, uint64_t count) override { acc.booleans(first, count, false); }
};

R run_predicate(const Request &req) {
    auto sr = setup(req, true);
    if (!sr.ok) return R::failure(sr.error.status, sr.error.message);
    Setup &s = sr.value;
    auto size_r = s.family->size();
    if (!size_r.ok) return R::failure(size_r.error.status, size_r.error.message);
    uint64_t size = size_r.value;
    Reduction reduction = parse_reduction(req.reduction);
    Shared shared;
    auto st = prepare_all(reduction, size, shared);
    if (!st.ok) return R::failure(st.error.status, st.error.message);

    if (s.impossible) {
        std::vector<Accumulator> accs;
        accs.emplace_back(reduction, &shared);
        accs[0].booleans(0, size, false);
        return assemble(req, reduction, accs, shared);
    }

    auto top_r = s.family->top_count();
    if (!top_r.ok) return R::failure(top_r.error.status, top_r.error.message);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, top_r.value));
    std::vector<Accumulator> accs;
    for (uint32_t t = 0; t < threads; ++t) accs.emplace_back(reduction, &shared);
    auto statuses = parallel_ranges(top_r.value, threads, [&](uint32_t t, uint64_t begin, uint64_t end) -> Status {
        PredicateVisitor visitor(s, accs[t]);
        auto walk = s.family->enumerate(visitor, begin, end);
        if (!walk.ok) return walk;
        if (visitor.bad_row) return fail(INTERNAL, "candidate row is not an ambient group element");
        return ok();
    });
    for (const auto &status : statuses)
        if (!status.ok) return R::failure(status.error.status, status.error.message);
    return assemble(req, reduction, accs, shared);
}

R run_multiset(const Request &req) {
    auto sr = setup(req, false);
    if (!sr.ok) return R::failure(sr.error.status, sr.error.message);
    const Setup &s = sr.value;
    auto size_r = s.family->size();
    if (!size_r.ok) return R::failure(size_r.error.status, size_r.error.message);
    uint64_t size = size_r.value;
    unsigned __int128 total = static_cast<unsigned __int128>(size) * s.group->v;
    if (total > std::numeric_limits<size_t>::max())
        return R::failure(OUT_OF_MEMORY, "difference_multiset output is too large");

    auto out = std::make_shared<Matrix>();
    out->p = NATURALS;
    out->count = size;
    out->rows = 1;
    out->cols = s.group->v;
    try {
        out->entries.assign(static_cast<size_t>(total), 0);
    } catch (const std::bad_alloc &) {
        return R::failure(OUT_OF_MEMORY, "difference_multiset output allocation failed");
    }

    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, size));
    auto statuses = parallel_ranges(size, threads, [&](uint32_t, uint64_t begin, uint64_t end) -> Status {
        Matrix member;
        std::vector<uint32_t> selected(s.k);
        std::vector<Entry> scratch;
        for (uint64_t i = begin; i < end; ++i) {
            auto st = s.family->member_into(i, member);
            if (!st.ok) return st;
            for (uint64_t j = 0; j < s.k; ++j) {
                selected[j] = s.group->element_index(member.entries.data() + j * s.group->n);
                if (selected[j] == s.group->v) return fail(INTERNAL, "candidate row is not an ambient group element");
            }
            Entry *counts = out->entries.data() + i * s.group->v;
            for (uint32_t x : selected)
                for (uint32_t y : selected)
                    ++counts[s.group->quotient(x, y, scratch)];
        }
        return ok();
    });
    for (const auto &status : statuses)
        if (!status.ok) return R::failure(status.error.status, status.error.message);

    auto object = std::make_shared<Object>();
    object->kind = "lk.naturals";
    object->matrix = out;
    return R::success(object);
}

R run(const Request &req) {
    if (req.op == "difference_multiset") return run_multiset(req);
    return run_predicate(req);
}

BackendRegistration registration{Backend{
    "difference_sets", "generic",
    [] { return true; },
    [](const Request &) { return true; },
    run,
    0}};

} // namespace
} // namespace lk::difference_sets
