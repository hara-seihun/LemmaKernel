#include "../../../../runtime/src/reduce.hpp"

#include <deque>
#include <map>
#include <numeric>
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace lk::cayley {
namespace {

using R = Result<std::shared_ptr<Object>>;
constexpr int INVALID = 1;
constexpr int INTERNAL = 4;

std::string perm_key(const Entry *p, uint64_t n) {
    return std::string(reinterpret_cast<const char *>(p), n * sizeof(Entry));
}

struct GroupModel {
    uint64_t action_n = 0;
    uint64_t order = 0;
    uint64_t identity = 0;
    std::vector<Entry> elements;
    std::unordered_map<std::string, uint32_t> index;
    std::vector<uint32_t> table;
    std::vector<uint32_t> inverse;

    uint32_t mul(uint64_t a, uint64_t b) const { return table[a * order + b]; }

    static Result<GroupModel> build(const Matrix &generators) {
        using G = Result<GroupModel>;
        auto closure = permutation_closure(generators, 1ULL << 26);
        if (!closure.ok) return G::failure(closure.error.status, closure.error.message);
        GroupModel g;
        g.action_n = generators.cols;
        g.elements = std::move(closure.value);
        g.order = g.elements.size() / g.action_n;
        for (uint64_t i = 0; i < g.order; ++i)
            g.index.emplace(perm_key(g.elements.data() + i * g.action_n, g.action_n), (uint32_t)i);
        std::vector<Entry> identity(g.action_n);
        std::iota(identity.begin(), identity.end(), 0);
        auto ei = g.index.find(perm_key(identity.data(), g.action_n));
        if (ei == g.index.end()) return G::failure(INTERNAL, "group closure has no identity");
        g.identity = ei->second;
        g.table.resize(g.order * g.order);
        std::vector<Entry> product(g.action_n);
        for (uint64_t a = 0; a < g.order; ++a) {
            const Entry *pa = g.elements.data() + a * g.action_n;
            for (uint64_t b = 0; b < g.order; ++b) {
                const Entry *pb = g.elements.data() + b * g.action_n;
                for (uint64_t x = 0; x < g.action_n; ++x) product[x] = pb[pa[x]];
                auto it = g.index.find(perm_key(product.data(), g.action_n));
                if (it == g.index.end()) return G::failure(INTERNAL, "group closure is not closed under composition");
                g.table[a * g.order + b] = it->second;
            }
        }
        g.inverse.assign(g.order, (uint32_t)g.identity);
        for (uint64_t a = 0; a < g.order; ++a)
            for (uint64_t b = 0; b < g.order; ++b)
                if (g.mul(a, b) == g.identity && g.mul(b, a) == g.identity) {
                    g.inverse[a] = (uint32_t)b;
                    break;
                }
        return G::success(std::move(g));
    }
};

struct Graph {
    uint64_t n = 0;
    std::vector<uint8_t> a;
    bool adj(uint64_t x, uint64_t y) const { return a[x * n + y] != 0; }
};

Result<std::vector<uint8_t>> connection_set(const GroupModel &g, const Matrix &member) {
    using S = Result<std::vector<uint8_t>>;
    if (member.cols != g.action_n)
        return S::failure(INVALID, "connection rows must be permutations on the same points as the group generators");
    std::vector<uint8_t> selected(g.order, 0);
    for (uint64_t r = 0; r < member.rows; ++r) {
        auto it = g.index.find(perm_key(member.entries.data() + r * member.cols, member.cols));
        if (it == g.index.end())
            return S::failure(INVALID, "connection row " + std::to_string(r) + " is not an element of G");
        uint64_t x = it->second;
        if (x == g.identity) return S::failure(INVALID, "the identity is not allowed in a connection set");
        if (selected[x]) return S::failure(INVALID, "connection rows must be distinct");
        selected[x] = 1;
    }
    return S::success(std::move(selected));
}

bool inverse_closed(const GroupModel &g, const std::vector<uint8_t> &s) {
    for (uint64_t x = 0; x < g.order; ++x)
        if (s[x] && !s[g.inverse[x]]) return false;
    return true;
}

Graph cayley_graph(const GroupModel &g, const std::vector<uint8_t> &s) {
    Graph graph{g.order, std::vector<uint8_t>(g.order * g.order, 0)};
    for (uint64_t x = 0; x < g.order; ++x)
        for (uint64_t y = x + 1; y < g.order; ++y) {
            uint64_t xy = g.mul(g.inverse[x], y);
            uint64_t yx = g.mul(g.inverse[y], x);
            graph.a[x * g.order + y] = graph.a[y * g.order + x] = s[xy] || s[yx];
        }
    return graph;
}

std::vector<int64_t> distances(const Graph &g, uint64_t start, int64_t block_a = -1, int64_t block_b = -1) {
    std::vector<int64_t> d(g.n, -1);
    std::deque<uint64_t> q;
    d[start] = 0;
    q.push_back(start);
    while (!q.empty()) {
        uint64_t x = q.front();
        q.pop_front();
        for (uint64_t y = 0; y < g.n; ++y) {
            if (!g.adj(x, y) || d[y] >= 0) continue;
            if (((int64_t)x == block_a && (int64_t)y == block_b) ||
                ((int64_t)x == block_b && (int64_t)y == block_a)) continue;
            d[y] = d[x] + 1;
            q.push_back(y);
        }
    }
    return d;
}

bool connected(const Graph &g) {
    auto d = distances(g, 0);
    return std::all_of(d.begin(), d.end(), [](int64_t x) { return x >= 0; });
}

bool regular_of_degree(const Graph &g, uint64_t degree) {
    for (uint64_t x = 0; x < g.n; ++x) {
        uint64_t d = 0;
        for (uint64_t y = 0; y < g.n; ++y) d += g.adj(x, y);
        if (d != degree) return false;
    }
    return true;
}

uint64_t graph_girth(const Graph &g) {
    uint64_t best = UINT64_MAX;
    for (uint64_t x = 0; x < g.n; ++x)
        for (uint64_t y = x + 1; y < g.n; ++y) {
            if (!g.adj(x, y)) continue;
            auto d = distances(g, x, (int64_t)x, (int64_t)y);
            if (d[y] >= 0) best = std::min(best, (uint64_t)d[y] + 1);
        }
    return best == UINT64_MAX ? 0 : best;
}

uint64_t graph_diameter(const Graph &g) {
    uint64_t diameter = 0;
    for (uint64_t x = 0; x < g.n; ++x) {
        auto d = distances(g, x);
        for (int64_t y : d) {
            if (y < 0) return 0;
            diameter = std::max(diameter, (uint64_t)y);
        }
    }
    return diameter;
}

struct Canonical {
    std::string key;
    uint64_t multiplicity = 0;
    bool overflow = false;
};

struct Canonicalizer {
    const Graph &g;
    Canonical best;

    explicit Canonicalizer(const Graph &graph) : g(graph) {}

    std::vector<uint32_t> refine(std::vector<uint32_t> colors) const {
        for (;;) {
            uint32_t classes = *std::max_element(colors.begin(), colors.end()) + 1;
            std::vector<std::vector<uint32_t>> signatures(g.n, std::vector<uint32_t>(classes + 1, 0));
            for (uint64_t v = 0; v < g.n; ++v) {
                signatures[v][0] = colors[v];
                for (uint64_t w = 0; w < g.n; ++w)
                    if (g.adj(v, w)) ++signatures[v][colors[w] + 1];
            }
            std::map<std::vector<uint32_t>, uint32_t> ids;
            for (const auto &s : signatures) ids.emplace(s, 0);
            uint32_t next_id = 0;
            for (auto &[_, id] : ids) id = next_id++;
            std::vector<uint32_t> next(g.n);
            for (uint64_t v = 0; v < g.n; ++v) next[v] = ids[signatures[v]];
            if (next == colors) return colors;
            colors.swap(next);
        }
    }

    std::string leaf_key(const std::vector<uint32_t> &colors) const {
        std::vector<uint64_t> order(g.n);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](uint64_t x, uint64_t y) { return colors[x] < colors[y]; });
        std::string key(g.n * g.n, '\0');
        for (uint64_t i = 0; i < g.n; ++i)
            for (uint64_t j = 0; j < g.n; ++j) key[i * g.n + j] = (char)g.adj(order[i], order[j]);
        return key;
    }

    void search(std::vector<uint32_t> colors) {
        colors = refine(std::move(colors));
        std::vector<uint64_t> sizes(g.n, 0);
        for (uint32_t c : colors) ++sizes[c];
        uint32_t cell = UINT32_MAX;
        for (uint32_t c = 0; c < g.n; ++c)
            if (sizes[c] > 1) { cell = c; break; }
        if (cell == UINT32_MAX) {
            std::string key = leaf_key(colors);
            if (best.multiplicity == 0 || key < best.key) best = Canonical{std::move(key), 1, false};
            else if (key == best.key) {
                if (best.multiplicity == UINT64_MAX) best.overflow = true;
                else ++best.multiplicity;
            }
            return;
        }
        for (uint64_t chosen = 0; chosen < g.n; ++chosen) {
            if (colors[chosen] != cell) continue;
            std::vector<uint32_t> split(g.n);
            for (uint64_t v = 0; v < g.n; ++v) {
                if (colors[v] < cell) split[v] = 2 * colors[v];
                else if (colors[v] == cell) split[v] = 2 * cell + (v == chosen ? 0 : 1);
                else split[v] = 2 * colors[v] + 1;
            }
            search(std::move(split));
        }
    }

    Result<Canonical> run() {
        using C = Result<Canonical>;
        if (g.n == 0) return C::failure(INTERNAL, "cannot canonicalize an empty graph");
        bool empty = true, complete = true;
        for (uint64_t x = 0; x < g.n; ++x)
            for (uint64_t y = x + 1; y < g.n; ++y) {
                empty &= !g.adj(x, y);
                complete &= g.adj(x, y);
            }
        if (empty || complete) {
            uint64_t fact = 1;
            for (uint64_t i = 2; i <= g.n; ++i) {
                if (fact > UINT64_MAX / i) return C::failure(INVALID, "graph automorphism order does not fit in 64 bits");
                fact *= i;
            }
            std::string key(g.n * g.n, '\0');
            if (complete)
                for (uint64_t x = 0; x < g.n; ++x)
                    for (uint64_t y = 0; y < g.n; ++y) key[x * g.n + y] = x != y;
            return C::success(Canonical{std::move(key), fact, false});
        }
        search(std::vector<uint32_t>(g.n, 0));
        if (best.overflow) return C::failure(INVALID, "graph automorphism order does not fit in 64 bits");
        return C::success(std::move(best));
    }
};

uint64_t element_order(const GroupModel &g, uint64_t x) {
    uint64_t y = g.identity;
    for (uint64_t k = 1; k <= g.order; ++k) {
        y = g.mul(x, y);
        if (y == g.identity) return k;
    }
    return 0;
}

std::vector<uint8_t> generated_by(const GroupModel &g, const std::vector<uint32_t> &gens) {
    std::vector<uint8_t> seen(g.order, 0);
    std::deque<uint32_t> q;
    seen[g.identity] = 1;
    q.push_back((uint32_t)g.identity);
    while (!q.empty()) {
        uint32_t a = q.front();
        q.pop_front();
        for (uint32_t x : gens) {
            uint32_t b = g.mul(x, a);
            if (!seen[b]) { seen[b] = 1; q.push_back(b); }
        }
    }
    return seen;
}

Result<std::vector<std::vector<uint32_t>>> group_automorphisms(const GroupModel &g) {
    using A = Result<std::vector<std::vector<uint32_t>>>;
    std::vector<uint32_t> gens;
    std::vector<uint8_t> span(g.order, 0);
    span[g.identity] = 1;
    for (uint32_t x = 0; x < g.order; ++x) {
        if (span[x]) continue;
        gens.push_back(x);
        span = generated_by(g, gens);
    }
    std::vector<uint64_t> orders(g.order);
    for (uint64_t x = 0; x < g.order; ++x) orders[x] = element_order(g, x);
    std::vector<std::vector<uint32_t>> candidates(gens.size());
    for (uint64_t i = 0; i < gens.size(); ++i)
        for (uint32_t x = 0; x < g.order; ++x)
            if (orders[x] == orders[gens[i]]) candidates[i].push_back(x);

    std::vector<std::vector<uint32_t>> auts;
    std::vector<uint32_t> images(gens.size());
    uint64_t tried = 0;
    bool too_many = false;
    auto extend = [&]() {
        if (++tried > (1ULL << 26)) { too_many = true; return; }
        std::vector<int64_t> phi(g.order, -1);
        phi[g.identity] = (int64_t)g.identity;
        std::deque<uint32_t> q;
        q.push_back((uint32_t)g.identity);
        bool ok = true;
        while (!q.empty() && ok) {
            uint32_t a = q.front();
            q.pop_front();
            for (uint64_t i = 0; i < gens.size(); ++i) {
                uint32_t b = g.mul(gens[i], a);
                uint32_t image = g.mul(images[i], (uint32_t)phi[a]);
                if (phi[b] < 0) { phi[b] = image; q.push_back(b); }
                else if ((uint32_t)phi[b] != image) { ok = false; break; }
            }
        }
        if (!ok || std::any_of(phi.begin(), phi.end(), [](int64_t x) { return x < 0; })) return;
        std::vector<uint8_t> used(g.order, 0);
        for (int64_t x : phi) {
            if (used[x]) return;
            used[x] = 1;
        }
        auts.emplace_back(phi.begin(), phi.end());
    };
    auto enumerate = [&](auto &self, uint64_t i) -> void {
        if (too_many) return;
        if (i == gens.size()) { extend(); return; }
        for (uint32_t x : candidates[i]) { images[i] = x; self(self, i + 1); if (too_many) return; }
    };
    enumerate(enumerate, 0);
    if (too_many) return A::failure(INVALID, "is_ci_set needs more than 2^26 candidate generator images for Aut(G)");
    if (auts.empty()) return A::failure(INTERNAL, "failed to find the identity automorphism of G");
    return A::success(std::move(auts));
}

std::string set_key(const std::vector<uint8_t> &s) {
    return std::string(reinterpret_cast<const char *>(s.data()), s.size());
}

struct CiData {
    std::vector<std::vector<uint8_t>> sets;
    std::vector<std::string> canonical;
    std::vector<std::vector<uint32_t>> automorphisms;
};

Result<CiData> build_ci_data(const GroupModel &g, uint64_t k) {
    using C = Result<CiData>;
    std::vector<std::vector<uint32_t>> atoms;
    std::vector<uint8_t> seen(g.order, 0);
    seen[g.identity] = 1;
    for (uint32_t x = 0; x < g.order; ++x) {
        if (seen[x]) continue;
        uint32_t y = g.inverse[x];
        seen[x] = seen[y] = 1;
        if (x == y) atoms.push_back({x});
        else atoms.push_back({std::min(x, y), std::max(x, y)});
    }
    std::sort(atoms.begin(), atoms.end());
    CiData data;
    std::vector<uint8_t> current(g.order, 0);
    bool too_many = false;
    auto choose = [&](auto &self, uint64_t i, uint64_t used) -> void {
        if (too_many || used > k) return;
        if (i == atoms.size()) {
            if (used == k) {
                if (data.sets.size() >= (1ULL << 24)) { too_many = true; return; }
                data.sets.push_back(current);
            }
            return;
        }
        uint64_t weight = atoms[i].size();
        if (used + weight <= k) {
            for (uint32_t x : atoms[i]) current[x] = 1;
            self(self, i + 1, used + weight);
            for (uint32_t x : atoms[i]) current[x] = 0;
        }
        self(self, i + 1, used);
    };
    choose(choose, 0, 0);
    if (too_many) return C::failure(INVALID, "is_ci_set has more than 2^24 inverse-closed connection sets of this size");
    auto auts = group_automorphisms(g);
    if (!auts.ok) return C::failure(auts.error.status, auts.error.message);
    data.automorphisms = std::move(auts.value);
    data.canonical.reserve(data.sets.size());
    for (const auto &s : data.sets) {
        Graph graph = cayley_graph(g, s);
        auto c = Canonicalizer{graph}.run();
        if (!c.ok) return C::failure(c.error.status, c.error.message);
        data.canonical.push_back(std::move(c.value.key));
    }
    return C::success(std::move(data));
}

Result<bool> is_ci_set(const GroupModel &g, const CiData &data, const std::vector<uint8_t> &s) {
    using B = Result<bool>;
    if (!inverse_closed(g, s)) return B::success(false);
    Graph graph = cayley_graph(g, s);
    auto canonical = Canonicalizer{graph}.run();
    if (!canonical.ok) return B::failure(canonical.error.status, canonical.error.message);
    std::unordered_set<std::string> orbit;
    std::vector<uint8_t> image(g.order);
    for (const auto &phi : data.automorphisms) {
        std::fill(image.begin(), image.end(), 0);
        for (uint64_t x = 0; x < g.order; ++x)
            if (s[x]) image[phi[x]] = 1;
        orbit.insert(set_key(image));
    }
    for (uint64_t i = 0; i < data.sets.size(); ++i)
        if (data.canonical[i] == canonical.value.key && !orbit.count(set_key(data.sets[i])))
            return B::success(false);
    return B::success(true);
}

Result<GroupModel> setup_group(const Request &req) {
    using G = Result<GroupModel>;
    auto it = req.handle_args.find("group");
    if (it == req.handle_args.end() || !it->second->matrix || it->second->matrix->p != 0)
        return G::failure(INVALID, "`group` must be a permutation group given by orbits.perms generators");
    return GroupModel::build(*it->second->matrix);
}

R run(const Request &req) {
    auto group = setup_group(req);
    if (!group.ok) return R::failure(group.error.status, group.error.message);
    auto size_r = req.family->size();
    if (!size_r.ok) return R::failure(size_r.error.status, size_r.error.message);
    uint64_t size = size_r.value;
    Reduction reduction = parse_reduction(req.reduction);
    Shared shared;
    auto prepared = prepare_all(reduction, size, shared);
    if (!prepared.ok) return R::failure(prepared.error.status, prepared.error.message);

    std::shared_ptr<CiData> ci;
    if (req.op == "is_ci_set") {
        auto built = build_ci_data(group.value, req.family->rows());
        if (!built.ok) return R::failure(built.error.status, built.error.message);
        ci = std::make_shared<CiData>(std::move(built.value));
    }

    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, size ? size : 1));
    std::vector<Accumulator> accs;
    for (uint32_t t = 0; t < threads; ++t) accs.emplace_back(reduction, &shared);
    auto statuses = parallel_ranges(size, threads, [&](uint32_t t, uint64_t begin, uint64_t end) -> Status {
        Matrix member;
        for (uint64_t i = begin; i < end; ++i) {
            if (accs[t].exhausted(i)) break;
            auto st = req.family->member_into(i, member);
            if (!st.ok) return st;
            auto selected = connection_set(group.value, member);
            if (!selected.ok) return fail(selected.error.status, selected.error.message);
            Graph graph = cayley_graph(group.value, selected.value);
            if (req.op == "connected") accs[t].boolean(i, connected(graph));
            else if (req.op == "is_regular_of_degree") accs[t].boolean(i, regular_of_degree(graph, req.int_args.at("degree")));
            else if (req.op == "girth") accs[t].integer(i, graph_girth(graph));
            else if (req.op == "diameter") accs[t].integer(i, graph_diameter(graph));
            else if (req.op == "aut_order") {
                auto c = Canonicalizer{graph}.run();
                if (!c.ok) return fail(c.error.status, c.error.message);
                accs[t].integer(i, c.value.multiplicity);
            } else if (req.op == "is_ci_set") {
                auto value = is_ci_set(group.value, *ci, selected.value);
                if (!value.ok) return fail(value.error.status, value.error.message);
                accs[t].boolean(i, value.value);
            } else return fail(INTERNAL, "unknown cayley operation " + req.op);
        }
        return ok();
    });
    for (const auto &st : statuses)
        if (!st.ok) return R::failure(st.error.status, st.error.message);
    return assemble(req, reduction, accs, shared);
}

BackendRegistration registration{Backend{
    "cayley", "generic",
    [] { return true; },
    [](const Request &) { return true; },
    run,
    0}};

} // namespace
} // namespace lk::cayley
