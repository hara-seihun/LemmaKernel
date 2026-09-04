#include "../../../../runtime/src/group_table.hpp"
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

Result<std::vector<std::vector<uint32_t>>> group_automorphisms(const GroupModel &g) {
    using A = Result<std::vector<std::vector<uint32_t>>>;
    auto automorphisms = group_table::automorphisms(g.table.data(), g.order);
    if (automorphisms.empty())
        return A::failure(INTERNAL, "failed to find the identity automorphism of G");
    return A::success(std::move(automorphisms));
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

struct NonCiWitnessData {
    std::vector<uint8_t> target;
    Graph target_graph;
    std::vector<uint32_t> isomorphism;
    std::vector<std::vector<uint32_t>> automorphisms;
};

Result<NonCiWitnessData> setup_non_ci_witness(const Request &req, const GroupModel &group,
                                                   bool enumerate_all_automorphisms) {
    using W = Result<NonCiWitnessData>;
    auto target_it = req.handle_args.find("target");
    auto map_it = req.handle_args.find("isomorphism");
    if (target_it == req.handle_args.end() || !target_it->second->matrix ||
        target_it->second->matrix->p != 0)
        return W::failure(INVALID, "target must be a nonempty orbits.perms connection set");
    if (map_it == req.handle_args.end() || !map_it->second->matrix ||
        map_it->second->matrix->p != 0)
        return W::failure(INVALID, "isomorphism must be one permutation");
    const Matrix &target_perms = *target_it->second->matrix;
    const Matrix &map = *map_it->second->matrix;
    if (target_perms.count == 0 || target_perms.cols != group.action_n)
        return W::failure(INVALID, "target must be a nonempty connection set on the group's points");
    if (map.count != 1 || map.cols != group.action_n)
        return W::failure(INVALID, "isomorphism must be one permutation on the group's points");

    Matrix target_member{0, 1, target_perms.count, target_perms.cols, target_perms.entries};
    auto selected = connection_set(group, target_member);
    if (!selected.ok) return W::failure(selected.error.status, selected.error.message);
    if (group.order != group.action_n)
        return W::failure(INVALID, "is_non_ci_witness needs a regular permutation representation of G");
    std::vector<uint32_t> point_to_group(group.action_n, UINT32_MAX);
    for (uint32_t element = 0; element < group.order; ++element) {
        uint32_t point = group.elements[element * group.action_n];
        if (point_to_group[point] != UINT32_MAX)
            return W::failure(INVALID, "is_non_ci_witness needs a regular permutation representation of G");
        point_to_group[point] = element;
    }
    std::vector<uint32_t> induced(group.order);
    for (uint32_t element = 0; element < group.order; ++element) {
        uint32_t point = group.elements[element * group.action_n];
        uint32_t image = map.entries[point];
        if (point_to_group[image] == UINT32_MAX)
            return W::failure(INTERNAL, "regular action point has no group element");
        induced[element] = point_to_group[image];
    }
    std::vector<std::vector<uint32_t>> automorphisms;
    if (enumerate_all_automorphisms) {
        auto found = group_automorphisms(group);
        if (!found.ok) return W::failure(found.error.status, found.error.message);
        automorphisms = std::move(found.value);
    }
    return W::success(NonCiWitnessData{
        selected.value, cayley_graph(group, selected.value), std::move(induced),
        std::move(automorphisms)});
}

Result<std::vector<std::vector<uint32_t>>> supplied_automorphisms(
    const Request &req, const GroupModel &group) {
    using A = Result<std::vector<std::vector<uint32_t>>>;
    auto it = req.handle_args.find("automorphisms");
    if (it == req.handle_args.end() || !it->second->matrix || it->second->matrix->p != 0)
        return A::failure(INVALID, "automorphisms must be permutation generators");
    const Matrix &generators = *it->second->matrix;
    if (generators.count == 0 || generators.cols != group.action_n)
        return A::failure(INVALID, "automorphisms must act on the regular representation's points");
    std::vector<uint32_t> point_to_group(group.action_n, UINT32_MAX);
    for (uint32_t element = 0; element < group.order; ++element)
        point_to_group[group.elements[element * group.action_n]] = element;
    std::vector<std::vector<uint32_t>> induced;
    induced.reserve(generators.count);
    for (uint64_t generator = 0; generator < generators.count; ++generator) {
        const Entry *point_map = generators.at(generator);
        std::vector<uint32_t> map(group.order);
        for (uint32_t element = 0; element < group.order; ++element) {
            uint32_t point = group.elements[element * group.action_n];
            map[element] = point_to_group[point_map[point]];
        }
        if (map[group.identity] != group.identity)
            return A::failure(INVALID, "a supplied automorphism does not fix the identity");
        for (uint32_t x = 0; x < group.order; ++x)
            for (uint32_t y = 0; y < group.order; ++y)
                if (map[group.mul(x, y)] != group.mul(map[x], map[y]))
                    return A::failure(INVALID, "a supplied permutation is not a group automorphism");
        induced.push_back(std::move(map));
    }
    return A::success(std::move(induced));
}

bool same_selected_set(const std::vector<uint8_t> &source, const std::vector<uint8_t> &target,
                       const std::vector<uint32_t> &mapping) {
    for (uint64_t x = 0; x < source.size(); ++x)
        if (source[x] != target[mapping[x]]) return false;
    return true;
}

bool separated_under(const std::vector<uint8_t> &source, const std::vector<uint8_t> &target,
                     const std::vector<std::vector<uint32_t>> &generators) {
    std::unordered_set<std::string> seen;
    std::deque<std::vector<uint8_t>> queue;
    seen.insert(set_key(source));
    queue.push_back(source);
    std::vector<uint8_t> image(source.size());
    while (!queue.empty()) {
        auto current = std::move(queue.front());
        queue.pop_front();
        if (current == target) return false;
        for (const auto &generator : generators) {
            std::fill(image.begin(), image.end(), 0);
            for (uint64_t x = 0; x < current.size(); ++x)
                if (current[x]) image[generator[x]] = 1;
            std::string key = set_key(image);
            if (seen.insert(key).second) queue.push_back(image);
        }
    }
    return true;
}

bool is_non_ci_witness(const GroupModel &group, const std::vector<uint8_t> &source,
                       const Graph &source_graph, const NonCiWitnessData &data) {
    if (!inverse_closed(group, source) || !inverse_closed(group, data.target)) return false;
    for (uint64_t x = 0; x < group.order; ++x)
        for (uint64_t y = 0; y < group.order; ++y)
            if (source_graph.adj(x, y) !=
                data.target_graph.adj(data.isomorphism[x], data.isomorphism[y])) return false;
    for (const auto &automorphism : data.automorphisms)
        if (same_selected_set(source, data.target, automorphism)) return false;
    return true;
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
    std::shared_ptr<NonCiWitnessData> non_ci_witness;
    std::vector<std::vector<uint32_t>> automorphism_generators;
    if (req.op == "is_non_ci_witness" || req.op == "is_separated_witness") {
        auto built = setup_non_ci_witness(req, group.value, req.op == "is_non_ci_witness");
        if (!built.ok) return R::failure(built.error.status, built.error.message);
        non_ci_witness = std::make_shared<NonCiWitnessData>(std::move(built.value));
        if (req.op == "is_separated_witness") {
            auto supplied = supplied_automorphisms(req, group.value);
            if (!supplied.ok) return R::failure(supplied.error.status, supplied.error.message);
            automorphism_generators = std::move(supplied.value);
        }
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
            } else if (req.op == "is_non_ci_witness" || req.op == "is_separated_witness") {
                bool value = inverse_closed(group.value, selected.value) &&
                             inverse_closed(group.value, non_ci_witness->target);
                if (value)
                    for (uint64_t x = 0; x < group.value.order && value; ++x)
                        for (uint64_t y = 0; y < group.value.order; ++y)
                            if (graph.adj(x, y) != non_ci_witness->target_graph.adj(
                                    non_ci_witness->isomorphism[x],
                                    non_ci_witness->isomorphism[y])) {
                                value = false;
                                break;
                            }
                if (value && req.op == "is_non_ci_witness")
                    value = is_non_ci_witness(group.value, selected.value, graph, *non_ci_witness);
                else if (value)
                    value = separated_under(
                        selected.value, non_ci_witness->target, automorphism_generators);
                accs[t].boolean(i, value);
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
