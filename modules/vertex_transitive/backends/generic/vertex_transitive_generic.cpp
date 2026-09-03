/* Portable graph-transitivity and Cayley-recognition backend.
 *
 * Automorphisms use the same colour-refined partial-isomorphism search as graph_iso. Regular
 * subgroups are generated through canonical increasing generator sequences. Every intermediate
 * subgroup is semiregular and has order at most n, since only such a subgroup can extend to a
 * regular one. */
#include "../../../../runtime/src/reduce.hpp"

#include <algorithm>
#include <numeric>
#include <unordered_map>

namespace lk::vertex_transitive {
namespace {

using R = Result<std::shared_ptr<Object>>;
using Perm = std::vector<Entry>;
constexpr int INVALID = 1;

struct Graph {
    uint64_t n = 0;
    std::vector<Entry> adjacency;

    Entry at(uint64_t i, uint64_t j) const { return adjacency[i * n + j]; }
};

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

std::vector<uint32_t> refine(const Graph &graph, std::vector<uint32_t> colours) {
    for (;;) {
        uint32_t colour_count = colours.empty() ? 0 : *std::max_element(colours.begin(), colours.end()) + 1;
        std::vector<std::vector<uint32_t>> signatures(graph.n);
        for (uint64_t v = 0; v < graph.n; ++v) {
            auto &signature = signatures[v];
            signature.assign(colour_count + 1, 0);
            signature[0] = colours[v];
            for (uint64_t w = 0; w < graph.n; ++w)
                if (graph.at(v, w)) ++signature[1 + colours[w]];
        }
        std::vector<uint64_t> order(graph.n);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](uint64_t a, uint64_t b) {
            return signatures[a] < signatures[b];
        });
        std::vector<uint32_t> next(graph.n, 0);
        uint32_t id = 0;
        for (uint64_t i = 0; i < graph.n; ++i) {
            if (i && signatures[order[i - 1]] != signatures[order[i]]) ++id;
            next[order[i]] = id;
        }
        if (next == colours) return colours;
        colours.swap(next);
    }
}

struct AutomorphismSearch {
    const Graph &graph;
    std::vector<uint32_t> stable_colours;
    std::vector<int64_t> image;
    std::vector<uint8_t> used_targets;
    std::vector<uint32_t> sources;
    std::vector<Perm> found;

    explicit AutomorphismSearch(const Graph &g)
        : graph(g), stable_colours(refine(g, std::vector<uint32_t>(g.n, 0))), image(g.n, -1),
          used_targets(g.n, 0) {}

    bool refined_pair(std::vector<uint32_t> &source_colours, std::vector<uint32_t> &target_colours) const {
        uint32_t k = sources.size();
        source_colours.resize(graph.n);
        target_colours.resize(graph.n);
        for (uint64_t v = 0; v < graph.n; ++v) {
            source_colours[v] = k + stable_colours[v];
            target_colours[v] = k + stable_colours[v];
        }
        for (uint32_t i = 0; i < k; ++i) {
            source_colours[sources[i]] = i;
            target_colours[image[sources[i]]] = i;
        }
        source_colours = refine(graph, std::move(source_colours));
        target_colours = refine(graph, std::move(target_colours));
        std::vector<uint32_t> source_counts(graph.n, 0), target_counts(graph.n, 0);
        for (uint64_t v = 0; v < graph.n; ++v) {
            ++source_counts[source_colours[v]];
            ++target_counts[target_colours[v]];
        }
        if (source_counts != target_counts) return false;
        for (uint32_t source : sources)
            if (source_colours[source] != target_colours[image[source]]) return false;
        return true;
    }

    bool adjacency_compatible(uint32_t source, uint32_t target) const {
        for (uint32_t other : sources)
            if (graph.at(source, other) != graph.at(target, image[other])) return false;
        return true;
    }

    void search() {
        if (sources.size() == graph.n) {
            Perm permutation(graph.n);
            for (uint64_t v = 0; v < graph.n; ++v) permutation[v] = (Entry)image[v];
            found.push_back(std::move(permutation));
            return;
        }
        std::vector<uint32_t> source_colours, target_colours;
        if (!refined_pair(source_colours, target_colours)) return;

        uint32_t source = graph.n;
        uint64_t best_count = graph.n + 1;
        for (uint32_t v = 0; v < graph.n; ++v) {
            if (image[v] >= 0) continue;
            uint64_t count = 0;
            for (uint32_t w = 0; w < graph.n; ++w)
                if (!used_targets[w] && target_colours[w] == source_colours[v]) ++count;
            if (count < best_count || (count == best_count && v < source)) {
                source = v;
                best_count = count;
            }
        }
        if (best_count == 0) return;
        for (uint32_t target = 0; target < graph.n; ++target) {
            if (used_targets[target] || target_colours[target] != source_colours[source] ||
                !adjacency_compatible(source, target)) continue;
            image[source] = target;
            used_targets[target] = 1;
            sources.push_back(source);
            search();
            sources.pop_back();
            used_targets[target] = 0;
            image[source] = -1;
        }
    }
};

std::vector<Perm> automorphisms(const Graph &graph) {
    AutomorphismSearch search(graph);
    search.search();
    std::sort(search.found.begin(), search.found.end());
    return search.found;
}

struct PermHash {
    size_t operator()(const Perm &g) const noexcept {
        size_t h = 1469598103934665603ULL;
        for (Entry x : g) {
            h ^= x;
            h *= 1099511628211ULL;
        }
        return h;
    }
};

uint64_t permutation_order(const Perm &g) {
    std::vector<uint8_t> seen(g.size(), 0);
    uint64_t order = 1;
    for (uint64_t start = 0; start < g.size(); ++start) {
        if (seen[start]) continue;
        uint64_t length = 0, x = start;
        while (!seen[x]) {
            seen[x] = 1;
            ++length;
            x = g[x];
        }
        order = std::lcm(order, length);
    }
    return order;
}

bool fixed_point_free(const Perm &g) {
    for (uint64_t i = 0; i < g.size(); ++i)
        if (g[i] == i) return false;
    return true;
}

class RegularSearch {
  public:
    explicit RegularSearch(const std::vector<Perm> &group)
        : elements(group), n(group.front().size()) {
        index.reserve(elements.size() * 2);
        for (uint32_t i = 0; i < elements.size(); ++i) index.emplace(elements[i], i);
        id = index.at(identity(n));
        for (uint32_t i = 0; i < elements.size(); ++i) {
            if (i == id || !fixed_point_free(elements[i])) continue;
            if (n % permutation_order(elements[i]) == 0) candidates.push_back(i);
        }
    }

    std::vector<std::vector<uint32_t>> all() {
        stop_at_first = false;
        std::vector<uint32_t> subgroup{id};
        search(subgroup, {}, id);
        std::sort(found.begin(), found.end());
        found.erase(std::unique(found.begin(), found.end()), found.end());
        return found;
    }

    bool any() {
        stop_at_first = true;
        std::vector<uint32_t> subgroup{id};
        search(subgroup, {}, id);
        return !found.empty();
    }

  private:
    const std::vector<Perm> &elements;
    uint64_t n;
    uint32_t id;
    std::unordered_map<Perm, uint32_t, PermHash> index;
    std::vector<uint32_t> candidates;
    std::vector<std::vector<uint32_t>> found;
    bool stop_at_first = false;

    bool closure(const std::vector<uint32_t> &generators, std::vector<uint32_t> &out) const {
        std::vector<uint8_t> seen(elements.size(), 0);
        out.clear();
        out.push_back(id);
        seen[id] = 1;
        for (size_t front = 0; front < out.size(); ++front) {
            for (uint32_t generator : generators) {
                auto it = index.find(compose(elements[out[front]], elements[generator]));
                if (it == index.end()) return false;
                uint32_t y = it->second;
                if (seen[y]) continue;
                if (y != id && !fixed_point_free(elements[y])) return false;
                seen[y] = 1;
                out.push_back(y);
                if (out.size() > n) return false;
            }
        }
        std::sort(out.begin(), out.end());
        return true;
    }

    void search(const std::vector<uint32_t> &subgroup, const std::vector<uint32_t> &generators,
                uint32_t after) {
        std::vector<uint8_t> member(elements.size(), 0);
        for (uint32_t x : subgroup) member[x] = 1;
        for (uint32_t g : candidates) {
            if (g <= after || member[g]) continue;
            auto next_generators = generators;
            next_generators.push_back(g);
            std::vector<uint32_t> larger;
            if (!closure(next_generators, larger)) continue;
            auto added = std::find_if(larger.begin(), larger.end(), [&](uint32_t x) { return !member[x]; });
            if (added == larger.end() || *added != g) continue;
            if (larger.size() == n) {
                found.push_back(std::move(larger));
                if (stop_at_first) return;
            } else {
                search(larger, next_generators, g);
                if (stop_at_first && !found.empty()) return;
            }
        }
    }
};

Status read_graph(const Family &family, uint64_t index, Graph &graph, Matrix &member) {
    auto status = family.member_into(index, member);
    if (!status.ok) return status;
    graph.n = member.rows;
    graph.adjacency = member.entries;
    for (uint64_t i = 0; i < graph.n; ++i) {
        if (graph.at(i, i)) return fail(INVALID, "simple graph adjacency matrices must have zero diagonal");
        for (uint64_t j = i + 1; j < graph.n; ++j)
            if (graph.at(i, j) != graph.at(j, i))
                return fail(INVALID, "graph adjacency matrix " + std::to_string(index) + " is not symmetric");
    }
    return ok();
}

bool vertex_transitive(const std::vector<Perm> &group, uint64_t n) {
    std::vector<uint8_t> reached(n, 0);
    for (const auto &g : group) reached[g[0]] = 1;
    return std::all_of(reached.begin(), reached.end(), [](uint8_t x) { return x != 0; });
}

bool arc_transitive(const Graph &graph, const std::vector<Perm> &group) {
    uint64_t arc_count = 0;
    std::pair<Entry, Entry> first{0, 0};
    bool have_first = false;
    for (uint64_t i = 0; i < graph.n; ++i)
        for (uint64_t j = 0; j < graph.n; ++j)
            if (graph.at(i, j)) {
                if (!have_first) first = {(Entry)i, (Entry)j};
                have_first = true;
                ++arc_count;
            }
    if (!have_first) return true;
    std::vector<uint8_t> reached(graph.n * graph.n, 0);
    uint64_t orbit_size = 0;
    for (const auto &g : group) {
        uint64_t image = (uint64_t)g[first.first] * graph.n + g[first.second];
        if (!reached[image]) {
            reached[image] = 1;
            ++orbit_size;
        }
    }
    return orbit_size == arc_count;
}

enum class ScalarOp { VertexTransitive, ArcTransitive, Cayley };

Status validate_request(const Family &family) {
    if (family.prime() != 2) return fail(INVALID, "graph adjacency matrices must be over F_2");
    if (family.rows() == 0) return fail(INVALID, "graphs must have at least one vertex");
    if (family.rows() != family.cols()) return fail(INVALID, "graph adjacency matrices must be square");
    if (family.rows() > 10) return fail(INVALID, "the generic backend accepts at most 10 vertices");
    return ok();
}

R run_scalar(const Request &req, ScalarOp op) {
    auto status = validate_request(*req.family);
    if (!status.ok) return R::failure(status.error.status, status.error.message);
    auto size_result = req.family->size();
    if (!size_result.ok) return R::failure(size_result.error.status, size_result.error.message);
    uint64_t size = size_result.value;
    if (size == 0) return R::failure(INVALID, "need at least one graph");

    Reduction reduction = parse_reduction(req.reduction);
    Shared shared;
    status = prepare_all(reduction, size, shared);
    if (!status.ok) return R::failure(status.error.status, status.error.message);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, size));
    std::vector<Accumulator> accs;
    for (uint32_t t = 0; t < threads; ++t) accs.emplace_back(reduction, &shared);
    auto statuses = parallel_ranges(size, threads, [&](uint32_t t, uint64_t begin, uint64_t end) -> Status {
        Matrix member;
        Graph graph;
        for (uint64_t i = begin; i < end; ++i) {
            if (accs[t].exhausted(i)) break;
            auto st = read_graph(*req.family, i, graph, member);
            if (!st.ok) return st;
            auto group = automorphisms(graph);
            bool value;
            if (op == ScalarOp::VertexTransitive) value = vertex_transitive(group, graph.n);
            else if (op == ScalarOp::ArcTransitive) value = arc_transitive(graph, group);
            else value = vertex_transitive(group, graph.n) && RegularSearch(group).any();
            accs[t].boolean(i, value);
        }
        return ok();
    });
    for (const auto &st : statuses)
        if (!st.ok) return R::failure(st.error.status, st.error.message);
    return assemble(req, reduction, accs, shared);
}

R run_regular_subgroups(const Request &req) {
    if (req.reduction != "all")
        return R::failure(INVALID, "regular_subgroups values only reduce with `all`");
    auto status = validate_request(*req.family);
    if (!status.ok) return R::failure(status.error.status, status.error.message);
    auto size_result = req.family->size();
    if (!size_result.ok) return R::failure(size_result.error.status, size_result.error.message);
    uint64_t size = size_result.value, n = req.family->rows();
    if (size == 0) return R::failure(INVALID, "need at least one graph");

    std::vector<std::vector<Perm>> values(size);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, size));
    auto statuses = parallel_ranges(size, threads, [&](uint32_t, uint64_t begin, uint64_t end) -> Status {
        Matrix member;
        Graph graph;
        for (uint64_t i = begin; i < end; ++i) {
            auto st = read_graph(*req.family, i, graph, member);
            if (!st.ok) return st;
            auto group = automorphisms(graph);
            auto subgroup_indices = RegularSearch(group).all();
            auto &out = values[i];
            out.reserve(subgroup_indices.size() * n);
            for (const auto &subgroup : subgroup_indices)
                for (uint32_t g : subgroup) out.push_back(group[g]);
        }
        return ok();
    });
    for (const auto &st : statuses)
        if (!st.ok) return R::failure(st.error.status, st.error.message);

    auto groups = std::make_shared<RegularSubgroups>();
    groups->count = size;
    groups->n = n;
    groups->offsets.push_back(0);
    for (const auto &member_groups : values) {
        for (const auto &g : member_groups)
            groups->entries.insert(groups->entries.end(), g.begin(), g.end());
        groups->offsets.push_back(groups->offsets.back() + member_groups.size() / n);
    }
    auto object = std::make_shared<Object>();
    object->kind = "vertex_transitive.regular_subgroups";
    object->regular_subgroups = groups;
    return R::success(object);
}

R run(const Request &req) {
    if (req.op == "is_vertex_transitive") return run_scalar(req, ScalarOp::VertexTransitive);
    if (req.op == "is_arc_transitive") return run_scalar(req, ScalarOp::ArcTransitive);
    if (req.op == "is_cayley") return run_scalar(req, ScalarOp::Cayley);
    if (req.op == "regular_subgroups") return run_regular_subgroups(req);
    return R::failure(4, "unknown vertex_transitive operation " + req.op);
}

BackendRegistration registration{Backend{
    "vertex_transitive", "generic",
    [] { return true; },
    [](const Request &) { return true; },
    run,
    0}};

} // namespace
} // namespace lk::vertex_transitive
