/* graph_iso generic backend: exact canonical labelling and automorphism groups.
 *
 * One-dimensional Weisfeiler-Lehman refinement supplies invariant colours after every
 * individualisation. Canonical search still considers every vertex order not removed by an
 * exact lower bound or by interchangeable twins, so its answer is the global lexicographic
 * minimum rather than a refinement-dependent choice. Automorphism search maps vertices only to
 * the matching refined colour and rejects a partial map as soon as adjacency or refinement makes
 * an extension impossible. */
#include "../../../../runtime/src/reduce.hpp"

#include <algorithm>
#include <numeric>

namespace lk::graph_iso {
namespace {

using R = Result<std::shared_ptr<Object>>;
constexpr int INVALID = 1;

struct Graph {
    uint64_t n;
    std::vector<Entry> adjacency;

    Entry at(uint64_t i, uint64_t j) const { return adjacency[i * n + j]; }
};

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

std::vector<Entry> relabel(const Graph &graph, const std::vector<uint32_t> &order) {
    std::vector<Entry> code(graph.n * graph.n);
    for (uint64_t i = 0; i < graph.n; ++i)
        for (uint64_t j = 0; j < graph.n; ++j)
            code[i * graph.n + j] = graph.at(order[i], order[j]);
    return code;
}

struct CanonicalSearch {
    const Graph &graph;
    std::vector<uint32_t> stable_colours;
    std::vector<uint8_t> used;
    std::vector<uint32_t> prefix;
    std::vector<Entry> best_code;
    std::vector<uint32_t> best_order;
    std::vector<uint8_t> twins;
    bool have_best = false;

    explicit CanonicalSearch(const Graph &g)
        : graph(g), stable_colours(refine(g, std::vector<uint32_t>(g.n, 0))), used(g.n, 0),
          twins(g.n * g.n, 0) {
        for (uint64_t u = 0; u < graph.n; ++u) {
            twins[u * graph.n + u] = 1;
            for (uint64_t v = u + 1; v < graph.n; ++v) {
                bool same = true;
                for (uint64_t i = 0; i < graph.n && same; ++i)
                    for (uint64_t j = 0; j < graph.n; ++j) {
                        uint64_t pi = i == u ? v : i == v ? u : i;
                        uint64_t pj = j == u ? v : j == v ? u : j;
                        if (graph.at(pi, pj) != graph.at(i, j)) { same = false; break; }
                    }
                twins[u * graph.n + v] = twins[v * graph.n + u] = same;
            }
        }
    }

    std::vector<Entry> lower_bound() const {
        uint64_t k = prefix.size();
        std::vector<Entry> bound(graph.n * graph.n, 0);
        std::vector<Entry> tail;
        for (uint64_t i = 0; i < k; ++i) {
            for (uint64_t j = 0; j < k; ++j)
                bound[i * graph.n + j] = graph.at(prefix[i], prefix[j]);
            tail.clear();
            for (uint64_t v = 0; v < graph.n; ++v)
                if (!used[v]) tail.push_back(graph.at(prefix[i], v));
            std::sort(tail.begin(), tail.end());
            std::copy(tail.begin(), tail.end(), bound.begin() + i * graph.n + k);
        }
        return bound;
    }

    bool prunable() const {
        if (!have_best) return false;
        auto bound = lower_bound();
        return std::lexicographical_compare(best_code.begin(), best_code.end(), bound.begin(), bound.end());
    }

    std::vector<uint32_t> individualised_colours() const {
        uint32_t k = prefix.size();
        std::vector<uint32_t> colours(graph.n);
        for (uint64_t v = 0; v < graph.n; ++v) colours[v] = k + stable_colours[v];
        for (uint32_t i = 0; i < k; ++i) colours[prefix[i]] = i;
        return refine(graph, std::move(colours));
    }

    void search() {
        if (prunable()) return;
        if (prefix.size() == graph.n) {
            auto candidate = relabel(graph, prefix);
            if (!have_best || candidate < best_code || (candidate == best_code && prefix < best_order)) {
                best_code = std::move(candidate);
                best_order = prefix;
                have_best = true;
            }
            return;
        }

        auto colours = individualised_colours();
        struct Candidate { uint32_t vertex, colour; std::vector<Entry> bound; };
        std::vector<Candidate> candidates;
        for (uint32_t v = 0; v < graph.n; ++v) {
            if (used[v]) continue;
            bool shadowed = false;
            for (uint32_t u = 0; u < v; ++u)
                if (!used[u] && twins[u * graph.n + v]) { shadowed = true; break; }
            if (shadowed) continue;
            used[v] = 1;
            prefix.push_back(v);
            candidates.push_back(Candidate{v, colours[v], lower_bound()});
            prefix.pop_back();
            used[v] = 0;
        }
        std::sort(candidates.begin(), candidates.end(), [](const Candidate &a, const Candidate &b) {
            if (a.bound != b.bound) return a.bound < b.bound;
            if (a.colour != b.colour) return a.colour < b.colour;
            return a.vertex < b.vertex;
        });
        for (const auto &candidate : candidates) {
            used[candidate.vertex] = 1;
            prefix.push_back(candidate.vertex);
            search();
            prefix.pop_back();
            used[candidate.vertex] = 0;
        }
    }
};

struct AutomorphismSearch {
    const Graph &graph;
    std::vector<uint32_t> stable_colours;
    std::vector<int64_t> image;
    std::vector<uint8_t> used_targets;
    std::vector<uint32_t> sources;
    std::vector<std::vector<uint32_t>> found;

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
        if (graph.at(source, source) != graph.at(target, target)) return false;
        for (uint32_t other : sources)
            if (graph.at(source, other) != graph.at(target, image[other])) return false;
        return true;
    }

    void search() {
        if (sources.size() == graph.n) {
            std::vector<uint32_t> permutation(graph.n);
            for (uint64_t v = 0; v < graph.n; ++v) permutation[v] = image[v];
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

Status read_graph(const Family &family, uint64_t index, Graph &graph, Matrix &member) {
    auto status = family.member_into(index, member);
    if (!status.ok) return status;
    graph.n = member.rows;
    graph.adjacency = member.entries;
    for (uint64_t i = 0; i < graph.n; ++i)
        for (uint64_t j = i + 1; j < graph.n; ++j)
            if (graph.at(i, j) != graph.at(j, i))
                return fail(INVALID, "graph adjacency matrix " + std::to_string(index) + " is not symmetric");
    return ok();
}

R run(const Request &req) {
    const Family &family = *req.family;
    if (family.prime() != 2) return R::failure(INVALID, "graph adjacency matrices must be over F_2");
    if (family.rows() == 0) return R::failure(INVALID, "graphs must have at least one vertex");
    if (family.rows() != family.cols()) return R::failure(INVALID, "graph adjacency matrices must be square");
    auto size_result = family.size();
    if (!size_result.ok) return R::failure(size_result.error.status, size_result.error.message);
    uint64_t size = size_result.value, n = family.rows();
    if (size == 0) return R::failure(INVALID, "need at least one graph");

    std::shared_ptr<Matrix> matrices;
    std::vector<std::vector<Entry>> group_entries;
    if (req.op == "canonical_form") {
        matrices = std::make_shared<Matrix>();
        matrices->p = 2; matrices->count = size; matrices->rows = n; matrices->cols = n;
        matrices->entries.resize(size * n * n);
    } else if (req.op == "canonical_label") {
        matrices = std::make_shared<Matrix>();
        matrices->p = 0; matrices->count = size; matrices->rows = 1; matrices->cols = n;
        matrices->entries.resize(size * n);
    } else if (req.op == "automorphism_group") {
        group_entries.resize(size);
    } else {
        return R::failure(4, "unknown graph_iso operation " + req.op);
    }

    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, size));
    auto statuses = parallel_ranges(size, threads, [&](uint32_t, uint64_t begin, uint64_t end) -> Status {
        Matrix member;
        Graph graph{0, {}};
        for (uint64_t i = begin; i < end; ++i) {
            auto status = read_graph(family, i, graph, member);
            if (!status.ok) return status;
            if (req.op == "automorphism_group") {
                AutomorphismSearch search(graph);
                search.search();
                std::sort(search.found.begin(), search.found.end());
                auto &out = group_entries[i];
                out.reserve(search.found.size() * n);
                for (const auto &g : search.found) out.insert(out.end(), g.begin(), g.end());
            } else {
                CanonicalSearch search(graph);
                search.search();
                if (req.op == "canonical_form")
                    std::copy(search.best_code.begin(), search.best_code.end(), matrices->at(i));
                else
                    std::copy(search.best_order.begin(), search.best_order.end(), matrices->at(i));
            }
        }
        return ok();
    });
    for (const auto &status : statuses)
        if (!status.ok) return R::failure(status.error.status, status.error.message);

    auto object = std::make_shared<Object>();
    if (matrices) {
        object->kind = matrix_kind(*matrices);
        object->matrix = matrices;
    } else {
        auto groups = std::make_shared<GraphGroups>();
        groups->count = size; groups->n = n; groups->offsets.push_back(0);
        for (auto &entries : group_entries) {
            groups->entries.insert(groups->entries.end(), entries.begin(), entries.end());
            groups->offsets.push_back(groups->offsets.back() + entries.size() / n);
        }
        object->kind = "graph_iso.groups";
        object->graph_groups = groups;
    }
    return R::success(object);
}

BackendRegistration registration{Backend{
    "graph_iso", "generic",
    [] { return true; },
    [](const Request &) { return true; },
    run,
    0}};

} // namespace
} // namespace lk::graph_iso
