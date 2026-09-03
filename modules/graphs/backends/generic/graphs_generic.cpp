#include "../../../../runtime/src/graph.hpp"
#include "../../../../runtime/src/reduce.hpp"

#include <numeric>
#include <queue>
#include <set>

namespace lk::graphs {
namespace {

using R = Result<std::shared_ptr<Object>>;
constexpr int INVALID = 1;

struct Graph {
    const Entry *a;
    uint64_t n;

    bool edge(uint64_t u, uint64_t v) const { return a[u * n + v] != 0; }

    std::vector<int> distances(uint64_t source) const {
        std::vector<int> d(n, -1);
        std::queue<uint64_t> q;
        d[source] = 0;
        q.push(source);
        while (!q.empty()) {
            uint64_t u = q.front();
            q.pop();
            for (uint64_t v = 0; v < n; ++v)
                if (edge(u, v) && d[v] < 0) {
                    d[v] = d[u] + 1;
                    q.push(v);
                }
        }
        return d;
    }

    bool connected() const {
        auto d = distances(0);
        return std::none_of(d.begin(), d.end(), [](int x) { return x < 0; });
    }

    uint64_t diameter() const {
        uint64_t answer = 0;
        for (uint64_t u = 0; u < n; ++u) {
            auto d = distances(u);
            if (std::any_of(d.begin(), d.end(), [](int x) { return x < 0; })) return n;
            answer = std::max<uint64_t>(answer, *std::max_element(d.begin(), d.end()));
        }
        return answer;
    }

    uint64_t girth() const {
        uint64_t answer = n + 1;
        for (uint64_t root = 0; root < n; ++root) {
            std::vector<int> d(n, -1), parent(n, -1);
            std::queue<uint64_t> q;
            d[root] = 0;
            q.push(root);
            while (!q.empty()) {
                uint64_t u = q.front();
                q.pop();
                for (uint64_t v = 0; v < n; ++v) {
                    if (!edge(u, v)) continue;
                    if (d[v] < 0) {
                        d[v] = d[u] + 1;
                        parent[v] = (int)u;
                        q.push(v);
                    } else if (parent[u] != (int)v) {
                        answer = std::min<uint64_t>(answer, d[u] + d[v] + 1);
                    }
                }
            }
        }
        return answer == n + 1 ? 0 : answer;
    }

    bool bipartite() const {
        std::vector<int> color(n, -1);
        for (uint64_t root = 0; root < n; ++root) {
            if (color[root] >= 0) continue;
            std::queue<uint64_t> q;
            color[root] = 0;
            q.push(root);
            while (!q.empty()) {
                uint64_t u = q.front();
                q.pop();
                for (uint64_t v = 0; v < n; ++v) {
                    if (!edge(u, v)) continue;
                    if (color[v] < 0) {
                        color[v] = 1 - color[u];
                        q.push(v);
                    } else if (color[v] == color[u]) return false;
                }
            }
        }
        return true;
    }

    uint64_t maximum_clique(bool complement = false) const {
        uint64_t best = 0;
        std::vector<uint64_t> candidates(n);
        std::iota(candidates.begin(), candidates.end(), 0);
        auto adjacent_for_search = [&](uint64_t u, uint64_t v) {
            return complement ? u != v && !edge(u, v) : edge(u, v);
        };
        auto expand = [&](auto &self, std::vector<uint64_t> remaining, uint64_t size) -> void {
            while (!remaining.empty()) {
                if (size + remaining.size() <= best) return;
                uint64_t v = remaining.back();
                remaining.pop_back();
                std::vector<uint64_t> next;
                for (uint64_t w : remaining)
                    if (adjacent_for_search(v, w)) next.push_back(w);
                if (next.empty()) best = std::max(best, size + 1);
                else self(self, std::move(next), size + 1);
            }
        };
        expand(expand, std::move(candidates), 0);
        return best;
    }

    uint64_t chromatic_number() const {
        std::vector<int> color(n, -1);
        uint64_t best = n;
        auto search = [&](auto &self, uint64_t colored, uint64_t used) -> void {
            if (used >= best) return;
            if (colored == n) {
                best = used;
                return;
            }
            uint64_t chosen = n, chosen_sat = 0, chosen_degree = 0;
            for (uint64_t v = 0; v < n; ++v) {
                if (color[v] >= 0) continue;
                std::set<int> seen_colors;
                uint64_t degree = 0;
                for (uint64_t w = 0; w < n; ++w)
                    if (edge(v, w)) {
                        ++degree;
                        if (color[w] >= 0) seen_colors.insert(color[w]);
                    }
                if (chosen == n || seen_colors.size() > chosen_sat ||
                    (seen_colors.size() == chosen_sat && degree > chosen_degree)) {
                    chosen = v;
                    chosen_sat = seen_colors.size();
                    chosen_degree = degree;
                }
            }
            for (uint64_t c = 0; c <= used && c < best; ++c) {
                bool conflict = false;
                for (uint64_t w = 0; w < n; ++w)
                    if (edge(chosen, w) && color[w] == (int)c) {
                        conflict = true;
                        break;
                    }
                if (conflict) continue;
                color[chosen] = (int)c;
                self(self, colored + 1, std::max(used, c + 1));
                color[chosen] = -1;
            }
        };
        search(search, 0, 0);
        return best;
    }

    std::vector<Entry> degree_sequence() const {
        std::vector<Entry> degrees(n, 0);
        for (uint64_t u = 0; u < n; ++u)
            for (uint64_t v = 0; v < n; ++v) degrees[u] += edge(u, v);
        std::sort(degrees.begin(), degrees.end(), std::greater<Entry>());
        return degrees;
    }
};

Status validate(const Matrix &m) {
    if (m.p != 2 || m.count != 1 || m.rows == 0 || m.rows != m.cols)
        return fail(INVALID, "graphs needs a nonempty square adjacency matrix over F_2");
    for (uint64_t i = 0; i < m.rows; ++i) {
        if (m.entries[i * m.rows + i]) return fail(INVALID, "graphs needs a simple undirected graph: diagonal entries must be zero");
        for (uint64_t j = i + 1; j < m.rows; ++j)
            if (m.entries[i * m.rows + j] != m.entries[j * m.rows + i])
                return fail(INVALID, "graphs needs a simple undirected graph: adjacency must be symmetric");
    }
    return ok();
}

enum class ScalarOp { Connected, Girth, Diameter, Chromatic, Clique, Independence, Bipartite };

R run_scalar(const Request &req, ScalarOp op) {
    auto size_r = req.family->size();
    if (!size_r.ok) return R::failure(size_r.error.status, size_r.error.message);
    uint64_t size = size_r.value;
    Reduction reduction = parse_reduction(req.reduction);
    Shared shared;
    auto prepared = prepare_all(reduction, size, shared);
    if (!prepared.ok) return R::failure(prepared.error.status, prepared.error.message);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, size ? size : 1));
    std::vector<Accumulator> accs;
    for (uint32_t t = 0; t < threads; ++t) accs.emplace_back(reduction, &shared);
    auto statuses = parallel_ranges(size, threads, [&](uint32_t t, uint64_t begin, uint64_t end) -> Status {
        Matrix member;
        for (uint64_t i = begin; i < end; ++i) {
            if (accs[t].exhausted(i)) break;
            auto st = req.family->member_into(i, member);
            if (!st.ok) return st;
            st = validate(member);
            if (!st.ok) return st;
            Graph g{member.entries.data(), member.rows};
            switch (op) {
            case ScalarOp::Connected: accs[t].boolean(i, g.connected()); break;
            case ScalarOp::Girth: accs[t].integer(i, g.girth()); break;
            case ScalarOp::Diameter: accs[t].integer(i, g.diameter()); break;
            case ScalarOp::Chromatic: accs[t].integer(i, g.chromatic_number()); break;
            case ScalarOp::Clique: accs[t].integer(i, g.maximum_clique()); break;
            case ScalarOp::Independence: accs[t].integer(i, g.maximum_clique(true)); break;
            case ScalarOp::Bipartite: accs[t].boolean(i, g.bipartite()); break;
            }
        }
        return ok();
    });
    for (const auto &st : statuses)
        if (!st.ok) return R::failure(st.error.status, st.error.message);
    return assemble(req, reduction, accs, shared);
}

R run_degree_sequences(const Request &req) {
    if (req.reduction != "all") return R::failure(INVALID, "degree_sequence values only reduce with `all`");
    auto size_r = req.family->size();
    if (!size_r.ok) return R::failure(size_r.error.status, size_r.error.message);
    auto out = std::make_shared<DegreeSequences>();
    out->count = size_r.value;
    out->n = req.family->rows();
    out->entries.resize(out->count * out->n);
    auto statuses = parallel_ranges(out->count, req.threads, [&](uint32_t, uint64_t begin, uint64_t end) -> Status {
        Matrix member;
        for (uint64_t i = begin; i < end; ++i) {
            auto st = req.family->member_into(i, member);
            if (!st.ok) return st;
            st = validate(member);
            if (!st.ok) return st;
            auto degrees = Graph{member.entries.data(), member.rows}.degree_sequence();
            std::copy(degrees.begin(), degrees.end(), out->entries.begin() + i * out->n);
        }
        return ok();
    });
    for (const auto &st : statuses)
        if (!st.ok) return R::failure(st.error.status, st.error.message);
    auto object = std::make_shared<Object>();
    object->kind = "graphs.degree_sequences";
    object->degree_sequences = out;
    return R::success(object);
}

R run_canonical_forms(const Request &req) {
    if (req.reduction != "all") return R::failure(INVALID, "canonical_form values only reduce with `all`");
    auto size_r = req.family->size();
    if (!size_r.ok) return R::failure(size_r.error.status, size_r.error.message);
    auto out = std::make_shared<Matrix>();
    out->p = 2;
    out->count = size_r.value;
    out->rows = out->cols = req.family->rows();
    out->entries.resize(out->count * out->rows * out->cols);
    auto statuses = parallel_ranges(out->count, req.threads, [&](uint32_t, uint64_t begin, uint64_t end) -> Status {
        Matrix member;
        for (uint64_t i = begin; i < end; ++i) {
            auto st = req.family->member_into(i, member);
            if (!st.ok) return st;
            st = validate(member);
            if (!st.ok) return st;
            auto canonical = graph::canonical(member.entries.data(), member.rows);
            std::copy(canonical.begin(), canonical.end(), out->entries.begin() + i * out->rows * out->cols);
        }
        return ok();
    });
    for (const auto &st : statuses)
        if (!st.ok) return R::failure(st.error.status, st.error.message);
    auto object = std::make_shared<Object>();
    object->kind = "gfp.matrix";
    object->matrix = out;
    return R::success(object);
}

R run(const Request &req) {
    if (req.op == "connected") return run_scalar(req, ScalarOp::Connected);
    if (req.op == "girth") return run_scalar(req, ScalarOp::Girth);
    if (req.op == "diameter") return run_scalar(req, ScalarOp::Diameter);
    if (req.op == "chromatic_number") return run_scalar(req, ScalarOp::Chromatic);
    if (req.op == "clique_number") return run_scalar(req, ScalarOp::Clique);
    if (req.op == "independence_number") return run_scalar(req, ScalarOp::Independence);
    if (req.op == "is_bipartite") return run_scalar(req, ScalarOp::Bipartite);
    if (req.op == "degree_sequence") return run_degree_sequences(req);
    if (req.op == "canonical_form") return run_canonical_forms(req);
    return R::failure(4, "unknown graphs operation " + req.op);
}

BackendRegistration registration{Backend{
    "graphs", "generic",
    [] { return true; },
    [](const Request &) { return true; },
    run,
    0}};

} // namespace
} // namespace lk::graphs
