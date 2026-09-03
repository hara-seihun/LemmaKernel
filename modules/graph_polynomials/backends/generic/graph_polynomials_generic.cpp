/* Exact graph polynomials for simple labelled edge sets.
 *
 * The family walker keeps the current edge prefix. Deletion-contraction caches live for the
 * whole request, so sibling members reuse every subgraph they have already solved. */
#include "../../../../runtime/src/registry.hpp"

#include <algorithm>
#include <bit>
#include <boost/multiprecision/cpp_int.hpp>
#include <limits>
#include <map>
#include <set>
#include <utility>

namespace lk::graph_polynomials {
namespace {

using Big = boost::multiprecision::cpp_int;
using Edge = std::pair<uint32_t, uint32_t>;
using Poly = std::vector<Big>;
using Degree = std::pair<uint32_t, uint32_t>;
using BPoly = std::map<Degree, Big>;
using R = Result<std::shared_ptr<Object>>;
constexpr int INVALID = 1;

struct Graph {
    uint32_t vertices;
    std::vector<Edge> edges;

    bool operator<(const Graph &other) const {
        return vertices < other.vertices || (vertices == other.vertices && edges < other.edges);
    }
};

Graph canonical(uint32_t vertices, std::vector<Edge> edges) {
    for (auto &[u, v] : edges) if (v < u) std::swap(u, v);
    std::sort(edges.begin(), edges.end());
    return Graph{vertices, std::move(edges)};
}

Graph contract(const Graph &graph, Edge edge, const std::vector<Edge> &rest) {
    const auto [u, v] = edge;
    auto rename = [=](uint32_t x) {
        uint32_t y = x == v ? u : x;
        return y > v ? y - 1 : y;
    };
    std::vector<Edge> edges;
    edges.reserve(rest.size());
    for (auto [a, b] : rest) edges.emplace_back(rename(a), rename(b));
    return canonical(graph.vertices - 1, std::move(edges));
}

Graph remove_endpoints(const Graph &graph, Edge edge, const std::vector<Edge> &rest) {
    const auto [u, v] = edge;
    auto rename = [=](uint32_t x) { return x - (u < x) - (v < x); };
    std::vector<Edge> edges;
    for (auto [a, b] : rest) {
        if (a == u || b == u || a == v || b == v) continue;
        edges.emplace_back(rename(a), rename(b));
    }
    return canonical(graph.vertices - 2, std::move(edges));
}

Poly monomial(size_t length, size_t degree) {
    Poly out(length);
    out[degree] = 1;
    return out;
}

Poly combine(const Poly &a, const Poly &b, size_t length, int sign = 1) {
    Poly out(length);
    for (size_t i = 0; i < length; ++i) {
        if (i < a.size()) out[i] += a[i];
        if (i < b.size()) out[i] += sign * b[i];
    }
    return out;
}

bool reachable(const std::vector<Edge> &edges, uint32_t source, uint32_t target, uint32_t vertices) {
    std::vector<uint8_t> seen(vertices);
    std::vector<uint32_t> queue{source};
    seen[source] = 1;
    for (size_t i = 0; i < queue.size(); ++i) {
        uint32_t at = queue[i];
        for (auto [u, v] : edges) {
            uint32_t next;
            if (u == at) next = v;
            else if (v == at) next = u;
            else continue;
            if (!seen[next]) {
                seen[next] = 1;
                queue.push_back(next);
            }
        }
    }
    return seen[target];
}

struct PolynomialVisitor final : Family::Visitor {
    const Request &request;
    uint32_t vertices;
    uint64_t edge_count;
    std::shared_ptr<Coefficients> output;
    std::vector<Edge> prefix;
    std::string error;
    std::map<Graph, Poly> chromatic_cache;
    std::map<Graph, Poly> matching_cache;
    std::map<Graph, BPoly> tutte_cache;
    std::map<Graph, Poly> characteristic_cache;

    PolynomialVisitor(const Request &request_, uint32_t vertices_, uint64_t edge_count_,
                      std::shared_ptr<Coefficients> output_)
        : request(request_), vertices(vertices_), edge_count(edge_count_), output(std::move(output_)) {}

    Step push(const Entry *row, Index, Index) override {
        prefix.emplace_back(row[0], row[1]);
        return Step::Descend;
    }

    void pop() override { prefix.pop_back(); }
    void take_all(Index, Index) override {}
    void skip_all(Index, Index) override {}

    Poly chromatic(const Graph &graph) {
        if (auto it = chromatic_cache.find(graph); it != chromatic_cache.end()) return it->second;
        Poly answer;
        if (std::any_of(graph.edges.begin(), graph.edges.end(), [](Edge e) { return e.first == e.second; })) {
            answer.assign(graph.vertices + 1, 0);
        } else if (graph.edges.empty()) {
            answer = monomial(graph.vertices + 1, graph.vertices);
        } else {
            Edge edge = graph.edges.back();
            std::vector<Edge> rest(graph.edges.begin(), graph.edges.end() - 1);
            answer = combine(chromatic(canonical(graph.vertices, rest)),
                             chromatic(contract(graph, edge, rest)), graph.vertices + 1, -1);
        }
        chromatic_cache.emplace(graph, answer);
        return answer;
    }

    Poly matching(const Graph &graph) {
        if (auto it = matching_cache.find(graph); it != matching_cache.end()) return it->second;
        Poly answer;
        if (graph.edges.empty()) {
            answer = monomial(graph.vertices + 1, graph.vertices);
        } else {
            Edge edge = graph.edges.back();
            std::vector<Edge> rest(graph.edges.begin(), graph.edges.end() - 1);
            answer = combine(matching(canonical(graph.vertices, rest)),
                             matching(remove_endpoints(graph, edge, rest)), graph.vertices + 1, -1);
        }
        matching_cache.emplace(graph, answer);
        return answer;
    }

    BPoly tutte(const Graph &graph) {
        if (auto it = tutte_cache.find(graph); it != tutte_cache.end()) return it->second;
        BPoly answer;
        if (graph.edges.empty()) {
            answer[{0, 0}] = 1;
        } else {
            Edge edge = graph.edges.back();
            std::vector<Edge> rest(graph.edges.begin(), graph.edges.end() - 1);
            if (edge.first == edge.second) {
                for (const auto &[degree, coefficient] : tutte(canonical(graph.vertices, rest)))
                    answer[{degree.first, degree.second + 1}] += coefficient;
            } else if (!reachable(rest, edge.first, edge.second, graph.vertices)) {
                for (const auto &[degree, coefficient] : tutte(contract(graph, edge, rest)))
                    answer[{degree.first + 1, degree.second}] += coefficient;
            } else {
                answer = tutte(canonical(graph.vertices, rest));
                for (const auto &[degree, coefficient] : tutte(contract(graph, edge, rest)))
                    answer[degree] += coefficient;
            }
        }
        tutte_cache.emplace(graph, answer);
        return answer;
    }

    Poly characteristic(const Graph &graph) {
        if (auto it = characteristic_cache.find(graph); it != characteristic_cache.end()) return it->second;
        const uint32_t n = graph.vertices;
        std::vector<uint8_t> adjacency((size_t)n * n);
        for (auto [u, v] : graph.edges) adjacency[(size_t)u * n + v] = adjacency[(size_t)v * n + u] = 1;
        std::vector<Poly> dp(1ULL << n);
        dp[0] = {1};
        for (uint64_t mask = 0; mask < (1ULL << n); ++mask) {
            if (dp[mask].empty()) continue;
            uint32_t row = std::popcount(mask);
            if (row == n) continue;
            for (uint32_t col = 0; col < n; ++col) {
                if (mask & (1ULL << col)) continue;
                uint32_t inversions = std::popcount(mask >> (col + 1));
                int sign = inversions & 1 ? -1 : 1;
                Poly term;
                if (row == col) {
                    term.assign(dp[mask].size() + 1, 0);
                    for (size_t i = 0; i < dp[mask].size(); ++i) term[i + 1] = sign * dp[mask][i];
                } else if (adjacency[(size_t)row * n + col]) {
                    term.resize(dp[mask].size());
                    for (size_t i = 0; i < dp[mask].size(); ++i) term[i] = -sign * dp[mask][i];
                } else {
                    continue;
                }
                uint64_t next = mask | (1ULL << col);
                dp[next] = combine(dp[next], term, std::max(dp[next].size(), term.size()));
            }
        }
        Poly answer = dp.back();
        answer.resize(n + 1);
        characteristic_cache.emplace(graph, answer);
        return answer;
    }

    void store(uint64_t index, const Poly &values) {
        static const Big low = std::numeric_limits<int64_t>::min();
        static const Big high = std::numeric_limits<int64_t>::max();
        if (values.size() != output->length) {
            error = "internal coefficient vector length mismatch";
            return;
        }
        for (size_t i = 0; i < values.size(); ++i) {
            if (values[i] < low || values[i] > high) {
                error = "a polynomial coefficient does not fit signed i64";
                return;
            }
            output->values[index * output->length + i] = values[i].convert_to<int64_t>();
        }
    }

    void leaf(Index index) override {
        if (!error.empty()) return;
        Graph graph = canonical(vertices, prefix);
        if (request.op == "chromatic") store(index, chromatic(graph));
        else if (request.op == "matching") store(index, matching(graph));
        else if (request.op == "characteristic") store(index, characteristic(graph));
        else if (request.op == "tutte") {
            BPoly terms = tutte(graph);
            Poly flat;
            flat.reserve((vertices + 1) * (edge_count + 1));
            for (uint32_t i = 0; i <= vertices; ++i)
                for (uint64_t j = 0; j <= edge_count; ++j)
                    flat.push_back(terms[{i, (uint32_t)j}]);
            store(index, flat);
        } else {
            error = "unknown graph_polynomials operation " + request.op;
        }
    }
};

R run(const Request &request) {
    auto vertices_it = request.int_args.find("vertices");
    if (vertices_it == request.int_args.end()) return R::failure(INVALID, "missing integer argument `vertices`");
    uint64_t vertices64 = vertices_it->second;
    if (vertices64 == 0 || vertices64 > 20) return R::failure(INVALID, "vertices must satisfy 1 <= vertices <= 20");
    const Family &family = *request.family;
    if (family.kind != Family::Kind::Subsets || family.cols() != 2)
        return R::failure(INVALID, "graph polynomial operations need subsets of two-column edge dictionaries");
    std::set<Edge> seen;
    for (uint64_t i = 0; i < family.data->count; ++i) {
        const Entry *edge = family.data->at(i);
        if (!(edge[0] < edge[1] && edge[1] < vertices64))
            return R::failure(INVALID, "edge dictionary rows must satisfy u < v < vertices");
        if (!seen.emplace(edge[0], edge[1]).second)
            return R::failure(INVALID, "edge dictionary contains a duplicate edge");
    }
    auto size = family.size();
    if (!size.ok) return R::failure(size.error.status, size.error.message);
    uint64_t length = request.op == "tutte" ? (vertices64 + 1) * (family.k + 1) : vertices64 + 1;
    if ((unsigned __int128)size.value * length > std::numeric_limits<size_t>::max())
        return R::failure(INVALID, "coefficient output is too large");
    auto coefficients = std::make_shared<Coefficients>();
    coefficients->count = size.value;
    coefficients->length = length;
    coefficients->values.resize(size.value * length);
    PolynomialVisitor visitor(request, (uint32_t)vertices64, family.k, coefficients);
    auto top = family.top_count();
    if (!top.ok) return R::failure(top.error.status, top.error.message);
    auto status = family.enumerate(visitor, 0, top.value);
    if (!status.ok) return R::failure(status.error.status, status.error.message);
    if (!visitor.error.empty()) return R::failure(INVALID, visitor.error);
    auto object = std::make_shared<Object>();
    object->kind = "graph_polynomials.coefficients";
    object->coefficients = std::move(coefficients);
    return R::success(object);
}

BackendRegistration registration{Backend{
    "graph_polynomials", "generic",
    [] { return true; },
    [](const Request &request) {
        auto n = request.int_args.find("vertices");
        return request.family->kind == Family::Kind::Subsets && request.family->cols() == 2 &&
               request.family->k <= 32 && n != request.int_args.end() && n->second <= 20;
    },
    run,
    0}};

} // namespace
} // namespace lk::graph_polynomials
