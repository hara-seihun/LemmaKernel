/* hypergraphs generic backend: finite uniform hypergraphs as natural-number edge matrices. */
#include "../../../../runtime/src/reduce.hpp"

#include <bit>
#include <functional>
#include <numeric>

namespace lk::hypergraphs {
namespace {

using R = Result<std::shared_ptr<Object>>;
constexpr int INVALID = 1;

struct Hypergraph {
    uint64_t vertices = 0;
    uint64_t uniformity = 0;
    std::vector<uint64_t> edges;
};

bool strictly_increasing(const Entry *edge, uint64_t k, uint64_t vertices) {
    for (uint64_t i = 0; i < k; ++i) {
        if (edge[i] >= vertices) return false;
        if (i && edge[i - 1] >= edge[i]) return false;
    }
    return true;
}

Status validate_edges(const Entry *entries, uint64_t count, uint64_t k, uint64_t vertices) {
    for (uint64_t i = 0; i < count; ++i) {
        const Entry *edge = entries + i * k;
        if (!strictly_increasing(edge, k, vertices))
            return fail(INVALID, "hypergraph: each edge must be strictly increasing with vertices in range");
        if (i) {
            const Entry *previous = entries + (i - 1) * k;
            if (!std::lexicographical_compare(previous, previous + k, edge, edge + k))
                return fail(INVALID, "hypergraph: edges must be in strict lexicographic order");
        }
    }
    return ok();
}

struct Setup {
    const Family *family = nullptr;
    uint64_t vertices = 0;
    uint64_t uniformity = 0;
};

Result<Setup> setup(const Request &req) {
    Setup out;
    out.family = req.family.get();
    auto vertices = req.int_args.find("vertices");
    if (vertices == req.int_args.end())
        return Result<Setup>::failure(INVALID, "hypergraph: missing `vertices`");
    out.vertices = vertices->second;
    if (out.vertices > 64)
        return Result<Setup>::failure(INVALID, "hypergraph: `vertices` must be at most 64");

    const Family &family = *out.family;
    if (family.kind != Family::Kind::Explicit && family.kind != Family::Kind::Subsets &&
        family.kind != Family::Kind::SubsetsOf)
        return Result<Setup>::failure(INVALID, "hypergraph operations require explicit, subsets, or subsets_of families");
    if (family.prime() != NATURALS)
        return Result<Setup>::failure(INVALID, "hypergraph edges must be natural-number matrices");

    auto size = family.size();
    if (!size.ok) return Result<Setup>::failure(size.error.status, size.error.message);
    if (size.value == 0 || family.rows() == 0)
        return Result<Setup>::failure(INVALID, "hypergraph families must contain nonempty edge sets");
    out.uniformity = family.cols();
    if (out.uniformity < 2 || out.uniformity > out.vertices)
        return Result<Setup>::failure(INVALID, "hypergraph uniformity must satisfy 2 <= uniformity <= vertices");

    Status valid = ok();
    if (family.kind == Family::Kind::Explicit) {
        for (uint64_t i = 0; i < family.data->count && valid.ok; ++i)
            valid = validate_edges(family.data->at(i), family.data->rows, out.uniformity, out.vertices);
    } else {
        valid = validate_edges(family.data->entries.data(), family.data->count, out.uniformity, out.vertices);
    }
    if (!valid.ok) return Result<Setup>::failure(valid.error.status, valid.error.message);
    return Result<Setup>::success(out);
}

Hypergraph decode(const Matrix &member, uint64_t vertices) {
    Hypergraph h{vertices, member.cols, {}};
    h.edges.reserve(member.rows);
    for (uint64_t i = 0; i < member.rows; ++i) {
        uint64_t edge = 0;
        for (uint64_t j = 0; j < member.cols; ++j) edge |= uint64_t{1} << member.entries[i * member.cols + j];
        h.edges.push_back(edge);
    }
    return h;
}

bool is_linear(const Hypergraph &h) {
    for (uint64_t i = 0; i < h.edges.size(); ++i)
        for (uint64_t j = i + 1; j < h.edges.size(); ++j)
            if (std::popcount(h.edges[i] & h.edges[j]) > 1) return false;
    return true;
}

bool proper_with(const Hypergraph &h, uint64_t colours, const std::vector<uint64_t> &order) {
    std::vector<int> colour(h.vertices, -1);
    std::function<bool(uint64_t)> search = [&](uint64_t position) {
        if (position == h.vertices) return true;
        uint64_t vertex = order[position];
        uint64_t choices = position == 0 ? 1 : colours;
        for (uint64_t c = 0; c < choices; ++c) {
            colour[vertex] = static_cast<int>(c);
            bool valid = true;
            for (uint64_t edge : h.edges) {
                int first = -1;
                bool complete = true, monochromatic = true;
                for (uint64_t bits = edge; bits; bits &= bits - 1) {
                    uint64_t v = std::countr_zero(bits);
                    if (colour[v] < 0) { complete = false; break; }
                    if (first < 0) first = colour[v];
                    else if (colour[v] != first) monochromatic = false;
                }
                if (complete && monochromatic) { valid = false; break; }
            }
            if (valid && search(position + 1)) return true;
            colour[vertex] = -1;
        }
        return false;
    };
    return search(0);
}

uint64_t colouring_number(const Hypergraph &h) {
    std::vector<uint64_t> degree(h.vertices, 0), order(h.vertices);
    for (uint64_t edge : h.edges)
        for (uint64_t bits = edge; bits; bits &= bits - 1) ++degree[std::countr_zero(bits)];
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](uint64_t a, uint64_t b) { return degree[a] > degree[b]; });
    for (uint64_t colours = 1; colours <= h.vertices; ++colours)
        if (proper_with(h, colours, order)) return colours;
    return h.vertices + 1;
}

bool has_berge_cycle(const Hypergraph &h, uint64_t length) {
    if (length < 2 || length > h.vertices || length > h.edges.size()) return false;
    std::vector<uint8_t> used_edge(h.edges.size(), 0);
    std::function<bool(uint64_t, uint64_t, uint64_t, uint64_t)> extend =
        [&](uint64_t start, uint64_t current, uint64_t depth, uint64_t used_vertices) {
            if (depth == length - 1) {
                uint64_t endpoints = (uint64_t{1} << start) | (uint64_t{1} << current);
                for (uint64_t i = 0; i < h.edges.size(); ++i)
                    if (!used_edge[i] && (h.edges[i] & endpoints) == endpoints) return true;
                return false;
            }
            uint64_t current_bit = uint64_t{1} << current;
            for (uint64_t i = 0; i < h.edges.size(); ++i) {
                if (used_edge[i] || !(h.edges[i] & current_bit)) continue;
                used_edge[i] = 1;
                for (uint64_t choices = h.edges[i] & ~used_vertices; choices; choices &= choices - 1) {
                    uint64_t next = std::countr_zero(choices);
                    if (extend(start, next, depth + 1, used_vertices | (uint64_t{1} << next))) {
                        used_edge[i] = 0;
                        return true;
                    }
                }
                used_edge[i] = 0;
            }
            return false;
        };
    for (uint64_t start = 0; start < h.vertices; ++start)
        if (extend(start, start, 0, uint64_t{1} << start)) return true;
    return false;
}

uint64_t berge_girth(const Hypergraph &h) {
    uint64_t bound = std::min<uint64_t>(h.vertices, h.edges.size());
    for (uint64_t length = 2; length <= bound; ++length)
        if (has_berge_cycle(h, length)) return length;
    return 0;
}

uint64_t binomial(uint64_t n, uint64_t k) {
    k = std::min(k, n - k);
    unsigned __int128 value = 1;
    for (uint64_t i = 1; i <= k; ++i) value = value * (n - k + i) / i;
    return static_cast<uint64_t>(value);
}

template <class Predicate>
bool any_subset(uint64_t n, uint64_t size, Predicate predicate) {
    if (size > n) return false;
    std::function<bool(uint64_t, uint64_t, uint64_t)> search =
        [&](uint64_t start, uint64_t left, uint64_t chosen) {
            if (left == 0) return predicate(chosen);
            for (uint64_t vertex = start; vertex <= n - left; ++vertex)
                if (search(vertex + 1, left - 1, chosen | (uint64_t{1} << vertex))) return true;
            return false;
        };
    return search(0, size, 0);
}

bool contains_clique(const Hypergraph &h, uint64_t size, bool red) {
    if (size > h.vertices) return false;
    uint64_t needed = binomial(size, h.uniformity);
    return any_subset(h.vertices, size, [&](uint64_t vertices) {
        uint64_t present = 0;
        for (uint64_t edge : h.edges) present += (edge & ~vertices) == 0;
        return red ? present == needed : present == 0;
    });
}

enum class Op { IsLinear, ColouringNumber, HasBergeCycle, BergeGirth, IsCliqueFree, IsRamseyColouring };

R run(const Request &req) {
    auto configured = setup(req);
    if (!configured.ok) return R::failure(configured.error.status, configured.error.message);
    const Setup &s = configured.value;

    Op op;
    if (req.op == "is_linear") op = Op::IsLinear;
    else if (req.op == "colouring_number") op = Op::ColouringNumber;
    else if (req.op == "has_berge_cycle") op = Op::HasBergeCycle;
    else if (req.op == "berge_girth") op = Op::BergeGirth;
    else if (req.op == "is_clique_free") op = Op::IsCliqueFree;
    else if (req.op == "is_ramsey_colouring") op = Op::IsRamseyColouring;
    else return R::failure(4, "unknown hypergraphs operation " + req.op);

    uint64_t length = req.int_args.count("length") ? req.int_args.at("length") : 0;
    uint64_t clique = req.int_args.count("clique_size") ? req.int_args.at("clique_size") : 0;
    uint64_t red = req.int_args.count("red_clique") ? req.int_args.at("red_clique") : 0;
    uint64_t blue = req.int_args.count("blue_clique") ? req.int_args.at("blue_clique") : 0;
    if (op == Op::HasBergeCycle && length < 2)
        return R::failure(INVALID, "has_berge_cycle: `length` must be at least 2");
    if (op == Op::IsCliqueFree && clique < s.uniformity)
        return R::failure(INVALID, "is_clique_free: `clique_size` must be at least the uniformity");
    if (op == Op::IsRamseyColouring && (red < s.uniformity || blue < s.uniformity))
        return R::failure(INVALID, "is_ramsey_colouring: clique sizes must be at least the uniformity");

    auto size = s.family->size();
    if (!size.ok) return R::failure(size.error.status, size.error.message);
    Reduction reduction = parse_reduction(req.reduction);
    Shared shared;
    auto prepared = prepare_all(reduction, size.value, shared);
    if (!prepared.ok) return R::failure(prepared.error.status, prepared.error.message);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, size.value));
    std::vector<Accumulator> accumulators;
    for (uint32_t thread = 0; thread < threads; ++thread) accumulators.emplace_back(reduction, &shared);

    auto statuses = parallel_ranges(size.value, threads, [&](uint32_t thread, uint64_t begin, uint64_t end) -> Status {
        Matrix member;
        for (uint64_t index = begin; index < end; ++index) {
            if (accumulators[thread].exhausted(index)) break;
            auto status = s.family->member_into(index, member);
            if (!status.ok) return status;
            Hypergraph h = decode(member, s.vertices);
            switch (op) {
            case Op::IsLinear: accumulators[thread].boolean(index, is_linear(h)); break;
            case Op::ColouringNumber: accumulators[thread].integer(index, colouring_number(h)); break;
            case Op::HasBergeCycle: accumulators[thread].boolean(index, has_berge_cycle(h, length)); break;
            case Op::BergeGirth: accumulators[thread].integer(index, berge_girth(h)); break;
            case Op::IsCliqueFree: accumulators[thread].boolean(index, !contains_clique(h, clique, true)); break;
            case Op::IsRamseyColouring:
                accumulators[thread].boolean(index, !contains_clique(h, red, true) && !contains_clique(h, blue, false));
                break;
            }
        }
        return ok();
    });
    for (const auto &status : statuses)
        if (!status.ok) return R::failure(status.error.status, status.error.message);
    return assemble(req, reduction, accumulators, shared);
}

BackendRegistration registration{Backend{
    "hypergraphs", "generic",
    [] { return true; },
    [](const Request &) { return true; },
    run,
    0}};

} // namespace
} // namespace lk::hypergraphs
