#include "../cayley_iso_common.hpp"
#include "../../../../runtime/src/graph.hpp"

namespace lk::cayley_iso {
namespace {

Result<std::vector<Entry>> canonical_graph(const GroupModel &group,
                                           const std::vector<uint64_t> &atom_words) {
    std::vector<uint8_t> selected = selected_elements(group, atom_words);
    std::vector<Entry> adjacency(group.order * group.order, 0);
    for (uint64_t x = 0; x < group.order; ++x)
        for (uint64_t y = x + 1; y < group.order; ++y) {
            bool edge = selected[group.mul(group.inverse[x], y)] != 0;
            adjacency[x * group.order + y] = adjacency[y * group.order + x] = edge;
        }
    return Result<std::vector<Entry>>::success(graph::canonical(adjacency.data(), group.order));
}

BackendRegistration registration{Backend{
    "cayley_iso", "generic",
    [] { return true; },
    [](const Request &req) { return req.family->kind == Family::Kind::GroupTables; },
    [](const Request &req) { return run_backend(req, canonical_graph); },
    0}};

} // namespace
} // namespace lk::cayley_iso
