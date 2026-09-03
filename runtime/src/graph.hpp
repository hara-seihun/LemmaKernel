#pragma once

#include "object.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace lk::graph {

inline bool adjacent(const Entry *a, uint64_t n, uint64_t u, uint64_t v) {
    return a[u * n + v] != 0;
}

inline bool lex_less(const std::vector<Entry> &a, const std::vector<Entry> &b) {
    return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
}

/* Find the lexicographically least row-major adjacency matrix over all vertex labellings.
 * The ordered cells contain vertices with equal adjacency signatures to the assigned prefix.
 * Splitting each cell into non-neighbours then neighbours fixes every lexically forced choice;
 * only vertices in the first cell need to be individualised. */
inline std::vector<Entry> canonical(const Entry *a, uint64_t n) {
    std::vector<Entry> best;
    std::vector<uint64_t> assigned;
    std::vector<std::vector<uint64_t>> initial(1);
    initial[0].resize(n);
    for (uint64_t i = 0; i < n; ++i) initial[0][i] = i;

    auto search = [&](auto &self, const std::vector<std::vector<uint64_t>> &cells) -> void {
        if (assigned.size() == n) {
            std::vector<Entry> candidate(n * n);
            for (uint64_t i = 0; i < n; ++i)
                for (uint64_t j = 0; j < n; ++j)
                    candidate[i * n + j] = a[assigned[i] * n + assigned[j]];
            if (best.empty() || lex_less(candidate, best)) best = std::move(candidate);
            return;
        }

        const auto &first = cells.front();
        std::vector<uint64_t> tried;
        for (uint64_t v : first) {
            bool twin = false;
            for (uint64_t u : tried) {
                bool same = true;
                for (const auto &cell : cells)
                    for (uint64_t w : cell)
                        if (w != u && w != v && adjacent(a, n, u, w) != adjacent(a, n, v, w)) {
                            same = false;
                            break;
                        }
                if (same) {
                    twin = true;
                    break;
                }
            }
            if (twin) continue;
            tried.push_back(v);
            assigned.push_back(v);
            std::vector<std::vector<uint64_t>> next;
            for (const auto &cell : cells) {
                std::vector<uint64_t> zero, one;
                for (uint64_t w : cell) {
                    if (w == v) continue;
                    (adjacent(a, n, v, w) ? one : zero).push_back(w);
                }
                if (!zero.empty()) next.push_back(std::move(zero));
                if (!one.empty()) next.push_back(std::move(one));
            }
            self(self, next);
            assigned.pop_back();
        }
    };
    search(search, initial);
    return best;
}

inline std::vector<std::pair<uint64_t, uint64_t>> edge_pairs(uint64_t n) {
    std::vector<std::pair<uint64_t, uint64_t>> edges;
    for (uint64_t i = 0; i < n; ++i)
        for (uint64_t j = i + 1; j < n; ++j) edges.emplace_back(i, j);
    return edges;
}

/* The first upper-triangle entry is the most significant bit, matching row-major lexical order. */
inline uint64_t upper_mask(const std::vector<Entry> &a, uint64_t n) {
    uint64_t mask = 0;
    for (auto [i, j] : edge_pairs(n)) mask = (mask << 1) | (a[i * n + j] != 0);
    return mask;
}

inline std::vector<Entry> from_upper_mask(uint64_t mask, uint64_t n) {
    auto edges = edge_pairs(n);
    std::vector<Entry> a(n * n, 0);
    for (uint64_t q = 0; q < edges.size(); ++q) {
        auto [i, j] = edges[q];
        Entry bit = (Entry)((mask >> (edges.size() - 1 - q)) & 1);
        a[i * n + j] = a[j * n + i] = bit;
    }
    return a;
}

} // namespace lk::graph
