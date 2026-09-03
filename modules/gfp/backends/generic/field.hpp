/* Portable F_p arithmetic and the reference Basis for walk.hpp. Any prime p < 2^32. */
#pragma once
#include "../walk.hpp"

namespace lk::gfp {

struct Field {
    uint64_t p;
    uint64_t barrett; /* floor(2^64 / p): one 64x64->128 multiply replaces a division */
    std::vector<Entry> inv_table; /* inverses when p is small enough to tabulate */

    explicit Field(uint64_t prime) : p(prime) {
        barrett = (uint64_t)(((unsigned __int128)1 << 64) / p);
        if (p <= (1u << 16)) {
            inv_table.assign(p, 0);
            for (uint64_t a = 1; a < p; ++a) inv_table[a] = (Entry)pow(a, p - 2);
        }
    }
    uint64_t reduce(uint64_t x) const {
        uint64_t q = (uint64_t)(((unsigned __int128)x * barrett) >> 64);
        uint64_t r = x - q * p;
        return r >= p ? r - p : r;
    }
    uint64_t pow(uint64_t a, uint64_t e) const {
        uint64_t r = 1;
        a %= p;
        while (e) {
            if (e & 1) r = (uint64_t)((unsigned __int128)r * a % p);
            a = (uint64_t)((unsigned __int128)a * a % p);
            e >>= 1;
        }
        return r;
    }
    Entry inverse(Entry a) const { return inv_table.empty() ? (Entry)pow(a, p - 2) : inv_table[a]; }

    /* row -= c * other. (p-c)*other + row < 2^64 for p < 2^32, so plain u64 arithmetic is exact. */
    void subtract_multiple(Entry *row, const Entry *other, Entry c, uint64_t cols) const {
        uint64_t m = p - c;
        for (uint64_t j = 0; j < cols; ++j) row[j] = (Entry)reduce((uint64_t)row[j] + m * other[j]);
    }
    void scale(Entry *row, Entry c, uint64_t cols) const {
        for (uint64_t j = 0; j < cols; ++j) row[j] = (Entry)reduce((uint64_t)row[j] * c);
    }
};

/* Echelon basis kept in insertion order: row i is zero at the pivots of the rows before it, has
 * leading entry 1 at its own pivot, and the pivot is its first nonzero column. */
struct EchelonBasis {
    Field f;
    uint64_t cols;
    std::vector<Entry> rows;
    std::vector<uint32_t> pivots;
    std::vector<Entry> scratch;
    mutable std::vector<uint64_t> order_buf;
    mutable std::vector<Entry> sorted_buf;
    mutable std::vector<uint32_t> piv_buf;

    EchelonBasis(uint64_t p, uint64_t c) : f(p), cols(c), scratch(c) {}
    uint64_t rank() const { return pivots.size(); }
    Entry negate(Entry a) const { return a ? (Entry)(f.p - a) : 0; }
    void clear() { rows.clear(); pivots.clear(); }
    Entry *row(uint64_t i) { return rows.data() + i * cols; }
    const Entry *row(uint64_t i) const { return rows.data() + i * cols; }

    void reduce_into(Entry *v) const {
        for (uint64_t i = 0; i < pivots.size(); ++i) {
            Entry c = v[pivots[i]];
            if (c) f.subtract_multiple(v, row(i), c, cols);
        }
    }
    bool add(const Entry *v) {
        std::copy(v, v + cols, scratch.begin());
        reduce_into(scratch.data());
        uint64_t lead = cols;
        for (uint64_t j = 0; j < cols; ++j)
            if (scratch[j]) { lead = j; break; }
        if (lead == cols) return false;
        if (scratch[lead] != 1) f.scale(scratch.data(), f.inverse(scratch[lead]), cols);
        rows.insert(rows.end(), scratch.begin(), scratch.end());
        pivots.push_back((uint32_t)lead);
        return true;
    }
    void remove_last() {
        rows.resize(rows.size() - cols);
        pivots.pop_back();
    }

    using Target = std::vector<Entry>;
    Target pack(const Entry *v) const { return Target(v, v + cols); }
    void reduce_by_last(Target &t) const {
        Entry c = t[pivots.back()];
        if (c) f.subtract_multiple(t.data(), row(rank() - 1), c, cols);
    }
    bool is_zero(const Target &t) const { return std::all_of(t.begin(), t.end(), [](Entry e) { return e == 0; }); }

    /* Back-substitute (row i clears its pivot from every earlier row; earlier rows are zero at
     * later pivots' columns only if the later pivot is to the right, which is exactly when the
     * entry can be nonzero), then sort rows by pivot. */
    void rref(std::vector<Entry> &out, std::vector<uint32_t> &piv) const {
        uint64_t r = rank();
        out = rows;
        piv = pivots;
        for (uint64_t i = 1; i < r; ++i)
            for (uint64_t e = 0; e < i; ++e) {
                Entry c = out[e * cols + piv[i]];
                if (c) f.subtract_multiple(out.data() + e * cols, out.data() + i * cols, c, cols);
            }
        auto &order = order_buf;
        order.resize(r);
        for (uint64_t i = 0; i < r; ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](uint64_t a, uint64_t b) { return piv[a] < piv[b]; });
        auto &sorted = sorted_buf;
        auto &sp = piv_buf;
        sorted.resize(r * cols);
        sp.resize(r);
        for (uint64_t i = 0; i < r; ++i) {
            std::copy(out.begin() + order[i] * cols, out.begin() + (order[i] + 1) * cols, sorted.begin() + i * cols);
            sp[i] = piv[order[i]];
        }
        out.swap(sorted);
        piv.swap(sp);
    }
};

} // namespace lk::gfp
