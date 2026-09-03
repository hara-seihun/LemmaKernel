#include "family.hpp"

#include <algorithm>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace lk {

namespace {

constexpr int INVALID = 1;
constexpr int INTERNAL = 4;

bool mul_overflows(uint64_t a, uint64_t b, uint64_t &out) {
    unsigned __int128 r = (unsigned __int128)a * b;
    out = (uint64_t)r;
    return r > UINT64_MAX;
}
bool add_overflows(uint64_t a, uint64_t b, uint64_t &out) {
    out = a + b;
    return out < a;
}

Result<uint64_t> binom(uint64_t n, uint64_t k) {
    if (k > n) return Result<uint64_t>::success(0);
    if (k > n - k) k = n - k;
    unsigned __int128 r = 1;
    for (uint64_t i = 1; i <= k; ++i) {
        r = r * (n - k + i) / i;
        if (r > UINT64_MAX) return Result<uint64_t>::failure(INVALID, "family size does not fit in 64 bits");
    }
    return Result<uint64_t>::success((uint64_t)r);
}

Result<uint64_t> pow_checked(uint64_t p, uint64_t e) {
    uint64_t r = 1;
    for (uint64_t i = 0; i < e; ++i)
        if (mul_overflows(r, p, r)) return Result<uint64_t>::failure(INVALID, "family size does not fit in 64 bits");
    return Result<uint64_t>::success(r);
}

struct Triple {
    uint64_t x, y, z;
    bool operator==(const Triple &) const = default;
};

struct TripleHash {
    size_t operator()(const Triple &s) const {
        size_t h = std::hash<uint64_t>{}(s.x);
        h ^= std::hash<uint64_t>{}(s.y) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<uint64_t>{}(s.z) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
    }
};

struct PartitionCounter {
    uint64_t cap;
    bool odd;
    bool overflow = false;
    std::unordered_map<Triple, uint64_t, TripleHash> memo;

    PartitionCounter(uint64_t total, uint64_t max_multiplicity, bool distinct, bool odd_parts)
        : cap(distinct ? 1 : (max_multiplicity ? max_multiplicity : total)), odd(odd_parts) {}

    uint64_t add(uint64_t a, uint64_t b) {
        unsigned __int128 s = (unsigned __int128)a + b;
        if (s > UINT64_MAX) { overflow = true; return UINT64_MAX; }
        return (uint64_t)s;
    }

    /* Count frequency vectors for available part sizes `part, part-1, ..., 1`. */
    uint64_t count(uint64_t rem, uint64_t part, uint64_t slots) {
        if (rem == 0) return 1;
        if (part == 0 || slots == 0) return 0;
        Triple key{rem, part, slots};
        if (auto it = memo.find(key); it != memo.end()) return it->second;
        uint64_t total = 0;
        if (odd && part % 2 == 0) {
            total = count(rem, part - 1, slots);
        } else {
            uint64_t max_count = std::min({cap, slots, rem / part});
            for (uint64_t c = max_count + 1; c-- > 0;)
                total = add(total, count(rem - c * part, part - 1, slots - c));
        }
        memo.emplace(key, total);
        return total;
    }

    uint64_t largest_branch(uint64_t total, uint64_t largest, uint64_t slots) {
        if ((odd && largest % 2 == 0) || largest > total || slots == 0) return 0;
        uint64_t answer = 0;
        uint64_t max_count = std::min({cap, slots, total / largest});
        for (uint64_t c = max_count + 1; c-- > 1;)
            answer = add(answer, count(total - c * largest, largest - 1, slots - c));
        return answer;
    }
};

struct CompositionCounter {
    uint64_t max_part;
    bool overflow = false;
    std::unordered_map<Triple, uint64_t, TripleHash> memo;

    explicit CompositionCounter(uint64_t maximum) : max_part(maximum) {}

    uint64_t count(uint64_t rem, uint64_t parts) {
        if (parts == 0) return rem == 0;
        if (rem < parts || (unsigned __int128)parts * max_part < rem) return 0;
        Triple key{rem, parts, 0};
        if (auto it = memo.find(key); it != memo.end()) return it->second;
        uint64_t total = 0;
        uint64_t hi = std::min(max_part, rem);
        for (uint64_t x = 1; x <= hi; ++x) {
            unsigned __int128 s = (unsigned __int128)total + count(rem - x, parts - 1);
            if (s > UINT64_MAX) { overflow = true; total = UINT64_MAX; break; }
            total = (uint64_t)s;
        }
        memo.emplace(key, total);
        return total;
    }
};

uint64_t effective_bound(uint64_t bound, uint64_t total) { return bound ? std::min(bound, total) : total; }

} // namespace

/* Pivot sets of a Grassmannian in lexicographic order with the leaf count under each. */
struct PivotTable {
    uint64_t n, h, p;
    std::vector<std::vector<uint32_t>> sets;
    std::vector<uint64_t> offsets; /* size sets.size()+1 */
    std::vector<std::vector<uint32_t>> free_counts; /* per set, per row */
};

namespace {

Result<PivotTable> build_pivot_table(uint64_t p, uint64_t n, uint64_t h) {
    PivotTable t{n, h, p, {}, {0}, {}};
    std::vector<uint32_t> c(h);
    for (uint64_t i = 0; i < h; ++i) c[i] = (uint32_t)i;
    for (;;) {
        std::vector<uint32_t> fr(h);
        uint64_t leaves = 1;
        for (uint64_t j = 0; j < h; ++j) {
            uint32_t f = 0;
            for (uint64_t col = c[j] + 1; col < n; ++col)
                if (!std::binary_search(c.begin(), c.end(), (uint32_t)col)) ++f;
            fr[j] = f;
            auto pw = pow_checked(p, f);
            if (!pw.ok) return Result<PivotTable>::failure(pw.error.status, pw.error.message);
            if (mul_overflows(leaves, pw.value, leaves))
                return Result<PivotTable>::failure(INVALID, "family size does not fit in 64 bits");
        }
        t.sets.push_back(c);
        t.free_counts.push_back(fr);
        uint64_t next;
        if (add_overflows(t.offsets.back(), leaves, next))
            return Result<PivotTable>::failure(INVALID, "family size does not fit in 64 bits");
        t.offsets.push_back(next);
        int64_t i = (int64_t)h - 1;
        while (i >= 0 && c[i] == n - h + i) --i;
        if (i < 0) break;
        ++c[i];
        for (uint64_t j = i + 1; j < h; ++j) c[j] = c[j - 1] + 1;
    }
    return Result<PivotTable>::success(std::move(t));
}

struct TransformVisitor : Family::Visitor {
    Family::Visitor &inner;
    const Matrix &c;
    uint64_t p, in_cols;
    std::vector<Entry> buf;
    TransformVisitor(Family::Visitor &v, const Matrix &cm, uint64_t prime, uint64_t incols)
        : inner(v), c(cm), p(prime), in_cols(incols), buf(cm.cols) {}
    Step push(const Entry *row, uint64_t first, uint64_t below) override {
        std::vector<uint64_t> acc(c.cols, 0);
        const Entry *cd = c.entries.data();
        for (uint64_t i = 0; i < in_cols; ++i) {
            uint64_t r = row[i];
            if (!r) continue;
            const Entry *crow = cd + i * c.cols;
            for (uint64_t j = 0; j < c.cols; ++j) acc[j] = (acc[j] + r * crow[j]) % p;
        }
        for (uint64_t j = 0; j < c.cols; ++j) buf[j] = (Entry)acc[j];
        return inner.push(buf.data(), first, below);
    }
    void pop() override { inner.pop(); }
    void leaf(uint64_t i) override { inner.leaf(i); }
    void take_all(uint64_t f, uint64_t n) override { inner.take_all(f, n); }
    void skip_all(uint64_t f, uint64_t n) override { inner.skip_all(f, n); }
};

} // namespace

const char *family_kind_name(Family::Kind k) {
    switch (k) {
    case Family::Kind::Explicit: return "explicit";
    case Family::Kind::Subsets: return "subsets";
    case Family::Kind::Grassmannian: return "grassmannian";
    case Family::Kind::AllMatrices: return "all_matrices";
    case Family::Kind::Transform: return "transform";
    case Family::Kind::Stack: return "stack";
    case Family::Kind::GroupElements: return "group_elements";
    case Family::Kind::SubsetsOf: return "subsets_of";
    case Family::Kind::SymmetricMatrices: return "symmetric_matrices";
    case Family::Kind::Range: return "range";
    case Family::Kind::Words: return "words";
    case Family::Kind::Partitions: return "partitions";
    case Family::Kind::Compositions: return "compositions";
    }
    return "?";
}

uint64_t Family::prime() const {
    if (kind == Kind::Range || kind == Kind::Words || kind == Kind::Partitions || kind == Kind::Compositions)
        return NATURALS;
    return child ? child->prime() : p;
}

uint64_t Family::rows() const {
    switch (kind) {
    case Kind::Explicit: return data->rows;
    case Kind::Subsets: return k;
    case Kind::Grassmannian: return h;
    case Kind::AllMatrices: return m;
    case Kind::Transform: return child->rows();
    case Kind::Stack: return child->rows() + data->rows;
    case Kind::GroupElements: return 1;
    case Kind::SubsetsOf: return k;
    case Kind::SymmetricMatrices: return n;
    case Kind::Range: return 1;
    case Kind::Words: return 1;
    case Kind::Partitions: return 1;
    case Kind::Compositions: return 1;
    }
    return 0;
}

uint64_t Family::cols() const {
    switch (kind) {
    case Kind::Explicit: return data->cols;
    case Kind::Subsets: return data->cols;
    case Kind::Grassmannian: return n;
    case Kind::AllMatrices: return n;
    case Kind::Transform: return data->cols;
    case Kind::Stack: return child->cols();
    case Kind::GroupElements: return data->cols;
    case Kind::SubsetsOf: return data->cols;
    case Kind::SymmetricMatrices: return n;
    case Kind::Range: return 1;
    case Kind::Words: return n;
    case Kind::Partitions: return n;
    case Kind::Compositions: return n;
    }
    return 0;
}

bool Family::is_explicit() const {
    return kind == Kind::Explicit;
}

Result<uint64_t> Family::size() const {
    switch (kind) {
    case Kind::Explicit: return Result<uint64_t>::success(data->count);
    case Kind::Subsets:
    case Kind::SubsetsOf: return binom(data->count, k);
    case Kind::Grassmannian: {
        auto t = pivot_table();
        if (!t.ok) return Result<uint64_t>::failure(t.error.status, t.error.message);
        return Result<uint64_t>::success(t.value->offsets.back());
    }
    case Kind::AllMatrices:
    case Kind::Words: return pow_checked(p, m * n);
    case Kind::SymmetricMatrices: return pow_checked(p, n * (n + 1) / 2);
    case Kind::Range: return Result<uint64_t>::success(b - a);
    case Kind::Partitions: {
        uint64_t largest = effective_bound(m, n), slots = effective_bound(k, n);
        PartitionCounter counter(n, h, a != 0, b != 0);
        uint64_t total = counter.count(n, largest, slots);
        if (counter.overflow) return Result<uint64_t>::failure(INVALID, "family size does not fit in 64 bits");
        return Result<uint64_t>::success(total);
    }
    case Kind::Compositions: {
        CompositionCounter counter(effective_bound(m, n));
        uint64_t total = 0;
        uint64_t first_parts = k ? k : 1, last_parts = k ? k : n;
        for (uint64_t parts = first_parts; parts <= last_parts; ++parts) {
            unsigned __int128 next = (unsigned __int128)total + counter.count(n, parts);
            if (next > UINT64_MAX) counter.overflow = true;
            total = next > UINT64_MAX ? UINT64_MAX : (uint64_t)next;
        }
        if (counter.overflow) return Result<uint64_t>::failure(INVALID, "family size does not fit in 64 bits");
        return Result<uint64_t>::success(total);
    }
    case Kind::Transform:
    case Kind::Stack: return child->size();
    case Kind::GroupElements: {
        auto g = group_elements();
        if (!g.ok) return Result<uint64_t>::failure(g.error.status, g.error.message);
        return Result<uint64_t>::success(g.value->size() / data->cols);
    }
    }
    return Result<uint64_t>::failure(INTERNAL, "unknown family kind");
}

Result<uint64_t> Family::top_count() const {
    switch (kind) {
    case Kind::Explicit: return Result<uint64_t>::success(data->count);
    case Kind::Subsets:
    case Kind::SubsetsOf: return Result<uint64_t>::success(data->count - k + 1);
    case Kind::Grassmannian: return binom(n, h);
    case Kind::AllMatrices:
    case Kind::Words:
    case Kind::SymmetricMatrices: return pow_checked(p, n);
    case Kind::Range: return size();
    case Kind::Partitions: return Result<uint64_t>::success(effective_bound(m, n));
    case Kind::Compositions: return Result<uint64_t>::success(k ? 1 : n);
    case Kind::Transform:
    case Kind::Stack: return child->top_count();
    case Kind::GroupElements: return size();
    }
    return Result<uint64_t>::failure(INTERNAL, "unknown family kind");
}

Result<Matrix> Family::member(uint64_t index) const {
    Matrix out;
    auto st = member_into(index, out);
    if (!st.ok) return Result<Matrix>::failure(st.error.status, st.error.message);
    return Result<Matrix>::success(std::move(out));
}

Status Family::member_into(uint64_t index, Matrix &out) const {
    auto sz = size();
    if (!sz.ok) return fail(sz.error.status, sz.error.message);
    if (index >= sz.value) return fail(INVALID, "member index out of range");
    out.p = prime();
    out.count = 1;
    out.rows = rows();
    out.cols = cols();
    switch (kind) {
    case Kind::Explicit:
        out.entries.assign(data->at(index), data->at(index) + out.rows * out.cols);
        break;
    case Kind::Subsets:
    case Kind::SubsetsOf: {
        uint64_t D = data->count, remaining = index, prev = 0;
        out.entries.resize(k * out.cols);
        for (uint64_t j = 0; j < k; ++j) {
            uint64_t c = prev;
            for (;; ++c) {
                auto below = binom(D - 1 - c, k - 1 - j);
                if (!below.ok) return fail(below.error.status, below.error.message);
                if (remaining < below.value) break;
                remaining -= below.value;
            }
            std::copy(data->at(0) + c * out.cols, data->at(0) + (c + 1) * out.cols, out.entries.begin() + j * out.cols);
            prev = c + 1;
        }
        break;
    }
    case Kind::Grassmannian: {
        auto tr = pivot_table();
        if (!tr.ok) return fail(tr.error.status, tr.error.message);
        const PivotTable &t = *tr.value;
        auto it = std::upper_bound(t.offsets.begin(), t.offsets.end(), index);
        uint64_t s = (uint64_t)(it - t.offsets.begin()) - 1;
        uint64_t rem = index - t.offsets[s];
        const auto &piv = t.sets[s];
        out.entries.assign(h * n, 0);
        /* The free entries, row-major, are the base-p digits of `rem`, most significant first;
         * peel them off from the last position backwards. */
        for (int64_t j = (int64_t)h - 1; j >= 0; --j) {
            Entry *row = out.entries.data() + j * n;
            row[piv[j]] = 1;
            int64_t next_piv = (int64_t)h - 1;
            for (int64_t col = (int64_t)n - 1; col > (int64_t)piv[j]; --col) {
                while (next_piv > j && (int64_t)piv[next_piv] > col) --next_piv;
                if (next_piv > j && (int64_t)piv[next_piv] == col) continue;
                row[col] = (Entry)(rem % p);
                rem /= p;
            }
        }
        break;
    }
    case Kind::AllMatrices:
    case Kind::Words: {
        out.entries.assign(m * n, 0);
        uint64_t rem = index;
        for (int64_t q = (int64_t)(m * n) - 1; q >= 0; --q) {
            out.entries[q] = (Entry)(rem % p);
            rem /= p;
        }
        break;
    }
    case Kind::SymmetricMatrices: {
        /* The upper triangle, row-major, holds the base-p digits of the index. */
        out.entries.assign(n * n, 0);
        uint64_t rem = index;
        for (int64_t i = (int64_t)n - 1; i >= 0; --i)
            for (int64_t j = (int64_t)n - 1; j >= i; --j) {
                Entry e = (Entry)(rem % p);
                rem /= p;
                out.entries[i * n + j] = e;
                out.entries[j * n + i] = e;
            }
        break;
    }
    case Kind::Range:
        out.entries.assign(1, (Entry)(a + index));
        break;
    case Kind::Partitions: {
        uint64_t rem = n, slots = effective_bound(k, n), largest = effective_bound(m, n), pos = 0;
        PartitionCounter counter(n, h, a != 0, b != 0);
        out.entries.assign(n, 0);
        for (uint64_t part = largest; part > 0 && rem > 0; --part) {
            if (b && part % 2 == 0) continue;
            uint64_t cap = a ? 1 : (h ? h : n);
            uint64_t max_count = std::min({cap, slots, rem / part});
            bool selected = false;
            for (uint64_t c = max_count + 1; c-- > 0;) {
                uint64_t below = counter.count(rem - c * part, part - 1, slots - c);
                if (index < below) {
                    for (uint64_t j = 0; j < c; ++j) out.entries[pos++] = (Entry)part;
                    rem -= c * part;
                    slots -= c;
                    selected = true;
                    break;
                }
                index -= below;
            }
            if (!selected) return fail(INTERNAL, "partition unranking failed");
        }
        if (rem != 0 || index != 0) return fail(INTERNAL, "partition unranking failed");
        break;
    }
    case Kind::Compositions: {
        uint64_t maximum = effective_bound(m, n), parts = k;
        CompositionCounter counter(maximum);
        if (parts == 0) {
            for (parts = 1; parts <= n; ++parts) {
                uint64_t below = counter.count(n, parts);
                if (index < below) break;
                index -= below;
            }
        }
        out.entries.assign(n, 0);
        uint64_t rem = n;
        for (uint64_t pos = 0; pos < parts; ++pos) {
            uint64_t hi = std::min(maximum, rem);
            bool selected = false;
            for (uint64_t x = hi + 1; x-- > 1;) {
                uint64_t below = counter.count(rem - x, parts - pos - 1);
                if (index < below) {
                    out.entries[pos] = (Entry)x;
                    rem -= x;
                    selected = true;
                    break;
                }
                index -= below;
            }
            if (!selected) return fail(INTERNAL, "composition unranking failed");
        }
        if (rem != 0 || index != 0) return fail(INTERNAL, "composition unranking failed");
        break;
    }
    case Kind::Transform: {
        auto inner = child->member(index);
        if (!inner.ok) return fail(inner.error.status, inner.error.message);
        out.entries.assign(out.rows * out.cols, 0);
        for (uint64_t r = 0; r < out.rows; ++r)
            for (uint64_t i = 0; i < inner.value.cols; ++i) {
                uint64_t v = inner.value.entries[r * inner.value.cols + i];
                if (!v) continue;
                for (uint64_t j = 0; j < out.cols; ++j)
                    out.entries[r * out.cols + j] =
                        (Entry)((out.entries[r * out.cols + j] + v * data->entries[i * data->cols + j]) % out.p);
            }
        break;
    }
    case Kind::Stack: {
        auto inner = child->member(index);
        if (!inner.ok) return fail(inner.error.status, inner.error.message);
        out.entries = inner.value.entries;
        out.entries.insert(out.entries.end(), data->entries.begin(), data->entries.end());
        break;
    }
    case Kind::GroupElements: {
        auto g = group_elements();
        if (!g.ok) return fail(g.error.status, g.error.message);
        out.entries.assign(g.value->begin() + index * out.cols, g.value->begin() + (index + 1) * out.cols);
        break;
    }
    }
    return ok();
}

Result<const PivotTable *> Family::pivot_table() const {
    using R = Result<const PivotTable *>;
    if (kind != Kind::Grassmannian) return R::failure(INVALID, "not a grassmannian family");
    if (auto *ready = pivots_ready.load(std::memory_order_acquire)) return R::success(ready);
    static std::mutex mu;
    std::lock_guard<std::mutex> lock(mu);
    if (!pivots) {
        auto t = build_pivot_table(p, n, h);
        if (!t.ok) return R::failure(t.error.status, t.error.message);
        pivots = std::make_shared<const PivotTable>(std::move(t.value));
        pivots_ready.store(pivots.get(), std::memory_order_release);
    }
    return R::success(pivots.get());
}

Result<const std::vector<Entry> *> Family::group_elements() const {
    using R = Result<const std::vector<Entry> *>;
    if (kind != Kind::GroupElements) return R::failure(INVALID, "not a group_elements family");
    if (auto *ready = elements_ready.load(std::memory_order_acquire)) return R::success(ready);
    static std::mutex mu;
    std::lock_guard<std::mutex> lock(mu);
    if (!elements) {
        auto c = permutation_closure(*data, 1ULL << 26);
        if (!c.ok) return R::failure(c.error.status, c.error.message);
        elements = std::make_shared<const std::vector<Entry>>(std::move(c.value));
        elements_ready.store(elements.get(), std::memory_order_release);
    }
    return R::success(elements.get());
}

Result<uint64_t> Family::index_of(const Matrix &mem) const {
    using R = Result<uint64_t>;
    if (mem.count != 1 || mem.rows != rows() || mem.cols != cols() || mem.p != prime())
        return R::failure(INVALID, "index_of: member has the wrong shape for this family");
    switch (kind) {
    case Kind::Subsets:
    case Kind::SubsetsOf: {
        /* Rows must be dictionary rows; the dictionary must have no duplicate rows. */
        uint64_t D = data->count, cols_ = data->cols;
        std::vector<uint64_t> idx(k);
        for (uint64_t j = 0; j < k; ++j) {
            uint64_t found = D;
            for (uint64_t c = 0; c < D; ++c)
                if (std::equal(data->at(c), data->at(c) + cols_, mem.entries.begin() + j * cols_)) { found = c; break; }
            if (found == D) return R::failure(INVALID, "index_of: row is not in the dictionary");
            idx[j] = found;
        }
        for (uint64_t j = 1; j < k; ++j)
            if (idx[j] <= idx[j - 1]) return R::failure(INVALID, "index_of: rows are not in increasing dictionary order");
        uint64_t index = 0, prev = 0;
        for (uint64_t j = 0; j < k; ++j) {
            for (uint64_t c = prev; c < idx[j]; ++c) {
                auto b = binom(D - 1 - c, k - 1 - j);
                if (!b.ok) return b;
                index += b.value;
            }
            prev = idx[j] + 1;
        }
        return R::success(index);
    }
    case Kind::Grassmannian: {
        auto tr = pivot_table();
        if (!tr.ok) return R::failure(tr.error.status, tr.error.message);
        const PivotTable &t = *tr.value;
        uint32_t piv_buf[64];
        std::vector<uint32_t> piv_heap;
        uint32_t *piv = h <= 64 ? piv_buf : (piv_heap.resize(h), piv_heap.data());
        for (uint64_t j = 0; j < h; ++j) {
            const Entry *row = mem.entries.data() + j * n;
            uint64_t lead = n;
            for (uint64_t c = 0; c < n; ++c) if (row[c]) { lead = c; break; }
            if (lead == n || row[lead] != 1 || (j > 0 && lead <= piv[j - 1]))
                return R::failure(INVALID, "index_of: member is not in reduced row echelon form");
            piv[j] = (uint32_t)lead;
        }
        auto less = [&](const std::vector<uint32_t> &a, const uint32_t *b) { return std::lexicographical_compare(a.begin(), a.end(), b, b + h); };
        auto it = std::lower_bound(t.sets.begin(), t.sets.end(), piv, less);
        if (it == t.sets.end() || !std::equal(it->begin(), it->end(), piv)) return R::failure(INTERNAL, "index_of: pivot set not found");
        uint64_t s = (uint64_t)(it - t.sets.begin());
        uint64_t index = t.offsets[s];
        uint64_t rem = 0;
        for (uint64_t j = 0; j < h; ++j) {
            const Entry *row = mem.entries.data() + j * n;
            for (uint64_t c = 0; c < n; ++c) {
                bool is_piv = std::binary_search(piv, piv + h, (uint32_t)c);
                if (is_piv || c < piv[j]) {
                    if (row[c] != (c == piv[j] ? 1u : 0u)) return R::failure(INVALID, "index_of: member is not in reduced row echelon form");
                } else rem = rem * p + row[c];
            }
        }
        return R::success(index + rem);
    }
    case Kind::AllMatrices:
    case Kind::Words: {
        uint64_t index = 0;
        for (Entry e : mem.entries) {
            if (e >= p) return R::failure(INVALID, "index_of: entry out of range");
            index = index * p + e;
        }
        return R::success(index);
    }
    case Kind::SymmetricMatrices: {
        uint64_t index = 0;
        for (uint64_t i = 0; i < n; ++i)
            for (uint64_t j = i; j < n; ++j) {
                if (mem.entries[i * n + j] != mem.entries[j * n + i]) return R::failure(INVALID, "index_of: matrix is not symmetric");
                index = index * p + mem.entries[i * n + j];
            }
        return R::success(index);
    }
    case Kind::Range: {
        uint64_t v = mem.entries[0];
        if (v < a || v >= b) return R::failure(INVALID, "index_of: value outside the range");
        return R::success(v - a);
    }
    case Kind::GroupElements: {
        auto g = group_elements();
        if (!g.ok) return R::failure(g.error.status, g.error.message);
        uint64_t nn = data->cols, count = g.value->size() / nn;
        uint64_t lo = 0, hi = count;
        while (lo < hi) {
            uint64_t mid = (lo + hi) / 2;
            const Entry *e = g.value->data() + mid * nn;
            if (std::lexicographical_compare(e, e + nn, mem.entries.begin(), mem.entries.end())) lo = mid + 1;
            else hi = mid;
        }
        if (lo < count && std::equal(g.value->data() + lo * nn, g.value->data() + (lo + 1) * nn, mem.entries.begin()))
            return R::success(lo);
        return R::failure(INVALID, "index_of: permutation is not in the group");
    }
    default:
        return R::failure(INVALID, std::string("index_of is not defined for ") + family_kind_name(kind) + " families");
    }
}

Status Family::enumerate(Visitor &v, uint64_t top_begin, uint64_t top_end) const {
    using Step = Visitor::Step;
    switch (kind) {
    case Kind::Explicit: {
        for (uint64_t i = top_begin; i < top_end; ++i) {
            const Entry *base = data->at(i);
            uint64_t pushed = 0;
            Step step = Step::Descend;
            for (uint64_t r = 0; r < data->rows; ++r) {
                step = v.push(base + r * data->cols, i, 1);
                ++pushed;
                if (step != Step::Descend) break;
            }
            if (step == Step::Descend) v.leaf(i);
            else if (step == Step::TakeAll) v.take_all(i, 1);
            else v.skip_all(i, 1);
            for (uint64_t r = 0; r < pushed; ++r) v.pop();
        }
        return ok();
    }
    case Kind::Subsets:
    case Kind::SubsetsOf: {
        uint64_t D = data->count, cols = data->cols;
        std::vector<std::vector<uint64_t>> tail(k + 1); /* tail[j][c] = C(D-1-c, k-1-j) */
        for (uint64_t j = 0; j < k; ++j) {
            tail[j].resize(D);
            for (uint64_t c = 0; c < D; ++c) {
                auto b = binom(D - 1 - c, k - 1 - j);
                if (!b.ok) return fail(b.error.status, b.error.message);
                tail[j][c] = b.value;
            }
        }
        uint64_t base_index = 0;
        for (uint64_t c = 0; c < top_begin; ++c) base_index += tail[0][c];
        struct Frame { uint64_t c, index; };
        std::vector<Frame> stack;
        auto descend = [&](auto &self, uint64_t depth, uint64_t prev, uint64_t index, uint64_t c_end) -> void {
            for (uint64_t c = prev; c < c_end; ++c) {
                uint64_t below = tail[depth][c];
                Step step = v.push(data->at(0) + c * cols, index, below);
                if (step == Step::Descend) {
                    if (depth + 1 == k) v.leaf(index);
                    else self(self, depth + 1, c + 1, index, D - (k - depth - 1) + 1);
                } else if (step == Step::TakeAll) {
                    v.take_all(index, below);
                } else {
                    v.skip_all(index, below);
                }
                v.pop();
                index += below;
            }
        };
        descend(descend, 0, top_begin, base_index, top_end);
        return ok();
    }
    case Kind::Grassmannian: {
        auto tr = pivot_table();
        if (!tr.ok) return fail(tr.error.status, tr.error.message);
        const PivotTable &t = *tr.value;
        std::vector<Entry> row(n);
        for (uint64_t s = top_begin; s < top_end; ++s) {
            const auto &piv = t.sets[s];
            const auto &fr = t.free_counts[s];
            std::vector<uint64_t> radix(h), below(h);
            for (uint64_t j = 0; j < h; ++j) radix[j] = pow_checked(p, fr[j]).value;
            for (int64_t j = (int64_t)h - 1; j >= 0; --j) below[j] = (j + 1 < (int64_t)h) ? below[j + 1] * radix[j + 1] : 1;
            std::vector<std::vector<uint64_t>> free_cols(h);
            for (uint64_t j = 0; j < h; ++j)
                for (uint64_t col = piv[j] + 1; col < n; ++col)
                    if (!std::binary_search(piv.begin(), piv.end(), (uint32_t)col)) free_cols[j].push_back(col);
            auto descend = [&](auto &self, uint64_t j, uint64_t index) -> void {
                std::vector<Entry> r(n, 0);
                r[piv[j]] = 1;
                const auto &fc = free_cols[j];
                for (uint64_t d = 0; d < radix[j]; ++d) {
                    Step step = v.push(r.data(), index, below[j]);
                    if (step == Step::Descend) {
                        if (j + 1 == h) v.leaf(index);
                        else self(self, j + 1, index);
                    } else if (step == Step::TakeAll) {
                        v.take_all(index, below[j]);
                    } else {
                        v.skip_all(index, below[j]);
                    }
                    v.pop();
                    index += below[j];
                    for (int64_t q = (int64_t)fc.size() - 1; q >= 0; --q) {
                        if (++r[fc[q]] < p) break;
                        r[fc[q]] = 0;
                    }
                }
            };
            descend(descend, 0, t.offsets[s]);
        }
        return ok();
    }
    case Kind::SymmetricMatrices: {
        /* Row j has n - j free entries (columns j..n-1); its first j entries are copies of column
         * j of the rows above. The partial matrix is kept in `mat`. */
        std::vector<uint64_t> below(n);
        for (int64_t j = (int64_t)n - 1; j >= 0; --j)
            below[j] = (j + 1 < (int64_t)n) ? below[j + 1] * pow_checked(p, n - j - 1).value : 1;
        std::vector<Entry> mat(n * n, 0);
        auto descend = [&](auto &self, uint64_t j, uint64_t index, uint64_t d_begin, uint64_t d_end) -> void {
            Entry *row = mat.data() + j * n;
            for (uint64_t c = 0; c < j; ++c) row[c] = mat[c * n + j];
            uint64_t rem = d_begin;
            for (int64_t q = (int64_t)n - 1; q >= (int64_t)j; --q) { row[q] = (Entry)(rem % p); rem /= p; }
            for (uint64_t d = d_begin; d < d_end; ++d) {
                Step step = v.push(row, index, below[j]);
                if (step == Step::Descend) {
                    if (j + 1 == n) v.leaf(index);
                    else self(self, j + 1, index, 0, pow_checked(p, n - j - 1).value);
                } else if (step == Step::TakeAll) {
                    v.take_all(index, below[j]);
                } else {
                    v.skip_all(index, below[j]);
                }
                v.pop();
                index += below[j];
                for (int64_t q = (int64_t)n - 1; q >= (int64_t)j; --q) {
                    if (++row[q] < p) break;
                    row[q] = 0;
                }
            }
        };
        descend(descend, 0, top_begin * below[0], top_begin, top_end);
        return ok();
    }
    case Kind::Range: {
        Entry e;
        for (uint64_t i = top_begin; i < top_end; ++i) {
            e = (Entry)(a + i);
            Step step = v.push(&e, i, 1);
            if (step == Step::Descend) v.leaf(i);
            else if (step == Step::TakeAll) v.take_all(i, 1);
            else v.skip_all(i, 1);
            v.pop();
        }
        return ok();
    }
    case Kind::Partitions: {
        uint64_t maximum = effective_bound(m, n), slots = effective_bound(k, n);
        PartitionCounter counter(n, h, a != 0, b != 0);
        uint64_t index = 0;
        for (uint64_t t = 0; t < top_begin; ++t)
            index += counter.largest_branch(n, maximum - t, slots);
        std::vector<Entry> row(n, 0);
        auto emit = [&](uint64_t i) {
            Step step = v.push(row.data(), i, 1);
            if (step == Step::Descend) v.leaf(i);
            else if (step == Step::TakeAll) v.take_all(i, 1);
            else v.skip_all(i, 1);
            v.pop();
        };
        auto descend = [&](auto &self, uint64_t part, uint64_t rem, uint64_t left, uint64_t pos) -> void {
            if (rem == 0) { emit(index++); return; }
            if (part == 0 || left == 0) return;
            if (b && part % 2 == 0) { self(self, part - 1, rem, left, pos); return; }
            uint64_t cap = a ? 1 : (h ? h : n);
            uint64_t max_count = std::min({cap, left, rem / part});
            for (uint64_t c = max_count + 1; c-- > 0;) {
                for (uint64_t j = 0; j < c; ++j) row[pos + j] = (Entry)part;
                self(self, part - 1, rem - c * part, left - c, pos + c);
                for (uint64_t j = 0; j < c; ++j) row[pos + j] = 0;
            }
        };
        for (uint64_t t = top_begin; t < top_end; ++t) {
            uint64_t largest = maximum - t;
            if ((b && largest % 2 == 0) || largest > n) continue;
            uint64_t cap = a ? 1 : (h ? h : n);
            uint64_t max_count = std::min({cap, slots, n / largest});
            for (uint64_t c = max_count + 1; c-- > 1;) {
                for (uint64_t j = 0; j < c; ++j) row[j] = (Entry)largest;
                descend(descend, largest - 1, n - c * largest, slots - c, c);
                for (uint64_t j = 0; j < c; ++j) row[j] = 0;
            }
        }
        return ok();
    }
    case Kind::Compositions: {
        uint64_t maximum = effective_bound(m, n);
        CompositionCounter counter(maximum);
        uint64_t index = 0;
        if (!k)
            for (uint64_t parts = 1; parts <= top_begin; ++parts) index += counter.count(n, parts);
        std::vector<Entry> row(n, 0);
        auto descend = [&](auto &self, uint64_t rem, uint64_t left, uint64_t pos) -> void {
            if (left == 0) {
                if (rem != 0) return;
                Step step = v.push(row.data(), index, 1);
                if (step == Step::Descend) v.leaf(index);
                else if (step == Step::TakeAll) v.take_all(index, 1);
                else v.skip_all(index, 1);
                v.pop();
                ++index;
                return;
            }
            uint64_t hi = std::min(maximum, rem);
            for (uint64_t x = hi + 1; x-- > 1;) {
                row[pos] = (Entry)x;
                self(self, rem - x, left - 1, pos + 1);
                row[pos] = 0;
            }
        };
        for (uint64_t t = top_begin; t < top_end; ++t) descend(descend, n, k ? k : t + 1, 0);
        return ok();
    }
    case Kind::AllMatrices:
    case Kind::Words: {
        auto per_row = pow_checked(p, n);
        if (!per_row.ok) return fail(per_row.error.status, per_row.error.message);
        std::vector<uint64_t> below(m);
        for (int64_t j = (int64_t)m - 1; j >= 0; --j) below[j] = (j + 1 < (int64_t)m) ? below[j + 1] * per_row.value : 1;
        auto descend = [&](auto &self, uint64_t j, uint64_t index, uint64_t d_begin, uint64_t d_end) -> void {
            std::vector<Entry> r(n, 0);
            uint64_t rem = d_begin;
            for (int64_t q = (int64_t)n - 1; q >= 0; --q) { r[q] = (Entry)(rem % p); rem /= p; }
            for (uint64_t d = d_begin; d < d_end; ++d) {
                Step step = v.push(r.data(), index, below[j]);
                if (step == Step::Descend) {
                    if (j + 1 == m) v.leaf(index);
                    else self(self, j + 1, index, 0, per_row.value);
                } else if (step == Step::TakeAll) {
                    v.take_all(index, below[j]);
                } else {
                    v.skip_all(index, below[j]);
                }
                v.pop();
                index += below[j];
                for (int64_t q = (int64_t)n - 1; q >= 0; --q) {
                    if (++r[q] < p) break;
                    r[q] = 0;
                }
            }
        };
        descend(descend, 0, top_begin * below[0], top_begin, top_end);
        return ok();
    }
    case Kind::Transform: {
        TransformVisitor tv(v, *data, prime(), child->cols());
        return child->enumerate(tv, top_begin, top_end);
    }
    case Kind::GroupElements: {
        auto g = group_elements();
        if (!g.ok) return fail(g.error.status, g.error.message);
        uint64_t nn = data->cols;
        for (uint64_t i = top_begin; i < top_end; ++i) {
            Step step = v.push(g.value->data() + i * nn, i, 1);
            if (step == Step::Descend) v.leaf(i);
            else if (step == Step::TakeAll) v.take_all(i, 1);
            else v.skip_all(i, 1);
            v.pop();
        }
        return ok();
    }
    case Kind::Stack: {
        /* Stacked rows are pushed first so the consumer reduces them once per enumerate call;
         * rank, span and rref do not depend on row order, and order-dependent operations are
         * explicit-only by contract. */
        uint64_t pushed = 0;
        Step step = Step::Descend;
        auto whole = child->size();
        if (!whole.ok) return fail(whole.error.status, whole.error.message);
        for (uint64_t r = 0; r < data->rows; ++r) {
            step = v.push(data->at(0) + r * data->cols, 0, whole.value);
            ++pushed;
            if (step != Step::Descend) break;
        }
        Status st = ok();
        if (step == Step::Descend) st = child->enumerate(v, top_begin, top_end);
        else {
            struct Counter : Visitor {
                uint64_t first = UINT64_MAX, count = 0;
                Step push(const Entry *, uint64_t, uint64_t) override { return Step::TakeAll; }
                void pop() override {}
                void leaf(uint64_t) override {}
                void take_all(uint64_t f, uint64_t n) override { first = std::min(first, f); count += n; }
                void skip_all(uint64_t, uint64_t) override {}
            } counter;
            st = child->enumerate(counter, top_begin, top_end);
            if (st.ok && counter.count) {
                if (step == Step::TakeAll) v.take_all(counter.first, counter.count);
                else v.skip_all(counter.first, counter.count);
            }
        }
        for (uint64_t r = 0; r < pushed; ++r) v.pop();
        return st;
    }
    }
    return fail(INTERNAL, "unknown family kind");
}

Result<std::shared_ptr<Family>> make_explicit(std::shared_ptr<Matrix> batch) {
    auto f = std::make_shared<Family>();
    f->kind = Family::Kind::Explicit;
    f->data = std::move(batch);
    f->p = f->data->p;
    return Result<std::shared_ptr<Family>>::success(f);
}

Result<std::shared_ptr<Family>> make_subsets(std::shared_ptr<Matrix> dictionary, uint64_t k) {
    /* The dictionary is a list of row vectors: either one rows x cols matrix or a batch of
     * 1 x cols vectors. Both are the same flat data; keep it as a batch of vectors. */
    if (dictionary->count == 1 && dictionary->rows != 1) {
        auto flat = std::make_shared<Matrix>(*dictionary);
        flat->count = dictionary->rows;
        flat->rows = 1;
        dictionary = flat;
    }
    if (dictionary->rows != 1)
        return Result<std::shared_ptr<Family>>::failure(INVALID, "subsets: the dictionary must be one rows x cols matrix or a batch of 1 x cols vectors");
    if (k == 0 || k > dictionary->count)
        return Result<std::shared_ptr<Family>>::failure(INVALID, "subsets: k must satisfy 1 <= k <= dictionary size");
    auto f = std::make_shared<Family>();
    f->kind = Family::Kind::Subsets;
    f->data = std::move(dictionary);
    f->p = f->data->p;
    f->k = k;
    auto sz = f->size();
    if (!sz.ok) return Result<std::shared_ptr<Family>>::failure(sz.error.status, sz.error.message);
    return Result<std::shared_ptr<Family>>::success(f);
}

Result<std::shared_ptr<Family>> make_grassmannian(uint64_t p, uint64_t n, uint64_t h) {
    if (p < 2 || p >= (1ULL << 32))
        return Result<std::shared_ptr<Family>>::failure(INVALID, "field size must satisfy 2 <= p < 2^32");
    if (h == 0 || h > n || n > 64)
        return Result<std::shared_ptr<Family>>::failure(INVALID, "grassmannian: need 1 <= h <= n <= 64");
    auto f = std::make_shared<Family>();
    f->kind = Family::Kind::Grassmannian;
    f->p = p;
    f->n = n;
    f->h = h;
    auto sz = f->size();
    if (!sz.ok) return Result<std::shared_ptr<Family>>::failure(sz.error.status, sz.error.message);
    return Result<std::shared_ptr<Family>>::success(f);
}

Result<std::shared_ptr<Family>> make_all_matrices(uint64_t p, uint64_t rows, uint64_t cols) {
    if (p < 2 || p >= (1ULL << 32))
        return Result<std::shared_ptr<Family>>::failure(INVALID, "field size must satisfy 2 <= p < 2^32");
    if (rows == 0 || cols == 0) return Result<std::shared_ptr<Family>>::failure(INVALID, "all_matrices: need rows, cols >= 1");
    auto f = std::make_shared<Family>();
    f->kind = Family::Kind::AllMatrices;
    f->p = p;
    f->m = rows;
    f->n = cols;
    auto sz = f->size();
    if (!sz.ok) return Result<std::shared_ptr<Family>>::failure(sz.error.status, sz.error.message);
    return Result<std::shared_ptr<Family>>::success(f);
}

Result<std::shared_ptr<Family>> make_transform(std::shared_ptr<Family> inner, std::shared_ptr<Matrix> c) {
    if (c->count != 1) return Result<std::shared_ptr<Family>>::failure(INVALID, "transform: C must be a single matrix");
    if (!is_prime(inner->prime()))
        return Result<std::shared_ptr<Family>>::failure(INVALID, "transform: matrix multiplication is defined over a prime field only");
    if (c->p != inner->prime()) return Result<std::shared_ptr<Family>>::failure(INVALID, "transform: prime mismatch");
    if (c->rows != inner->cols())
        return Result<std::shared_ptr<Family>>::failure(INVALID, "transform: C must have as many rows as the members have columns");
    auto f = std::make_shared<Family>();
    f->kind = Family::Kind::Transform;
    f->child = std::move(inner);
    f->data = std::move(c);
    return Result<std::shared_ptr<Family>>::success(f);
}

Result<std::shared_ptr<Family>> make_stack(std::shared_ptr<Family> inner, std::shared_ptr<Matrix> rows) {
    if (rows->count != 1) return Result<std::shared_ptr<Family>>::failure(INVALID, "stack: rows must be a single matrix");
    if (rows->p != inner->prime()) return Result<std::shared_ptr<Family>>::failure(INVALID, "stack: prime mismatch");
    if (rows->cols != inner->cols())
        return Result<std::shared_ptr<Family>>::failure(INVALID, "stack: rows must have as many columns as the members");
    auto f = std::make_shared<Family>();
    f->kind = Family::Kind::Stack;
    f->child = std::move(inner);
    f->data = std::move(rows);
    return Result<std::shared_ptr<Family>>::success(f);
}

Result<std::shared_ptr<Family>> make_subsets_of(std::shared_ptr<Family> inner, uint64_t k) {
    using R = Result<std::shared_ptr<Family>>;
    auto sz = inner->size();
    if (!sz.ok) return R::failure(sz.error.status, sz.error.message);
    if (sz.value > (1ULL << 22)) return R::failure(INVALID, "subsets_of: the inner family has more than 2^22 members");
    if (k == 0 || k > sz.value) return R::failure(INVALID, "subsets_of: k must satisfy 1 <= k <= inner family size");
    auto dict = std::make_shared<Matrix>();
    dict->p = inner->prime();
    dict->count = sz.value;
    dict->rows = 1;
    dict->cols = inner->rows() * inner->cols();
    dict->entries.resize(dict->count * dict->cols);
    Matrix tmp;
    for (uint64_t i = 0; i < sz.value; ++i) {
        auto st = inner->member_into(i, tmp);
        if (!st.ok) return R::failure(st.error.status, st.error.message);
        std::copy(tmp.entries.begin(), tmp.entries.end(), dict->entries.begin() + i * dict->cols);
    }
    auto f = std::make_shared<Family>();
    f->kind = Family::Kind::SubsetsOf;
    f->child = std::move(inner);
    f->data = std::move(dict);
    f->p = f->data->p;
    f->k = k;
    auto total = f->size();
    if (!total.ok) return R::failure(total.error.status, total.error.message);
    return R::success(f);
}

Result<std::shared_ptr<Family>> make_symmetric_matrices(uint64_t p, uint64_t n) {
    using R = Result<std::shared_ptr<Family>>;
    if (p < 2 || p >= (1ULL << 32)) return R::failure(INVALID, "field size must satisfy 2 <= p < 2^32");
    if (n == 0) return R::failure(INVALID, "symmetric_matrices: need n >= 1");
    auto f = std::make_shared<Family>();
    f->kind = Family::Kind::SymmetricMatrices;
    f->p = p;
    f->n = n;
    auto sz = f->size();
    if (!sz.ok) return R::failure(sz.error.status, sz.error.message);
    return R::success(f);
}

Result<std::shared_ptr<Family>> make_range(uint64_t a, uint64_t b) {
    using R = Result<std::shared_ptr<Family>>;
    if (a >= b) return R::failure(INVALID, "range: need a < b");
    if (b > (1ULL << 32)) return R::failure(INVALID, "range: values must be < 2^32");
    auto f = std::make_shared<Family>();
    f->kind = Family::Kind::Range;
    f->a = a;
    f->b = b;
    return R::success(f);
}

Result<std::shared_ptr<Family>> make_words(uint64_t alphabet, uint64_t length) {
    using R = Result<std::shared_ptr<Family>>;
    if (alphabet < 2 || alphabet >= (1ULL << 32)) return R::failure(INVALID, "words: need 2 <= alphabet < 2^32");
    if (length == 0) return R::failure(INVALID, "words: need length >= 1");
    auto f = std::make_shared<Family>();
    f->kind = Family::Kind::Words;
    f->p = alphabet;
    f->m = 1;
    f->n = length;
    auto sz = f->size();
    if (!sz.ok) return R::failure(sz.error.status, sz.error.message);
    return R::success(f);
}

Result<std::shared_ptr<Family>> make_partitions(uint64_t total, uint64_t max_part, uint64_t max_parts,
                                                uint64_t max_multiplicity, uint64_t distinct, uint64_t odd) {
    using R = Result<std::shared_ptr<Family>>;
    if (total == 0 || total > 4096) return R::failure(INVALID, "partitions: need 1 <= total <= 4096");
    if (max_part > UINT32_MAX) return R::failure(INVALID, "partitions: max_part must be < 2^32");
    if (max_parts > total) return R::failure(INVALID, "partitions: max_parts must be at most total");
    if (max_multiplicity > total) return R::failure(INVALID, "partitions: max_multiplicity must be at most total");
    if (distinct > 1 || odd > 1) return R::failure(INVALID, "partitions: distinct and odd must be 0 or 1");
    auto f = std::make_shared<Family>();
    f->kind = Family::Kind::Partitions;
    f->n = total;
    f->m = max_part;
    f->k = max_parts;
    f->h = max_multiplicity;
    f->a = distinct;
    f->b = odd;
    auto sz = f->size();
    if (!sz.ok) return R::failure(sz.error.status, sz.error.message);
    if (sz.value == 0) return R::failure(INVALID, "partitions: constraints admit no partition");
    return R::success(f);
}

Result<std::shared_ptr<Family>> make_compositions(uint64_t total, uint64_t parts, uint64_t max_part) {
    using R = Result<std::shared_ptr<Family>>;
    if (total == 0 || total > 4096) return R::failure(INVALID, "compositions: need 1 <= total <= 4096");
    if (parts > total) return R::failure(INVALID, "compositions: parts must be at most total");
    if (max_part > UINT32_MAX) return R::failure(INVALID, "compositions: max_part must be < 2^32");
    auto f = std::make_shared<Family>();
    f->kind = Family::Kind::Compositions;
    f->n = total;
    f->k = parts;
    f->m = max_part;
    auto sz = f->size();
    if (!sz.ok) return R::failure(sz.error.status, sz.error.message);
    if (sz.value == 0) return R::failure(INVALID, "compositions: constraints admit no composition");
    return R::success(f);
}

Result<std::vector<Entry>> permutation_closure(const Matrix &generators, uint64_t limit) {
    using R = Result<std::vector<Entry>>;
    uint64_t n = generators.cols;
    if (generators.p != 0 || generators.rows != 1) return R::failure(INVALID, "generators must be a batch of permutations");
    for (uint64_t g = 0; g < generators.count; ++g) {
        std::vector<bool> seen(n, false);
        for (uint64_t i = 0; i < n; ++i) {
            Entry e = generators.entries[g * n + i];
            if (e >= n || seen[e]) return R::failure(INVALID, "generator " + std::to_string(g) + " is not a permutation of 0.." + std::to_string(n - 1));
            seen[e] = true;
        }
    }
    std::vector<Entry> store;
    struct Hash {
        const std::vector<Entry> *store; uint64_t n;
        size_t operator()(uint64_t i) const {
            uint64_t h = 1469598103934665603ULL;
            for (uint64_t j = 0; j < n; ++j) { h ^= (*store)[i * n + j]; h *= 1099511628211ULL; }
            return (size_t)h;
        }
    };
    struct Eq {
        const std::vector<Entry> *store; uint64_t n;
        bool operator()(uint64_t a, uint64_t b) const { return std::equal(store->begin() + a * n, store->begin() + (a + 1) * n, store->begin() + b * n); }
    };
    std::unordered_set<uint64_t, Hash, Eq> seen(64, Hash{&store, n}, Eq{&store, n});
    for (uint64_t i = 0; i < n; ++i) store.push_back((Entry)i); /* identity */
    seen.insert(0);
    std::vector<Entry> tmp(n);
    for (uint64_t front = 0; front < store.size() / n; ++front) {
        for (uint64_t g = 0; g < generators.count; ++g) {
            const Entry *gen = generators.entries.data() + g * n;
            for (uint64_t i = 0; i < n; ++i) tmp[i] = gen[store[front * n + i]]; /* x |-> gen(front(x)) */
            uint64_t idx = store.size() / n;
            store.insert(store.end(), tmp.begin(), tmp.end());
            if (!seen.insert(idx).second) store.resize(idx * n);
            else if (idx + 1 > limit) return R::failure(INVALID, "group has more than " + std::to_string(limit) + " elements");
        }
    }
    uint64_t count = store.size() / n;
    std::vector<uint64_t> order(count);
    for (uint64_t i = 0; i < count; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](uint64_t a, uint64_t b) {
        return std::lexicographical_compare(store.begin() + a * n, store.begin() + (a + 1) * n, store.begin() + b * n, store.begin() + (b + 1) * n);
    });
    std::vector<Entry> sorted(store.size());
    for (uint64_t i = 0; i < count; ++i) std::copy(store.begin() + order[i] * n, store.begin() + (order[i] + 1) * n, sorted.begin() + i * n);
    return R::success(std::move(sorted));
}

Result<std::shared_ptr<Family>> make_group_elements(std::shared_ptr<Matrix> generators) {
    if (generators->p != 0 || generators->rows != 1)
        return Result<std::shared_ptr<Family>>::failure(INVALID, "group_elements: generators must be an orbits.perms batch");
    if (generators->count == 0) return Result<std::shared_ptr<Family>>::failure(INVALID, "group_elements: need at least one generator");
    auto f = std::make_shared<Family>();
    f->kind = Family::Kind::GroupElements;
    f->data = std::move(generators);
    f->p = 0;
    f->n = f->data->cols;
    return Result<std::shared_ptr<Family>>::success(f);
}

} // namespace lk
