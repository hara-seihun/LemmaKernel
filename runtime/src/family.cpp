#include "family.hpp"

#include <algorithm>
#include <mutex>
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
    Step push(const Entry *row) override {
        std::vector<uint64_t> acc(c.cols, 0);
        const Entry *cd = c.entries.data();
        for (uint64_t i = 0; i < in_cols; ++i) {
            uint64_t r = row[i];
            if (!r) continue;
            const Entry *crow = cd + i * c.cols;
            for (uint64_t j = 0; j < c.cols; ++j) acc[j] = (acc[j] + r * crow[j]) % p;
        }
        for (uint64_t j = 0; j < c.cols; ++j) buf[j] = (Entry)acc[j];
        return inner.push(buf.data());
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
    }
    return "?";
}

uint64_t Family::prime() const { return child ? child->prime() : p; }

uint64_t Family::rows() const {
    switch (kind) {
    case Kind::Explicit: return data->rows;
    case Kind::Subsets: return k;
    case Kind::Grassmannian: return h;
    case Kind::AllMatrices: return m;
    case Kind::Transform: return child->rows();
    case Kind::Stack: return child->rows() + data->rows;
    case Kind::GroupElements: return 1;
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
    }
    return 0;
}

bool Family::is_explicit() const {
    return kind == Kind::Explicit;
}

Result<uint64_t> Family::size() const {
    switch (kind) {
    case Kind::Explicit: return Result<uint64_t>::success(data->count);
    case Kind::Subsets: return binom(data->count, k);
    case Kind::Grassmannian: {
        auto t = pivot_table();
        if (!t.ok) return Result<uint64_t>::failure(t.error.status, t.error.message);
        return Result<uint64_t>::success(t.value->offsets.back());
    }
    case Kind::AllMatrices: return pow_checked(p, m * n);
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
    case Kind::Subsets: return Result<uint64_t>::success(data->count - k + 1);
    case Kind::Grassmannian: return binom(n, h);
    case Kind::AllMatrices: return pow_checked(p, n);
    case Kind::Transform:
    case Kind::Stack: return child->top_count();
    case Kind::GroupElements: return size();
    }
    return Result<uint64_t>::failure(INTERNAL, "unknown family kind");
}

Result<Matrix> Family::member(uint64_t index) const {
    auto sz = size();
    if (!sz.ok) return Result<Matrix>::failure(sz.error.status, sz.error.message);
    if (index >= sz.value) return Result<Matrix>::failure(INVALID, "member index out of range");
    Matrix out;
    out.p = prime();
    out.count = 1;
    out.rows = rows();
    out.cols = cols();
    switch (kind) {
    case Kind::Explicit:
        out.entries.assign(data->at(index), data->at(index) + out.rows * out.cols);
        break;
    case Kind::Subsets: {
        uint64_t D = data->count, remaining = index, prev = 0;
        out.entries.resize(k * out.cols);
        for (uint64_t j = 0; j < k; ++j) {
            uint64_t c = prev;
            for (;; ++c) {
                auto below = binom(D - 1 - c, k - 1 - j);
                if (!below.ok) return Result<Matrix>::failure(below.error.status, below.error.message);
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
        if (!tr.ok) return Result<Matrix>::failure(tr.error.status, tr.error.message);
        const PivotTable &t = *tr.value;
        auto it = std::upper_bound(t.offsets.begin(), t.offsets.end(), index);
        uint64_t s = (uint64_t)(it - t.offsets.begin()) - 1;
        uint64_t rem = index - t.offsets[s];
        const auto &piv = t.sets[s];
        const auto &fr = t.free_counts[s];
        out.entries.assign(h * n, 0);
        std::vector<uint64_t> radix(h);
        for (uint64_t j = 0; j < h; ++j) radix[j] = pow_checked(p, fr[j]).value;
        for (uint64_t j = 0; j < h; ++j) {
            uint64_t below = 1;
            for (uint64_t i = j + 1; i < h; ++i) below *= radix[i];
            uint64_t digits = rem / below;
            rem %= below;
            Entry *row = out.entries.data() + j * n;
            row[piv[j]] = 1;
            std::vector<uint64_t> cols_free;
            for (uint64_t col = piv[j] + 1; col < n; ++col)
                if (!std::binary_search(piv.begin(), piv.end(), (uint32_t)col)) cols_free.push_back(col);
            for (int64_t q = (int64_t)cols_free.size() - 1; q >= 0; --q) {
                row[cols_free[q]] = (Entry)(digits % p);
                digits /= p;
            }
        }
        break;
    }
    case Kind::AllMatrices: {
        out.entries.assign(m * n, 0);
        uint64_t rem = index;
        for (int64_t q = (int64_t)(m * n) - 1; q >= 0; --q) {
            out.entries[q] = (Entry)(rem % p);
            rem /= p;
        }
        break;
    }
    case Kind::Transform: {
        auto inner = child->member(index);
        if (!inner.ok) return inner;
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
        if (!inner.ok) return inner;
        out.entries = inner.value.entries;
        out.entries.insert(out.entries.end(), data->entries.begin(), data->entries.end());
        break;
    }
    case Kind::GroupElements: {
        auto g = group_elements();
        if (!g.ok) return Result<Matrix>::failure(g.error.status, g.error.message);
        out.entries.assign(g.value->begin() + index * out.cols, g.value->begin() + (index + 1) * out.cols);
        break;
    }
    }
    return Result<Matrix>::success(std::move(out));
}

Result<const PivotTable *> Family::pivot_table() const {
    using R = Result<const PivotTable *>;
    if (kind != Kind::Grassmannian) return R::failure(INVALID, "not a grassmannian family");
    static std::mutex mu;
    std::lock_guard<std::mutex> lock(mu);
    if (!pivots) {
        auto t = build_pivot_table(p, n, h);
        if (!t.ok) return R::failure(t.error.status, t.error.message);
        pivots = std::make_shared<const PivotTable>(std::move(t.value));
    }
    return R::success(pivots.get());
}

Result<const std::vector<Entry> *> Family::group_elements() const {
    using R = Result<const std::vector<Entry> *>;
    if (kind != Kind::GroupElements) return R::failure(INVALID, "not a group_elements family");
    static std::mutex mu;
    std::lock_guard<std::mutex> lock(mu);
    if (!elements) {
        auto c = permutation_closure(*data, 1ULL << 26);
        if (!c.ok) return R::failure(c.error.status, c.error.message);
        elements = std::make_shared<const std::vector<Entry>>(std::move(c.value));
    }
    return R::success(elements.get());
}

Result<uint64_t> Family::index_of(const Matrix &mem) const {
    using R = Result<uint64_t>;
    if (mem.count != 1 || mem.rows != rows() || mem.cols != cols() || mem.p != prime())
        return R::failure(INVALID, "index_of: member has the wrong shape for this family");
    switch (kind) {
    case Kind::Subsets: {
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
        std::vector<uint32_t> piv(h);
        for (uint64_t j = 0; j < h; ++j) {
            const Entry *row = mem.entries.data() + j * n;
            uint64_t lead = n;
            for (uint64_t c = 0; c < n; ++c) if (row[c]) { lead = c; break; }
            if (lead == n || row[lead] != 1 || (j > 0 && lead <= piv[j - 1]))
                return R::failure(INVALID, "index_of: member is not in reduced row echelon form");
            piv[j] = (uint32_t)lead;
        }
        auto it = std::lower_bound(t.sets.begin(), t.sets.end(), piv);
        if (it == t.sets.end() || *it != piv) return R::failure(INTERNAL, "index_of: pivot set not found");
        uint64_t s = (uint64_t)(it - t.sets.begin());
        uint64_t index = t.offsets[s];
        uint64_t rem = 0;
        for (uint64_t j = 0; j < h; ++j) {
            const Entry *row = mem.entries.data() + j * n;
            for (uint64_t c = 0; c < n; ++c) {
                bool is_piv = std::binary_search(piv.begin(), piv.end(), (uint32_t)c);
                if (is_piv || c < piv[j]) {
                    if (row[c] != (c == piv[j] ? 1u : 0u)) return R::failure(INVALID, "index_of: member is not in reduced row echelon form");
                } else rem = rem * p + row[c];
            }
        }
        return R::success(index + rem);
    }
    case Kind::AllMatrices: {
        uint64_t index = 0;
        for (Entry e : mem.entries) index = index * p + e;
        return R::success(index);
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
                step = v.push(base + r * data->cols);
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
    case Kind::Subsets: {
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
                Step step = v.push(data->at(0) + c * cols);
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
                    Step step = v.push(r.data());
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
    case Kind::AllMatrices: {
        auto per_row = pow_checked(p, n);
        if (!per_row.ok) return fail(per_row.error.status, per_row.error.message);
        std::vector<uint64_t> below(m);
        for (int64_t j = (int64_t)m - 1; j >= 0; --j) below[j] = (j + 1 < (int64_t)m) ? below[j + 1] * per_row.value : 1;
        auto descend = [&](auto &self, uint64_t j, uint64_t index, uint64_t d_begin, uint64_t d_end) -> void {
            std::vector<Entry> r(n, 0);
            uint64_t rem = d_begin;
            for (int64_t q = (int64_t)n - 1; q >= 0; --q) { r[q] = (Entry)(rem % p); rem /= p; }
            for (uint64_t d = d_begin; d < d_end; ++d) {
                Step step = v.push(r.data());
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
            Step step = v.push(g.value->data() + i * nn);
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
        for (uint64_t r = 0; r < data->rows; ++r) {
            step = v.push(data->at(0) + r * data->cols);
            ++pushed;
            if (step != Step::Descend) break;
        }
        Status st = ok();
        if (step == Step::Descend) st = child->enumerate(v, top_begin, top_end);
        else {
            struct Counter : Visitor {
                uint64_t first = UINT64_MAX, count = 0;
                Step push(const Entry *) override { return Step::TakeAll; }
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
    if (!is_prime(p)) return Result<std::shared_ptr<Family>>::failure(INVALID, "p is not prime");
    if (p >= (1ULL << 32)) return Result<std::shared_ptr<Family>>::failure(INVALID, "p must be < 2^32");
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
    if (!is_prime(p)) return Result<std::shared_ptr<Family>>::failure(INVALID, "p is not prime");
    if (p >= (1ULL << 32)) return Result<std::shared_ptr<Family>>::failure(INVALID, "p must be < 2^32");
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
