#include "family.hpp"

#include <algorithm>

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

/* Pivot sets of a Grassmannian in lexicographic order with the leaf count under each. */
struct PivotTable {
    uint64_t n, h, p;
    std::vector<std::vector<uint32_t>> sets;
    std::vector<uint64_t> offsets; /* size sets.size()+1 */
    std::vector<std::vector<uint32_t>> free_counts; /* per set, per row */
};

Result<PivotTable> pivot_table(uint64_t p, uint64_t n, uint64_t h) {
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
        auto t = pivot_table(p, n, h);
        if (!t.ok) return Result<uint64_t>::failure(t.error.status, t.error.message);
        return Result<uint64_t>::success(t.value.offsets.back());
    }
    case Kind::AllMatrices: return pow_checked(p, m * n);
    case Kind::Transform:
    case Kind::Stack: return child->size();
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
        auto t = pivot_table(p, n, h);
        if (!t.ok) return Result<Matrix>::failure(t.error.status, t.error.message);
        auto it = std::upper_bound(t.value.offsets.begin(), t.value.offsets.end(), index);
        uint64_t s = (uint64_t)(it - t.value.offsets.begin()) - 1;
        uint64_t rem = index - t.value.offsets[s];
        const auto &piv = t.value.sets[s];
        const auto &fr = t.value.free_counts[s];
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
    }
    return Result<Matrix>::success(std::move(out));
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
        auto t = pivot_table(p, n, h);
        if (!t.ok) return fail(t.error.status, t.error.message);
        std::vector<Entry> row(n);
        for (uint64_t s = top_begin; s < top_end; ++s) {
            const auto &piv = t.value.sets[s];
            const auto &fr = t.value.free_counts[s];
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
            descend(descend, 0, t.value.offsets[s]);
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

} // namespace lk
