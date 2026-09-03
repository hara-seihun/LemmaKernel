#include "../../../gfp/backends/generic/field.hpp"

#include <optional>

namespace lk::char_poly {
namespace {

using gfp::Field;
using Poly = std::vector<Entry>;
using Mat = std::vector<Entry>;
using R = Result<std::shared_ptr<Object>>;
constexpr int INVALID = 1;
constexpr int INTERNAL = 4;
constexpr uint64_t EXPENSIVE_LIMIT = 1ULL << 20;

void trim(Poly &f) {
    while (!f.empty() && f.back() == 0) f.pop_back();
}

Entry coefficient(const Poly &f, size_t i) { return i < f.size() ? f[i] : 0; }

Poly poly_add(const Field &field, const Poly &a, const Poly &b) {
    Poly out(std::max(a.size(), b.size()));
    for (size_t i = 0; i < out.size(); ++i) out[i] = (Entry)field.reduce((uint64_t)coefficient(a, i) + coefficient(b, i));
    trim(out);
    return out;
}

Poly poly_neg(const Field &field, const Poly &a) {
    Poly out(a.size());
    for (size_t i = 0; i < a.size(); ++i) out[i] = a[i] ? (Entry)(field.p - a[i]) : 0;
    return out;
}

Poly poly_sub(const Field &field, const Poly &a, const Poly &b) { return poly_add(field, a, poly_neg(field, b)); }

Poly poly_mul(const Field &field, const Poly &a, const Poly &b) {
    if (a.empty() || b.empty()) return {};
    Poly out(a.size() + b.size() - 1, 0);
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < b.size(); ++j)
            out[i + j] = (Entry)field.reduce((uint64_t)out[i + j] + (uint64_t)a[i] * b[j]);
    trim(out);
    return out;
}

Poly poly_scale(const Field &field, const Poly &a, Entry c) {
    Poly out = a;
    field.scale(out.data(), c, out.size());
    trim(out);
    return out;
}

Poly poly_power(const Field &field, const Poly &a, uint64_t exponent) {
    Poly out{1}, base = a;
    while (exponent) {
        if (exponent & 1) out = poly_mul(field, out, base);
        exponent >>= 1;
        if (exponent) base = poly_mul(field, base, base);
    }
    return out;
}

Poly monic(const Field &field, Poly f) {
    trim(f);
    if (!f.empty() && f.back() != 1) f = poly_scale(field, f, field.inverse(f.back()));
    return f;
}

std::pair<Poly, Poly> poly_divmod(const Field &field, Poly dividend, Poly divisor) {
    divisor = monic(field, std::move(divisor));
    trim(dividend);
    Poly quotient;
    while (!divisor.empty() && dividend.size() >= divisor.size()) {
        size_t shift = dividend.size() - divisor.size();
        Poly term(shift + 1, 0);
        term.back() = dividend.back();
        quotient = poly_add(field, quotient, term);
        dividend = poly_sub(field, dividend, poly_mul(field, term, divisor));
    }
    return {std::move(quotient), std::move(dividend)};
}

Poly derivative(const Field &field, const Poly &f) {
    if (f.size() < 2) return {};
    Poly out(f.size() - 1);
    for (size_t i = 1; i < f.size(); ++i) out[i - 1] = (Entry)((unsigned __int128)i * f[i] % field.p);
    trim(out);
    return out;
}

Poly poly_gcd(const Field &field, Poly a, Poly b) {
    trim(a);
    trim(b);
    while (!b.empty()) {
        Poly r = poly_divmod(field, a, b).second;
        a = std::move(b);
        b = std::move(r);
    }
    return monic(field, std::move(a));
}

Mat identity(uint64_t n) {
    Mat out(n * n, 0);
    for (uint64_t i = 0; i < n; ++i) out[i * n + i] = 1;
    return out;
}

Mat matrix_mul(const Field &field, const Mat &a, const Mat &b, uint64_t n) {
    Mat out(n * n, 0);
    for (uint64_t i = 0; i < n; ++i)
        for (uint64_t k = 0; k < n; ++k) {
            Entry x = a[i * n + k];
            if (!x) continue;
            for (uint64_t j = 0; j < n; ++j)
                out[i * n + j] = (Entry)field.reduce((uint64_t)out[i * n + j] + (uint64_t)x * b[k * n + j]);
        }
    return out;
}

uint64_t matrix_rank(const Field &field, Mat a, uint64_t rows, uint64_t cols) {
    uint64_t rank = 0;
    for (uint64_t c = 0; c < cols && rank < rows; ++c) {
        uint64_t pivot = rank;
        while (pivot < rows && a[pivot * cols + c] == 0) ++pivot;
        if (pivot == rows) continue;
        if (pivot != rank)
            std::swap_ranges(a.begin() + pivot * cols, a.begin() + (pivot + 1) * cols, a.begin() + rank * cols);
        Entry *prow = a.data() + rank * cols;
        if (prow[c] != 1) field.scale(prow, field.inverse(prow[c]), cols);
        for (uint64_t i = rank + 1; i < rows; ++i) {
            Entry factor = a[i * cols + c];
            if (factor) field.subtract_multiple(a.data() + i * cols, prow, factor, cols);
        }
        ++rank;
    }
    return rank;
}

/* Similarity-reduce to upper Hessenberg form, then use the leading-principal-minor recurrence. */
Poly characteristic_polynomial(const Field &field, const Entry *entries, uint64_t n) {
    Mat h(entries, entries + n * n);
    for (uint64_t k = 1; k < n; ++k) {
        uint64_t pivot = k;
        while (pivot < n && h[pivot * n + k - 1] == 0) ++pivot;
        if (pivot == n) continue;
        if (pivot != k) {
            for (uint64_t j = 0; j < n; ++j) std::swap(h[pivot * n + j], h[k * n + j]);
            for (uint64_t i = 0; i < n; ++i) std::swap(h[i * n + pivot], h[i * n + k]);
        }
        Entry inverse = field.inverse(h[k * n + k - 1]);
        for (uint64_t i = k + 1; i < n; ++i) {
            Entry q = (Entry)field.reduce((uint64_t)h[i * n + k - 1] * inverse);
            if (!q) continue;
            field.subtract_multiple(h.data() + i * n, h.data() + k * n, q, n);
            for (uint64_t r = 0; r < n; ++r)
                h[r * n + k] = (Entry)field.reduce((uint64_t)h[r * n + k] + (uint64_t)q * h[r * n + i]);
        }
    }
    std::vector<Poly> leading(n + 1);
    leading[0] = {1};
    for (uint64_t size = 1; size <= n; ++size) {
        Entry diagonal = h[(size - 1) * n + size - 1];
        Poly x_minus{diagonal ? (Entry)(field.p - diagonal) : 0, 1};
        Poly value = poly_mul(field, x_minus, leading[size - 1]);
        Entry product = 1;
        for (int64_t j = (int64_t)size - 2; j >= 0; --j) {
            product = (Entry)field.reduce((uint64_t)product * h[(j + 1) * n + j]);
            Entry scalar = (Entry)field.reduce((uint64_t)h[j * n + size - 1] * product);
            if (scalar) value = poly_sub(field, value, poly_scale(field, leading[j], scalar));
        }
        leading[size] = std::move(value);
    }
    return leading[n];
}

std::optional<Poly> relation_at(const Field &field, const std::vector<Mat> &powers, uint64_t n, uint64_t degree) {
    uint64_t equations = n * n, width = degree + 1;
    Mat augmented(equations * width);
    for (uint64_t row = 0; row < equations; ++row) {
        for (uint64_t col = 0; col < degree; ++col) augmented[row * width + col] = powers[col][row];
        Entry target = powers[degree][row];
        augmented[row * width + degree] = target ? (Entry)(field.p - target) : 0;
    }
    uint64_t rank = 0;
    std::vector<uint64_t> pivots;
    for (uint64_t col = 0; col < degree && rank < equations; ++col) {
        uint64_t pivot = rank;
        while (pivot < equations && augmented[pivot * width + col] == 0) ++pivot;
        if (pivot == equations) continue;
        if (pivot != rank)
            std::swap_ranges(augmented.begin() + pivot * width, augmented.begin() + (pivot + 1) * width,
                             augmented.begin() + rank * width);
        Entry *prow = augmented.data() + rank * width;
        if (prow[col] != 1) field.scale(prow, field.inverse(prow[col]), width);
        for (uint64_t row = 0; row < equations; ++row) {
            if (row == rank) continue;
            Entry factor = augmented[row * width + col];
            if (factor) field.subtract_multiple(augmented.data() + row * width, prow, factor, width);
        }
        pivots.push_back(col);
        ++rank;
    }
    for (uint64_t row = rank; row < equations; ++row) {
        bool zero = true;
        for (uint64_t col = 0; col < degree; ++col) zero &= augmented[row * width + col] == 0;
        if (zero && augmented[row * width + degree]) return std::nullopt;
    }
    Poly relation(degree + 1, 0);
    relation[degree] = 1;
    for (uint64_t row = 0; row < pivots.size(); ++row) relation[pivots[row]] = augmented[row * width + degree];
    return relation;
}

Poly minimal_polynomial(const Field &field, const Entry *entries, uint64_t n) {
    Mat a(entries, entries + n * n);
    std::vector<Mat> powers{identity(n)};
    for (uint64_t d = 1; d <= n; ++d) {
        powers.push_back(matrix_mul(field, powers.back(), a, n));
        auto relation = relation_at(field, powers, n, d);
        if (relation) return *relation;
    }
    return {};
}

void factor_polynomial(const Field &field, const Poly &f, std::vector<Poly> &out) {
    uint64_t degree = f.size() - 1;
    for (uint64_t d = 1; d <= degree / 2; ++d) {
        uint64_t count = 1;
        for (uint64_t i = 0; i < d; ++i) count *= field.p;
        for (uint64_t index = 0; index < count; ++index) {
            Poly candidate(d + 1, 0);
            candidate[d] = 1;
            uint64_t rem = index;
            for (int64_t i = (int64_t)d - 1; i >= 0; --i) { candidate[i] = (Entry)(rem % field.p); rem /= field.p; }
            auto qr = poly_divmod(field, f, candidate);
            if (qr.second.empty()) {
                factor_polynomial(field, candidate, out);
                factor_polynomial(field, qr.first, out);
                return;
            }
        }
    }
    out.push_back(monic(field, f));
}

Mat evaluate_at_matrix(const Field &field, const Mat &a, const Poly &f, uint64_t n) {
    Mat value(n * n, 0), scalar(n * n);
    for (auto it = f.rbegin(); it != f.rend(); ++it) {
        value = matrix_mul(field, value, a, n);
        for (uint64_t i = 0; i < n; ++i) value[i * n + i] = (Entry)field.reduce((uint64_t)value[i * n + i] + *it);
    }
    return value;
}

Mat matrix_power(const Field &field, const Mat &a, uint64_t n, uint64_t exponent) {
    Mat out = identity(n), base = a;
    while (exponent) {
        if (exponent & 1) out = matrix_mul(field, out, base, n);
        exponent >>= 1;
        if (exponent) base = matrix_mul(field, base, base, n);
    }
    return out;
}

std::vector<uint64_t> block_exponents(const Field &field, const Mat &a, const Poly &factor,
                                      uint64_t multiplicity, uint64_t n) {
    Mat evaluated = evaluate_at_matrix(field, a, factor, n);
    uint64_t degree = factor.size() - 1;
    auto at_least = [&](uint64_t k) {
        if (k == 0 || k > multiplicity) return uint64_t{0};
        uint64_t now = n - matrix_rank(field, matrix_power(field, evaluated, n, k), n, n);
        uint64_t before = n - matrix_rank(field, matrix_power(field, evaluated, n, k - 1), n, n);
        return (now - before) / degree;
    };
    std::vector<uint64_t> out;
    for (uint64_t k = 1; k <= multiplicity; ++k)
        for (uint64_t copies = at_least(k) - at_least(k + 1); copies; --copies) out.push_back(k);
    return out;
}

std::vector<Poly> invariant_factors(const Field &field, const Entry *entries, uint64_t n) {
    Mat a(entries, entries + n * n);
    std::vector<Poly> factors;
    factor_polynomial(field, characteristic_polynomial(field, entries, n), factors);
    std::sort(factors.begin(), factors.end());
    struct Primary { Poly factor; std::vector<uint64_t> exponents; };
    std::vector<Primary> primary;
    for (size_t i = 0; i < factors.size();) {
        size_t end = i + 1;
        while (end < factors.size() && factors[end] == factors[i]) ++end;
        primary.push_back({factors[i], block_exponents(field, a, factors[i], end - i, n)});
        i = end;
    }
    size_t count = 0;
    for (const auto &x : primary) count = std::max(count, x.exponents.size());
    std::vector<Poly> out(count, Poly{1});
    for (size_t j = 0; j < count; ++j)
        for (const auto &x : primary) {
            size_t offset = count - x.exponents.size();
            if (j >= offset) out[j] = poly_mul(field, out[j], poly_power(field, x.factor, x.exponents[j - offset]));
        }
    return out;
}

Mat companion(const Field &field, const Poly &f) {
    uint64_t n = f.size() - 1;
    Mat out(n * n, 0);
    for (uint64_t i = 0; i + 1 < n; ++i) out[i * n + i + 1] = 1;
    for (uint64_t j = 0; j < n; ++j) out[(n - 1) * n + j] = f[j] ? (Entry)(field.p - f[j]) : 0;
    return out;
}

Mat rational_form(const Field &field, const Entry *entries, uint64_t n) {
    auto factors = invariant_factors(field, entries, n);
    Mat out(n * n, 0);
    uint64_t offset = 0;
    for (const auto &factor : factors) {
        Mat block = companion(field, factor);
        uint64_t d = factor.size() - 1;
        for (uint64_t i = 0; i < d; ++i)
            std::copy(block.begin() + i * d, block.begin() + (i + 1) * d, out.begin() + (offset + i) * n + offset);
        offset += d;
    }
    return out;
}

Mat conjugacy_label(const Field &field, const Entry *entries, uint64_t n) {
    auto factors = invariant_factors(field, entries, n);
    Mat out(n * (n + 1), 0);
    for (uint64_t i = 0; i < factors.size(); ++i)
        std::copy(factors[i].begin(), factors[i].end(), out.begin() + i * (n + 1));
    return out;
}

uint64_t power_bound(uint64_t p, uint64_t n, uint64_t limit) {
    uint64_t value = 1;
    for (uint64_t i = 0; i < n; ++i) {
        if (value > limit / p) return limit + 1;
        value *= p;
    }
    return value;
}

uint64_t element_order(const Field &field, const Entry *entries, uint64_t n) {
    Mat a(entries, entries + n * n);
    if (matrix_rank(field, a, n, n) != n) return 0;
    Mat one = identity(n), power = one;
    uint64_t bound = power_bound(field.p, n, EXPENSIVE_LIMIT);
    for (uint64_t order = 1; order <= bound; ++order) {
        power = matrix_mul(field, power, a, n);
        if (power == one) return order;
    }
    return 0;
}

bool expensive(const std::string &op) {
    return op == "rational_canonical_form" || op == "conjugacy_label" || op == "element_order";
}

bool accepts(const Request &req) {
    uint64_t p = req.family->prime(), n = req.family->rows();
    if (!is_prime(p) || p >= (1ULL << 32) || n == 0) return false;
    return !expensive(req.op) || power_bound(p, n, EXPENSIVE_LIMIT) <= EXPENSIVE_LIMIT;
}

R run_values(const Request &req, uint64_t size, uint64_t n, const std::string &op) {
    uint64_t rows = op == "charpoly" || op == "minpoly" ? 1 : n;
    uint64_t cols = op == "charpoly" || op == "minpoly" || op == "conjugacy_label" ? n + 1 : n;
    auto result = std::make_shared<Matrix>();
    result->p = req.family->prime(); result->count = size; result->rows = rows; result->cols = cols;
    result->entries.assign(size * rows * cols, 0);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, size ? size : 1));
    auto statuses = parallel_ranges(size, threads, [&](uint32_t, uint64_t begin, uint64_t end) -> Status {
        Field field(req.family->prime());
        Matrix member;
        for (uint64_t i = begin; i < end; ++i) {
            auto st = req.family->member_into(i, member);
            if (!st.ok) return st;
            if (op == "charpoly" || op == "minpoly") {
                Poly f = op == "charpoly" ? characteristic_polynomial(field, member.entries.data(), n)
                                           : minimal_polynomial(field, member.entries.data(), n);
                std::copy(f.begin(), f.end(), result->entries.begin() + i * (n + 1));
            } else {
                Mat value = op == "rational_canonical_form" ? rational_form(field, member.entries.data(), n)
                                                             : conjugacy_label(field, member.entries.data(), n);
                std::copy(value.begin(), value.end(), result->entries.begin() + i * rows * cols);
            }
        }
        return ok();
    });
    for (const auto &st : statuses) if (!st.ok) return R::failure(st.error.status, st.error.message);
    auto object = std::make_shared<Object>();
    object->kind = "gfp.matrix";
    object->matrix = result;
    return R::success(object);
}

R run_reduced(const Request &req, uint64_t size, uint64_t n) {
    Reduction reduction = parse_reduction(req.reduction);
    Shared shared;
    auto prepared = prepare_all(reduction, size, shared);
    if (!prepared.ok) return R::failure(prepared.error.status, prepared.error.message);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, size ? size : 1));
    std::vector<Accumulator> accumulators;
    for (uint32_t i = 0; i < threads; ++i) accumulators.emplace_back(reduction, &shared);
    auto statuses = parallel_ranges(size, threads, [&](uint32_t thread, uint64_t begin, uint64_t end) -> Status {
        Field field(req.family->prime());
        Matrix member;
        for (uint64_t i = begin; i < end; ++i) {
            if (accumulators[thread].exhausted(i)) break;
            auto st = req.family->member_into(i, member);
            if (!st.ok) return st;
            Poly minimal = minimal_polynomial(field, member.entries.data(), n);
            if (req.op == "is_regular") accumulators[thread].boolean(i, minimal.size() == n + 1);
            else if (req.op == "is_semisimple")
                accumulators[thread].boolean(i, poly_gcd(field, minimal, derivative(field, minimal)) == Poly{1});
            else accumulators[thread].integer(i, element_order(field, member.entries.data(), n));
        }
        return ok();
    });
    for (const auto &st : statuses) if (!st.ok) return R::failure(st.error.status, st.error.message);
    return assemble(req, reduction, accumulators, shared);
}

R run(const Request &req) {
    if (req.family->rows() != req.family->cols()) return R::failure(INVALID, "char_poly operations need square matrices");
    auto size = req.family->size();
    if (!size.ok) return R::failure(size.error.status, size.error.message);
    uint64_t n = req.family->rows();
    if (req.op == "charpoly" || req.op == "minpoly" || req.op == "rational_canonical_form" || req.op == "conjugacy_label")
        return run_values(req, size.value, n, req.op);
    if (req.op == "is_regular" || req.op == "is_semisimple" || req.op == "element_order")
        return run_reduced(req, size.value, n);
    return R::failure(INTERNAL, "unknown char_poly operation " + req.op);
}

BackendRegistration registration{Backend{
    "char_poly", "generic", [] { return true; }, accepts, run, 0}};

} // namespace
} // namespace lk::char_poly
