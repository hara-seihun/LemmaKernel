/* Polynomial-basis linear algebra over GF(q), q < 2^32. */
#include "../../../gfp/backends/walk.hpp"

#include <array>

namespace lk::gfq {
namespace {

constexpr int INVALID = 1;
constexpr int INTERNAL = 4;
using R = Result<std::shared_ptr<Object>>;

uint64_t smallest_divisor(uint64_t q) {
    if ((q & 1) == 0) return 2;
    for (uint64_t p = 3; p <= q / p; p += 2)
        if (q % p == 0) return p;
    return q;
}

struct Field {
    uint64_t q = 0, p = 0, d = 0;
    std::vector<Entry> modulus;

    Entry add(Entry a, Entry b) const {
        uint64_t out = 0, place = 1;
        for (uint64_t i = 0; i < d; ++i) {
            out += ((a % p + b % p) % p) * place;
            a = (Entry)(a / p);
            b = (Entry)(b / p);
            place *= p;
        }
        return (Entry)out;
    }

    Entry negate(Entry a) const {
        uint64_t out = 0, place = 1;
        for (uint64_t i = 0; i < d; ++i) {
            out += ((p - a % p) % p) * place;
            a = (Entry)(a / p);
            place *= p;
        }
        return (Entry)out;
    }

    Entry multiply(Entry a, Entry b) const {
        std::array<uint64_t, 64> aa{}, bb{}, coeff{};
        for (uint64_t i = 0; i < d; ++i) {
            aa[i] = a % p;
            bb[i] = b % p;
            a = (Entry)(a / p);
            b = (Entry)(b / p);
        }
        for (uint64_t i = 0; i < d; ++i)
            for (uint64_t j = 0; j < d; ++j)
                coeff[i + j] = (coeff[i + j] + aa[i] * bb[j]) % p;
        for (int64_t k = (int64_t)(2 * d - 2); k >= (int64_t)d; --k) {
            uint64_t lead = coeff[k];
            for (uint64_t j = 0; j < d; ++j)
                coeff[k - d + j] = (coeff[k - d + j] + p - lead * modulus[j] % p) % p;
        }
        uint64_t out = 0;
        for (int64_t i = (int64_t)d - 1; i >= 0; --i) out = out * p + coeff[i];
        return (Entry)out;
    }

    Entry power(Entry a, uint64_t e) const {
        Entry out = 1;
        while (e) {
            if (e & 1) out = multiply(out, a);
            a = multiply(a, a);
            e >>= 1;
        }
        return out;
    }

    Entry inverse(Entry a) const { return power(a, q - 2); }

    void subtract_multiple(Entry *row, const Entry *other, Entry c, uint64_t cols) const {
        Entry nc = negate(c);
        for (uint64_t j = 0; j < cols; ++j) row[j] = add(row[j], multiply(nc, other[j]));
    }

    void scale(Entry *row, Entry c, uint64_t cols) const {
        for (uint64_t j = 0; j < cols; ++j) row[j] = multiply(row[j], c);
    }
};

std::vector<Entry> polynomial_remainder(std::vector<Entry> f, const std::vector<Entry> &g, uint64_t p) {
    uint64_t degree = g.size() - 1;
    for (int64_t k = (int64_t)f.size() - 1; k >= (int64_t)degree; --k) {
        uint64_t lead = f[k];
        for (uint64_t j = 0; j < degree; ++j)
            f[k - degree + j] = (Entry)((f[k - degree + j] + p - lead * g[j] % p) % p);
        f[k] = 0;
    }
    f.resize(degree);
    return f;
}

bool is_irreducible(const Field &field) {
    for (uint64_t degree = 1; degree <= field.d / 2; ++degree) {
        uint64_t count = 1;
        for (uint64_t i = 0; i < degree; ++i) count *= field.p;
        for (uint64_t code = 0; code < count; ++code) {
            std::vector<Entry> divisor(degree + 1, 0);
            uint64_t x = code;
            for (uint64_t i = 0; i < degree; ++i) {
                divisor[i] = (Entry)(x % field.p);
                x /= field.p;
            }
            divisor[degree] = 1;
            auto rem = polynomial_remainder(field.modulus, divisor, field.p);
            if (std::all_of(rem.begin(), rem.end(), [](Entry a) { return a == 0; })) return false;
        }
    }
    return true;
}

Result<Field> parse_field(const Request &req) {
    auto it = req.handle_args.find("modulus");
    if (it == req.handle_args.end() || !it->second->matrix)
        return Result<Field>::failure(INVALID, "`modulus` must be a field polynomial vector");
    const Matrix &m = *it->second->matrix;
    uint64_t q = req.family->prime();
    if (m.p != q || m.count != 1 || m.rows != 1 || m.cols < 2)
        return Result<Field>::failure(INVALID, "modulus must be one vector tagged with the same q as the family");
    Field field;
    field.q = q;
    field.p = smallest_divisor(q);
    field.d = m.cols - 1;
    field.modulus = m.entries;
    if (!is_prime(field.p)) return Result<Field>::failure(INVALID, "the base p is not prime");
    uint64_t power = 1;
    for (uint64_t i = 0; i < field.d; ++i) {
        if (power > UINT64_MAX / field.p) {
            power = UINT64_MAX;
            break;
        }
        power *= field.p;
    }
    if (power != q) return Result<Field>::failure(INVALID, "q must equal p^d for the modulus degree d");
    if (field.d > 31) return Result<Field>::failure(INVALID, "extension degree exceeds the q < 2^32 representation");
    if (field.modulus.back() != 1 || std::any_of(field.modulus.begin(), field.modulus.end(), [&](Entry a) { return a >= field.p; }))
        return Result<Field>::failure(INVALID, "modulus must be monic with coefficients in F_p");
    if (!is_irreducible(field)) return Result<Field>::failure(INVALID, "modulus must be irreducible over F_p");
    return Result<Field>::success(std::move(field));
}

struct EchelonBasis {
    Field field;
    uint64_t cols;
    std::vector<Entry> rows;
    std::vector<uint32_t> pivots;
    std::vector<Entry> scratch;
    mutable std::vector<uint64_t> order_buf;
    mutable std::vector<Entry> sorted_buf;
    mutable std::vector<uint32_t> piv_buf;

    EchelonBasis(Field f, uint64_t c) : field(std::move(f)), cols(c), scratch(c) {}
    uint64_t rank() const { return pivots.size(); }
    Entry negate(Entry a) const { return field.negate(a); }
    Entry *row(uint64_t i) { return rows.data() + i * cols; }
    const Entry *row(uint64_t i) const { return rows.data() + i * cols; }

    void reduce_into(Entry *v) const {
        for (uint64_t i = 0; i < pivots.size(); ++i) {
            Entry c = v[pivots[i]];
            if (c) field.subtract_multiple(v, row(i), c, cols);
        }
    }
    bool add(const Entry *v) {
        std::copy(v, v + cols, scratch.begin());
        reduce_into(scratch.data());
        uint64_t lead = cols;
        for (uint64_t j = 0; j < cols; ++j)
            if (scratch[j]) { lead = j; break; }
        if (lead == cols) return false;
        if (scratch[lead] != 1) field.scale(scratch.data(), field.inverse(scratch[lead]), cols);
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
    void reduce_by_last(Target &target) const {
        Entry c = target[pivots.back()];
        if (c) field.subtract_multiple(target.data(), row(rank() - 1), c, cols);
    }
    bool is_zero(const Target &target) const {
        return std::all_of(target.begin(), target.end(), [](Entry a) { return a == 0; });
    }

    void rref(std::vector<Entry> &out, std::vector<uint32_t> &piv) const {
        uint64_t r = rank();
        out = rows;
        piv = pivots;
        for (uint64_t i = 1; i < r; ++i)
            for (uint64_t earlier = 0; earlier < i; ++earlier) {
                Entry c = out[earlier * cols + piv[i]];
                if (c) field.subtract_multiple(out.data() + earlier * cols, out.data() + i * cols, c, cols);
            }
        order_buf.resize(r);
        for (uint64_t i = 0; i < r; ++i) order_buf[i] = i;
        std::sort(order_buf.begin(), order_buf.end(), [&](uint64_t a, uint64_t b) { return piv[a] < piv[b]; });
        sorted_buf.resize(r * cols);
        piv_buf.resize(r);
        for (uint64_t i = 0; i < r; ++i) {
            std::copy(out.begin() + order_buf[i] * cols, out.begin() + (order_buf[i] + 1) * cols,
                      sorted_buf.begin() + i * cols);
            piv_buf[i] = piv[order_buf[i]];
        }
        out.swap(sorted_buf);
        piv.swap(piv_buf);
    }
};

uint64_t gauss_jordan(const Field &field, Entry *a, uint64_t rows, uint64_t cols_a,
                      uint64_t cols_total, std::vector<uint32_t> &pivots) {
    uint64_t r = 0;
    pivots.clear();
    for (uint64_t c = 0; c < cols_a && r < rows; ++c) {
        uint64_t pivot = rows;
        for (uint64_t i = r; i < rows; ++i)
            if (a[i * cols_total + c]) { pivot = i; break; }
        if (pivot == rows) continue;
        if (pivot != r)
            std::swap_ranges(a + pivot * cols_total, a + (pivot + 1) * cols_total, a + r * cols_total);
        Entry *pivot_row = a + r * cols_total;
        if (pivot_row[c] != 1) field.scale(pivot_row, field.inverse(pivot_row[c]), cols_total);
        for (uint64_t i = 0; i < rows; ++i) {
            if (i == r) continue;
            Entry factor = a[i * cols_total + c];
            if (factor) field.subtract_multiple(a + i * cols_total, pivot_row, factor, cols_total);
        }
        pivots.push_back((uint32_t)c);
        ++r;
    }
    return r;
}

template <class Fn> void parallel_members(uint64_t count, uint32_t threads, Fn fn) {
    if (count == 0) return;
    threads = std::max<uint32_t>(1, std::min<uint64_t>(threads, count));
    std::atomic<uint64_t> next{0};
    auto work = [&] {
        for (;;) {
            uint64_t i = next.fetch_add(1);
            if (i >= count) break;
            fn(i);
        }
    };
    if (threads == 1) work();
    else {
        std::vector<std::thread> pool;
        for (uint32_t i = 0; i < threads; ++i) pool.emplace_back(work);
        for (auto &thread : pool) thread.join();
    }
}

R run_explicit(const Request &req, const Field &field) {
    const Matrix &a = *req.family->data;
    auto object = std::make_shared<Object>();
    if (req.op == "inverse") {
        if (a.rows != a.cols) return R::failure(INVALID, "inverse needs square members");
        auto output = std::make_shared<Inverses>();
        output->p = a.p;
        output->count = a.count;
        output->n = a.rows;
        output->invertible.assign(a.count, 0);
        output->entries.assign(a.count * a.rows * a.rows, 0);
        uint64_t n = a.rows, total_cols = 2 * n;
        parallel_members(a.count, req.threads, [&](uint64_t member) {
            std::vector<Entry> augmented(n * total_cols, 0);
            for (uint64_t row = 0; row < n; ++row) {
                std::copy(a.at(member) + row * n, a.at(member) + (row + 1) * n,
                          augmented.begin() + row * total_cols);
                augmented[row * total_cols + n + row] = 1;
            }
            std::vector<uint32_t> pivots;
            if (gauss_jordan(field, augmented.data(), n, n, total_cols, pivots) != n) return;
            output->invertible[member] = 1;
            for (uint64_t row = 0; row < n; ++row)
                std::copy(augmented.begin() + row * total_cols + n, augmented.begin() + (row + 1) * total_cols,
                          output->entries.begin() + (member * n + row) * n);
        });
        object->kind = "gfp.inverses";
        object->inverses = output;
        return R::success(object);
    }
    if (req.op == "solve") {
        auto it = req.handle_args.find("rhs");
        if (it == req.handle_args.end() || !it->second->matrix)
            return R::failure(INVALID, "solve needs a matrix argument `rhs`");
        const Matrix &b = *it->second->matrix;
        if (b.p != a.p || b.count != a.count || b.rows != 1 || b.cols != a.rows)
            return R::failure(INVALID, "rhs must be count vectors of member rows entries over the same field");
        auto output = std::make_shared<Solutions>();
        output->p = a.p;
        output->count = a.count;
        output->length = a.cols;
        output->solvable.assign(a.count, 0);
        output->entries.assign(a.count * a.cols, 0);
        uint64_t total_cols = a.cols + 1;
        parallel_members(a.count, req.threads, [&](uint64_t member) {
            std::vector<Entry> augmented(a.rows * total_cols, 0);
            for (uint64_t row = 0; row < a.rows; ++row) {
                std::copy(a.at(member) + row * a.cols, a.at(member) + (row + 1) * a.cols,
                          augmented.begin() + row * total_cols);
                augmented[row * total_cols + a.cols] = b.entries[member * a.rows + row];
            }
            std::vector<uint32_t> pivots;
            uint64_t rank = gauss_jordan(field, augmented.data(), a.rows, a.cols, total_cols, pivots);
            for (uint64_t row = rank; row < a.rows; ++row)
                if (augmented[row * total_cols + a.cols]) return;
            output->solvable[member] = 1;
            for (uint64_t row = 0; row < rank; ++row)
                output->entries[member * a.cols + pivots[row]] = augmented[row * total_cols + a.cols];
        });
        object->kind = "gfp.solutions";
        object->solutions = output;
        return R::success(object);
    }
    return R::failure(INTERNAL, "unknown explicit gfq operation " + req.op);
}

R run(const Request &req) {
    auto parsed = parse_field(req);
    if (!parsed.ok) return R::failure(parsed.error.status, parsed.error.message);
    gfp::Query query;
    if (!gfp::parse_query(req.op, query)) return run_explicit(req, parsed.value);
    return gfp::run_walk<EchelonBasis>(
        req, query, parse_reduction(req.reduction),
        [&](uint64_t, uint64_t cols) { return EchelonBasis(parsed.value, cols); });
}

BackendRegistration registration{Backend{
    "gfq", "generic",
    [] { return true; },
    [](const Request &req) {
        uint64_t q = req.family->prime();
        return q >= 2 && q < (1ULL << 32);
    },
    run,
    0}};

} // namespace
} // namespace lk::gfq
