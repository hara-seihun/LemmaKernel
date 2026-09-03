#include "../../../gfp/backends/generic/field.hpp"

namespace lk::bilinear_invariants {
namespace {

using gfp::Field;
using Mat = std::vector<Entry>;
using R = Result<std::shared_ptr<Object>>;
constexpr int INVALID = 1;
constexpr int INTERNAL = 4;

struct FormInfo {
    uint64_t rank;
    Entry discriminant;
    bool alternating;
};

Entry add(const Field &field, Entry a, Entry b) {
    return (Entry)field.reduce((uint64_t)a + b);
}

Entry subtract(const Field &field, Entry a, Entry b) {
    return (Entry)field.reduce((uint64_t)a + (b ? field.p - b : 0));
}

Entry multiply(const Field &field, Entry a, Entry b) {
    return (Entry)field.reduce((uint64_t)a * b);
}

bool symmetric(const Matrix &a) {
    for (uint64_t i = 0; i < a.rows; ++i)
        for (uint64_t j = i + 1; j < a.cols; ++j)
            if (a.entries[i * a.cols + j] != a.entries[j * a.cols + i]) return false;
    return true;
}

bool alternating(const Matrix &a) {
    for (uint64_t i = 0; i < a.rows; ++i) {
        if (a.entries[i * a.cols + i] != 0) return false;
        for (uint64_t j = i + 1; j < a.cols; ++j)
            if (((uint64_t)a.entries[i * a.cols + j] + a.entries[j * a.cols + i]) % a.p != 0) return false;
    }
    return true;
}

void congruence_swap(Mat &a, uint64_t n, uint64_t i, uint64_t j) {
    if (i == j) return;
    for (uint64_t c = 0; c < n; ++c) std::swap(a[i * n + c], a[j * n + c]);
    for (uint64_t r = 0; r < n; ++r) std::swap(a[r * n + i], a[r * n + j]);
}

FormInfo form_info(const Field &field, const Matrix &member) {
    uint64_t n = member.rows;
    bool is_alternating = alternating(member);
    Mat a = member.entries;
    uint64_t k = 0, rank = 0;
    Entry discriminant = 1;
    while (k < n) {
        uint64_t diagonal = n;
        for (uint64_t i = k; i < n; ++i)
            if (a[i * n + i]) { diagonal = i; break; }
        if (diagonal != n) {
            congruence_swap(a, n, k, diagonal);
            Entry d = a[k * n + k];
            discriminant = multiply(field, discriminant, d);
            Entry inverse = field.inverse(d);
            for (uint64_t i = k + 1; i < n; ++i)
                for (uint64_t j = k + 1; j < n; ++j) {
                    Entry term = multiply(field, multiply(field, a[i * n + k], inverse), a[k * n + j]);
                    a[i * n + j] = subtract(field, a[i * n + j], term);
                }
            ++rank;
            ++k;
            continue;
        }
        uint64_t first = n, second = n;
        for (uint64_t i = k; i < n && first == n; ++i)
            for (uint64_t j = i + 1; j < n; ++j)
                if (a[i * n + j]) { first = i; second = j; break; }
        if (first == n) break;
        congruence_swap(a, n, k, first);
        congruence_swap(a, n, k + 1, second);
        Entry x = a[k * n + k + 1], y = a[(k + 1) * n + k];
        Entry xy = multiply(field, x, y);
        Entry block_determinant = xy ? (Entry)(field.p - xy) : 0;
        discriminant = multiply(field, discriminant, block_determinant);
        Entry inverse_x = field.inverse(x), inverse_y = field.inverse(y);
        for (uint64_t i = k + 2; i < n; ++i)
            for (uint64_t j = k + 2; j < n; ++j) {
                Entry left = multiply(field, multiply(field, a[i * n + k], inverse_y), a[(k + 1) * n + j]);
                Entry right = multiply(field, multiply(field, a[i * n + k + 1], inverse_x), a[k * n + j]);
                a[i * n + j] = subtract(field, a[i * n + j], add(field, left, right));
            }
        rank += 2;
        k += 2;
    }
    return {rank, rank ? discriminant : Entry{0}, is_alternating};
}

uint64_t power_mod(uint64_t base, uint64_t exponent, uint64_t p) {
    uint64_t value = 1;
    while (exponent) {
        if (exponent & 1) value = (uint64_t)((unsigned __int128)value * base % p);
        exponent >>= 1;
        if (exponent) base = (uint64_t)((unsigned __int128)base * base % p);
    }
    return value;
}

uint64_t square_class(uint64_t x, uint64_t p) {
    if (x == 0) return 0;
    return power_mod(x, (p - 1) / 2, p) == 1 ? 1 : 2;
}

Entry least_nonsquare(uint64_t p) {
    for (uint64_t x = 2; x < p; ++x)
        if (square_class(x, p) == 2) return (Entry)x;
    return 1;
}

Mat congruence_label(const Field &field, uint64_t n, const FormInfo &info) {
    Mat out(n * n, 0);
    if (info.alternating) {
        for (uint64_t i = 0; i < info.rank; i += 2) {
            out[i * n + i + 1] = 1;
            out[(i + 1) * n + i] = (Entry)(field.p - 1);
        }
        return out;
    }
    Entry last = 1;
    if (field.p != 2 && square_class(info.discriminant, field.p) == 2) last = least_nonsquare(field.p);
    for (uint64_t i = 0; i < info.rank; ++i) out[i * n + i] = i + 1 == info.rank ? last : 1;
    return out;
}

Status validate_member(const Matrix &member) {
    if (!symmetric(member) && !alternating(member))
        return fail(INVALID, "each matrix must be symmetric or alternating");
    return ok();
}

R run_values(const Request &req, uint64_t size, uint64_t n) {
    auto result = std::make_shared<Matrix>();
    result->p = req.family->prime();
    result->count = size;
    result->rows = n;
    result->cols = n;
    result->entries.assign(size * n * n, 0);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, size ? size : 1));
    auto statuses = parallel_ranges(size, threads, [&](uint32_t, uint64_t begin, uint64_t end) -> Status {
        Field field(req.family->prime());
        Matrix member;
        for (uint64_t i = begin; i < end; ++i) {
            auto status = req.family->member_into(i, member);
            if (!status.ok) return status;
            status = validate_member(member);
            if (!status.ok) return status;
            Mat label = congruence_label(field, n, form_info(field, member));
            std::copy(label.begin(), label.end(), result->entries.begin() + i * n * n);
        }
        return ok();
    });
    for (const auto &status : statuses)
        if (!status.ok) return R::failure(status.error.status, status.error.message);
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
            auto status = req.family->member_into(i, member);
            if (!status.ok) return status;
            status = validate_member(member);
            if (!status.ok) return status;
            FormInfo info = form_info(field, member);
            if (req.op == "is_nondegenerate") accumulators[thread].boolean(i, info.rank == n);
            else if (req.op == "is_alternating") accumulators[thread].boolean(i, info.alternating);
            else if (req.op == "rank") accumulators[thread].integer(i, info.rank);
            else if (req.op == "radical_dimension") accumulators[thread].integer(i, n - info.rank);
            else if (req.op == "determinant") accumulators[thread].integer(i, info.rank == n ? info.discriminant : 0);
            else if (req.op == "determinant_class")
                accumulators[thread].integer(i, info.rank == n ? square_class(info.discriminant, field.p) : 0);
            else if (req.op == "discriminant_class")
                accumulators[thread].integer(i, info.rank ? square_class(info.discriminant, field.p) : 0);
            else return fail(INTERNAL, "unknown bilinear_invariants operation " + req.op);
        }
        return ok();
    });
    for (const auto &status : statuses)
        if (!status.ok) return R::failure(status.error.status, status.error.message);
    return assemble(req, reduction, accumulators, shared);
}

R run(const Request &req) {
    if (req.family->rows() == 0 || req.family->rows() != req.family->cols())
        return R::failure(INVALID, "bilinear_invariants needs nonempty square matrices");
    auto size = req.family->size();
    if (!size.ok) return R::failure(size.error.status, size.error.message);
    if (req.op == "congruence_label") return run_values(req, size.value, req.family->rows());
    return run_reduced(req, size.value, req.family->rows());
}

BackendRegistration registration{Backend{
    "bilinear_invariants", "generic", [] { return true; },
    [](const Request &req) { return is_prime(req.family->prime()) && req.family->prime() < (1ULL << 32); },
    run, 0}};

} // namespace
} // namespace lk::bilinear_invariants
