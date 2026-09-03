/* Portable invariants of symmetric forms over odd prime fields. */
#include "../../../gfp/backends/generic/field.hpp"
#include "../../../../runtime/src/reduce.hpp"

namespace lk::quadratic_forms {
namespace {

using R = Result<std::shared_ptr<Object>>;
constexpr int INVALID = 1;

struct Classification {
    uint64_t rank = 0;
    Entry discriminant = 1;
    bool hyperbolic = true;
    uint64_t type = 0;
    uint64_t witt_index = 0;
};

bool symmetric(const Entry *a, uint64_t n) {
    for (uint64_t i = 0; i < n; ++i)
        for (uint64_t j = i + 1; j < n; ++j)
            if (a[i * n + j] != a[j * n + i]) return false;
    return true;
}

void swap_coordinates(std::vector<Entry> &a, uint64_t n, uint64_t left, uint64_t right) {
    if (left == right) return;
    for (uint64_t j = 0; j < n; ++j) std::swap(a[left * n + j], a[right * n + j]);
    for (uint64_t i = 0; i < n; ++i) std::swap(a[i * n + left], a[i * n + right]);
}

void add_coordinate(std::vector<Entry> &a, uint64_t n, uint64_t i, uint64_t j, const gfp::Field &field) {
    for (uint64_t c = 0; c < n; ++c)
        a[i * n + c] = (Entry)field.reduce((uint64_t)a[i * n + c] + a[j * n + c]);
    for (uint64_t r = 0; r < n; ++r)
        a[r * n + i] = (Entry)field.reduce((uint64_t)a[r * n + i] + a[r * n + j]);
}

Classification classify(const Entry *entries, uint64_t n, const gfp::Field &field) {
    std::vector<Entry> a(entries, entries + n * n);
    Entry disc = 1;
    uint64_t rank = 0;
    for (uint64_t k = 0; k < n; ++k) {
        uint64_t pivot = n;
        for (uint64_t i = k; i < n; ++i)
            if (a[i * n + i]) { pivot = i; break; }
        if (pivot == n) {
            uint64_t left = n, right = n;
            for (uint64_t i = k; i < n && left == n; ++i)
                for (uint64_t j = i + 1; j < n; ++j)
                    if (a[i * n + j]) { left = i; right = j; break; }
            if (left == n) break;
            add_coordinate(a, n, left, right, field);
            pivot = left;
        }
        swap_coordinates(a, n, k, pivot);
        Entry d = a[k * n + k];
        disc = (Entry)field.reduce((uint64_t)disc * d);
        ++rank;
        Entry inverse = field.inverse(d);
        for (uint64_t i = k + 1; i < n; ++i) {
            Entry scaled = (Entry)field.reduce((uint64_t)a[i * n + k] * inverse);
            for (uint64_t j = k + 1; j < n; ++j) {
                Entry correction = (Entry)field.reduce((uint64_t)scaled * a[k * n + j]);
                a[i * n + j] = (Entry)field.reduce((uint64_t)a[i * n + j] +
                                                    (correction ? field.p - correction : 0));
            }
        }
        for (uint64_t i = k + 1; i < n; ++i) a[i * n + k] = a[k * n + i] = 0;
    }

    Entry signed_disc = disc;
    if ((rank / 2) & 1) signed_disc = disc ? (Entry)(field.p - disc) : 0;
    bool hyperbolic = field.pow(signed_disc, (field.p - 1) / 2) == 1;
    uint64_t type = rank & 1 ? 2 : hyperbolic ? 0 : 1;
    uint64_t half = rank / 2;
    uint64_t witt = (rank & 1) || hyperbolic ? half : half - 1;
    return {rank, disc, hyperbolic, type, witt};
}

bool isometric(const Classification &left, const Classification &right, const gfp::Field &field) {
    return left.rank == right.rank &&
           field.pow(left.discriminant, (field.p - 1) / 2) ==
               field.pow(right.discriminant, (field.p - 1) / 2);
}

Result<uint64_t> checked_projective_points(uint64_t p, uint64_t n) {
    using N = Result<uint64_t>;
    uint64_t total = 0, power = 1;
    for (uint64_t i = 0; i < n; ++i) {
        if (total > UINT64_MAX - power)
            return N::failure(INVALID, "isotropic point count does not fit in 64 bits");
        total += power;
        if (i + 1 < n) {
            if (power > UINT64_MAX / p)
                return N::failure(INVALID, "isotropic point count does not fit in 64 bits");
            power *= p;
        }
    }
    return N::success(total);
}

Result<uint64_t> checked_power(uint64_t p, uint64_t exponent) {
    using N = Result<uint64_t>;
    uint64_t value = 1;
    for (uint64_t i = 0; i < exponent; ++i) {
        if (value > UINT64_MAX / p)
            return N::failure(INVALID, "isotropic point count does not fit in 64 bits");
        value *= p;
    }
    return N::success(value);
}

Result<uint64_t> isotropic_points(uint64_t p, uint64_t n, const Classification &form) {
    using N = Result<uint64_t>;
    if (form.rank == 0) return checked_projective_points(p, n);
    auto base = checked_projective_points(p, n - 1);
    if (!base.ok || (form.rank & 1)) return base;
    uint64_t exponent = n - form.rank + form.rank / 2 - 1;
    auto correction = checked_power(p, exponent);
    if (!correction.ok) return correction;
    if (form.hyperbolic) {
        if (base.value > UINT64_MAX - correction.value)
            return N::failure(INVALID, "isotropic point count does not fit in 64 bits");
        return N::success(base.value + correction.value);
    }
    if (correction.value > base.value)
        return N::failure(4, "invalid elliptic point-count classification");
    return N::success(base.value - correction.value);
}

Status validate_request(const Request &req) {
    const Family &family = *req.family;
    if (family.kind != Family::Kind::Explicit && family.kind != Family::Kind::SymmetricMatrices)
        return fail(INVALID, "quadratic_forms operations need explicit or symmetric_matrices families");
    uint64_t p = family.prime();
    if (p >= (1ULL << 32) || (p & 1) == 0 || !is_prime(p))
        return fail(INVALID, "quadratic_forms requires an odd prime below 2^32");
    if (family.rows() != family.cols()) return fail(INVALID, "quadratic_forms requires square members");
    return ok();
}

Result<std::vector<Entry>> comparison_form(const Request &req, uint64_t n) {
    using V = Result<std::vector<Entry>>;
    auto it = req.handle_args.find("other");
    if (it == req.handle_args.end() || !it->second->matrix)
        return V::failure(INVALID, "other must be one symmetric matrix over the same field with the same n x n shape");
    const Matrix &source = *it->second->matrix;
    uint64_t rows = source.rows == 1 ? source.count : source.rows;
    if (source.p != req.family->prime() || rows != n || source.cols != n || source.entries.size() < n * n)
        return V::failure(INVALID, "other must be one symmetric matrix over the same field with the same n x n shape");
    std::vector<Entry> out(source.entries.begin(), source.entries.begin() + n * n);
    if (!symmetric(out.data(), n))
        return V::failure(INVALID, "other must be one symmetric matrix over the same field with the same n x n shape");
    return V::success(std::move(out));
}

enum class ScalarOp { FormType, Rank, WittIndex, IsIsometric, IsotropicPointCount };

R run_scalar(const Request &req, ScalarOp op) {
    const Family &family = *req.family;
    auto size_result = family.size();
    if (!size_result.ok) return R::failure(size_result.error.status, size_result.error.message);
    uint64_t size = size_result.value, n = family.cols();
    gfp::Field field(family.prime());

    Classification other_class;
    if (op == ScalarOp::IsIsometric) {
        auto other = comparison_form(req, n);
        if (!other.ok) return R::failure(other.error.status, other.error.message);
        other_class = classify(other.value.data(), n, field);
    }

    Reduction reduction = parse_reduction(req.reduction);
    Shared shared;
    auto prepared = prepare_all(reduction, size, shared);
    if (!prepared.ok) return R::failure(prepared.error.status, prepared.error.message);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, size ? size : 1));
    std::vector<Accumulator> accumulators;
    for (uint32_t t = 0; t < threads; ++t) accumulators.emplace_back(reduction, &shared);

    auto statuses = parallel_ranges(size, threads, [&](uint32_t thread, uint64_t begin, uint64_t end) -> Status {
        Matrix member;
        for (uint64_t i = begin; i < end; ++i) {
            if (accumulators[thread].exhausted(i)) break;
            auto st = family.member_into(i, member);
            if (!st.ok) return st;
            if (!symmetric(member.entries.data(), n)) return fail(INVALID, "quadratic_forms requires symmetric members");
            Classification form = classify(member.entries.data(), n, field);
            if (op == ScalarOp::IsIsometric) {
                accumulators[thread].boolean(i, isometric(form, other_class, field));
            } else if (op == ScalarOp::FormType) {
                accumulators[thread].integer(i, form.type);
            } else if (op == ScalarOp::Rank) {
                accumulators[thread].integer(i, form.rank);
            } else if (op == ScalarOp::WittIndex) {
                accumulators[thread].integer(i, form.witt_index);
            } else {
                auto count = isotropic_points(field.p, n, form);
                if (!count.ok) return fail(count.error.status, count.error.message);
                accumulators[thread].integer(i, count.value);
            }
        }
        return ok();
    });
    for (const auto &st : statuses)
        if (!st.ok) return R::failure(st.error.status, st.error.message);
    return assemble(req, reduction, accumulators, shared);
}

R run_radical(const Request &req) {
    if (req.reduction != "all") return R::failure(INVALID, "radical values only reduce with `all`");
    const Family &family = *req.family;
    auto size_result = family.size();
    if (!size_result.ok) return R::failure(size_result.error.status, size_result.error.message);
    uint64_t size = size_result.value, n = family.cols();
    std::vector<std::vector<Entry>> rows(size);
    std::vector<uint64_t> row_counts(size);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, size ? size : 1));
    auto statuses = parallel_ranges(size, threads, [&](uint32_t, uint64_t begin, uint64_t end) -> Status {
        Matrix member;
        gfp::EchelonBasis workspace(family.prime(), n);
        std::vector<Entry> basis;
        std::vector<uint32_t> pivots;
        for (uint64_t i = begin; i < end; ++i) {
            auto st = family.member_into(i, member);
            if (!st.ok) return st;
            if (!symmetric(member.entries.data(), n)) return fail(INVALID, "quadratic_forms requires symmetric members");
            workspace.clear();
            for (uint64_t r = 0; r < n; ++r) workspace.add(member.entries.data() + r * n);
            workspace.rref(basis, pivots);
            std::vector<uint8_t> is_pivot(n, 0);
            for (uint32_t pivot : pivots) is_pivot[pivot] = 1;
            auto &out = rows[i];
            for (uint64_t free = 0; free < n; ++free) {
                if (is_pivot[free]) continue;
                size_t start = out.size();
                out.resize(start + n, 0);
                out[start + free] = 1;
                for (uint64_t r = 0; r < pivots.size(); ++r) {
                    Entry value = basis[r * n + free];
                    out[start + pivots[r]] = value ? (Entry)(family.prime() - value) : 0;
                }
            }
            row_counts[i] = n - pivots.size();
        }
        return ok();
    });
    for (const auto &st : statuses)
        if (!st.ok) return R::failure(st.error.status, st.error.message);

    auto values = std::make_shared<Basis>();
    values->p = family.prime();
    values->count = size;
    values->cols = n;
    values->offsets.push_back(0);
    for (uint64_t i = 0; i < size; ++i) {
        values->entries.insert(values->entries.end(), rows[i].begin(), rows[i].end());
        values->offsets.push_back(values->offsets.back() + row_counts[i]);
    }
    auto object = std::make_shared<Object>();
    object->kind = "gfp.basis";
    object->basis = values;
    return R::success(object);
}

R run(const Request &req) {
    auto valid = validate_request(req);
    if (!valid.ok) return R::failure(valid.error.status, valid.error.message);
    if (req.op == "form_type") return run_scalar(req, ScalarOp::FormType);
    if (req.op == "rank") return run_scalar(req, ScalarOp::Rank);
    if (req.op == "radical") return run_radical(req);
    if (req.op == "witt_index") return run_scalar(req, ScalarOp::WittIndex);
    if (req.op == "is_isometric") return run_scalar(req, ScalarOp::IsIsometric);
    if (req.op == "isotropic_point_count") return run_scalar(req, ScalarOp::IsotropicPointCount);
    return R::failure(4, "unknown quadratic_forms operation " + req.op);
}

BackendRegistration registration{Backend{
    "quadratic_forms", "generic",
    [] { return true; },
    [](const Request &) { return true; },
    run,
    0}};

} // namespace
} // namespace lk::quadratic_forms
