/* Portable linear-code computations over prime fields.
 *
 * Every generator matrix is reduced to the unique nonzero rref basis. Codeword enumeration uses
 * the reflected q-ary Gray code: one coefficient changes by +1 or -1 at every step, so the next
 * word costs one scaled row addition instead of a matrix-vector product. */
#include "../../../gfp/backends/generic/field.hpp"
#include "../../../../runtime/src/reduce.hpp"

#include <numeric>

namespace lk::linear_codes {
namespace {

using R = Result<std::shared_ptr<Object>>;
constexpr int INVALID = 1;

Result<uint64_t> checked_power(uint64_t base, uint64_t exponent) {
    uint64_t value = 1;
    for (uint64_t i = 0; i < exponent; ++i) {
        if (base && value > UINT64_MAX / base)
            return Result<uint64_t>::failure(INVALID, "enumeration size does not fit in 64 bits");
        value *= base;
    }
    return Result<uint64_t>::success(value);
}

void canonical_basis(gfp::EchelonBasis &workspace, const Entry *rows, uint64_t row_count, uint64_t n,
                     std::vector<Entry> &out, std::vector<uint32_t> &pivots) {
    workspace.clear();
    for (uint64_t r = 0; r < row_count; ++r) workspace.add(rows + r * n);
    workspace.rref(out, pivots);
}

std::vector<Entry> dual_basis(const gfp::Field &field, const std::vector<Entry> &basis,
                              const std::vector<uint32_t> &pivots, uint64_t n) {
    std::vector<uint8_t> is_pivot(n, 0);
    for (uint32_t c : pivots) is_pivot[c] = 1;
    std::vector<Entry> out;
    out.reserve((n - pivots.size()) * n);
    for (uint64_t c = 0; c < n; ++c) {
        if (is_pivot[c]) continue;
        size_t start = out.size();
        out.resize(start + n, 0);
        out[start + c] = 1;
        for (uint64_t r = 0; r < pivots.size(); ++r) {
            Entry e = basis[r * n + c];
            out[start + pivots[r]] = e ? (Entry)(field.p - e) : 0;
        }
    }
    return out;
}

uint64_t weight(const std::vector<Entry> &word) {
    return std::count_if(word.begin(), word.end(), [](Entry x) { return x != 0; });
}

template <class Fn>
Status gray_codewords(const gfp::Field &field, const std::vector<Entry> &basis, uint64_t n, Fn fn) {
    uint64_t k = n ? basis.size() / n : 0;
    auto count = checked_power(field.p, k);
    if (!count.ok) return fail(count.error.status, count.error.message);
    std::vector<Entry> word(n, 0);
    if (!fn(word)) return ok();
    std::vector<Entry> digits(k, 0);
    std::vector<int8_t> direction(k, 1);
    for (uint64_t step = 1; step < count.value; ++step) {
        uint64_t i = 0;
        while (i < k) {
            int64_t next = (int64_t)digits[i] + direction[i];
            if (0 <= next && next < (int64_t)field.p) {
                Entry delta = direction[i] > 0 ? 1 : (Entry)(field.p - 1);
                digits[i] = (Entry)next;
                for (uint64_t j = 0; j < i; ++j) direction[j] = -direction[j];
                for (uint64_t j = 0; j < n; ++j)
                    word[j] = (Entry)field.reduce((uint64_t)word[j] + (uint64_t)delta * basis[i * n + j]);
                break;
            }
            ++i;
        }
        if (!fn(word)) return ok();
    }
    return ok();
}

Status weight_enumerator(const gfp::Field &field, const std::vector<Entry> &basis, uint64_t n,
                         std::vector<uint64_t> &out) {
    out.assign(n + 1, 0);
    return gray_codewords(field, basis, n, [&](const std::vector<Entry> &word) {
        ++out[weight(word)];
        return true;
    });
}

Status minimum_distance(const gfp::Field &field, const std::vector<Entry> &basis, uint64_t n, uint64_t &out) {
    out = n + 1;
    auto st = gray_codewords(field, basis, n, [&](const std::vector<Entry> &word) {
        uint64_t w = weight(word);
        if (w && w < out) out = w;
        return out != 1;
    });
    if (st.ok && out == n + 1) out = 0;
    return st;
}

uint64_t distance(const Entry *left, const Entry *right, uint64_t n) {
    uint64_t d = 0;
    for (uint64_t i = 0; i < n; ++i) d += left[i] != right[i];
    return d;
}

Status covering_radius(const gfp::Field &field, const std::vector<Entry> &basis, uint64_t n, uint64_t &out) {
    auto code_count = checked_power(field.p, n ? basis.size() / n : 0);
    if (!code_count.ok) return fail(code_count.error.status, code_count.error.message);
    if (n && code_count.value > SIZE_MAX / n) return fail(INVALID, "codewords do not fit in memory");
    std::vector<Entry> code;
    code.reserve(code_count.value * n);
    auto st = gray_codewords(field, basis, n, [&](const std::vector<Entry> &word) {
        code.insert(code.end(), word.begin(), word.end());
        return true;
    });
    if (!st.ok) return st;

    std::vector<Entry> identity(n * n, 0);
    for (uint64_t i = 0; i < n; ++i) identity[i * n + i] = 1;
    out = 0;
    return gray_codewords(field, identity, n, [&](const std::vector<Entry> &word) {
        uint64_t nearest = n;
        for (uint64_t i = 0; i < code_count.value && nearest; ++i)
            nearest = std::min(nearest, distance(word.data(), code.data() + i * n, n));
        out = std::max(out, nearest);
        return out != n;
    });
}

bool self_dual(gfp::EchelonBasis &workspace, const std::vector<Entry> &basis,
               const std::vector<uint32_t> &pivots, uint64_t n) {
    if (pivots.size() * 2 != n) return false;
    std::vector<Entry> d = dual_basis(workspace.f, basis, pivots, n), canonical;
    std::vector<uint32_t> dual_pivots;
    canonical_basis(workspace, d.data(), n - pivots.size(), n, canonical, dual_pivots);
    return canonical == basis;
}

uint64_t automorphism_order(gfp::EchelonBasis &workspace, const std::vector<Entry> &basis, uint64_t n) {
    uint64_t k = n ? basis.size() / n : 0;
    std::vector<uint64_t> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::vector<Entry> image(k * n), canonical;
    std::vector<uint32_t> pivots;
    uint64_t order = 0;
    do {
        for (uint64_t r = 0; r < k; ++r)
            for (uint64_t j = 0; j < n; ++j) image[r * n + j] = basis[r * n + perm[j]];
        canonical_basis(workspace, image.data(), k, n, canonical, pivots);
        order += canonical == basis;
    } while (std::next_permutation(perm.begin(), perm.end()));
    return order;
}

enum class ScalarOp { MinimumDistance, IsSelfDual, CoveringRadius, IsMds, AutOrder };

R run_scalar(const Request &req, ScalarOp op) {
    const Family &family = *req.family;
    auto size_result = family.size();
    if (!size_result.ok) return R::failure(size_result.error.status, size_result.error.message);
    uint64_t size = size_result.value, n = family.cols();
    Reduction reduction = parse_reduction(req.reduction);
    Shared shared;
    auto prepared = prepare_all(reduction, size, shared);
    if (!prepared.ok) return R::failure(prepared.error.status, prepared.error.message);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, size ? size : 1));
    std::vector<Accumulator> accumulators;
    for (uint32_t t = 0; t < threads; ++t) accumulators.emplace_back(reduction, &shared);

    auto statuses = parallel_ranges(size, threads, [&](uint32_t thread, uint64_t begin, uint64_t end) -> Status {
        Matrix member;
        gfp::EchelonBasis workspace(family.prime(), n);
        std::vector<Entry> basis;
        std::vector<uint32_t> pivots;
        for (uint64_t i = begin; i < end; ++i) {
            if (accumulators[thread].exhausted(i)) break;
            auto st = family.member_into(i, member);
            if (!st.ok) return st;
            canonical_basis(workspace, member.entries.data(), member.rows, n, basis, pivots);
            if (op == ScalarOp::IsSelfDual) {
                accumulators[thread].boolean(i, self_dual(workspace, basis, pivots, n));
                continue;
            }
            if (op == ScalarOp::IsMds) {
                uint64_t d;
                st = minimum_distance(workspace.f, basis, n, d);
                if (!st.ok) return st;
                accumulators[thread].boolean(i, !pivots.empty() && d == n - pivots.size() + 1);
                continue;
            }
            uint64_t value = 0;
            if (op == ScalarOp::MinimumDistance) st = minimum_distance(workspace.f, basis, n, value);
            else if (op == ScalarOp::CoveringRadius) st = covering_radius(workspace.f, basis, n, value);
            else value = automorphism_order(workspace, basis, n);
            if (!st.ok) return st;
            accumulators[thread].integer(i, value);
        }
        return ok();
    });
    for (const auto &st : statuses)
        if (!st.ok) return R::failure(st.error.status, st.error.message);
    return assemble(req, reduction, accumulators, shared);
}

R run_weight_enumerator(const Request &req) {
    if (req.reduction != "all") return R::failure(INVALID, "weight_enumerator values only reduce with `all`");
    const Family &family = *req.family;
    auto size_result = family.size();
    if (!size_result.ok) return R::failure(size_result.error.status, size_result.error.message);
    uint64_t size = size_result.value, n = family.cols();
    if ((unsigned __int128)size * (n + 1) > SIZE_MAX / sizeof(uint64_t))
        return R::failure(INVALID, "weight enumerators do not fit in memory");
    auto values = std::make_shared<WeightEnumerators>();
    values->count = size;
    values->n = n;
    values->coefficients.assign(size * (n + 1), 0);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, size ? size : 1));
    auto statuses = parallel_ranges(size, threads, [&](uint32_t, uint64_t begin, uint64_t end) -> Status {
        Matrix member;
        gfp::EchelonBasis workspace(family.prime(), n);
        std::vector<Entry> basis;
        std::vector<uint32_t> pivots;
        std::vector<uint64_t> enumerator;
        for (uint64_t i = begin; i < end; ++i) {
            auto st = family.member_into(i, member);
            if (!st.ok) return st;
            canonical_basis(workspace, member.entries.data(), member.rows, n, basis, pivots);
            st = weight_enumerator(workspace.f, basis, n, enumerator);
            if (!st.ok) return st;
            std::copy(enumerator.begin(), enumerator.end(), values->coefficients.begin() + i * (n + 1));
        }
        return ok();
    });
    for (const auto &st : statuses)
        if (!st.ok) return R::failure(st.error.status, st.error.message);
    auto object = std::make_shared<Object>();
    object->kind = "linear_codes.weight_enumerators";
    object->weight_enumerators = values;
    return R::success(object);
}

R run_dual(const Request &req) {
    if (req.reduction != "all") return R::failure(INVALID, "dual values only reduce with `all`");
    const Family &family = *req.family;
    auto size_result = family.size();
    if (!size_result.ok) return R::failure(size_result.error.status, size_result.error.message);
    uint64_t size = size_result.value, n = family.cols();
    std::vector<std::vector<Entry>> rows(size);
    std::vector<uint64_t> row_counts(size, 0);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, size ? size : 1));
    auto statuses = parallel_ranges(size, threads, [&](uint32_t, uint64_t begin, uint64_t end) -> Status {
        Matrix member;
        gfp::EchelonBasis workspace(family.prime(), n);
        std::vector<Entry> basis;
        std::vector<uint32_t> pivots;
        for (uint64_t i = begin; i < end; ++i) {
            auto st = family.member_into(i, member);
            if (!st.ok) return st;
            canonical_basis(workspace, member.entries.data(), member.rows, n, basis, pivots);
            rows[i] = dual_basis(workspace.f, basis, pivots, n);
            row_counts[i] = n - pivots.size();
        }
        return ok();
    });
    for (const auto &st : statuses)
        if (!st.ok) return R::failure(st.error.status, st.error.message);

    auto values = std::make_shared<lk::Basis>();
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
    if (req.op == "minimum_distance") return run_scalar(req, ScalarOp::MinimumDistance);
    if (req.op == "weight_enumerator") return run_weight_enumerator(req);
    if (req.op == "dual") return run_dual(req);
    if (req.op == "is_self_dual") return run_scalar(req, ScalarOp::IsSelfDual);
    if (req.op == "covering_radius") return run_scalar(req, ScalarOp::CoveringRadius);
    if (req.op == "is_mds") return run_scalar(req, ScalarOp::IsMds);
    if (req.op == "aut_order") return run_scalar(req, ScalarOp::AutOrder);
    return R::failure(4, "unknown linear_codes operation " + req.op);
}

BackendRegistration registration{Backend{
    "linear_codes", "generic",
    [] { return true; },
    [](const Request &req) {
        uint64_t p = req.family->prime();
        return is_prime(p) && (req.op != "aut_order" || req.family->cols() <= 10);
    },
    run,
    0}};

} // namespace
} // namespace lk::linear_codes
