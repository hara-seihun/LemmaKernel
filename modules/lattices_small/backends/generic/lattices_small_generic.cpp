/* Integral lattices, portable backend.
 *
 * Gram entries are exact signed i32 values. Fincke-Pohst uses an LDL decomposition only to
 * choose branches; every accepted vector is checked again with exact integer arithmetic, then
 * the output is sorted by signed coordinates. */
#include "../../../../runtime/src/reduce.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>

namespace lk::lattices_small {
namespace {

using R = Result<std::shared_ptr<Object>>;
constexpr int INVALID = 1;

struct Gram {
    uint64_t n = 0;
    std::vector<int64_t> a;
    std::vector<long double> l;
    std::vector<long double> d;
};

Result<__int128> determinant(const std::vector<int64_t> &source, uint64_t stride, uint64_t n) {
    if (n == 0) return Result<__int128>::success(1);
    std::vector<__int128> a(n * n);
    for (uint64_t i = 0; i < n; ++i)
        for (uint64_t j = 0; j < n; ++j) a[i * n + j] = source[i * stride + j];
    __int128 previous = 1;
    for (uint64_t k = 0; k + 1 < n; ++k) {
        __int128 pivot = a[k * n + k];
        if (pivot == 0) return Result<__int128>::success(0);
        for (uint64_t i = k + 1; i < n; ++i)
            for (uint64_t j = k + 1; j < n; ++j) {
                __int128 left, right, numerator;
                if (__builtin_mul_overflow(a[i * n + j], pivot, &left) ||
                    __builtin_mul_overflow(a[i * n + k], a[k * n + j], &right) ||
                    __builtin_sub_overflow(left, right, &numerator))
                    return Result<__int128>::failure(INVALID, "Gram determinant exceeds the exact-arithmetic limit");
                a[i * n + j] = numerator / previous;
            }
        previous = pivot;
    }
    return Result<__int128>::success(a[(n - 1) * n + n - 1]);
}

Result<Gram> decode_gram(const Matrix &m) {
    using G = Result<Gram>;
    if (m.p != GRAMS || m.count != 1 || m.rows == 0 || m.rows != m.cols)
        return G::failure(INVALID, "member must be one square lattices.gram matrix");
    if (m.rows > 16) return G::failure(INVALID, "Gram rank must be at most 16");
    Gram g;
    g.n = m.rows;
    g.a.resize(g.n * g.n);
    for (uint64_t i = 0; i < g.a.size(); ++i) g.a[i] = decode_signed(m.entries[i]);
    for (uint64_t i = 0; i < g.n; ++i)
        for (uint64_t j = 0; j < g.n; ++j)
            if (g.a[i * g.n + j] != g.a[j * g.n + i])
                return G::failure(INVALID, "Gram matrix must be symmetric");
    for (uint64_t k = 1; k <= g.n; ++k) {
        auto d = determinant(g.a, g.n, k);
        if (!d.ok) return G::failure(d.error.status, d.error.message);
        if (d.value <= 0) return G::failure(INVALID, "Gram matrix must be positive definite");
    }
    g.l.assign(g.n * g.n, 0);
    g.d.assign(g.n, 0);
    for (uint64_t i = 0; i < g.n; ++i) {
        g.l[i * g.n + i] = 1;
        for (uint64_t j = 0; j < i; ++j) {
            long double value = (long double)g.a[i * g.n + j];
            for (uint64_t k = 0; k < j; ++k)
                value -= g.l[i * g.n + k] * g.d[k] * g.l[j * g.n + k];
            g.l[i * g.n + j] = value / g.d[j];
        }
        long double value = (long double)g.a[i * g.n + i];
        for (uint64_t k = 0; k < i; ++k)
            value -= g.l[i * g.n + k] * g.l[i * g.n + k] * g.d[k];
        if (!(value > 0) || !std::isfinite(value))
            return G::failure(INVALID, "Gram matrix is too ill-conditioned for Fincke-Pohst branching");
        g.d[i] = value;
    }
    return G::success(std::move(g));
}

__int128 norm(const Gram &g, const std::vector<int64_t> &x) {
    __int128 q = 0;
    for (uint64_t i = 0; i < g.n; ++i)
        for (uint64_t j = 0; j < g.n; ++j)
            q += (__int128)x[i] * g.a[i * g.n + j] * x[j];
    return q;
}

Result<std::vector<std::vector<int64_t>>> enumerate(const Gram &g, uint64_t bound, bool include_zero) {
    using V = Result<std::vector<std::vector<int64_t>>>;
    std::vector<std::vector<int64_t>> out;
    std::vector<int64_t> x(g.n, 0);
    std::string error;
    std::function<void(int64_t, long double)> visit = [&](int64_t i, long double remaining) {
        if (!error.empty()) return;
        if (i < 0) {
            __int128 q = norm(g, x);
            if (q >= 0 && (unsigned __int128)q <= bound && (include_zero || q != 0)) out.push_back(x);
            return;
        }
        long double shift = 0;
        for (uint64_t j = (uint64_t)i + 1; j < g.n; ++j) shift += g.l[j * g.n + i] * x[j];
        long double radius = std::sqrt(std::max((long double)0, remaining / g.d[i]));
        long double low = -shift - radius, high = -shift + radius;
        if (low < INT32_MIN || high > INT32_MAX) {
            error = "a short-vector coordinate does not fit i32";
            return;
        }
        int64_t first = std::max<int64_t>(INT32_MIN, (int64_t)std::ceil(low) - 1);
        int64_t last = std::min<int64_t>(INT32_MAX, (int64_t)std::floor(high) + 1);
        for (int64_t xi = first; xi <= last; ++xi) {
            x[i] = xi;
            long double y = xi + shift;
            long double next = remaining - g.d[i] * y * y;
            long double tolerance = 1e-12L * (1 + bound + std::fabs(remaining));
            if (next >= -tolerance) visit(i - 1, std::max((long double)0, next));
        }
    };
    visit((int64_t)g.n - 1, (long double)bound);
    if (!error.empty()) return V::failure(INVALID, error);
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return V::success(std::move(out));
}

Result<uint64_t> lattice_minimum(const Gram &g) {
    uint64_t upper = UINT64_MAX;
    for (uint64_t i = 0; i < g.n; ++i) upper = std::min<uint64_t>(upper, (uint64_t)g.a[i * g.n + i]);
    auto vs = enumerate(g, upper, false);
    if (!vs.ok) return Result<uint64_t>::failure(vs.error.status, vs.error.message);
    uint64_t best = upper;
    for (const auto &x : vs.value) best = std::min<uint64_t>(best, (uint64_t)norm(g, x));
    return Result<uint64_t>::success(best);
}

Result<uint64_t> scalar_value(const std::string &op, const Gram &g) {
    if (op == "minimum") return lattice_minimum(g);
    if (op == "kissing_number") {
        auto minimum = lattice_minimum(g);
        if (!minimum.ok) return minimum;
        auto vs = enumerate(g, minimum.value, false);
        if (!vs.ok) return Result<uint64_t>::failure(vs.error.status, vs.error.message);
        uint64_t count = 0;
        for (const auto &x : vs.value) count += norm(g, x) == minimum.value;
        return Result<uint64_t>::success(count);
    }
    if (op == "is_unimodular") {
        auto d = determinant(g.a, g.n, g.n);
        if (!d.ok) return Result<uint64_t>::failure(d.error.status, d.error.message);
        return Result<uint64_t>::success(d.value == 1);
    }
    if (op == "is_even") {
        for (uint64_t i = 0; i < g.n; ++i) if (g.a[i * g.n + i] % 2 != 0) return Result<uint64_t>::success(0);
        return Result<uint64_t>::success(1);
    }
    return Result<uint64_t>::failure(4, "unknown scalar lattices_small operation " + op);
}

R run_scalar(const Request &req) {
    auto size = req.family->size();
    if (!size.ok) return R::failure(size.error.status, size.error.message);
    Reduction reduction = parse_reduction(req.reduction);
    Shared shared;
    auto prepared = prepare_all(reduction, size.value, shared);
    if (!prepared.ok) return R::failure(prepared.error.status, prepared.error.message);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, size.value ? size.value : 1));
    std::vector<Accumulator> accumulators;
    for (uint32_t t = 0; t < threads; ++t) accumulators.emplace_back(reduction, &shared);
    auto statuses = parallel_ranges(size.value, threads, [&](uint32_t t, uint64_t begin, uint64_t end) -> Status {
        Matrix member;
        for (uint64_t i = begin; i < end; ++i) {
            if (accumulators[t].exhausted(i)) break;
            auto materialised = req.family->member_into(i, member);
            if (!materialised.ok) return materialised;
            auto gram = decode_gram(member);
            if (!gram.ok) return fail(gram.error.status, gram.error.message);
            auto value = scalar_value(req.op, gram.value);
            if (!value.ok) return fail(value.error.status, value.error.message);
            if (req.op == "is_unimodular" || req.op == "is_even") accumulators[t].boolean(i, value.value != 0);
            else accumulators[t].integer(i, value.value);
        }
        return ok();
    });
    for (const auto &status : statuses) if (!status.ok) return R::failure(status.error.status, status.error.message);
    return assemble(req, reduction, accumulators, shared);
}

Result<uint64_t> request_bound(const Request &req) {
    auto it = req.int_args.find("bound");
    if (it == req.int_args.end()) return Result<uint64_t>::failure(INVALID, "missing bound");
    if (it->second > 1000000) return Result<uint64_t>::failure(INVALID, "bound must be at most 1000000");
    return Result<uint64_t>::success(it->second);
}

R run_theta(const Request &req) {
    if (req.reduction != "all") return R::failure(INVALID, "theta_series values only reduce with all");
    auto bound = request_bound(req);
    if (!bound.ok) return R::failure(bound.error.status, bound.error.message);
    auto size = req.family->size();
    if (!size.ok) return R::failure(size.error.status, size.error.message);
    unsigned __int128 words = (unsigned __int128)size.value * (bound.value + 1);
    if (words > SIZE_MAX / sizeof(uint64_t)) return R::failure(INVALID, "theta-series output is too large");
    auto theta = std::make_shared<ThetaSeries>();
    theta->count = size.value; theta->bound = bound.value;
    theta->coefficients.assign((size_t)words, 0);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, size.value ? size.value : 1));
    auto statuses = parallel_ranges(size.value, threads, [&](uint32_t, uint64_t begin, uint64_t end) -> Status {
        Matrix member;
        for (uint64_t i = begin; i < end; ++i) {
            auto materialised = req.family->member_into(i, member);
            if (!materialised.ok) return materialised;
            auto gram = decode_gram(member);
            if (!gram.ok) return fail(gram.error.status, gram.error.message);
            auto vectors = enumerate(gram.value, bound.value, true);
            if (!vectors.ok) return fail(vectors.error.status, vectors.error.message);
            uint64_t *coefficients = theta->coefficients.data() + i * (bound.value + 1);
            for (const auto &x : vectors.value) ++coefficients[(uint64_t)norm(gram.value, x)];
        }
        return ok();
    });
    for (const auto &status : statuses) if (!status.ok) return R::failure(status.error.status, status.error.message);
    auto out = std::make_shared<Object>();
    out->kind = "lattices.theta_series";
    out->theta_series = theta;
    return R::success(out);
}

R run_short_vectors(const Request &req) {
    if (req.reduction != "all") return R::failure(INVALID, "short_vectors values only reduce with all");
    auto bound = request_bound(req);
    if (!bound.ok) return R::failure(bound.error.status, bound.error.message);
    auto size = req.family->size();
    if (!size.ok) return R::failure(size.error.status, size.error.message);
    uint64_t n = req.family->rows();
    std::vector<std::vector<std::vector<int64_t>>> all(size.value);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, size.value ? size.value : 1));
    auto statuses = parallel_ranges(size.value, threads, [&](uint32_t, uint64_t begin, uint64_t end) -> Status {
        Matrix member;
        for (uint64_t i = begin; i < end; ++i) {
            auto materialised = req.family->member_into(i, member);
            if (!materialised.ok) return materialised;
            auto gram = decode_gram(member);
            if (!gram.ok) return fail(gram.error.status, gram.error.message);
            auto vectors = enumerate(gram.value, bound.value, false);
            if (!vectors.ok) return fail(vectors.error.status, vectors.error.message);
            all[i] = std::move(vectors.value);
        }
        return ok();
    });
    for (const auto &status : statuses) if (!status.ok) return R::failure(status.error.status, status.error.message);
    auto vectors = std::make_shared<ShortVectors>();
    vectors->count = size.value; vectors->n = n; vectors->bound = bound.value; vectors->offsets.push_back(0);
    for (const auto &batch : all) {
        for (const auto &x : batch)
            for (int64_t coordinate : x) {
                Entry encoded;
                if (!encode_signed(coordinate, encoded)) return R::failure(INVALID, "a short-vector coordinate does not fit i32");
                vectors->entries.push_back(encoded);
            }
        vectors->offsets.push_back(vectors->offsets.back() + batch.size());
    }
    auto out = std::make_shared<Object>();
    out->kind = "lattices.short_vectors";
    out->short_vectors = vectors;
    return R::success(out);
}

R run(const Request &req) {
    if (req.op == "theta_series") return run_theta(req);
    if (req.op == "short_vectors") return run_short_vectors(req);
    return run_scalar(req);
}

BackendRegistration registration{Backend{
    "lattices_small", "generic",
    [] { return true; },
    [](const Request &req) { return req.family->prime() == GRAMS && req.family->rows() <= 16; },
    run,
    0}};

} // namespace
} // namespace lk::lattices_small
