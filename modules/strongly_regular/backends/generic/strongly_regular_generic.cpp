/* Portable strongly regular graph tests on square F_2 adjacency matrices. */
#include "../../../../runtime/src/reduce.hpp"

#include <bit>
#include <cmath>
#include <optional>
#include <boost/multiprecision/cpp_int.hpp>

namespace lk::strongly_regular {
namespace {

using R = Result<std::shared_ptr<Object>>;
using Big = boost::multiprecision::cpp_int;
constexpr int INVALID = 1;

struct Parameters {
    uint64_t v, k, lambda, mu;
};

struct Spectrum {
    uint64_t k;
    int64_t delta;
    uint64_t discriminant, multiplicity_plus, multiplicity_minus;
};

std::optional<Parameters> analyse(const Matrix &m, std::vector<uint64_t> &bits) {
    uint64_t v = m.rows;
    if (v == 0 || m.cols != v) return std::nullopt;
    uint64_t words = (v + 63) / 64;
    bits.assign(v * words, 0);
    std::vector<uint64_t> degrees(v, 0);
    for (uint64_t i = 0; i < v; ++i) {
        if (m.entries[i * v + i]) return std::nullopt;
        for (uint64_t j = 0; j < v; ++j) {
            Entry a = m.entries[i * v + j];
            if (a != m.entries[j * v + i]) return std::nullopt;
            if (a) {
                bits[i * words + j / 64] |= 1ULL << (j % 64);
                ++degrees[i];
            }
        }
    }
    uint64_t k = degrees[0];
    if (k == 0 || k >= v - 1 || std::any_of(degrees.begin(), degrees.end(), [&](uint64_t d) { return d != k; }))
        return std::nullopt;
    bool have_lambda = false, have_mu = false;
    uint64_t lambda = 0, mu = 0;
    for (uint64_t i = 0; i < v; ++i) {
        for (uint64_t j = i + 1; j < v; ++j) {
            uint64_t common = 0;
            for (uint64_t w = 0; w < words; ++w)
                common += std::popcount(bits[i * words + w] & bits[j * words + w]);
            if (m.entries[i * v + j]) {
                if (have_lambda && common != lambda) return std::nullopt;
                lambda = common;
                have_lambda = true;
            } else {
                if (have_mu && common != mu) return std::nullopt;
                mu = common;
                have_mu = true;
            }
        }
    }
    if (!have_lambda || !have_mu) return std::nullopt;
    return Parameters{v, k, lambda, mu};
}

uint64_t integer_sqrt(uint64_t n) {
    uint64_t q = (uint64_t)std::sqrt((long double)n);
    while ((unsigned __int128)(q + 1) * (q + 1) <= n) ++q;
    while ((unsigned __int128)q * q > n) --q;
    return q;
}

std::optional<Spectrum> spectrum_of(const Parameters &p) {
    __int128 delta = (__int128)p.lambda - p.mu;
    __int128 d_signed = delta * delta + 4 * ((__int128)p.k - p.mu);
    if (d_signed <= 0 || d_signed > UINT64_MAX || delta < INT64_MIN || delta > INT64_MAX) return std::nullopt;
    uint64_t d = (uint64_t)d_signed;
    __int128 imbalance = 2 * (__int128)p.k + (__int128)(p.v - 1) * delta;
    uint64_t q = integer_sqrt(d);
    __int128 f, g;
    if ((unsigned __int128)q * q == d) {
        __int128 base = (__int128)(p.v - 1) * q;
        __int128 denominator = 2 * (__int128)q;
        if (base - imbalance < 0 || base + imbalance < 0 ||
            (base - imbalance) % denominator != 0 || (base + imbalance) % denominator != 0)
            return std::nullopt;
        f = (base - imbalance) / denominator;
        g = (base + imbalance) / denominator;
    } else {
        if (imbalance != 0 || (p.v - 1) % 2) return std::nullopt;
        f = g = (p.v - 1) / 2;
    }
    if (f > UINT64_MAX || g > UINT64_MAX) return std::nullopt;
    return Spectrum{p.k, (int64_t)delta, d, (uint64_t)f, (uint64_t)g};
}

struct Quad {
    Big a, b;
};

Quad add(const Quad &x, const Quad &y) { return {x.a + y.a, x.b + y.b}; }
Quad sub(const Quad &x, const Quad &y) { return {x.a - y.a, x.b - y.b}; }
Quad scale(const Big &c, const Quad &x) { return {c * x.a, c * x.b}; }
Quad mul(const Quad &x, const Quad &y, const Big &d) {
    return {x.a * y.a + x.b * y.b * d, x.a * y.b + x.b * y.a};
}
Quad cube(const Quad &x, const Big &d) { return mul(mul(x, x, d), x, d); }

bool nonnegative(const Quad &x, const Big &d) {
    if (x.a >= 0) return x.b >= 0 || x.a * x.a >= x.b * x.b * d;
    return x.b > 0 && x.b * x.b * d >= x.a * x.a;
}

bool krein_one(const Parameters &p, int sign) {
    Big delta = Big(p.lambda) - p.mu;
    Big d = delta * delta + 4 * (Big(p.k) - p.mu);
    Big c = Big(p.v - p.k - 1) * (p.v - p.k - 1);
    Big kk = Big(p.k) * p.k;
    Quad x{delta, sign}, x_plus_two{delta + 2, sign};
    Quad value = sub(scale(c, add(cube(x, d), {8 * kk, 0})), scale(kk, cube(x_plus_two, d)));
    return nonnegative(value, d);
}

bool krein_bound(const Parameters &p) { return krein_one(p, 1) && krein_one(p, -1); }

bool absolute_bound(const Parameters &p) {
    if (p.mu == 0 || p.v + p.lambda == 2 * p.k) return true;
    auto s = spectrum_of(p);
    if (!s) return false;
    Big v = p.v, f = s->multiplicity_plus, g = s->multiplicity_minus;
    return 2 * v <= f * (f + 3) && 2 * v <= g * (g + 3);
}

R run_values(const Request &req, uint64_t size, bool spectra) {
    auto o = std::make_shared<Object>();
    std::shared_ptr<SrgParams> params;
    std::shared_ptr<SrgSpectra> spectrum;
    if (spectra) {
        spectrum = std::make_shared<SrgSpectra>();
        spectrum->count = size;
        spectrum->present.assign(size, 0);
        spectrum->k.assign(size, 0);
        spectrum->delta_negative.assign(size, 0);
        spectrum->delta_abs.assign(size, 0);
        spectrum->discriminant.assign(size, 0);
        spectrum->multiplicity_plus.assign(size, 0);
        spectrum->multiplicity_minus.assign(size, 0);
    } else {
        params = std::make_shared<SrgParams>();
        params->count = size;
        params->present.assign(size, 0);
        params->values.assign(size * 4, 0);
    }
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, size ? size : 1));
    std::vector<Matrix> members(threads);
    std::vector<std::vector<uint64_t>> bits(threads);
    auto statuses = parallel_ranges(size, threads, [&](uint32_t t, uint64_t begin, uint64_t end) -> Status {
        for (uint64_t i = begin; i < end; ++i) {
            auto st = req.family->member_into(i, members[t]);
            if (!st.ok) return st;
            auto p = analyse(members[t], bits[t]);
            if (!p) continue;
            if (!spectra) {
                params->present[i] = 1;
                params->values[4 * i] = p->v;
                params->values[4 * i + 1] = p->k;
                params->values[4 * i + 2] = p->lambda;
                params->values[4 * i + 3] = p->mu;
            } else if (auto s = spectrum_of(*p)) {
                spectrum->present[i] = 1;
                spectrum->k[i] = s->k;
                spectrum->delta_negative[i] = s->delta < 0;
                spectrum->delta_abs[i] = s->delta < 0 ? (uint64_t)(-(__int128)s->delta) : (uint64_t)s->delta;
                spectrum->discriminant[i] = s->discriminant;
                spectrum->multiplicity_plus[i] = s->multiplicity_plus;
                spectrum->multiplicity_minus[i] = s->multiplicity_minus;
            }
        }
        return ok();
    });
    for (const auto &st : statuses)
        if (!st.ok) return R::failure(st.error.status, st.error.message);
    if (spectra) {
        o->kind = "strongly_regular.spectra";
        o->srg_spectra = spectrum;
    } else {
        o->kind = "strongly_regular.params";
        o->srg_params = params;
    }
    return R::success(o);
}

R run_boolean(const Request &req, uint64_t size) {
    Reduction reduction = parse_reduction(req.reduction);
    Shared shared;
    auto prepared = prepare_all(reduction, size, shared);
    if (!prepared.ok) return R::failure(prepared.error.status, prepared.error.message);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, size ? size : 1));
    std::vector<Matrix> members(threads);
    std::vector<std::vector<uint64_t>> bits(threads);
    std::vector<Accumulator> accs;
    for (uint32_t t = 0; t < threads; ++t) accs.emplace_back(reduction, &shared);
    auto statuses = parallel_ranges(size, threads, [&](uint32_t t, uint64_t begin, uint64_t end) -> Status {
        for (uint64_t i = begin; i < end; ++i) {
            if (accs[t].exhausted(i)) break;
            auto st = req.family->member_into(i, members[t]);
            if (!st.ok) return st;
            auto p = analyse(members[t], bits[t]);
            bool value = false;
            if (p) {
                if (req.op == "is_srg") value = true;
                else if (req.op == "krein_bound") value = krein_bound(*p);
                else if (req.op == "absolute_bound") value = absolute_bound(*p);
            }
            accs[t].boolean(i, value);
        }
        return ok();
    });
    for (const auto &st : statuses)
        if (!st.ok) return R::failure(st.error.status, st.error.message);
    return assemble(req, reduction, accs, shared);
}

R run(const Request &req) {
    if (req.family->prime() != 2 || req.family->rows() != req.family->cols())
        return R::failure(INVALID, "strongly_regular needs square matrices over F_2");
    auto size = req.family->size();
    if (!size.ok) return R::failure(size.error.status, size.error.message);
    if (req.op == "srg_params") return run_values(req, size.value, false);
    if (req.op == "spectrum") return run_values(req, size.value, true);
    if (req.op == "is_srg" || req.op == "krein_bound" || req.op == "absolute_bound")
        return run_boolean(req, size.value);
    return R::failure(4, "unknown strongly_regular operation " + req.op);
}

BackendRegistration registration{Backend{
    "strongly_regular", "generic",
    [] { return true; },
    [](const Request &req) { return req.family->prime() == 2 && req.family->rows() == req.family->cols(); },
    run,
    0}};

} // namespace
} // namespace lk::strongly_regular
