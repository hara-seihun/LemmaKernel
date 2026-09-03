/* polynomials_fq generic backend: portable C++ for any prime p < 2^32.
 *
 * A member is one row of d entries, the monic polynomial x^d + sum a_i x^i. Inside, a polynomial
 * is its ascending coefficient vector with no trailing zero, so {} is 0 and {1} is 1.
 *
 * Irreducibility and factorisation use the Frobenius, not trial division: x^(p^d) mod f is built
 * up by repeated p-th powers, and gcd(f, x^(p^d) - x) collects the factors of degree d. Repeated
 * factors are separated first by Musser's squarefree decomposition, which also handles the
 * derivative vanishing (f is then a p-th power). Roots are Horner evaluation at every element,
 * gcd is Euclid, and the order of x is the walk x, x^2, x^3, ... until it returns to 1, which is
 * why `order` and `is_primitive` refuse a residue ring of 2^32 elements or more. */
#include "../../../../runtime/src/reduce.hpp"
#include "../../../gfp/backends/generic/field.hpp"

namespace lk::polynomials_fq {
namespace {

using R = Result<std::shared_ptr<Object>>;
constexpr int INVALID = 1;
constexpr int INTERNAL = 4;
constexpr uint64_t WALK_LIMIT = 1ULL << 32; /* size of the residue ring `order` will walk */
constexpr uint64_t MAX_MATERIALISED = 1ULL << 26;

/* Ascending coefficients over F_p, no trailing zero. */
using Poly = std::vector<Entry>;

struct Ring {
    gfp::Field f;
    uint64_t p;

    explicit Ring(uint64_t prime) : f(prime), p(prime) {}

    static void trim(Poly &a) {
        while (!a.empty() && a.back() == 0) a.pop_back();
    }
    /* deg a, for a nonzero a. */
    static uint64_t degree(const Poly &a) { return a.size() - 1; }

    Poly sub(const Poly &a, const Poly &b) const {
        Poly out(std::max(a.size(), b.size()), 0);
        for (size_t i = 0; i < out.size(); ++i) {
            uint64_t x = i < a.size() ? a[i] : 0, y = i < b.size() ? b[i] : 0;
            out[i] = (Entry)f.reduce(x + (p - y));
        }
        trim(out);
        return out;
    }

    Poly mul(const Poly &a, const Poly &b) const {
        if (a.empty() || b.empty()) return {};
        Poly out(a.size() + b.size() - 1, 0);
        for (size_t i = 0; i < a.size(); ++i) {
            if (!a[i]) continue;
            for (size_t j = 0; j < b.size(); ++j)
                out[i + j] = (Entry)f.reduce((uint64_t)out[i + j] + (uint64_t)a[i] * b[j]);
        }
        trim(out);
        return out;
    }

    /* r -= (leading term of r / leading term of b) * b, repeatedly. `binv` inverts b's lead. */
    void mod_inplace(Poly &r, const Poly &b, Entry binv) const {
        while (r.size() >= b.size() && !r.empty()) {
            Entry c = (Entry)f.reduce((uint64_t)r.back() * binv);
            size_t k = r.size() - b.size();
            for (size_t i = 0; i < b.size(); ++i)
                r[k + i] = (Entry)f.reduce((uint64_t)r[k + i] + (uint64_t)(p - c) * b[i]);
            trim(r);
        }
    }

    Poly mod(Poly a, const Poly &b) const {
        trim(a);
        if (b.empty()) return a;
        mod_inplace(a, b, f.inverse(b.back()));
        return a;
    }

    /* Quotient of a by a nonzero b. */
    Poly div(const Poly &a, const Poly &b) const {
        Poly r = a;
        trim(r);
        if (b.empty() || r.size() < b.size()) return {};
        Poly q(r.size() - b.size() + 1, 0);
        Entry binv = f.inverse(b.back());
        while (r.size() >= b.size() && !r.empty()) {
            Entry c = (Entry)f.reduce((uint64_t)r.back() * binv);
            size_t k = r.size() - b.size();
            q[k] = c;
            for (size_t i = 0; i < b.size(); ++i)
                r[k + i] = (Entry)f.reduce((uint64_t)r[k + i] + (uint64_t)(p - c) * b[i]);
            trim(r);
        }
        trim(q);
        return q;
    }

    void make_monic(Poly &a) const {
        if (a.empty() || a.back() == 1) return;
        Entry c = f.inverse(a.back());
        for (auto &x : a) x = (Entry)f.reduce((uint64_t)x * c);
    }

    Poly gcd(Poly a, Poly b) const {
        trim(a);
        trim(b);
        while (!b.empty()) {
            Poly r = mod(a, b);
            a.swap(b);
            b.swap(r);
        }
        make_monic(a);
        return a;
    }

    /* base^e mod m, for m of degree at least 1. */
    Poly powmod(Poly base, uint64_t e, const Poly &m) const {
        Poly result{1};
        base = mod(base, m);
        while (e) {
            if (e & 1) result = mod(mul(result, base), m);
            e >>= 1;
            if (e) base = mod(mul(base, base), m);
        }
        return result;
    }

    Entry eval(const Poly &a, uint64_t x) const {
        uint64_t acc = 0;
        for (size_t i = a.size(); i-- > 0;) acc = f.reduce(acc * x + a[i]);
        return (Entry)acc;
    }

    Poly derivative(const Poly &a) const {
        Poly out;
        for (size_t i = 1; i < a.size(); ++i)
            out.push_back((Entry)f.reduce((uint64_t)a[i] * (i % p)));
        trim(out);
        return out;
    }

    /* a is a p-th power (only exponents divisible by p occur); take its p-th root, using that
     * c^(1/p) = c in F_p. */
    Poly pth_root(const Poly &a) const {
        Poly out;
        for (size_t i = 0; i * p < a.size(); ++i) out.push_back(a[i * p]);
        trim(out);
        return out;
    }

    /* No irreducible factor of degree at most deg/2, so f is irreducible. */
    bool irreducible(const Poly &a) const {
        if (a.size() < 2) return false;
        uint64_t n = degree(a);
        if (n == 1) return true;
        const Poly x{0, 1};
        Poly h = powmod(x, p, a);
        for (uint64_t d = 1; 2 * d <= n; ++d) {
            if (gcd(a, sub(h, x)).size() > 1) return false;
            h = powmod(h, p, a);
        }
        return true;
    }

    /* Degrees of the irreducible factors of a monic squarefree s, each repeated `multiplicity`
     * times: gcd(s, x^(p^d) - x) is the product of the factors of degree d. */
    void distinct_degrees(Poly s, uint64_t multiplicity, std::vector<uint64_t> &out) const {
        const Poly x{0, 1};
        Poly h = x;
        uint64_t d = 0;
        while (s.size() > 1 && 2 * (d + 1) <= degree(s)) {
            ++d;
            h = powmod(h, p, s);
            Poly g = gcd(s, sub(h, x));
            if (g.size() > 1) {
                for (uint64_t i = 0; i < (degree(g) / d) * multiplicity; ++i) out.push_back(d);
                s = div(s, g);
                h = mod(h, s);
            }
        }
        if (s.size() > 1)
            for (uint64_t i = 0; i < multiplicity; ++i) out.push_back(degree(s));
    }

    /* Musser: peel off the factors of each multiplicity, then recurse into the p-th power part. */
    std::vector<uint64_t> factor_degrees(Poly a) const {
        std::vector<uint64_t> out;
        uint64_t e = 1;
        while (a.size() > 1) {
            Poly d = derivative(a);
            if (d.empty()) { /* a is a p-th power */
                a = pth_root(a);
                e *= p;
                continue;
            }
            Poly c = gcd(a, d);
            Poly w = div(a, c);
            for (uint64_t i = 1; w.size() > 1; ++i) {
                Poly y = gcd(w, c);
                Poly z = div(w, y);
                if (z.size() > 1) distinct_degrees(z, i * e, out);
                w = y;
                c = div(c, y);
            }
            a = pth_root(c);
            e *= p;
        }
        std::sort(out.begin(), out.end());
        return out;
    }

    void roots(const Poly &a, std::vector<uint64_t> &out) const {
        out.clear();
        for (uint64_t x = 0; x < p; ++x)
            if (eval(a, x) == 0) out.push_back(x);
    }

    /* Writing a = x^h g with g(0) != 0, the least e >= 1 with g | x^e - 1. */
    uint64_t order(const Poly &a) const {
        size_t h = 0;
        while (h < a.size() && a[h] == 0) ++h;
        Poly g(a.begin() + h, a.end());
        if (g.size() <= 1) return 1; /* the residue ring is trivial */
        Entry ginv = f.inverse(g.back());
        const Poly one{1};
        Poly cur = mod(Poly{0, 1}, g);
        uint64_t e = 1;
        while (cur != one) {
            cur.insert(cur.begin(), 0);
            mod_inplace(cur, g, ginv);
            ++e;
        }
        return e;
    }

    /* Irreducible, not divisible by x, and x runs through every nonzero residue. */
    bool primitive(const Poly &a, uint64_t units) const {
        return a.size() >= 2 && a[0] != 0 && irreducible(a) && order(a) == units;
    }
};

enum class Op { IsIrreducible, FactorisationDegrees, IsPrimitive, Order, Roots, RootCount, Gcd };

Result<Op> parse_op(const std::string &name) {
    static const std::map<std::string, Op> names{
        {"is_irreducible", Op::IsIrreducible}, {"factorisation_degrees", Op::FactorisationDegrees},
        {"is_primitive", Op::IsPrimitive}, {"order", Op::Order}, {"roots", Op::Roots},
        {"root_count", Op::RootCount}, {"gcd", Op::Gcd}};
    auto it = names.find(name);
    if (it == names.end()) return Result<Op>::failure(INTERNAL, "unknown polynomials_fq operation " + name);
    return Result<Op>::success(it->second);
}

bool walks_powers(Op op) { return op == Op::Order || op == Op::IsPrimitive; }
bool materialises(Op op) { return op == Op::Roots || op == Op::Gcd || op == Op::FactorisationDegrees; }

/* p^d, saturating at WALK_LIMIT. */
uint64_t ring_size(uint64_t p, uint64_t d) {
    uint64_t size = 1;
    for (uint64_t i = 0; i < d; ++i) {
        if (size > WALK_LIMIT / p) return WALK_LIMIT;
        size *= p;
    }
    return size;
}

/* The monic polynomial a member row denotes: its coefficients, then the leading 1. */
Poly polynomial_of(const Entry *row, uint64_t d) {
    Poly a(row, row + d);
    a.push_back(1);
    Ring::trim(a); /* a leading 1 never trims, but a zero-column member would */
    return a;
}

R run_materialised(const Request &req, const Ring &ring, Op op, uint64_t d, uint64_t size) {
    if (req.reduction != "all")
        return R::failure(INVALID, "polynomials_fq." + req.op + " values only reduce with `all`");
    if (size > MAX_MATERIALISED)
        return R::failure(INVALID, "family too large to materialise");
    Poly other;
    if (op == Op::Gcd) {
        auto it = req.handle_args.find("other");
        if (it == req.handle_args.end() || !it->second->matrix)
            return R::failure(INVALID, "gcd needs a gfp.matrix argument `other`");
        const Matrix &m = *it->second->matrix;
        if (m.p != ring.p || m.count != 1 || m.rows != 1)
            return R::failure(INVALID, "gcd needs `other` to be one 1 x k row over the same prime");
        other = polynomial_of(m.entries.data(), m.cols);
    }
    std::vector<std::vector<uint64_t>> per(size);
    auto statuses = parallel_ranges(size, std::max<uint32_t>(1, req.threads), [&](uint32_t, uint64_t begin, uint64_t end) -> Status {
        Matrix member;
        std::vector<uint64_t> values;
        for (uint64_t index = begin; index < end; ++index) {
            auto st = req.family->member_into(index, member);
            if (!st.ok) return st;
            Poly a = polynomial_of(member.entries.data(), d);
            switch (op) {
            case Op::FactorisationDegrees:
                per[index] = ring.factor_degrees(a);
                break;
            case Op::Roots:
                ring.roots(a, values);
                per[index] = values;
                break;
            default: {
                Poly g = ring.gcd(a, other);
                if (!g.empty()) g.pop_back(); /* the leading 1 is implied, as in a member row */
                per[index].assign(g.begin(), g.end());
                break;
            }
            }
        }
        return ok();
    });
    for (const auto &status : statuses)
        if (!status.ok) return R::failure(status.error.status, status.error.message);

    std::vector<uint64_t> offsets{0};
    offsets.reserve(size + 1);
    for (const auto &xs : per) offsets.push_back(offsets.back() + xs.size());
    auto o = std::make_shared<Object>();
    if (op == Op::FactorisationDegrees) {
        auto out = std::make_shared<Degrees>();
        out->count = size;
        out->offsets = std::move(offsets);
        out->values.reserve(out->offsets.back());
        for (const auto &xs : per) out->values.insert(out->values.end(), xs.begin(), xs.end());
        o->kind = "polynomials_fq.degrees";
        o->degrees = out;
    } else {
        auto out = std::make_shared<Elements>();
        out->p = ring.p;
        out->count = size;
        out->offsets = std::move(offsets);
        out->values.reserve(out->offsets.back());
        for (const auto &xs : per)
            for (uint64_t x : xs) out->values.push_back((Entry)x);
        o->kind = "polynomials_fq.elements";
        o->elements = out;
    }
    return R::success(o);
}

R run(const Request &req) {
    uint64_t p = req.family->prime();
    if (p < 2 || p >= (1ULL << 32))
        return R::failure(INVALID, "polynomials_fq needs a prime field with p < 2^32");
    if (req.family->rows() != 1)
        return R::failure(INVALID, "polynomials_fq reads every member as one polynomial, so a member must have exactly one row");
    auto parsed = parse_op(req.op);
    if (!parsed.ok) return R::failure(parsed.error.status, parsed.error.message);
    Op op = parsed.value;
    uint64_t d = req.family->cols();
    uint64_t units = ring_size(p, d);
    if (walks_powers(op) && units >= WALK_LIMIT)
        return R::failure(INVALID, "polynomials_fq." + req.op + " walks the powers of x and the residue ring "
                                   "F_p[x]/(f) is too large: it needs fewer than 2^32 elements");
    auto size = req.family->size();
    if (!size.ok) return R::failure(size.error.status, size.error.message);

    Ring ring(p);
    if (materialises(op)) return run_materialised(req, ring, op, d, size.value);

    Reduction reduction = parse_reduction(req.reduction);
    Shared shared;
    auto prepared = prepare_all(reduction, size.value, shared);
    if (!prepared.ok) return R::failure(prepared.error.status, prepared.error.message);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, size.value ? size.value : 1));
    std::vector<Accumulator> accumulators;
    for (uint32_t t = 0; t < threads; ++t) accumulators.emplace_back(reduction, &shared);

    auto statuses = parallel_ranges(size.value, threads, [&](uint32_t thread, uint64_t begin, uint64_t end) -> Status {
        Matrix member;
        std::vector<uint64_t> values;
        for (uint64_t index = begin; index < end; ++index) {
            if (accumulators[thread].exhausted(index)) break;
            auto st = req.family->member_into(index, member);
            if (!st.ok) return st;
            Poly a = polynomial_of(member.entries.data(), d);
            switch (op) {
            case Op::IsIrreducible:
                accumulators[thread].boolean(index, ring.irreducible(a));
                break;
            case Op::IsPrimitive:
                accumulators[thread].boolean(index, ring.primitive(a, units - 1));
                break;
            case Op::Order:
                accumulators[thread].integer(index, ring.order(a));
                break;
            default:
                ring.roots(a, values);
                accumulators[thread].integer(index, values.size());
                break;
            }
        }
        return ok();
    });
    for (const auto &status : statuses)
        if (!status.ok) return R::failure(status.error.status, status.error.message);
    return assemble(req, reduction, accumulators, shared);
}

BackendRegistration registration{Backend{
    "polynomials_fq", "generic",
    [] { return true; },
    [](const Request &req) { return req.family->prime() >= 2 && req.family->prime() < (1ULL << 32); },
    run,
    0}};

} // namespace
} // namespace lk::polynomials_fq
