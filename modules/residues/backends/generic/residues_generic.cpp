/* Portable arithmetic in Z/n over a range of residues.
 *
 * A request fixes one modulus, so everything that depends on it alone is computed once: the
 * factorisation of n and of phi(n), and, for discrete logarithms, the reduction chain of a
 * non-unit base together with the baby-step table. Each member is then a handful of modular
 * exponentiations. `least_primitive_root` is the exception: its member is the modulus, so it
 * pays for its own factorisation. */
#include "../../../../runtime/src/reduce.hpp"

#include <optional>
#include <unordered_map>

namespace lk::residues {
namespace {

using R = Result<std::shared_ptr<Object>>;
constexpr int INVALID = 1;
constexpr int INTERNAL = 4;
constexpr uint64_t MAX_MODULUS = 1ULL << 32;

enum class Op { Order, PrimitiveRoot, QuadraticResidue, DiscreteLog, Legendre, Jacobi, LeastPrimitiveRoot };

Result<Op> parse_op(const std::string &name) {
    static const std::map<std::string, Op> names{
        {"multiplicative_order", Op::Order}, {"is_primitive_root", Op::PrimitiveRoot},
        {"is_quadratic_residue", Op::QuadraticResidue}, {"discrete_log", Op::DiscreteLog},
        {"legendre", Op::Legendre}, {"jacobi", Op::Jacobi},
        {"least_primitive_root", Op::LeastPrimitiveRoot}};
    auto it = names.find(name);
    if (it == names.end()) return Result<Op>::failure(INTERNAL, "unknown residues operation " + name);
    return Result<Op>::success(it->second);
}

/* Every modulus is below 2^32, so a product of two residues fits in 64 bits. */

uint64_t gcd64(uint64_t a, uint64_t b) {
    while (b) {
        uint64_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

uint64_t pow_mod(uint64_t a, uint64_t e, uint64_t n) {
    uint64_t r = 1 % n;
    a %= n;
    while (e) {
        if (e & 1) r = r * a % n;
        a = a * a % n;
        e >>= 1;
    }
    return r;
}

/* a^-1 mod m by the extended Euclidean algorithm; gcd(a, m) must be 1. */
uint64_t inv_mod(uint64_t a, uint64_t m) {
    if (m == 1) return 0;
    int64_t t = 0, new_t = 1, r = (int64_t)m, new_r = (int64_t)(a % m);
    while (new_r) {
        int64_t q = r / new_r;
        int64_t tmp = t - q * new_t;
        t = new_t;
        new_t = tmp;
        tmp = r - q * new_r;
        r = new_r;
        new_r = tmp;
    }
    return (uint64_t)(t < 0 ? t + (int64_t)m : t);
}

using Factorisation = std::vector<std::pair<uint64_t, unsigned>>;

Factorisation factor(uint64_t n) {
    Factorisation out;
    for (uint64_t d = 2; d * d <= n; ++d) {
        if (n % d) continue;
        unsigned e = 0;
        while (n % d == 0) {
            n /= d;
            ++e;
        }
        out.push_back({d, e});
    }
    if (n > 1) out.push_back({n, 1});
    return out;
}

uint64_t totient_of(const Factorisation &f) {
    uint64_t phi = 1;
    for (auto [p, e] : f) {
        phi *= p - 1;
        for (unsigned i = 1; i < e; ++i) phi *= p;
    }
    return phi;
}

/* One modulus and everything about it that every residue shares. */
struct Modulus {
    uint64_t n, phi;
    Factorisation f;               /* n = prod p^e */
    std::vector<uint64_t> phi_primes; /* the distinct primes of phi(n) */

    explicit Modulus(uint64_t modulus) : n(modulus), f(factor(modulus)) {
        phi = totient_of(f);
        for (auto [q, e] : factor(phi)) {
            (void)e;
            phi_primes.push_back(q);
        }
    }

    /* The least k >= 1 with a^k = 1, found by removing prime factors from phi(n); 0 for a
     * non-unit, whose powers never reach 1. */
    uint64_t order(uint64_t a) const {
        if (gcd64(a, n) != 1) return 0;
        uint64_t e = phi, one = 1 % n;
        for (uint64_t q : phi_primes)
            while (e % q == 0 && pow_mod(a, e / q, n) == one) e /= q;
        return e;
    }

    bool primitive_root(uint64_t a) const {
        if (gcd64(a, n) != 1) return false;
        uint64_t one = 1 % n;
        for (uint64_t q : phi_primes)
            if (pow_mod(a, phi / q, n) == one) return false;
        return true;
    }

    /* A unit is a square mod n exactly when it is a square modulo every prime power of n:
     * Euler's criterion for odd primes, and 1 mod 4 or 1 mod 8 for the powers of two. */
    bool quadratic_residue(uint64_t a) const {
        if (gcd64(a, n) != 1) return false;
        for (auto [p, e] : f) {
            if (p == 2) {
                if (e == 2 && a % 4 != 1) return false;
                if (e >= 3 && a % 8 != 1) return false;
            } else if (pow_mod(a, (p - 1) / 2, p) != 1) {
                return false;
            }
        }
        return true;
    }

    /* (Z/n)* is cyclic exactly for n = 1, 2, 4, p^k and 2 p^k with p an odd prime. */
    bool cyclic() const {
        if (n <= 2 || n == 4) return true;
        if (f.size() == 1) return f[0].first != 2;
        return f.size() == 2 && f[0].first == 2 && f[0].second == 1;
    }

    uint64_t least_primitive_root() const {
        if (!cyclic()) return 0;
        for (uint64_t g = 1; g <= n; ++g)
            if (primitive_root(g)) return g;
        return 0;
    }
};

/* The Legendre symbol by Euler's criterion, encoded 0, 1, 2 with 2 for -1. */
uint64_t legendre_sym(uint64_t a, uint64_t p) {
    a %= p;
    if (a == 0) return 0;
    return pow_mod(a, (p - 1) / 2, p) == 1 ? 1 : 2;
}

/* The Jacobi symbol by quadratic reciprocity, in the same encoding. */
uint64_t jacobi_sym(uint64_t a, uint64_t n) {
    a %= n;
    bool negative = false;
    while (a) {
        while ((a & 1) == 0) {
            a >>= 1;
            uint64_t r = n & 7;
            if (r == 3 || r == 5) negative = !negative;
        }
        std::swap(a, n);
        if ((a & 3) == 3 && (n & 3) == 3) negative = !negative;
        a %= n;
    }
    return n == 1 ? (negative ? 2 : 1) : 0;
}

/* Least x with base^x = target in Z/n, or n when there is none.
 *
 * When the base is a unit this is baby-step giant-step: with S = ceil(sqrt(n)), the table of
 * base^j for j < S is shared by every member and each member walks target * base^-iS through it,
 * i ascending, so the first hit is the least x.
 *
 * A non-unit base is not invertible, so the equation is first reduced: while g = gcd(base, m)
 * exceeds 1, a solution with x > t forces g | b, and dividing g out of b and m leaves the same
 * problem one power down with an accumulated constant k. The chain of (g, m, k) depends only on
 * the base and the modulus, so it is built once too; only the divisions of the target are per
 * member, and x = t is tested at each link on the way down. */
struct Dlog {
    struct Link {
        uint64_t g, m, k;
    };
    uint64_t n, base, m, k, step, giant;
    std::vector<Link> chain;
    std::unordered_map<uint64_t, uint64_t> baby; /* k * base^j mod m -> least j < step */

    Dlog(uint64_t modulus, uint64_t b) : n(modulus), base(b % modulus), m(modulus), k(1 % modulus) {
        for (uint64_t g = gcd64(base, m); g > 1; g = gcd64(base, m)) {
            chain.push_back({g, m, k});
            m /= g;
            k = k * (base / g) % m;
        }
        step = 1;
        while (step * step < m) ++step;
        uint64_t v = k;
        for (uint64_t j = 0; j < step; ++j) {
            baby.emplace(v, j);
            v = v * (base % m) % m;
        }
        giant = inv_mod(pow_mod(base, step, m), m);
    }

    uint64_t solve(uint64_t target) const {
        uint64_t b = target % n;
        for (uint64_t t = 0; t < chain.size(); ++t) {
            if (b == chain[t].k) return t;
            if (b % chain[t].g) return n;
            b /= chain[t].g;
        }
        uint64_t cur = b % m;
        for (uint64_t i = 0; i <= step; ++i) {
            auto it = baby.find(cur);
            if (it != baby.end()) {
                uint64_t x = i * step + it->second + chain.size();
                return x < n ? x : n;
            }
            cur = cur * giant % m;
        }
        return n;
    }
};

R run(const Request &req) {
    if (req.family->kind != Family::Kind::Range)
        return R::failure(INVALID, "residues operations are defined on range families only");
    auto parsed = parse_op(req.op);
    if (!parsed.ok) return R::failure(parsed.error.status, parsed.error.message);
    Op op = parsed.value;

    uint64_t modulus = 1;
    if (op == Op::LeastPrimitiveRoot) {
        if (req.family->a < 1)
            return R::failure(INVALID, "least_primitive_root needs moduli of at least 1");
    } else {
        auto found = req.int_args.find("modulus");
        if (found == req.int_args.end()) return R::failure(INTERNAL, "residues: no modulus argument");
        modulus = found->second;
        if (modulus < 1) return R::failure(INVALID, "residues needs a modulus of at least 1");
        if (modulus >= MAX_MODULUS) return R::failure(INVALID, "residues needs a modulus less than 2^32");
    }

    Modulus mod(modulus);
    if (op == Op::Legendre && (modulus % 2 == 0 || mod.f.size() != 1 || mod.f[0].second != 1))
        return R::failure(INVALID, "legendre needs an odd prime modulus");
    if (op == Op::Jacobi && modulus % 2 == 0)
        return R::failure(INVALID, "jacobi needs an odd modulus");

    std::optional<Dlog> dlog;
    if (op == Op::DiscreteLog) {
        auto found = req.int_args.find("base");
        if (found == req.int_args.end()) return R::failure(INTERNAL, "residues: no base argument");
        dlog.emplace(modulus, found->second);
    }

    auto size = req.family->size();
    if (!size.ok) return R::failure(size.error.status, size.error.message);
    Reduction reduction = parse_reduction(req.reduction);
    Shared shared;
    auto prepared = prepare_all(reduction, size.value, shared);
    if (!prepared.ok) return R::failure(prepared.error.status, prepared.error.message);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, size.value ? size.value : 1));
    std::vector<Accumulator> accumulators;
    for (uint32_t t = 0; t < threads; ++t) accumulators.emplace_back(reduction, &shared);

    auto statuses = parallel_ranges(size.value, threads, [&](uint32_t thread, uint64_t begin, uint64_t end) -> Status {
        Accumulator &acc = accumulators[thread];
        for (uint64_t index = begin; index < end; ++index) {
            if (acc.exhausted(index)) break;
            /* the member of a range family, checked above, is the single natural a + index */
            uint64_t value = req.family->a + index;
            uint64_t a = value % modulus;
            switch (op) {
            case Op::Order: acc.integer(index, mod.order(a)); break;
            case Op::PrimitiveRoot: acc.boolean(index, mod.primitive_root(a)); break;
            case Op::QuadraticResidue: acc.boolean(index, mod.quadratic_residue(a)); break;
            case Op::DiscreteLog: acc.integer(index, dlog->solve(a)); break;
            case Op::Legendre: acc.integer(index, legendre_sym(a, modulus)); break;
            case Op::Jacobi: acc.integer(index, jacobi_sym(a, modulus)); break;
            case Op::LeastPrimitiveRoot: acc.integer(index, Modulus(value).least_primitive_root()); break;
            }
        }
        return ok();
    });
    for (const auto &status : statuses)
        if (!status.ok) return R::failure(status.error.status, status.error.message);
    return assemble(req, reduction, accumulators, shared);
}

BackendRegistration registration{Backend{
    "residues", "generic",
    [] { return true; },
    [](const Request &req) { return req.family->kind == Family::Kind::Range; },
    run,
    0}};

} // namespace
} // namespace lk::residues
