/* elliptic_curves_fp generic backend: portable C++ for any prime 3 < p < 2^32.
 *
 * A member is the pair (a, b) of y^2 = x^3 + a*x + b. Two tables are built once per request and
 * shared by every member: how many square roots each element of F_p has (and the least one, when
 * the group law needs points), and nothing else; everything per member is a single pass over x.
 * That pass walks x = 0, 1, ... by third differences, so x^3 + a*x + b costs three additions
 * rather than two multiplications.
 *
 * Point counting, the isomorphism orbit and the group law all enumerate F_p, so the backend
 * accepts only primes it can walk (see prime_limit); larger primes need a different algorithm,
 * not a bigger machine, and the runtime says no backend accepts the request. */
#include "../../../../runtime/src/reduce.hpp"
#include "../../../gfp/backends/generic/field.hpp"

#include <numeric>

namespace lk::elliptic_curves_fp {
namespace {

using R = Result<std::shared_ptr<Object>>;
constexpr int INVALID = 1;
constexpr int INTERNAL = 4;

enum class Op { PointCount, Nonsingular, Supersingular, JInvariant, IsCanonical, ClassSize, GroupStructure };

bool parse_op(const std::string &name, Op &out) {
    static const std::map<std::string, Op> names{
        {"point_count", Op::PointCount}, {"nonsingular", Op::Nonsingular},
        {"supersingular", Op::Supersingular}, {"j_invariant", Op::JInvariant},
        {"is_canonical", Op::IsCanonical}, {"class_size", Op::ClassSize},
        {"group_structure", Op::GroupStructure}};
    auto it = names.find(name);
    if (it == names.end()) return false;
    out = it->second;
    return true;
}

/* The largest prime this backend will enumerate, per operation. */
uint64_t prime_limit(Op op) {
    switch (op) {
    case Op::PointCount:
    case Op::Supersingular: return 1ULL << 24;   /* one pass over F_p per member, plus a p-byte table */
    case Op::IsCanonical:
    case Op::ClassSize: return 1ULL << 22;       /* one pass over F_p^* per member */
    case Op::GroupStructure: return 1ULL << 20;  /* the point list and its orders, per member */
    default: return (1ULL << 32) - 1;            /* nonsingular and j_invariant are O(1) per member */
    }
}

bool needs_roots(Op op) { return op == Op::GroupStructure; }
bool needs_root_counts(Op op) {
    return op == Op::PointCount || op == Op::Supersingular || op == Op::GroupStructure;
}

/* A point of E(F_p): the point at infinity, or an affine (x, y). */
struct Pt {
    uint64_t x = 0, y = 0;
    bool inf = true;
};

struct Curves {
    gfp::Field f;
    uint64_t p;
    std::vector<uint8_t> root_count; /* #{y : y^2 = v}, for v in 0..p-1 */
    std::vector<Entry> root;         /* the least such y, where there is one */

    explicit Curves(uint64_t prime) : f(prime), p(prime) {}

    /* Square roots in F_p, once per request: y and p - y are the roots of y^2, so half of F_p^*
     * is enough, and walking y upwards leaves the least root in `root`. */
    void build(bool with_roots) {
        root_count.assign(p, 0);
        root_count[0] = 1;
        if (with_roots) root.assign(p, 0);
        for (uint64_t y = 1; 2 * y < p; ++y) {
            uint64_t v = f.reduce(y * y);
            root_count[v] = 2;
            if (with_roots) root[v] = (Entry)y;
        }
    }

    uint64_t reduce(uint64_t x) const { return f.reduce(x); }
    uint64_t inverse(uint64_t x) const { return f.inverse((Entry)x); }

    /* 4*a^3 + 27*b^2 in F_p: zero exactly on the singular pairs, since the discriminant is
     * -16 times it and p > 3. */
    uint64_t singular_form(uint64_t a, uint64_t b) const {
        return reduce(4 * reduce(reduce(a * a) * a) + 27 * reduce(b * b));
    }

    /* 1728*4*a^3 / (4*a^3 + 27*b^2), or p when the pair is singular. */
    uint64_t j_invariant(uint64_t a, uint64_t b) const {
        uint64_t d = singular_form(a, b);
        if (d == 0) return p;
        return reduce(reduce(6912 % p * reduce(reduce(a * a) * a)) * inverse(d));
    }

    /* #E(F_p), the affine solutions plus the point at infinity. `v` walks x^3 + a*x + b by its
     * third differences: f(x+1) - f(x) = 3x^2 + 3x + 1 + a, whose own differences are 6x + 6
     * and 6. */
    uint64_t point_count(uint64_t a, uint64_t b) const {
        uint64_t v = b % p, d1 = reduce(1 + a), d2 = 6 % p, d3 = 6 % p;
        uint64_t total = 1;
        for (uint64_t x = 0; x < p; ++x) {
            total += root_count[v];
            v += d1;
            if (v >= p) v -= p;
            d1 += d2;
            if (d1 >= p) d1 -= p;
            d2 += d3;
            if (d2 >= p) d2 -= p;
        }
        return total;
    }

    void points(uint64_t a, uint64_t b, std::vector<Pt> &out) const {
        out.clear();
        uint64_t v = b % p, d1 = reduce(1 + a), d2 = 6 % p, d3 = 6 % p;
        for (uint64_t x = 0; x < p; ++x) {
            if (root_count[v] == 1) {
                out.push_back(Pt{x, 0, false});
            } else if (root_count[v] == 2) {
                uint64_t y = root[v];
                out.push_back(Pt{x, y, false});
                out.push_back(Pt{x, p - y, false});
            }
            v += d1;
            if (v >= p) v -= p;
            d1 += d2;
            if (d1 >= p) d1 -= p;
            d2 += d3;
            if (d2 >= p) d2 -= p;
        }
    }

    Pt add(uint64_t a, const Pt &P, const Pt &Q) const {
        if (P.inf) return Q;
        if (Q.inf) return P;
        if (P.x == Q.x && (P.y + Q.y) % p == 0) return Pt{};
        uint64_t l;
        if (P.x == Q.x) {
            l = reduce(reduce(3 * reduce(P.x * P.x) + a) * inverse(reduce(2 * P.y)));
        } else {
            uint64_t dx = reduce(Q.x + p - P.x), dy = reduce(Q.y + p - P.y);
            l = reduce(dy * inverse(dx));
        }
        uint64_t x3 = reduce(reduce(l * l) + 2 * p - P.x - Q.x);
        uint64_t y3 = reduce(reduce(l * reduce(P.x + p - x3)) + p - P.y);
        return Pt{x3, y3, false};
    }

    Pt multiply(uint64_t a, uint64_t k, Pt P) const {
        Pt acc;
        while (k) {
            if (k & 1) acc = add(a, acc, P);
            P = add(a, P, P);
            k >>= 1;
        }
        return acc;
    }

    /* The order of P, from a multiple `n` of it and the primes dividing `n`. */
    uint64_t order(uint64_t a, const Pt &P, uint64_t n, const std::vector<uint64_t> &primes) const {
        uint64_t m = n;
        for (uint64_t q : primes)
            while (m % q == 0 && multiply(a, m / q, P).inf) m /= q;
        return m;
    }

    /* The isomorphism class of (a, b) is {(u^4*a, u^6*b) : u in F_p^*}. Its stabiliser has the
     * same size for every member of the class, so the class has (p-1)/|stabiliser| elements. */
    bool is_canonical(uint64_t a, uint64_t b) const {
        for (uint64_t u = 2; u < p; ++u) {
            uint64_t u2 = reduce(u * u), u4 = reduce(u2 * u2);
            uint64_t ia = reduce(u4 * a);
            if (ia > a) continue;
            uint64_t ib = reduce(reduce(u4 * u2) * b);
            if (ia < a || ib < b) return false;
        }
        return true;
    }

    uint64_t class_size(uint64_t a, uint64_t b) const {
        uint64_t fixed = 0;
        for (uint64_t u = 1; u < p; ++u) {
            uint64_t u2 = reduce(u * u), u4 = reduce(u2 * u2);
            if (reduce(u4 * a) == a && reduce(reduce(u4 * u2) * b) == b) ++fixed;
        }
        return (p - 1) / fixed;
    }

    /* Invariant factors (n1, n2) of E(F_p), n1 | n2. The exponent of a finite abelian group is
     * the order of one of its elements, so n2 is the largest point order and n1 is #E / n2. */
    void group_structure(uint64_t a, uint64_t b, std::vector<Pt> &scratch, uint64_t &n1, uint64_t &n2) const {
        if (singular_form(a, b) == 0) {
            n1 = n2 = 0;
            return;
        }
        points(a, b, scratch);
        uint64_t n = scratch.size() + 1;
        std::vector<uint64_t> primes;
        uint64_t rest = n;
        for (uint64_t q = 2; q * q <= rest; ++q)
            if (rest % q == 0) {
                primes.push_back(q);
                while (rest % q == 0) rest /= q;
            }
        if (rest > 1) primes.push_back(rest);
        uint64_t exponent = 1;
        for (const Pt &P : scratch) {
            if (exponent == n) break;
            exponent = std::lcm(exponent, order(a, P, n, primes));
        }
        n1 = n / exponent;
        n2 = exponent;
    }
};

Status check_request(const Request &req, Op op) {
    const Family &fam = *req.family;
    uint64_t p = fam.prime();
    if (p <= 3 || p >= (1ULL << 32))
        return fail(INVALID, "elliptic_curves_fp needs a prime p > 3 (short Weierstrass form needs 6 invertible), got p = " +
                                 std::to_string(p));
    if (p > prime_limit(op))
        return fail(INVALID, "elliptic_curves_fp." + req.op + " enumerates F_p; this backend stops at p = " +
                                 std::to_string(prime_limit(op)));
    if (fam.rows() != 1 || fam.cols() != 2)
        return fail(INVALID, "elliptic_curves_fp needs members that are 1 x 2 matrices (a, b), got " +
                                 std::to_string(fam.rows()) + " x " + std::to_string(fam.cols()));
    return ok();
}

R run_group_structure(const Request &req, Curves &curves) {
    const Family &fam = *req.family;
    if (req.reduction != "all") return R::failure(INVALID, "group_structure values only reduce with `all`");
    auto size_r = fam.size();
    if (!size_r.ok) return R::failure(size_r.error.status, size_r.error.message);
    uint64_t size = size_r.value;
    if (size > (1ULL << 30)) return R::failure(INVALID, "family too large to materialise");
    auto groups = std::make_shared<CurveGroups>();
    groups->count = size;
    groups->orders.assign(2 * size, 0);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, size ? size : 1));
    auto statuses = parallel_ranges(size, threads, [&](uint32_t, uint64_t begin, uint64_t end) -> Status {
        Matrix member;
        std::vector<Pt> scratch;
        for (uint64_t index = begin; index < end; ++index) {
            auto st = fam.member_into(index, member);
            if (!st.ok) return st;
            uint64_t n1 = 0, n2 = 0;
            curves.group_structure(member.entries[0], member.entries[1], scratch, n1, n2);
            groups->orders[2 * index] = n1;
            groups->orders[2 * index + 1] = n2;
        }
        return ok();
    });
    for (const auto &status : statuses)
        if (!status.ok) return R::failure(status.error.status, status.error.message);
    auto o = std::make_shared<Object>();
    o->kind = "elliptic_curves_fp.group";
    o->curve_groups = groups;
    return R::success(o);
}

R run(const Request &req) {
    Op op;
    if (!parse_op(req.op, op)) return R::failure(INTERNAL, "unknown elliptic_curves_fp operation " + req.op);
    auto valid = check_request(req, op);
    if (!valid.ok) return R::failure(valid.error.status, valid.error.message);

    const Family &fam = *req.family;
    Curves curves(fam.prime());
    if (needs_root_counts(op)) curves.build(needs_roots(op));
    if (op == Op::GroupStructure) return run_group_structure(req, curves);

    auto size_r = fam.size();
    if (!size_r.ok) return R::failure(size_r.error.status, size_r.error.message);
    uint64_t size = size_r.value;
    Reduction reduction = parse_reduction(req.reduction);
    Shared shared;
    auto prepared = prepare_all(reduction, size, shared);
    if (!prepared.ok) return R::failure(prepared.error.status, prepared.error.message);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, size ? size : 1));
    std::vector<Accumulator> accumulators;
    for (uint32_t t = 0; t < threads; ++t) accumulators.emplace_back(reduction, &shared);

    auto statuses = parallel_ranges(size, threads, [&](uint32_t thread, uint64_t begin, uint64_t end) -> Status {
        Matrix member;
        Accumulator &acc = accumulators[thread];
        for (uint64_t index = begin; index < end; ++index) {
            if (acc.exhausted(index)) break;
            auto st = fam.member_into(index, member);
            if (!st.ok) return st;
            uint64_t a = member.entries[0], b = member.entries[1];
            switch (op) {
            case Op::PointCount: acc.integer(index, curves.point_count(a, b)); break;
            case Op::Nonsingular: acc.boolean(index, curves.singular_form(a, b) != 0); break;
            case Op::Supersingular:
                acc.boolean(index, curves.singular_form(a, b) != 0 && curves.point_count(a, b) == curves.p + 1);
                break;
            case Op::JInvariant: acc.integer(index, curves.j_invariant(a, b)); break;
            case Op::IsCanonical: acc.boolean(index, curves.is_canonical(a, b)); break;
            case Op::ClassSize: acc.integer(index, curves.class_size(a, b)); break;
            case Op::GroupStructure: return fail(INTERNAL, "group_structure is not a reduction operation");
            }
        }
        return ok();
    });
    for (const auto &status : statuses)
        if (!status.ok) return R::failure(status.error.status, status.error.message);
    return assemble(req, reduction, accumulators, shared);
}

BackendRegistration registration{Backend{
    "elliptic_curves_fp", "generic",
    [] { return true; },
    /* Small primes and misshapen members are accepted here and refused by run() with a message
     * that says what is wrong; primes past what this backend can enumerate are not accepted at
     * all, so the runtime reports that no backend takes the request. */
    [](const Request &req) {
        Op op;
        if (!parse_op(req.op, op)) return false;
        uint64_t p = req.family->prime();
        return p >= 2 && p < (1ULL << 32) && p <= prime_limit(op);
    },
    run,
    0}};

} // namespace
} // namespace lk::elliptic_curves_fp
