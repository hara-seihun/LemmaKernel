/* Portable continued fractions, fundamental units and class numbers over families of naturals.
 *
 * Every member is one natural number below 2^32, so the whole of the continued-fraction walk of
 * sqrt(d) runs in 64-bit arithmetic: the complete quotients (m + sqrt d)/q have 0 < m < sqrt d
 * and 0 < q < 2 sqrt d, and the partial quotients are below 2 sqrt d. What does grow is the
 * convergent, so the recurrence is accumulated in 128-bit arithmetic and the walk stops as soon
 * as a coordinate passes 2^64 - 1: the request is refused, never truncated.
 *
 * There is nothing a request's members share, so each one is answered on its own and the work
 * is split over member ranges. */
#include "../../../../runtime/src/reduce.hpp"

#include <cmath>

namespace lk::continued_fractions_and_pell {
namespace {

using R = Result<std::shared_ptr<Object>>;
constexpr int INVALID = 1;
constexpr int INTERNAL = 4;

enum class Op { Period, PeriodMax, PeriodSum, Expansion, NegativePell, Unit, Pell, ClassNumber };

bool parse_op(const std::string &name, Op &out) {
    static const std::map<std::string, Op> names{
        {"cf_period", Op::Period}, {"cf_period_max", Op::PeriodMax}, {"cf_period_sum", Op::PeriodSum},
        {"cf_expansion", Op::Expansion}, {"negative_pell", Op::NegativePell},
        {"fundamental_unit", Op::Unit}, {"pell_fundamental", Op::Pell}, {"class_number", Op::ClassNumber}};
    auto it = names.find(name);
    if (it == names.end()) return false;
    out = it->second;
    return true;
}

bool materialised(Op op) { return op == Op::Expansion || op == Op::Unit || op == Op::Pell; }

/* floor(sqrt(n)) for n < 2^32, corrected so that no rounding of the double matters. */
uint64_t isqrt(uint64_t n) {
    if (n == 0) return 0;
    uint64_t x = (uint64_t)std::sqrt((double)n);
    while (x > 0 && x * x > n) --x;
    while ((x + 1) * (x + 1) <= n) ++x;
    return x;
}

/* The period a_1 .. a_L of sqrt(d) in one pass, without storing it. */
struct Period {
    uint64_t length = 0, largest = 0, total = 0;
};

Period period_stats(uint64_t d) {
    Period out;
    uint64_t a0 = isqrt(d);
    if (a0 * a0 == d) return out;
    for (uint64_t m = a0, q = d - a0 * a0;;) {
        uint64_t a = (a0 + m) / q;
        ++out.length;
        out.largest = std::max(out.largest, a);
        out.total += a;
        if (q == 1) return out;
        m = q * a - m;
        q = (d - m * m) / q;
    }
}

/* a_0 followed by exactly one period; just [a_0] for a perfect square. */
void expansion_terms(uint64_t d, std::vector<uint64_t> &out) {
    uint64_t a0 = isqrt(d);
    out.assign(1, a0);
    if (a0 * a0 == d) return;
    for (uint64_t m = a0, q = d - a0 * a0;;) {
        uint64_t a = (a0 + m) / q;
        out.push_back(a);
        if (q == 1) return;
        m = q * a - m;
        q = (d - m * m) / q;
    }
}

enum class UnitState { None, Found, TooWide };

/* The fundamental unit of Z[sqrt d] as (x, y) with x^2 - d y^2 = (-1)^L.
 *
 * It is the convergent p_{L-1}/q_{L-1}, so the walk feeds a_0, a_1, ..., a_{L-1} into the
 * convergent recurrence and stops when it reaches the last period term, which is the one whose
 * complete quotient has denominator 1. */
UnitState fundamental_unit(uint64_t d, uint64_t &x, uint64_t &y, bool &negative) {
    uint64_t a0 = isqrt(d);
    if (a0 * a0 == d) return UnitState::None;
    unsigned __int128 p = a0, q = 1, pp = 1, qq = 0; /* the convergent of [a_0] */
    uint64_t length = 0;
    for (uint64_t m = a0, s = d - a0 * a0;;) {
        uint64_t a = (a0 + m) / s;
        ++length;
        if (s == 1) break; /* a is a_L: the convergent already holds p_{L-1}/q_{L-1} */
        unsigned __int128 np = a * p + pp, nq = a * q + qq;
        if (np > UINT64_MAX || nq > UINT64_MAX) return UnitState::TooWide;
        pp = p; qq = q; p = np; q = nq;
        m = s * a - m;
        s = (d - m * m) / s;
    }
    x = (uint64_t)p;
    y = (uint64_t)q;
    negative = length % 2 == 1;
    return UnitState::Found;
}

/* The least x, y > 0 with x^2 - d y^2 = 1: the unit, or its square when the unit has norm -1.
 * (x + y sqrt d)^2 = (2 x^2 + 1) + 2 x y sqrt d, and y < x, so x < 2^32 is what both need. */
UnitState pell_fundamental(uint64_t d, uint64_t &x, uint64_t &y) {
    bool negative = false;
    UnitState state = fundamental_unit(d, x, y, negative);
    if (state != UnitState::Found || !negative) return state;
    if (x >= (1ULL << 32)) return UnitState::TooWide;
    unsigned __int128 nx = 2 * (unsigned __int128)x * x + 1, ny = 2 * (unsigned __int128)x * y;
    if (nx > UINT64_MAX || ny > UINT64_MAX) return UnitState::TooWide;
    x = (uint64_t)nx;
    y = (uint64_t)ny;
    return UnitState::Found;
}

uint64_t gcd64(uint64_t a, uint64_t b) {
    while (b) {
        uint64_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

/* h(-n): primitive reduced positive definite forms (a, b, c) with b^2 - 4ac = -n, that is
 * -a < b <= a <= c and b >= 0 when a = c. Reduction pairs (a, b, c) with (a, -b, c), so a form
 * with 0 < b < a < c is counted twice. n = 4ac - b^2 >= 3a^2 bounds the outer loop. */
uint64_t class_number(uint64_t n) {
    if (n == 0 || (n % 4 != 0 && n % 4 != 3)) return 0;
    uint64_t total = 0;
    for (uint64_t a = 1; a <= isqrt(n / 3); ++a)
        for (uint64_t b = 0; b <= a; ++b) {
            uint64_t t = b * b + n;
            if (t % (4 * a)) continue;
            uint64_t c = t / (4 * a);
            if (c < a || gcd64(gcd64(a, b), c) != 1) continue;
            total += (b == 0 || b == a || c == a) ? 1 : 2;
        }
    return total;
}

/* The natural number a member carries. A range member is its index offset by `a`; anything else
 * is read out of the family. */
struct Members {
    const Family &fam;
    bool range;
    explicit Members(const Family &f) : fam(f), range(f.kind == Family::Kind::Range) {}
    Status at(uint64_t index, Matrix &scratch, uint64_t &out) const {
        if (range) {
            out = fam.a + index;
            return ok();
        }
        auto st = fam.member_into(index, scratch);
        if (!st.ok) return st;
        out = scratch.entries[0];
        return ok();
    }
};

Status check_family(const Family &fam) {
    if (fam.prime() != NATURALS || fam.rows() != 1 || fam.cols() != 1)
        return fail(INVALID, "continued_fractions_and_pell needs 1 x 1 natural-number members, got " +
                                 std::to_string(fam.rows()) + " x " + std::to_string(fam.cols()));
    return ok();
}

/* cf_expansion, fundamental_unit and pell_fundamental produce one object per member, so they
 * only reduce with `all` and the whole family is materialised. */
R run_materialised(const Request &req, Op op, uint64_t size) {
    if (req.reduction != "all")
        return R::failure(INVALID, "continued_fractions_and_pell." + req.op + " values only reduce with `all`");
    if (size > (1ULL << 30)) return R::failure(INVALID, "family too large to materialise");
    Members members(*req.family);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, size ? size : 1));

    if (op == Op::Expansion) {
        std::vector<std::vector<uint64_t>> terms(size);
        auto statuses = parallel_ranges(size, threads, [&](uint32_t, uint64_t begin, uint64_t end) -> Status {
            Matrix scratch;
            for (uint64_t index = begin; index < end; ++index) {
                uint64_t d = 0;
                auto st = members.at(index, scratch, d);
                if (!st.ok) return st;
                expansion_terms(d, terms[index]);
            }
            return ok();
        });
        for (const auto &status : statuses)
            if (!status.ok) return R::failure(status.error.status, status.error.message);
        auto e = std::make_shared<ContinuedFractions>();
        e->count = size;
        e->offsets.reserve(size + 1);
        e->offsets.push_back(0);
        for (const auto &t : terms) {
            e->values.insert(e->values.end(), t.begin(), t.end());
            e->offsets.push_back(e->values.size());
        }
        auto o = std::make_shared<Object>();
        o->kind = "continued_fractions_and_pell.expansion";
        o->continued_fractions = e;
        return R::success(o);
    }

    auto u = std::make_shared<QuadraticUnits>();
    u->count = size;
    u->solvable.assign(size, 0);
    u->negative.assign(size, 0);
    u->pairs.assign(2 * size, 0);
    auto statuses = parallel_ranges(size, threads, [&](uint32_t, uint64_t begin, uint64_t end) -> Status {
        Matrix scratch;
        for (uint64_t index = begin; index < end; ++index) {
            uint64_t d = 0;
            auto st = members.at(index, scratch, d);
            if (!st.ok) return st;
            uint64_t x = 0, y = 0;
            bool negative = false;
            UnitState state = op == Op::Unit ? fundamental_unit(d, x, y, negative) : pell_fundamental(d, x, y);
            if (state == UnitState::TooWide)
                return fail(INVALID, "continued_fractions_and_pell." + req.op + " for d = " + std::to_string(d) +
                                         " exceeds 2^64 - 1; no member is truncated, so the request is refused");
            if (state == UnitState::None) continue;
            u->solvable[index] = 1;
            u->negative[index] = op == Op::Unit && negative ? 1 : 0;
            u->pairs[2 * index] = x;
            u->pairs[2 * index + 1] = y;
        }
        return ok();
    });
    for (const auto &status : statuses)
        if (!status.ok) return R::failure(status.error.status, status.error.message);
    auto o = std::make_shared<Object>();
    o->kind = "continued_fractions_and_pell.unit";
    o->quadratic_units = u;
    return R::success(o);
}

R run(const Request &req) {
    Op op;
    if (!parse_op(req.op, op))
        return R::failure(INTERNAL, "unknown continued_fractions_and_pell operation " + req.op);
    auto valid = check_family(*req.family);
    if (!valid.ok) return R::failure(valid.error.status, valid.error.message);
    auto size_r = req.family->size();
    if (!size_r.ok) return R::failure(size_r.error.status, size_r.error.message);
    uint64_t size = size_r.value;
    if (materialised(op)) return run_materialised(req, op, size);

    Members members(*req.family);
    Reduction reduction = parse_reduction(req.reduction);
    Shared shared;
    auto prepared = prepare_all(reduction, size, shared);
    if (!prepared.ok) return R::failure(prepared.error.status, prepared.error.message);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, size ? size : 1));
    std::vector<Accumulator> accumulators;
    for (uint32_t t = 0; t < threads; ++t) accumulators.emplace_back(reduction, &shared);

    auto statuses = parallel_ranges(size, threads, [&](uint32_t thread, uint64_t begin, uint64_t end) -> Status {
        Accumulator &acc = accumulators[thread];
        Matrix scratch;
        for (uint64_t index = begin; index < end; ++index) {
            if (acc.exhausted(index)) break;
            uint64_t value = 0;
            auto st = members.at(index, scratch, value);
            if (!st.ok) return st;
            if (op == Op::ClassNumber) {
                acc.integer(index, class_number(value));
                continue;
            }
            Period period = period_stats(value);
            switch (op) {
            case Op::Period: acc.integer(index, period.length); break;
            case Op::PeriodMax: acc.integer(index, period.largest); break;
            case Op::PeriodSum: acc.integer(index, period.total); break;
            case Op::NegativePell: acc.boolean(index, period.length % 2 == 1); break;
            default: return fail(INTERNAL, "continued_fractions_and_pell: materialised op in the reduction path");
            }
        }
        return ok();
    });
    for (const auto &status : statuses)
        if (!status.ok) return R::failure(status.error.status, status.error.message);
    return assemble(req, reduction, accumulators, shared);
}

BackendRegistration registration{Backend{
    "continued_fractions_and_pell", "generic",
    [] { return true; },
    /* Any family of naturals is accepted here; members of the wrong shape are refused by run()
     * with a message that says what is wrong. A family over a prime field is not this module's
     * business at all, so it is not accepted and the runtime reports that no backend takes it. */
    [](const Request &req) { return req.family->prime() == NATURALS; },
    run,
    0}};

} // namespace
} // namespace lk::continued_fractions_and_pell
