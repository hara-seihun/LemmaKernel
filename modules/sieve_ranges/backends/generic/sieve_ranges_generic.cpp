/* sieve_ranges generic backend: the arithmetic functions of a whole interval, by sieving.
 *
 * Members are single natural numbers below 2^32, so every prime factor beyond the square root of
 * the largest member is the last one left after dividing by the smaller primes. The primes up to
 * that square root are sieved once per request; the interval is then walked in blocks, and each
 * prime strikes out its own multiples in the block and is divided out of them. A member costs a
 * few divisions instead of a trial division up to its own square root.
 *
 * Everything the module computes is a product or a count over the factors, so one pass
 * accumulates whichever the requested operation needs. `explicit` batches are not consecutive, so
 * they are trial-divided by the same primes.
 */
#include "../../../../runtime/src/reduce.hpp"

#include <cmath>

namespace lk::sieve_ranges {
namespace {

using R = Result<std::shared_ptr<Object>>;
constexpr int INVALID = 1;
constexpr int INTERNAL = 4;
/* No number below 2^32 has ten distinct prime factors: 2*3*5*7*11*13*17*19*23 = 223092870 and
 * the next primorial is 6469693230. */
constexpr uint64_t MAX_DISTINCT_PRIMES = 10;
constexpr uint64_t BLOCK = 1 << 15;

enum class Op { IsPrime, IsSquarefree, Factorisation, Omega, BigOmega, Totient, Sigma, DivisorCount,
                Mobius, LargestPrimeFactor };

Result<Op> parse_op(const std::string &name) {
    static const std::map<std::string, Op> names{
        {"is_prime", Op::IsPrime}, {"is_squarefree", Op::IsSquarefree}, {"factorisation", Op::Factorisation},
        {"omega", Op::Omega}, {"big_omega", Op::BigOmega}, {"totient", Op::Totient}, {"sigma", Op::Sigma},
        {"divisor_count", Op::DivisorCount}, {"mobius", Op::Mobius},
        {"largest_prime_factor", Op::LargestPrimeFactor}};
    auto it = names.find(name);
    if (it == names.end()) return Result<Op>::failure(INTERNAL, "unknown sieve_ranges operation " + name);
    return Result<Op>::success(it->second);
}

uint64_t isqrt(uint64_t n) {
    auto r = (uint64_t)std::sqrt((double)n);
    while (r && r * r > n) --r;
    while ((r + 1) * (r + 1) <= n) ++r;
    return r;
}

std::vector<uint32_t> primes_upto(uint64_t limit) {
    std::vector<uint32_t> out;
    if (limit < 2) return out;
    std::vector<uint8_t> composite(limit + 1, 0);
    for (uint64_t i = 2; i <= limit; ++i) {
        if (composite[i]) continue;
        out.push_back((uint32_t)i);
        for (uint64_t j = i * i; j <= limit; j += i) composite[j] = 1;
    }
    return out;
}

/* One member under construction: what the factors seen so far say about it. `acc` is the
 * multiplicative accumulator of whichever product the operation wants. */
struct Cell {
    uint64_t n = 0, rem = 0, acc = 1;
    uint32_t omega = 0, big = 0, largest = 0;
    uint8_t squarefree = 1;
};

/* Factors of every member of a block, when the operation asks for them. */
struct Factors {
    std::vector<uint32_t> primes; /* count * MAX_DISTINCT_PRIMES */
    std::vector<uint8_t> exponents;
    std::vector<uint8_t> counts;

    void reset(uint64_t count) {
        primes.assign(count * MAX_DISTINCT_PRIMES, 0);
        exponents.assign(count * MAX_DISTINCT_PRIMES, 0);
        counts.assign(count, 0);
    }
};

/* Record one prime power `p^e` of member `i`. Primes arrive in increasing order. */
inline Status take(Op op, Cell &c, uint64_t p, uint32_t e, Factors *factors, uint64_t i) {
    ++c.omega;
    c.big += e;
    c.largest = (uint32_t)p;
    if (e > 1) c.squarefree = 0;
    switch (op) {
    case Op::Totient: {
        uint64_t f = p - 1;
        for (uint32_t k = 1; k < e; ++k) f *= p;
        c.acc *= f;
        break;
    }
    case Op::Sigma: {
        uint64_t term = 1, power = 1;
        for (uint32_t k = 0; k < e; ++k) {
            power *= p;
            term += power;
        }
        c.acc *= term;
        break;
    }
    case Op::DivisorCount: c.acc *= e + 1; break;
    case Op::Factorisation: {
        uint64_t slot = factors->counts[i];
        if (slot >= MAX_DISTINCT_PRIMES) return fail(INTERNAL, "more distinct prime factors than a member below 2^32 can have");
        factors->primes[i * MAX_DISTINCT_PRIMES + slot] = (uint32_t)p;
        factors->exponents[i * MAX_DISTINCT_PRIMES + slot] = (uint8_t)e;
        factors->counts[i] = (uint8_t)(slot + 1);
        break;
    }
    default: break;
    }
    return ok();
}

/* The operation's value for a finished member. */
inline uint64_t value_of(Op op, const Cell &c) {
    if (c.n == 0) return op == Op::Mobius ? 1 : 0; /* mu(0) = 0, shifted to 1; everything else 0 */
    switch (op) {
    case Op::IsPrime: return c.omega == 1 && c.big == 1;
    case Op::IsSquarefree: return c.squarefree;
    case Op::Omega: return c.omega;
    case Op::BigOmega: return c.big;
    case Op::Totient:
    case Op::Sigma:
    case Op::DivisorCount: return c.acc;
    case Op::Mobius: return c.squarefree ? (c.omega % 2 ? 0 : 2) : 1;
    case Op::LargestPrimeFactor: return c.largest;
    default: return 0;
    }
}

inline void start(Cell &c, uint64_t n) {
    c = Cell{};
    c.n = n;
    c.rem = n;
}

/* Finish a member whose small factors have all been divided out: what is left is 1 or a prime. */
inline Status finish(Op op, Cell &c, Factors *factors, uint64_t i) {
    if (c.n && c.rem > 1) return take(op, c, c.rem, 1, factors, i);
    return ok();
}

/* Factor the members with indices [begin, end) of a family into `cells`, by sieving a contiguous
 * range or by trial division of an explicit batch. */
Status factor_members(Op op, const Family &family, const std::vector<uint32_t> &primes, uint64_t begin,
                      uint64_t end, std::vector<Cell> &cells, Factors *factors) {
    uint64_t count = end - begin;
    cells.assign(count, Cell{});
    if (factors) factors->reset(count);

    if (family.kind == Family::Kind::Range) {
        uint64_t lo = family.a + begin;
        for (uint64_t i = 0; i < count; ++i) start(cells[i], lo + i);
        for (uint32_t p : primes) {
            /* The first multiple of p at or above lo, never 0: 0 is divisible by everything. */
            uint64_t first = std::max<uint64_t>(p, ((lo + p - 1) / p) * p);
            for (uint64_t n = first; n < lo + count; n += p) {
                Cell &c = cells[n - lo];
                uint32_t e = 0;
                while (c.rem % p == 0) {
                    c.rem /= p;
                    ++e;
                }
                auto st = take(op, c, p, e, factors, n - lo);
                if (!st.ok) return st;
            }
        }
    } else {
        const Matrix &batch = *family.data;
        for (uint64_t i = 0; i < count; ++i) {
            Cell &c = cells[i];
            start(c, batch.entries[begin + i]);
            for (uint32_t p : primes) {
                if ((uint64_t)p * p > c.rem) break;
                if (c.rem % p) continue;
                uint32_t e = 0;
                while (c.rem % p == 0) {
                    c.rem /= p;
                    ++e;
                }
                auto st = take(op, c, p, e, factors, i);
                if (!st.ok) return st;
            }
        }
    }
    for (uint64_t i = 0; i < count; ++i) {
        auto st = finish(op, cells[i], factors, i);
        if (!st.ok) return st;
    }
    return ok();
}

/* The largest member, which fixes how far the prime sieve has to run. */
Result<uint64_t> largest_member(const Family &family) {
    if (family.kind == Family::Kind::Range) return Result<uint64_t>::success(family.b - 1);
    uint64_t best = 0;
    for (Entry e : family.data->entries) best = std::max<uint64_t>(best, e);
    return Result<uint64_t>::success(best);
}

Status validate(const Family &family) {
    if (family.prime() != NATURALS)
        return fail(INVALID, "sieve_ranges operations need members that are natural numbers (lk.naturals), not elements of a field");
    if (family.rows() != 1 || family.cols() != 1)
        return fail(INVALID, "sieve_ranges operations need 1 x 1 members, one natural number each; these are " +
                                 std::to_string(family.rows()) + " x " + std::to_string(family.cols()));
    return ok();
}

R run_factorisation(const Request &req, const std::vector<uint32_t> &primes, uint64_t size) {
    uint64_t blocks = (size + BLOCK - 1) / BLOCK;
    std::vector<std::vector<uint64_t>> pairs(blocks);
    std::vector<std::vector<uint8_t>> counts(blocks);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, blocks ? blocks : 1));
    auto statuses = parallel_ranges(blocks, threads, [&](uint32_t, uint64_t first, uint64_t last) -> Status {
        std::vector<Cell> cells;
        Factors factors;
        for (uint64_t b = first; b < last; ++b) {
            uint64_t begin = b * BLOCK, end = std::min(size, begin + BLOCK);
            auto st = factor_members(Op::Factorisation, *req.family, primes, begin, end, cells, &factors);
            if (!st.ok) return st;
            counts[b] = factors.counts;
            auto &out = pairs[b];
            out.clear();
            for (uint64_t i = 0; i < end - begin; ++i)
                for (uint64_t j = 0; j < factors.counts[i]; ++j) {
                    out.push_back(factors.primes[i * MAX_DISTINCT_PRIMES + j]);
                    out.push_back(factors.exponents[i * MAX_DISTINCT_PRIMES + j]);
                }
        }
        return ok();
    });
    for (const auto &status : statuses)
        if (!status.ok) return R::failure(status.error.status, status.error.message);

    auto f = std::make_shared<Factorisation>();
    f->count = size;
    f->offsets.reserve(size + 1);
    f->offsets.push_back(0);
    uint64_t total = 0;
    for (uint64_t b = 0; b < blocks; ++b)
        for (uint8_t c : counts[b]) f->offsets.push_back(total += c);
    f->pairs.reserve(2 * total);
    for (uint64_t b = 0; b < blocks; ++b) f->pairs.insert(f->pairs.end(), pairs[b].begin(), pairs[b].end());

    auto o = std::make_shared<Object>();
    o->kind = "sieve_ranges.factorisation";
    o->factorisation = f;
    return R::success(o);
}

R run(const Request &req) {
    auto valid = validate(*req.family);
    if (!valid.ok) return R::failure(valid.error.status, valid.error.message);
    auto parsed = parse_op(req.op);
    if (!parsed.ok) return R::failure(parsed.error.status, parsed.error.message);
    Op op = parsed.value;

    auto size_r = req.family->size();
    if (!size_r.ok) return R::failure(size_r.error.status, size_r.error.message);
    uint64_t size = size_r.value;
    auto largest = largest_member(*req.family);
    if (!largest.ok) return R::failure(largest.error.status, largest.error.message);
    std::vector<uint32_t> primes = primes_upto(isqrt(largest.value));

    if (op == Op::Factorisation) {
        if (req.reduction != "all") return R::failure(INVALID, "factorisation values only reduce with `all`");
        return run_factorisation(req, primes, size);
    }

    Reduction reduction = parse_reduction(req.reduction);
    Shared shared;
    auto prepared = prepare_all(reduction, size, shared);
    if (!prepared.ok) return R::failure(prepared.error.status, prepared.error.message);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, size ? size : 1));
    std::vector<Accumulator> accumulators;
    for (uint32_t t = 0; t < threads; ++t) accumulators.emplace_back(reduction, &shared);
    bool boolean = op == Op::IsPrime || op == Op::IsSquarefree;

    auto statuses = parallel_ranges(size, threads, [&](uint32_t thread, uint64_t begin, uint64_t end) -> Status {
        Accumulator &acc = accumulators[thread];
        std::vector<Cell> cells;
        for (uint64_t block = begin; block < end; block += BLOCK) {
            if (acc.exhausted(block)) break;
            uint64_t last = std::min(end, block + BLOCK);
            auto st = factor_members(op, *req.family, primes, block, last, cells, nullptr);
            if (!st.ok) return st;
            for (uint64_t i = 0; i < last - block; ++i) {
                uint64_t value = value_of(op, cells[i]);
                if (boolean) acc.boolean(block + i, value != 0);
                else acc.integer(block + i, value);
            }
        }
        return ok();
    });
    for (const auto &status : statuses)
        if (!status.ok) return R::failure(status.error.status, status.error.message);
    return assemble(req, reduction, accumulators, shared);
}

BackendRegistration registration{Backend{
    "sieve_ranges", "generic",
    [] { return true; },
    [](const Request &) { return true; },
    run,
    0}};

} // namespace
} // namespace lk::sieve_ranges
