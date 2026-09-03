/* Portable additive-combinatorics predicates on finite sets of naturals.
 *
 * The family is walked depth first, one element at a time, and every quantity is carried along
 * the prefix: adding x to a prefix P costs O(|P|) because the Schur count, the sumset, the
 * difference multiplicities and the progressions through x are all updates rather than
 * recomputations. Membership is a byte per value of the ambient group, so a lookup is one load.
 *
 * Sum-freedom, the Sidon property, progression-freedom and small doubling are subset-closed:
 * once a prefix fails, every set below it fails, and the whole subtree is answered false without
 * being visited. The integer operations have no such structure and are computed at the leaves.
 *
 * Difference codes match the Lean reference: (x + n - y) % n in Z/n, and x + shift - y over the
 * integers, where shift is an upper bound for the elements. Only counts of codes escape, so the
 * shift the backend picks (the largest dictionary element) need not be the reference's. */
#include "../../../../runtime/src/reduce.hpp"

#include <algorithm>

namespace lk::sum_free_and_additive {
namespace {

using R = Result<std::shared_ptr<Object>>;
constexpr int INVALID = 1;
constexpr int INTERNAL = 4;
/* One byte (membership, sumset) or four (difference multiplicities) per value of the ambient
 * group, per thread. */
constexpr uint64_t MAX_SPAN = 1ULL << 22;

enum class Op { IsSumFree, IsSidon, IsApFree, SumsetSize, DifferenceSetSize, SchurTripleCount,
                MaxDifferenceMultiplicity, IsSmallDoubling };

Result<Op> parse_op(const std::string &name) {
    static const std::map<std::string, Op> names{
        {"is_sum_free", Op::IsSumFree}, {"is_sidon", Op::IsSidon}, {"is_ap_free", Op::IsApFree},
        {"sumset_size", Op::SumsetSize}, {"difference_set_size", Op::DifferenceSetSize},
        {"schur_triple_count", Op::SchurTripleCount},
        {"max_difference_multiplicity", Op::MaxDifferenceMultiplicity},
        {"is_small_doubling", Op::IsSmallDoubling}};
    auto it = names.find(name);
    if (it == names.end()) return Result<Op>::failure(INTERNAL, "unknown sum_free_and_additive operation " + name);
    return Result<Op>::success(it->second);
}

bool is_boolean(Op op) {
    return op == Op::IsSumFree || op == Op::IsSidon || op == Op::IsApFree || op == Op::IsSmallDoubling;
}
bool needs_sumset(Op op) { return op == Op::SumsetSize || op == Op::IsSmallDoubling; }
bool needs_differences(Op op) {
    return op == Op::DifferenceSetSize || op == Op::IsSidon || op == Op::MaxDifferenceMultiplicity;
}

/* The ambient group: the integers when modulus == 0, otherwise Z/modulus. Every value the walker
 * indexes with (an element, a sum, a difference code) is below `span`. */
struct Ambient {
    uint64_t modulus = 0, shift = 0, span = 0;

    uint64_t add(uint64_t x, uint64_t y) const { return modulus ? (x + y) % modulus : x + y; }
    uint64_t code(uint64_t x, uint64_t y) const { /* x - y */
        return modulus ? (x + modulus - y) % modulus : x + shift - y;
    }
    uint64_t zero_code() const { return modulus ? 0 : shift; }
    /* x - y as an element, when it is one: over the integers a smaller x has no difference. */
    bool difference(uint64_t x, uint64_t y, uint64_t &out) const {
        if (modulus) { out = (x + modulus - y) % modulus; return true; }
        if (x < y) return false;
        out = x - y;
        return true;
    }
};

/* One member's worth of incremental state, one stack entry per element of the prefix. */
struct Level {
    uint64_t schur = 0;        /* ordered pairs of the prefix whose sum is in the prefix */
    uint64_t sumset = 0;       /* |P + P| */
    uint64_t differences = 0;  /* |P - P| */
    uint64_t multiplicity = 0; /* the most popular nonzero difference of P */
    bool has_ap = false;
    uint64_t sum_undo = 0, diff_undo = 0; /* marks into the undo logs */
};

struct Walker : Family::Visitor {
    Op op;
    Reduction reduction;
    Ambient amb;
    uint64_t length, bound_num, bound_den, member_size;
    bool prune;
    Shared *shared;
    Accumulator acc;

    std::vector<uint32_t> vals;              /* the prefix, in the order pushed */
    std::vector<uint8_t> present;            /* present[v]: v is in the prefix */
    std::vector<uint8_t> in_sumset;          /* in_sumset[s]: s is a sum of the prefix */
    std::vector<uint32_t> multiplicities;    /* per difference code, over the prefix */
    std::vector<uint32_t> sum_log, diff_log; /* values to undo on pop */
    std::vector<Level> levels;

    Walker(Op o, Reduction r, const Ambient &a, uint64_t len, uint64_t num, uint64_t den,
           uint64_t rows, Shared *s)
        : op(o), reduction(r), amb(a), length(len), bound_num(num), bound_den(den),
          member_size(rows), prune(is_boolean(o)), shared(s), acc(r, s) {
        present.assign(amb.span, 0);
        if (needs_sumset(op)) in_sumset.assign(amb.span, 0);
        if (needs_differences(op)) multiplicities.assign(amb.span, 0);
        levels.push_back(Level{});
    }

    bool member(uint64_t v) const { return v < amb.span && present[v]; }

    /* Terms of the progression through x with common difference d that lie in the prefix, both
     * ways, counted up to `length`. */
    uint64_t progression_through(uint64_t x, uint64_t d) const {
        uint64_t total = 1, cur = x;
        for (uint64_t i = 1; i < length; ++i) {
            cur = amb.add(cur, d);
            if (!member(cur)) break;
            ++total;
        }
        cur = x;
        for (uint64_t i = total; i < length; ++i) {
            uint64_t next;
            if (!amb.difference(cur, d, next) || !member(next)) break;
            cur = next;
            ++total;
        }
        return total;
    }

    /* A progression of `length` terms through x: it has a neighbour in the prefix, so its common
     * difference is the distance from x to some other element. */
    bool ap_through(uint64_t x) const {
        for (uint32_t y : vals) {
            if (y == x) continue;
            uint64_t d = amb.modulus ? amb.code(y, x) : (y > x ? y - x : x - y);
            if (progression_through(x, d) >= length) return true;
        }
        return false;
    }

    Step push(const Entry *row, uint64_t first, uint64_t) override {
        uint64_t x = row[0];
        Level next = levels.back();
        next.sum_undo = sum_log.size();
        next.diff_undo = diff_log.size();
        const std::vector<uint32_t> &prefix = vals; /* still P; x is added below */

        if (op == Op::IsSumFree || op == Op::SchurTripleCount) {
            /* New ordered pairs: (x,y) and (y,x) for y in P, and (x,x); plus the pairs of P whose
             * sum is x, which only now lands in the set. */
            for (uint32_t y : prefix) {
                uint64_t s = amb.add(x, y);
                if (s == x || member(s)) next.schur += 2;
                uint64_t d;
                if (amb.difference(x, y, d) && member(d)) next.schur += 1;
            }
            uint64_t twice = amb.add(x, x);
            if (twice == x || member(twice)) next.schur += 1;
        }
        if (needs_sumset(op)) {
            for (uint32_t y : prefix) {
                uint64_t s = amb.add(x, y);
                if (!in_sumset[s]) { in_sumset[s] = 1; sum_log.push_back((uint32_t)s); ++next.sumset; }
            }
            uint64_t twice = amb.add(x, x);
            if (!in_sumset[twice]) { in_sumset[twice] = 1; sum_log.push_back((uint32_t)twice); ++next.sumset; }
        }
        if (needs_differences(op)) {
            for (uint32_t y : prefix) {
                for (uint64_t c : {amb.code(x, y), amb.code(y, x)}) {
                    uint32_t m = ++multiplicities[c];
                    diff_log.push_back((uint32_t)c);
                    if (m == 1) ++next.differences;
                    next.multiplicity = std::max<uint64_t>(next.multiplicity, m);
                }
            }
            uint64_t zero = amb.zero_code(); /* the pair (x,x); its difference is never counted */
            if (++multiplicities[zero] == 1) ++next.differences;
            diff_log.push_back((uint32_t)zero);
        }

        vals.push_back((uint32_t)x);
        present[x] = 1;
        if (op == Op::IsApFree && !next.has_ap) next.has_ap = ap_through(x);
        levels.push_back(next);

        if (!prune) return Step::Descend;
        if (acc.exhausted(first)) return Step::Skip;
        return failed(next) ? Step::Skip : Step::Descend;
    }

    /* Whether a prefix already decides the answer false for every set below it. */
    bool failed(const Level &l) const {
        switch (op) {
        case Op::IsSumFree: return l.schur != 0;
        case Op::IsSidon: return l.multiplicity > 1;
        case Op::IsApFree: return l.has_ap;
        case Op::IsSmallDoubling: return (unsigned __int128)l.sumset * bound_den >
                                         (unsigned __int128)bound_num * member_size;
        default: return false;
        }
    }

    void pop() override {
        const Level &l = levels.back();
        while (sum_log.size() > l.sum_undo) { in_sumset[sum_log.back()] = 0; sum_log.pop_back(); }
        while (diff_log.size() > l.diff_undo) { --multiplicities[diff_log.back()]; diff_log.pop_back(); }
        present[vals.back()] = 0;
        vals.pop_back();
        levels.pop_back();
    }

    void leaf(uint64_t index) override {
        const Level &l = levels.back();
        switch (op) {
        case Op::IsSumFree: acc.boolean(index, l.schur == 0); break;
        case Op::IsSidon: acc.boolean(index, l.multiplicity <= 1); break;
        case Op::IsApFree: acc.boolean(index, !l.has_ap); break;
        case Op::IsSmallDoubling: acc.boolean(index, !failed(l)); break;
        case Op::SumsetSize: acc.integer(index, l.sumset); break;
        case Op::DifferenceSetSize: acc.integer(index, l.differences); break;
        case Op::SchurTripleCount: acc.integer(index, l.schur); break;
        case Op::MaxDifferenceMultiplicity: acc.integer(index, l.multiplicity); break;
        }
    }

    void take_all(uint64_t first, uint64_t n) override { acc.booleans(first, n, true); }
    void skip_all(uint64_t first, uint64_t n) override { acc.booleans(first, n, false); }
};

/* What a member (`explicit`) or the dictionary (`subsets`, `subsets_of`) has to be: one element
 * per row, every element inside the ambient group, no element twice. The elements bound the span
 * of the ambient group, so the first pass measures them and the duplicate check follows once the
 * span is known and a byte per value is affordable. */
const char *set_name(const Family &fam) {
    return fam.kind == Family::Kind::Explicit ? "member" : "dictionary";
}

Status scan(const Family &fam, uint64_t modulus, uint64_t &largest) {
    const Matrix &data = *fam.data;
    if (data.cols != 1)
        return fail(INVALID, std::string(set_name(fam)) + " must have one column: one element per row");
    for (uint64_t i = 0; i < data.count * data.rows; ++i) {
        uint64_t v = data.entries[i];
        if (modulus && v >= modulus)
            return fail(INVALID, std::string(set_name(fam)) + " has an element that is not below the modulus");
        largest = std::max(largest, v);
    }
    return ok();
}

Status check_distinct(const Family &fam, uint64_t span) {
    const Matrix &data = *fam.data;
    /* one row per element for a dictionary; `rows` of them per member of an explicit batch */
    uint64_t sets = fam.kind == Family::Kind::Explicit ? data.count : 1;
    uint64_t per_set = fam.kind == Family::Kind::Explicit ? data.rows : data.count;
    std::vector<uint8_t> seen(span, 0);
    for (uint64_t s = 0; s < sets; ++s) {
        const Entry *first = data.entries.data() + s * per_set;
        bool repeated = false;
        for (uint64_t i = 0; i < per_set && !repeated; ++i) repeated = seen[first[i]]++;
        for (uint64_t i = 0; i < per_set; ++i) seen[first[i]] = 0;
        if (repeated) return fail(INVALID, std::string(set_name(fam)) + " contains a duplicate element");
    }
    return ok();
}

R run(const Request &req) {
    const Family &fam = *req.family;
    if (fam.kind != Family::Kind::Explicit && fam.kind != Family::Kind::Subsets &&
        fam.kind != Family::Kind::SubsetsOf)
        return R::failure(INVALID, "sum_free_and_additive is defined on explicit, subsets and subsets_of families only");
    if (fam.prime() != NATURALS)
        return R::failure(INVALID, "sum_free_and_additive members must be lk.naturals");
    auto parsed = parse_op(req.op);
    if (!parsed.ok) return R::failure(parsed.error.status, parsed.error.message);
    Op op = parsed.value;

    uint64_t modulus = req.int_args.at("modulus");
    uint64_t length = req.int_args.count("length") ? req.int_args.at("length") : 0;
    uint64_t bound_num = req.int_args.count("bound_num") ? req.int_args.at("bound_num") : 0;
    uint64_t bound_den = req.int_args.count("bound_den") ? req.int_args.at("bound_den") : 1;
    if (op == Op::IsApFree && length < 2)
        return R::failure(INVALID, "is_ap_free needs a progression length of at least 2");
    if (op == Op::IsSmallDoubling && bound_den < 1)
        return R::failure(INVALID, "is_small_doubling needs bound_den >= 1");

    uint64_t largest = 0;
    auto scanned = scan(fam, modulus, largest);
    if (!scanned.ok) return R::failure(scanned.error.status, scanned.error.message);
    Ambient amb;
    amb.modulus = modulus;
    amb.shift = modulus ? 0 : largest;
    amb.span = modulus ? modulus : 2 * largest + 1;
    if (amb.span > MAX_SPAN)
        return R::failure(INVALID, "the generic backend needs a modulus below 2^22, or elements below 2^21 when the modulus is 0");
    auto distinct = check_distinct(fam, amb.span);
    if (!distinct.ok) return R::failure(distinct.error.status, distinct.error.message);

    auto size_r = fam.size();
    if (!size_r.ok) return R::failure(size_r.error.status, size_r.error.message);
    auto tops_r = fam.top_count();
    if (!tops_r.ok) return R::failure(tops_r.error.status, tops_r.error.message);
    Reduction reduction = parse_reduction(req.reduction);
    Shared shared;
    auto prepared = prepare_all(reduction, size_r.value, shared);
    if (!prepared.ok) return R::failure(prepared.error.status, prepared.error.message);

    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, tops_r.value));
    std::vector<Walker> walkers;
    walkers.reserve(threads);
    for (uint32_t t = 0; t < threads; ++t)
        walkers.emplace_back(op, reduction, amb, length, bound_num, bound_den, fam.rows(), &shared);
    auto statuses = parallel_ranges(tops_r.value, threads, [&](uint32_t t, uint64_t begin, uint64_t end) {
        return fam.enumerate(walkers[t], begin, end);
    });
    for (const auto &st : statuses)
        if (!st.ok) return R::failure(st.error.status, st.error.message);

    std::vector<Accumulator> accs;
    for (auto &w : walkers) accs.push_back(w.acc);
    return assemble(req, reduction, accs, shared);
}

BackendRegistration registration{Backend{
    "sum_free_and_additive", "generic",
    [] { return true; },
    /* The whole module's domain: what a set can arrive as. Members that are not naturals, or
     * are not one element per row, are refused by name in run(). */
    [](const Request &req) {
        return req.family->kind == Family::Kind::Explicit || req.family->kind == Family::Kind::Subsets ||
               req.family->kind == Family::Kind::SubsetsOf;
    },
    run,
    0}};

} // namespace
} // namespace lk::sum_free_and_additive
