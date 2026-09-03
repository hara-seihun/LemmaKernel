/* spreads_and_partitions generic backend: portable C++ for any prime p < 2^32.
 *
 * Every operation reads one member as a set of subspaces of F_p^n and asks how they meet, so the
 * whole backend is one depth-first walk of the family that keeps an echelon basis per component
 * of the current prefix. A family's members are pushed one row at a time, and a row is one
 * component: when the new component meets an earlier one, or is zero, or changes the dimension,
 * or pushes the covered count past p^n - 1, no completion of that prefix can be a partial spread
 * (a spread, a partition), so the whole subtree is skipped without visiting a member of it. That
 * is where the time goes: on the 324,632 five-sets of lines of PG(3,2) the walk decides all of
 * them from a few thousand nodes.
 *
 * Costs, so the next backend knows what to beat: pushing a component costs h reductions to build
 * its basis and at most depth * h more to test it against the earlier components, all of them
 * n-entry rows. When p^n is small (the interesting case: p = 2, n <= 20) a component is a set of
 * (p^h - 1)/(p - 1) projective points and "meets trivially" is one AND of two bitsets; that
 * replaces every elimination above and is the obvious next backend.
 *
 * `intersecting_pairs` is an integer operation, so no subtree can be decided early and the walk
 * never prunes; it costs the same per node and visits every member. */
#include "../../../gfp/backends/generic/field.hpp"

namespace lk::spreads {
namespace {

using R = Result<std::shared_ptr<Object>>;
constexpr int INVALID = 1;
constexpr int INTERNAL = 4;

enum class Op { PartialSpread, Spread, Partition, IntersectingPairs, Packing };

bool parse_op(const std::string &name, Op &op) {
    static const std::map<std::string, Op> table{
        {"is_partial_spread", Op::PartialSpread}, {"is_spread", Op::Spread},
        {"is_vector_space_partition", Op::Partition}, {"intersecting_pairs", Op::IntersectingPairs},
        {"is_packing", Op::Packing}};
    auto it = table.find(name);
    if (it == table.end()) return false;
    op = it->second;
    return true;
}

/* p^e, or false on overflow of 128 bits. */
bool pow_checked(uint64_t p, uint64_t e, unsigned __int128 &out) {
    unsigned __int128 r = 1;
    for (uint64_t i = 0; i < e; ++i) {
        if (r > (~(unsigned __int128)0) / p) return false;
        r *= p;
    }
    out = r;
    return true;
}

/* The Gaussian binomial [n choose h]_p, the number of h-dimensional subspaces of F_p^n. Every
 * partial product is itself a Gaussian binomial, so the divisions are exact. */
bool gauss_binomial(uint64_t p, uint64_t n, uint64_t h, unsigned __int128 &out) {
    unsigned __int128 c = 1;
    for (uint64_t i = 0; i < h; ++i) {
        unsigned __int128 num, den;
        if (!pow_checked(p, n - i, num) || !pow_checked(p, i + 1, den)) return false;
        if (num == 0 || c > (~(unsigned __int128)0) / (num - 1)) return false;
        c = c * (num - 1) / (den - 1);
    }
    out = c;
    return true;
}

/* An echelon basis of a subspace of F_p^n: rows in insertion order, row i reduced by the rows
 * before it with leading entry 1 at pivots[i]. `rank()` is the dimension it spans. */
struct Echelon {
    uint64_t n = 0;
    std::vector<Entry> rows;
    std::vector<uint32_t> pivots;

    void clear() { rows.clear(); pivots.clear(); }
    uint64_t rank() const { return pivots.size(); }
    const Entry *row(uint64_t i) const { return rows.data() + i * n; }
    /* `v` is n entries of scratch and is destroyed. False when v was already in the span. */
    bool add(const gfp::Field &f, Entry *v) {
        for (uint64_t i = 0; i < pivots.size(); ++i) {
            Entry c = v[pivots[i]];
            if (c) f.subtract_multiple(v, row(i), c, n);
        }
        uint64_t lead = n;
        for (uint64_t j = 0; j < n; ++j)
            if (v[j]) { lead = j; break; }
        if (lead == n) return false;
        if (v[lead] != 1) f.scale(v, f.inverse(v[lead]), n);
        rows.insert(rows.end(), v, v + n);
        pivots.push_back((uint32_t)lead);
        return true;
    }
    /* The reduced row echelon basis, rank * n entries appended to `out`: the canonical form of
     * the subspace, and so the only thing that decides when two components are equal. */
    void rref_into(const gfp::Field &f, std::vector<Entry> &out) const {
        uint64_t r = rank(), base = out.size();
        std::vector<uint64_t> order(r);
        for (uint64_t i = 0; i < r; ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](uint64_t a, uint64_t b) { return pivots[a] < pivots[b]; });
        for (uint64_t i = 0; i < r; ++i) out.insert(out.end(), row(order[i]), row(order[i]) + n);
        for (uint64_t i = 1; i < r; ++i) {
            uint32_t piv = pivots[order[i]];
            for (uint64_t e = 0; e < i; ++e) {
                Entry c = out[base + e * n + piv];
                if (c) f.subtract_multiple(out.data() + base + e * n, out.data() + base + i * n, c, n);
            }
        }
    }
};

/* What the walk carries: `n` entries per component, the operation's thresholds, and one
 * Echelon per component of the prefix. Depth d has state[d]; state[0] is the empty prefix. */
struct Walker : Family::Visitor {
    Op op;
    Reduction reduction;
    uint64_t p, n, h, blocks;      /* blocks: components per row (1 except for packings) */
    unsigned __int128 target;      /* p^n - 1 */
    unsigned __int128 total;       /* packings: [n choose h]_p */
    gfp::Field f;
    Accumulator acc;

    uint64_t rows;                              /* components (rows) of one member */
    uint64_t depth = 0;
    std::vector<unsigned __int128> powers;      /* p^d for d = 0..n */
    std::vector<Echelon> comps;                 /* one per component of the prefix */
    std::vector<uint8_t> okay;                  /* prefix still consistent with the predicate */
    std::vector<uint64_t> pairs;                /* intersecting pairs so far */
    std::vector<unsigned __int128> cover;       /* Σ (p^dim - 1) so far */
    std::vector<uint64_t> canon_len;            /* packings: canonical forms held after depth d */
    std::vector<Entry> canon;                   /* packings: their rref bases, h * n entries each */
    std::vector<Entry> scratch;
    Echelon merged;

    Walker(Op o, Reduction r, uint64_t prime, uint64_t nn, uint64_t hh, uint64_t b, uint64_t member_rows,
           unsigned __int128 t, unsigned __int128 tot, Shared *shared)
        : op(o), reduction(r), p(prime), n(nn), h(hh), blocks(b), target(t), total(tot), f(prime),
          acc(r, shared), rows(member_rows), scratch(nn) {
        powers.resize(n + 1);
        powers[0] = 1;
        for (uint64_t d = 1; d <= n; ++d) powers[d] = powers[d - 1] * p;
        comps.resize((member_rows + 1) * blocks);
        for (auto &c : comps) c.n = n;
        merged.n = n;
        okay.assign(member_rows + 1, 1);
        pairs.assign(member_rows + 1, 0);
        cover.assign(member_rows + 1, 0);
        canon_len.assign(member_rows + 1, 0);
    }

    /* The basis of the component held in `row`'s block `b`, into comps[slot]. */
    uint64_t build(const Entry *row, uint64_t b, uint64_t slot) {
        Echelon &c = comps[slot];
        c.clear();
        for (uint64_t i = 0; i < h; ++i) {
            std::copy(row + (b * h + i) * n, row + (b * h + i + 1) * n, scratch.begin());
            c.add(f, scratch.data());
        }
        return c.rank();
    }

    /* Do the components at `a` and `b` meet in more than 0? Their ranks add exactly when they
     * do not, so feed one basis into a copy of the other. */
    bool meets(const Echelon &a, const Echelon &b) {
        merged.rows = a.rows;
        merged.pivots = a.pivots;
        for (uint64_t i = 0; i < b.rank(); ++i) {
            std::copy(b.row(i), b.row(i) + n, scratch.begin());
            if (!merged.add(f, scratch.data())) return true;
        }
        return false;
    }

    /* One row of a packing candidate: `blocks` components that must form a spread of F_p^n by
     * h-subspaces, all of them new. */
    bool push_packing(const Entry *row, uint64_t held, uint64_t &canonical) {
        unsigned __int128 row_cover = 0;
        for (uint64_t b = 0; b < blocks; ++b) {
            if (build(row, b, held + b) != h) return false;
            row_cover += powers[h] - 1;
            for (uint64_t i = 0; i < b; ++i)
                if (meets(comps[held + i], comps[held + b])) return false;
            uint64_t at = canonical * h * n;
            canon.resize(at);
            comps[held + b].rref_into(f, canon);
            for (uint64_t q = 0; q < canonical; ++q)
                if (std::equal(canon.begin() + q * h * n, canon.begin() + (q + 1) * h * n, canon.begin() + at))
                    return false;
            ++canonical;
        }
        return row_cover == target && (unsigned __int128)canonical <= total;
    }

    Step push(const Entry *row, Index first, Index) override {
        uint64_t d = depth++;
        uint64_t held = d * blocks; /* components of the prefix before this row */
        bool ok = okay[d] != 0;
        uint64_t pair_count = pairs[d];
        unsigned __int128 cov = cover[d];
        uint64_t canonical = canon_len[d];
        if (op == Op::Packing) {
            if (ok) ok = push_packing(row, held, canonical);
            canon.resize(canonical * h * n);
        } else if (op == Op::IntersectingPairs) {
            /* an integer per member, so no prefix is ever decided: `ok` stays true, the walk
             * descends everywhere, and every member is visited. */
            build(row, 0, held);
            for (uint64_t i = 0; i < held; ++i)
                if (meets(comps[i], comps[held])) ++pair_count;
        } else {
            uint64_t dim = build(row, 0, held);
            cov += powers[dim] - 1;
            if (dim == 0) ok = false;
            if (op != Op::Partition && d > 0 && dim != comps[0].rank()) ok = false;
            for (uint64_t i = 0; ok && i < held; ++i)
                if (meets(comps[i], comps[held])) ok = false;
            if (op == Op::Spread || op == Op::Partition) {
                /* too much cover can never come back, and neither can too little: what is left to
                 * push is at most one full component per remaining row. */
                unsigned __int128 largest = op == Op::Spread ? powers[dim] - 1 : powers[std::min(h, n)] - 1;
                if (cov > target || cov + (unsigned __int128)(rows - depth) * largest < target) ok = false;
            }
        }
        okay[depth] = ok ? 1 : 0;
        pairs[depth] = pair_count;
        cover[depth] = cov;
        canon_len[depth] = canonical;
        if (acc.exhausted(first)) return Step::Skip;
        return ok ? Step::Descend : Step::Skip;
    }

    void pop() override {
        --depth;
        if (op == Op::Packing) canon.resize(canon_len[depth] * h * n);
    }

    void leaf(Index index) override {
        switch (op) {
        case Op::IntersectingPairs: acc.integer(index, pairs[depth]); break;
        case Op::PartialSpread: acc.boolean(index, okay[depth] != 0); break;
        case Op::Spread:
        case Op::Partition: acc.boolean(index, okay[depth] != 0 && cover[depth] == target); break;
        case Op::Packing: acc.boolean(index, okay[depth] != 0 && (unsigned __int128)canon_len[depth] == total); break;
        }
    }

    void take_all(Index first, Index count) override { acc.booleans(first, count, true); }
    void skip_all(Index first, Index count) override { acc.booleans(first, count, false); }
};

R run(const Request &req) {
    Op op;
    if (!parse_op(req.op, op)) return R::failure(INTERNAL, "unknown spreads_and_partitions operation " + req.op);
    const Family &fam = *req.family;
    uint64_t p = fam.prime(), cols = fam.cols(), rows = fam.rows();
    uint64_t n = req.int_args.count("n") ? req.int_args.at("n") : 0;
    uint64_t h = op == Op::Packing ? (req.int_args.count("h") ? req.int_args.at("h") : 0) : (n ? cols / n : 0);
    if (n == 0) return R::failure(INVALID, "the ambient dimension n must be at least 1");
    if (op == Op::Packing && h == 0) return R::failure(INVALID, "the component dimension h must be at least 1");
    uint64_t unit = op == Op::Packing ? n * h : n;
    if (cols % unit != 0)
        return R::failure(INVALID, (op == Op::Packing ? "n * h = " : "n = ") + std::to_string(unit) +
                                       " does not divide the " + std::to_string(cols) + " columns of a member");
    uint64_t blocks = op == Op::Packing ? cols / unit : 1;
    unsigned __int128 target = 0, total = 0;
    if (!pow_checked(p, n, target)) return R::failure(INVALID, "p^n does not fit in 128 bits");
    target -= 1;
    if (op == Op::Packing && !gauss_binomial(p, n, h, total))
        return R::failure(INVALID, "the number of h-subspaces of F_p^n does not fit in 128 bits");

    auto size_r = fam.size();
    if (!size_r.ok) return R::failure(size_r.error.status, size_r.error.message);
    auto tops_r = fam.top_count();
    if (!tops_r.ok) return R::failure(tops_r.error.status, tops_r.error.message);
    uint64_t size = size_r.value, tops = tops_r.value;

    Reduction reduction = parse_reduction(req.reduction);
    Shared shared;
    auto st = prepare_all(reduction, size, shared);
    if (!st.ok) return R::failure(st.error.status, st.error.message);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, tops ? tops : 1));
    std::vector<Walker> walkers;
    walkers.reserve(threads);
    for (uint32_t t = 0; t < threads; ++t)
        walkers.emplace_back(op, reduction, p, n, h, blocks, rows, target, total, &shared);
    auto statuses = parallel_ranges(tops, threads, [&](uint32_t t, uint64_t begin, uint64_t end) {
        return fam.enumerate(walkers[t], begin, end);
    });
    for (const auto &s : statuses)
        if (!s.ok) return R::failure(s.error.status, s.error.message);
    std::vector<Accumulator> accs;
    for (auto &w : walkers) accs.push_back(w.acc);
    return assemble(req, reduction, accs, shared);
}

BackendRegistration registration{Backend{
    "spreads_and_partitions", "generic",
    [] { return true; },
    [](const Request &req) { return req.family->prime() != 0 && req.family->prime() < (1ULL << 32); },
    run,
    0}};

} // namespace
} // namespace lk::spreads
