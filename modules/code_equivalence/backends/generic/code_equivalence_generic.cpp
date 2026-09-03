/* Portable canonical forms of linear codes under monomial equivalence.
 *
 * The representative of a class is its code of least index in the Grassmannian order, and that
 * code is always systematic: a code of dimension k has an information set, and moving it to the
 * first k coordinates gives the least possible pivot set {0,...,k-1}. So the search does not
 * touch the whole monomial group. It runs over the information sets of the member's rref basis;
 * each one, reduced to the identity, fixes the image up to a monomial map of the k rows and a
 * scale and an order of the n-k remaining columns. For a fixed row map the columns are then
 * settled: scale each to leading entry one, and sort them, because reading [I | A] row by row
 * compares the columns lexicographically.
 *
 * The maps that reach the representative are the automorphisms of the code, one coset of them,
 * so counting them as the search goes gives the automorphism order and the orbit size exactly,
 * without ever enumerating the orbit. */
#include "../../../gfp/backends/generic/field.hpp"
#include "../../../../runtime/src/reduce.hpp"

#include <algorithm>
#include <numeric>

namespace lk::code_equivalence {
namespace {

using R = Result<std::shared_ptr<Object>>;
constexpr int INVALID = 1;

enum class Op { IsCanonical, CanonicalIndex, CanonicalForm, OrbitSize, AutOrder };

Result<uint64_t> checked_mul(uint64_t a, uint64_t b, const char *what) {
    if (a && b > UINT64_MAX / a) return Result<uint64_t>::failure(INVALID, std::string(what) + " does not fit in 64 bits");
    return Result<uint64_t>::success(a * b);
}

/* (p-1)^n n! with scalars, n! without: the order of the group that acts. */
Result<uint64_t> group_order(uint64_t p, uint64_t n, bool scalars) {
    uint64_t order = 1;
    for (uint64_t i = 2; i <= n; ++i) {
        auto step = checked_mul(order, i, "the group order");
        if (!step.ok) return step;
        order = step.value;
    }
    if (scalars)
        for (uint64_t i = 0; i < n; ++i) {
            auto step = checked_mul(order, p - 1, "the group order");
            if (!step.ok) return step;
            order = step.value;
        }
    return Result<uint64_t>::success(order);
}

/* Number of free entries above the pivot set: the count of (row, column) pairs an rref shape
 * with these pivots leaves open. */
uint64_t free_positions(const std::vector<uint64_t> &pivots, uint64_t n) {
    uint64_t total = 0;
    for (uint64_t i = 0; i < pivots.size(); ++i)
        for (uint64_t c = pivots[i] + 1; c < n; ++c)
            total += std::find(pivots.begin(), pivots.end(), c) == pivots.end();
    return total;
}

/* The next k-subset of {0,...,n-1} in lexicographic order; false when there is none. */
bool next_subset(std::vector<uint64_t> &s, uint64_t n) {
    uint64_t k = s.size();
    for (uint64_t i = k; i-- > 0;)
        if (s[i] + (k - i) < n) {
            ++s[i];
            for (uint64_t j = i + 1; j < k; ++j) s[j] = s[j - 1] + 1;
            return true;
        }
    return false;
}

/* Size of grassmannian(p, n, k), by [n,k] = [n-1,k-1] + p^k [n-1,k]. An index only means
 * anything while it fits in 64 bits, so overflow is an error rather than a wrap. */
Result<uint64_t> grassmannian_size(uint64_t p, uint64_t n, uint64_t k) {
    auto too_big = [] { return Result<uint64_t>::failure(INVALID, "the Grassmannian at this dimension does not fit in 64 bits"); };
    if (k > n) return Result<uint64_t>::success(0);
    std::vector<unsigned __int128> row(k + 1, 0);
    row[0] = 1;
    for (uint64_t i = 1; i <= n; ++i)
        for (uint64_t j = std::min(i, k); j > 0; --j) {
            unsigned __int128 power = 1;
            for (uint64_t t = 0; t < j; ++t) {
                power *= p;
                if (power > UINT64_MAX) return too_big();
            }
            row[j] = row[j - 1] + power * row[j];
            if (row[j] > UINT64_MAX) return too_big();
        }
    return Result<uint64_t>::success((uint64_t)row[k]);
}

struct Worker {
    uint64_t p, n;
    gfp::Field field;
    gfp::EchelonBasis workspace;
    Matrix member;
    std::vector<Entry> basis;      /* k x n rref of the member */
    std::vector<uint32_t> pivots;
    std::vector<Entry> sub, square, inverse, w, columns, best_columns;
    std::vector<uint64_t> set, complement, permutation, scale, order, best_order;
    uint64_t k = 0;

    Worker(uint64_t prime, uint64_t cols) : p(prime), n(cols), field(prime), workspace(prime, cols) {}

    Status load(const Family &family, uint64_t index) {
        auto st = family.member_into(index, member);
        if (!st.ok) return st;
        workspace.clear();
        for (uint64_t r = 0; r < member.rows; ++r) workspace.add(member.entries.data() + r * n);
        workspace.rref(basis, pivots);
        k = pivots.size();
        return ok();
    }

    /* Index of the member's own code in grassmannian(p, n, k). */
    uint64_t own_index() const {
        uint64_t index = 0;
        std::vector<uint64_t> s(k), piv(pivots.begin(), pivots.end());
        std::iota(s.begin(), s.end(), 0);
        while (s != piv) {
            uint64_t block = 1;
            for (uint64_t i = free_positions(s, n); i-- > 0;) block *= p;
            index += block;
            next_subset(s, n);
        }
        for (uint64_t i = 0; i < k; ++i)
            for (uint64_t c = pivots[i] + 1; c < n; ++c)
                if (std::find(piv.begin(), piv.end(), c) == piv.end())
                    index = index * p + basis[i * n + c];
        return index;
    }

    /* m^-1 into `inverse`, false when m is singular. */
    bool invert(const std::vector<Entry> &m) {
        inverse.assign(k * k, 0);
        square = m;
        for (uint64_t i = 0; i < k; ++i) inverse[i * k + i] = 1;
        for (uint64_t c = 0; c < k; ++c) {
            uint64_t pivot = k;
            for (uint64_t r = c; r < k; ++r)
                if (square[r * k + c]) { pivot = r; break; }
            if (pivot == k) return false;
            if (pivot != c)
                for (uint64_t j = 0; j < k; ++j) {
                    std::swap(square[pivot * k + j], square[c * k + j]);
                    std::swap(inverse[pivot * k + j], inverse[c * k + j]);
                }
            Entry f = field.inverse(square[c * k + c]);
            if (f != 1) {
                field.scale(square.data() + c * k, f, k);
                field.scale(inverse.data() + c * k, f, k);
            }
            for (uint64_t r = 0; r < k; ++r) {
                Entry e = square[r * k + c];
                if (r == c || !e) continue;
                field.subtract_multiple(square.data() + r * k, square.data() + c * k, e, k);
                field.subtract_multiple(inverse.data() + r * k, inverse.data() + c * k, e, k);
            }
        }
        return true;
    }

    /* Scale a column to leading entry one; the zero column stays zero. */
    void normalise(Entry *v) {
        for (uint64_t i = 0; i < k; ++i)
            if (v[i]) {
                if (v[i] != 1) field.scale(v, field.inverse(v[i]), k);
                return;
            }
    }

    bool column_less(uint64_t a, uint64_t b) const {
        for (uint64_t i = 0; i < k; ++i) {
            if (columns[a * k + i] != columns[b * k + i]) return columns[a * k + i] < columns[b * k + i];
        }
        return false;
    }

    bool column_equal(uint64_t a, uint64_t b) const {
        for (uint64_t i = 0; i < k; ++i)
            if (columns[a * k + i] != columns[b * k + i]) return false;
        return true;
    }

    /* How many (order, scale) choices of the free columns produce exactly this arrangement:
     * equal columns may be swapped, and a zero column accepts any of the p-1 scales. */
    uint64_t arrangements(bool scalars) const {
        uint64_t m = complement.size(), total = 1;
        for (uint64_t t = 0; t < m;) {
            uint64_t run = 1;
            while (t + run < m && column_equal(order[t], order[t + run])) ++run;
            for (uint64_t i = 2; i <= run; ++i) total *= i;
            bool zero = true;
            for (uint64_t i = 0; i < k; ++i) zero &= columns[order[t] * k + i] == 0;
            if (scalars && zero)
                for (uint64_t i = 0; i < run; ++i) total *= p - 1;
            t += run;
        }
        return total;
    }

    /* The least index in the class, the basis attaining it, and how many maps reach it. */
    struct Answer {
        uint64_t least = 0;
        uint64_t automorphisms = 0;
    };

    Answer search(bool scalars, bool count_maps, uint64_t stop_below) {
        uint64_t m = n - k;
        Answer answer{UINT64_MAX, 0};
        set.resize(k);
        std::iota(set.begin(), set.end(), 0);
        square.resize(k * k);
        w.assign(k * n, 0);
        columns.assign(m * k, 0);
        permutation.resize(k);
        scale.assign(k, 1);
        order.resize(m);
        sub.assign(k * k, 0);
        do {
            for (uint64_t i = 0; i < k; ++i)
                for (uint64_t j = 0; j < k; ++j) sub[i * k + j] = basis[i * n + set[j]];
            if (!invert(sub)) continue;
            for (uint64_t i = 0; i < k; ++i)
                for (uint64_t c = 0; c < n; ++c) {
                    uint64_t x = 0;
                    for (uint64_t t = 0; t < k; ++t) x += (uint64_t)inverse[i * k + t] * basis[t * n + c];
                    w[i * n + c] = (Entry)field.reduce(x);
                }
            complement.clear();
            for (uint64_t c = 0; c < n; ++c)
                if (std::find(set.begin(), set.end(), c) == set.end()) complement.push_back(c);

            std::iota(permutation.begin(), permutation.end(), 0);
            do {
                std::fill(scale.begin(), scale.end(), 1);
                while (true) {
                    for (uint64_t t = 0; t < m; ++t) {
                        for (uint64_t i = 0; i < k; ++i)
                            columns[t * k + i] =
                                (Entry)field.reduce(scale[i] * w[permutation[i] * n + complement[t]]);
                        if (scalars) normalise(columns.data() + t * k);
                    }
                    std::iota(order.begin(), order.end(), 0);
                    std::sort(order.begin(), order.end(), [&](uint64_t a, uint64_t b) { return column_less(a, b); });
                    uint64_t index = 0;
                    for (uint64_t i = 0; i < k; ++i)
                        for (uint64_t t = 0; t < m; ++t) index = index * p + columns[order[t] * k + i];
                    if (index < answer.least) {
                        answer.least = index;
                        answer.automorphisms = count_maps ? arrangements(scalars) : 0;
                        best_columns = columns;
                        best_order = order;
                        if (answer.least < stop_below) return answer;
                    } else if (count_maps && index == answer.least) {
                        answer.automorphisms += arrangements(scalars);
                    }
                    if (!scalars) break;
                    uint64_t i = k;
                    while (i-- > 0) {
                        if (++scale[i] < p) break;
                        scale[i] = 1;
                    }
                    if (i == (uint64_t)-1) break;
                }
            } while (std::next_permutation(permutation.begin(), permutation.end()));
        } while (next_subset(set, n));
        return answer;
    }

    /* [I_k | A] for the representative the last search found. */
    std::vector<Entry> representative() const {
        uint64_t m = n - k;
        std::vector<Entry> rows(k * n, 0);
        for (uint64_t i = 0; i < k; ++i) {
            rows[i * n + i] = 1;
            for (uint64_t t = 0; t < m; ++t) rows[i * n + k + t] = best_columns[best_order[t] * k + i];
        }
        return rows;
    }
};

struct Setup {
    bool scalars = false;
    uint64_t order = 0;
};

Result<Setup> setup(const Request &req) {
    auto flag = req.int_args.find("scalars");
    if (flag == req.int_args.end() || flag->second > 1)
        return Result<Setup>::failure(INVALID, "`scalars` must be 1 for the monomial group or 0 for coordinate permutations only");
    Setup s;
    s.scalars = flag->second == 1;
    auto order = group_order(req.family->prime(), req.family->cols(), s.scalars);
    if (!order.ok) return Result<Setup>::failure(order.error.status, order.error.message);
    s.order = order.value;
    return Result<Setup>::success(s);
}

/* The ranking is only defined while the Grassmannian at every dimension a member can have fits
 * in 64 bits; check the widest one the family can reach before answering anything. */
Status check_sizes(const Family &family) {
    uint64_t p = family.prime(), n = family.cols();
    for (uint64_t k = 0; k <= n; ++k) {
        auto size = grassmannian_size(p, n, k);
        if (!size.ok) return fail(size.error.status, size.error.message);
    }
    return ok();
}

R run_reduced(const Request &req, Op op) {
    auto config = setup(req);
    if (!config.ok) return R::failure(config.error.status, config.error.message);
    const Family &family = *req.family;
    auto sized = family.size();
    if (!sized.ok) return R::failure(sized.error.status, sized.error.message);
    auto checked = check_sizes(family);
    if (!checked.ok) return R::failure(checked.error.status, checked.error.message);
    uint64_t size = sized.value, n = family.cols(), p = family.prime();

    Reduction reduction = parse_reduction(req.reduction);
    Shared shared;
    auto prepared = prepare_all(reduction, size, shared);
    if (!prepared.ok) return R::failure(prepared.error.status, prepared.error.message);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, size ? size : 1));
    std::vector<Accumulator> accumulators;
    for (uint32_t t = 0; t < threads; ++t) accumulators.emplace_back(reduction, &shared);

    bool count_maps = op == Op::OrbitSize || op == Op::AutOrder;
    auto statuses = parallel_ranges(size, threads, [&](uint32_t thread, uint64_t begin, uint64_t end) -> Status {
        Worker worker(p, n);
        for (uint64_t i = begin; i < end; ++i) {
            if (accumulators[thread].exhausted(i)) break;
            auto st = worker.load(family, i);
            if (!st.ok) return st;
            uint64_t own = worker.own_index();
            auto answer = worker.search(config.value.scalars, count_maps,
                                        op == Op::IsCanonical ? own : 0);
            if (op == Op::IsCanonical) accumulators[thread].boolean(i, own == answer.least);
            else if (op == Op::CanonicalIndex) accumulators[thread].integer(i, answer.least);
            else if (op == Op::AutOrder) accumulators[thread].integer(i, answer.automorphisms);
            else accumulators[thread].integer(i, config.value.order / answer.automorphisms);
        }
        return ok();
    });
    for (const auto &st : statuses)
        if (!st.ok) return R::failure(st.error.status, st.error.message);
    return assemble(req, reduction, accumulators, shared);
}

R run_canonical_form(const Request &req) {
    if (req.reduction != "all") return R::failure(INVALID, "canonical_form values only reduce with `all`");
    auto config = setup(req);
    if (!config.ok) return R::failure(config.error.status, config.error.message);
    const Family &family = *req.family;
    auto sized = family.size();
    if (!sized.ok) return R::failure(sized.error.status, sized.error.message);
    auto checked = check_sizes(family);
    if (!checked.ok) return R::failure(checked.error.status, checked.error.message);
    uint64_t size = sized.value, n = family.cols(), p = family.prime();

    std::vector<std::vector<Entry>> rows(size);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, size ? size : 1));
    auto statuses = parallel_ranges(size, threads, [&](uint32_t, uint64_t begin, uint64_t end) -> Status {
        Worker worker(p, n);
        for (uint64_t i = begin; i < end; ++i) {
            auto st = worker.load(family, i);
            if (!st.ok) return st;
            worker.search(config.value.scalars, false, 0);
            rows[i] = worker.representative();
        }
        return ok();
    });
    for (const auto &st : statuses)
        if (!st.ok) return R::failure(st.error.status, st.error.message);

    auto values = std::make_shared<lk::Basis>();
    values->p = p;
    values->count = size;
    values->cols = n;
    values->offsets.push_back(0);
    for (uint64_t i = 0; i < size; ++i) {
        values->entries.insert(values->entries.end(), rows[i].begin(), rows[i].end());
        values->offsets.push_back(values->offsets.back() + (n ? rows[i].size() / n : 0));
    }
    auto object = std::make_shared<Object>();
    object->kind = "gfp.basis";
    object->basis = values;
    return R::success(object);
}

R run(const Request &req) {
    if (req.op == "is_canonical") return run_reduced(req, Op::IsCanonical);
    if (req.op == "canonical_index") return run_reduced(req, Op::CanonicalIndex);
    if (req.op == "canonical_form") return run_canonical_form(req);
    if (req.op == "orbit_size") return run_reduced(req, Op::OrbitSize);
    if (req.op == "aut_order") return run_reduced(req, Op::AutOrder);
    return R::failure(4, "unknown code_equivalence operation " + req.op);
}

BackendRegistration registration{Backend{
    "code_equivalence", "generic",
    [] { return true; },
    [](const Request &req) { return is_prime(req.family->prime()); },
    run,
    0}};

} // namespace
} // namespace lk::code_equivalence
