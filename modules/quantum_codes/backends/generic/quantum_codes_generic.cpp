/* Portable backend for binary symplectic stabiliser codes.
 *
 * Rows generate C <= F_2^(2n), with coordinates (x|z). Boolean operations use bit-packed
 * elimination. Distance enumerates errors by increasing symplectic weight and stops at the first
 * word in C^perp_s outside C, rather than scanning all 4^n words when the distance is small. */
#include "../../../../runtime/src/reduce.hpp"

#include <algorithm>
#include <functional>

namespace lk::quantum_codes {
namespace {

using R = Result<std::shared_ptr<Object>>;
using Bits = std::vector<uint64_t>;
constexpr int INVALID = 1;
constexpr int INTERNAL = 4;

inline bool bit(const Bits &v, uint64_t i) { return (v[i >> 6] >> (i & 63)) & 1; }

inline void set_bit(Bits &v, uint64_t i, bool value) {
    uint64_t mask = 1ULL << (i & 63);
    if (value) v[i >> 6] |= mask;
    else v[i >> 6] &= ~mask;
}

inline void xor_into(Bits &a, const Bits &b) {
    for (size_t i = 0; i < a.size(); ++i) a[i] ^= b[i];
}

inline bool parity_dot(const Bits &a, const Bits &b) {
    unsigned parity = 0;
    for (size_t i = 0; i < a.size(); ++i) parity ^= (unsigned)__builtin_parityll(a[i] & b[i]);
    return parity != 0;
}

Bits pack_row(const Entry *row, uint64_t cols) {
    Bits out((cols + 63) / 64, 0);
    for (uint64_t c = 0; c < cols; ++c)
        if (row[c]) set_bit(out, c, true);
    return out;
}

struct BinaryBasis {
    uint64_t cols;
    std::vector<Bits> rows;
    std::vector<uint64_t> pivots;

    explicit BinaryBasis(uint64_t n) : cols(n) {}

    void reduce(Bits &v) const {
        for (size_t i = 0; i < rows.size(); ++i)
            if (bit(v, pivots[i])) xor_into(v, rows[i]);
    }

    bool add(Bits v) {
        reduce(v);
        uint64_t pivot = cols;
        for (uint64_t c = 0; c < cols; ++c)
            if (bit(v, c)) { pivot = c; break; }
        if (pivot == cols) return false;
        rows.push_back(std::move(v));
        pivots.push_back(pivot);
        return true;
    }

    bool contains(const Bits &v) const {
        Bits reduced = v;
        reduce(reduced);
        return std::all_of(reduced.begin(), reduced.end(), [](uint64_t word) { return word == 0; });
    }

    uint64_t rank() const { return rows.size(); }
};

bool dual_contained_in_code(std::vector<Bits> checks, const BinaryBasis &code, uint64_t cols) {
    uint64_t r = 0;
    std::vector<uint64_t> pivots;
    for (uint64_t c = 0; c < cols && r < checks.size(); ++c) {
        uint64_t pivot = r;
        while (pivot < checks.size() && !bit(checks[pivot], c)) ++pivot;
        if (pivot == checks.size()) continue;
        std::swap(checks[r], checks[pivot]);
        for (uint64_t i = 0; i < checks.size(); ++i)
            if (i != r && bit(checks[i], c)) xor_into(checks[i], checks[r]);
        pivots.push_back(c);
        ++r;
    }
    checks.resize(r);
    std::vector<uint8_t> is_pivot(cols, 0);
    for (uint64_t c : pivots) is_pivot[c] = 1;
    for (uint64_t free = 0; free < cols; ++free) {
        if (is_pivot[free]) continue;
        Bits v((cols + 63) / 64, 0);
        set_bit(v, free, true);
        for (uint64_t i = 0; i < pivots.size(); ++i)
            if (bit(checks[i], free)) set_bit(v, pivots[i], true);
        if (!code.contains(v)) return false;
    }
    return true;
}

uint64_t binary_rank(const Matrix &m, uint64_t first_col, uint64_t cols) {
    BinaryBasis basis(cols);
    for (uint64_t r = 0; r < m.rows; ++r)
        basis.add(pack_row(m.entries.data() + r * m.cols + first_col, cols));
    return basis.rank();
}

bool self_orthogonal(const Matrix &m) {
    uint64_t n = m.cols / 2;
    const Entry *a = m.entries.data();
    for (uint64_t r = 0; r < m.rows; ++r)
        for (uint64_t s = r; s < m.rows; ++s) {
            unsigned inner = 0;
            for (uint64_t i = 0; i < n; ++i)
                inner ^= (unsigned)((a[r * m.cols + i] & a[s * m.cols + n + i]) ^
                                    (a[r * m.cols + n + i] & a[s * m.cols + i]));
            if (inner) return false;
        }
    return true;
}

bool css(const Matrix &m) {
    uint64_t n = m.cols / 2;
    return binary_rank(m, 0, n) + binary_rank(m, n, n) == binary_rank(m, 0, m.cols);
}

uint64_t symplectic_distance(const Matrix &m) {
    uint64_t n = m.cols / 2;
    BinaryBasis code(m.cols);
    std::vector<Bits> checks;
    checks.reserve(m.rows);
    for (uint64_t r = 0; r < m.rows; ++r) {
        const Entry *row = m.entries.data() + r * m.cols;
        code.add(pack_row(row, m.cols));
        Bits check((m.cols + 63) / 64, 0);
        for (uint64_t i = 0; i < n; ++i) {
            if (row[n + i]) set_bit(check, i, true);
            if (row[i]) set_bit(check, n + i, true);
        }
        checks.push_back(std::move(check));
    }

    /* Detect C^perp_s <= C before searching. Lagrangian stabilisers and all coisotropic input
     * spaces have no logical word; without this check they require a complete 4^n scan. */
    if (code.rank() >= n && dual_contained_in_code(checks, code, m.cols)) return n + 1;

    Bits candidate((m.cols + 63) / 64, 0);
    auto accepted = [&]() {
        for (const Bits &check : checks)
            if (parity_dot(candidate, check)) return false;
        return !code.contains(candidate);
    };

    for (uint64_t weight = 1; weight <= n; ++weight) {
        std::vector<uint64_t> support(weight);
        std::function<bool(size_t)> assign;
        assign = [&](size_t depth) {
            if (depth == support.size()) return accepted();
            uint64_t q = support[depth];
            for (unsigned symbol = 1; symbol <= 3; ++symbol) {
                set_bit(candidate, q, symbol & 1);
                set_bit(candidate, n + q, symbol & 2);
                if (assign(depth + 1)) return true;
            }
            set_bit(candidate, q, false);
            set_bit(candidate, n + q, false);
            return false;
        };
        std::function<bool(uint64_t, size_t)> choose;
        choose = [&](uint64_t start, size_t depth) {
            if (depth == support.size()) return assign(0);
            uint64_t remaining = support.size() - depth;
            for (uint64_t q = start; q + remaining <= n; ++q) {
                support[depth] = q;
                if (choose(q + 1, depth + 1)) return true;
            }
            return false;
        };
        if (choose(0, 0)) return weight;
    }
    return n + 1;
}

enum class Op { IsSelfOrthogonal, Distance, IsCss };

R run(const Request &req) {
    const Family &family = *req.family;
    if (family.prime() != 2) return R::failure(INVALID, "quantum_codes is defined over F_2");
    if (family.cols() % 2 != 0) return R::failure(INVALID, "quantum_codes needs an even number 2n of columns");

    Op op;
    if (req.op == "is_self_orthogonal") op = Op::IsSelfOrthogonal;
    else if (req.op == "distance") op = Op::Distance;
    else if (req.op == "is_css") op = Op::IsCss;
    else return R::failure(INTERNAL, "unknown quantum_codes operation " + req.op);

    auto size_result = family.size();
    if (!size_result.ok) return R::failure(size_result.error.status, size_result.error.message);
    uint64_t size = size_result.value;
    Reduction reduction = parse_reduction(req.reduction);
    Shared shared;
    auto prepared = prepare_all(reduction, size, shared);
    if (!prepared.ok) return R::failure(prepared.error.status, prepared.error.message);

    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, size ? size : 1));
    std::vector<Accumulator> accumulators;
    std::vector<Matrix> members(threads);
    for (uint32_t t = 0; t < threads; ++t) accumulators.emplace_back(reduction, &shared);
    auto statuses = parallel_ranges(size, threads, [&](uint32_t t, uint64_t begin, uint64_t end) -> Status {
        for (uint64_t i = begin; i < end; ++i) {
            if (accumulators[t].exhausted(i)) break;
            auto status = family.member_into(i, members[t]);
            if (!status.ok) return status;
            if (op == Op::Distance) accumulators[t].integer(i, symplectic_distance(members[t]));
            else accumulators[t].boolean(i, op == Op::IsSelfOrthogonal ? self_orthogonal(members[t]) : css(members[t]));
        }
        return ok();
    });
    for (const auto &status : statuses)
        if (!status.ok) return R::failure(status.error.status, status.error.message);
    return assemble(req, reduction, accumulators, shared);
}

BackendRegistration registration{Backend{
    "quantum_codes", "generic",
    [] { return true; },
    [](const Request &) { return true; },
    run,
    0}};

} // namespace
} // namespace lk::quantum_codes
