/* matchings_and_flows generic backend: portable exact algorithms over matrix families. */
#include "../../../../runtime/src/reduce.hpp"

#include <bit>
#include <boost/multiprecision/cpp_int.hpp>
#include <limits>
#include <queue>

namespace lk::matchings_and_flows {
namespace {

using boost::multiprecision::cpp_int;
using R = Result<std::shared_ptr<Object>>;
constexpr int INVALID = 1;

Result<uint64_t> checked_u64(const cpp_int &value, const char *operation) {
    if (value < 0 || value > cpp_int(std::numeric_limits<uint64_t>::max()))
        return Result<uint64_t>::failure(INVALID, std::string(operation) + " result does not fit in 64 bits");
    return Result<uint64_t>::success(value.convert_to<uint64_t>());
}

/* Ryser's formula in Gray-code order. Only O(n) arbitrary-precision integers are kept. */
Result<uint64_t> permanent(const Matrix &m) {
    const uint64_t n = m.rows;
    if (n >= 63)
        return Result<uint64_t>::failure(INVALID, "perfect_matching_count accepts fewer than 63 rows");
    std::vector<cpp_int> row_sums(n, 0);
    cpp_int answer = 0;
    uint64_t previous_gray = 0;
    const uint64_t end = uint64_t{1} << n;
    for (uint64_t step = 1; step < end; ++step) {
        const uint64_t gray = step ^ (step >> 1);
        const uint64_t changed = gray ^ previous_gray;
        const unsigned column = std::countr_zero(changed);
        const bool added = (gray & changed) != 0;
        for (uint64_t i = 0; i < n; ++i) {
            Entry a = m.entries[i * n + column];
            if (added) row_sums[i] += a;
            else row_sums[i] -= a;
        }
        cpp_int term = 1;
        for (const auto &sum : row_sums) term *= sum;
        if ((n - std::popcount(gray)) & 1) answer -= term;
        else answer += term;
        previous_gray = gray;
    }
    return checked_u64(answer, "perfect_matching_count");
}

cpp_int determinant(std::vector<cpp_int> a, uint64_t n) {
    if (n == 0) return 1;
    cpp_int previous = 1;
    bool negate = false;
    for (uint64_t k = 0; k + 1 < n; ++k) {
        uint64_t pivot_row = k;
        while (pivot_row < n && a[pivot_row * n + k] == 0) ++pivot_row;
        if (pivot_row == n) return 0;
        if (pivot_row != k) {
            for (uint64_t j = 0; j < n; ++j) std::swap(a[k * n + j], a[pivot_row * n + j]);
            negate = !negate;
        }
        const cpp_int pivot = a[k * n + k];
        for (uint64_t i = k + 1; i < n; ++i) {
            for (uint64_t j = k + 1; j < n; ++j)
                a[i * n + j] = (a[i * n + j] * pivot - a[i * n + k] * a[k * n + j]) / previous;
            a[i * n + k] = 0;
        }
        previous = pivot;
    }
    cpp_int out = a[(n - 1) * n + n - 1];
    return negate ? -out : out;
}

bool is_symmetric(const Matrix &m) {
    const uint64_t n = m.rows;
    for (uint64_t i = 0; i < n; ++i)
        for (uint64_t j = i + 1; j < n; ++j)
            if (m.entries[i * n + j] != m.entries[j * n + i]) return false;
    return true;
}

Result<uint64_t> spanning_tree_count(const Matrix &m) {
    const uint64_t n = m.rows;
    if (!is_symmetric(m))
        return Result<uint64_t>::failure(INVALID, "spanning_tree_count requires every member to be symmetric");
    const uint64_t minor_n = n - 1;
    std::vector<cpp_int> minor(minor_n * minor_n);
    for (uint64_t i = 1; i < n; ++i) {
        cpp_int degree = 0;
        for (uint64_t j = 0; j < n; ++j)
            if (i != j) degree += cpp_int(m.entries[i * n + j]);
        for (uint64_t j = 1; j < n; ++j)
            minor[(i - 1) * minor_n + (j - 1)] = i == j ? degree : -cpp_int(m.entries[i * n + j]);
    }
    return checked_u64(determinant(std::move(minor), minor_n), "spanning_tree_count");
}

/* Edmonds-Karp on a dense residual capacity matrix. Separate opposite arcs are represented by
 * their combined residual capacities, which is the standard residual-network construction. */
Result<uint64_t> max_flow(const Matrix &m, uint64_t source, uint64_t sink) {
    const uint64_t n = m.rows;
    std::vector<uint64_t> residual(m.entries.begin(), m.entries.end());
    std::vector<uint64_t> parent(n);
    unsigned __int128 total = 0;
    for (;;) {
        std::fill(parent.begin(), parent.end(), n);
        parent[source] = source;
        std::queue<uint64_t> queue;
        queue.push(source);
        while (!queue.empty() && parent[sink] == n) {
            uint64_t u = queue.front();
            queue.pop();
            for (uint64_t v = 0; v < n; ++v) {
                if (parent[v] == n && residual[u * n + v] != 0) {
                    parent[v] = u;
                    queue.push(v);
                }
            }
        }
        if (parent[sink] == n) break;
        uint64_t amount = std::numeric_limits<uint64_t>::max();
        for (uint64_t v = sink; v != source; v = parent[v])
            amount = std::min(amount, residual[parent[v] * n + v]);
        for (uint64_t v = sink; v != source; v = parent[v]) {
            uint64_t u = parent[v];
            residual[u * n + v] -= amount;
            if (residual[v * n + u] > std::numeric_limits<uint64_t>::max() - amount)
                return Result<uint64_t>::failure(INVALID, "max_flow residual capacity does not fit in 64 bits");
            residual[v * n + u] += amount;
        }
        total += amount;
        if (total > std::numeric_limits<uint64_t>::max())
            return Result<uint64_t>::failure(INVALID, "max_flow result does not fit in 64 bits");
    }
    return Result<uint64_t>::success(static_cast<uint64_t>(total));
}

enum class Op { PerfectMatchingCount, SpanningTreeCount, MaxFlow };

R run(const Request &req) {
    const Family &family = *req.family;
    if (family.rows() != family.cols())
        return R::failure(INVALID, "matchings_and_flows operations require square matrices");
    const uint64_t n = family.rows();
    Op op;
    uint64_t source = 0, sink = 0;
    if (req.op == "perfect_matching_count") op = Op::PerfectMatchingCount;
    else if (req.op == "spanning_tree_count") op = Op::SpanningTreeCount;
    else if (req.op == "max_flow") {
        op = Op::MaxFlow;
        source = req.int_args.at("source");
        sink = req.int_args.at("sink");
        if (source == sink) return R::failure(INVALID, "max_flow source and sink must be distinct");
        if (source >= n || sink >= n) return R::failure(INVALID, "max_flow source and sink must be vertex indices below the matrix size");
    } else return R::failure(4, "unknown matchings_and_flows operation " + req.op);

    auto size_result = family.size();
    if (!size_result.ok) return R::failure(size_result.error.status, size_result.error.message);
    const uint64_t size = size_result.value;
    Reduction reduction = parse_reduction(req.reduction);
    Shared shared;
    auto prepared = prepare_all(reduction, size, shared);
    if (!prepared.ok) return R::failure(prepared.error.status, prepared.error.message);
    const uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, size ? size : 1));
    std::vector<Accumulator> accumulators;
    for (uint32_t t = 0; t < threads; ++t) accumulators.emplace_back(reduction, &shared);

    auto statuses = parallel_ranges(size, threads, [&](uint32_t thread, uint64_t begin, uint64_t end) -> Status {
        Matrix member;
        for (uint64_t i = begin; i < end; ++i) {
            auto status = family.member_into(i, member);
            if (!status.ok) return status;
            Result<uint64_t> value;
            switch (op) {
            case Op::PerfectMatchingCount: value = permanent(member); break;
            case Op::SpanningTreeCount: value = spanning_tree_count(member); break;
            case Op::MaxFlow: value = max_flow(member, source, sink); break;
            }
            if (!value.ok) return fail(value.error.status, value.error.message);
            accumulators[thread].integer(i, value.value);
        }
        return ok();
    });
    for (const auto &status : statuses)
        if (!status.ok) return R::failure(status.error.status, status.error.message);
    return assemble(req, reduction, accumulators, shared);
}

BackendRegistration registration{Backend{
    "matchings_and_flows", "generic",
    [] { return true; },
    [](const Request &) { return true; },
    run,
    0}};

} // namespace
} // namespace lk::matchings_and_flows
