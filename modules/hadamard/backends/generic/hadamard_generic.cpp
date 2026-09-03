/* Portable backend for binary matrices read as signs, with 0 = +1 and 1 = -1. */
#include "../../../../runtime/src/reduce.hpp"

#include <algorithm>
#include <numeric>

namespace lk::hadamard {
namespace {

using R = Result<std::shared_ptr<Object>>;
constexpr int INVALID = 1;

bool is_hadamard(const Entry *a, uint64_t rows, uint64_t cols) {
    if (rows != cols) return false;
    for (uint64_t i = 0; i < rows; ++i)
        for (uint64_t j = 0; j < i; ++j) {
            uint64_t different = 0;
            for (uint64_t k = 0; k < cols; ++k) different += a[i * cols + k] != a[j * cols + k];
            if (different * 2 != cols) return false;
        }
    return true;
}

bool is_skew(const Entry *a, uint64_t rows, uint64_t cols) {
    if (rows != cols) return false;
    for (uint64_t i = 0; i < rows; ++i) {
        if (a[i * cols + i] != 0) return false;
        for (uint64_t j = 0; j < i; ++j)
            if (a[i * cols + j] == a[j * cols + i]) return false;
    }
    return true;
}

bool is_regular(const Entry *a, uint64_t rows, uint64_t cols) {
    if (rows != cols) return false;
    uint64_t target = 0;
    if (rows != 0)
        for (uint64_t j = 0; j < cols; ++j) target += a[j];
    for (uint64_t i = 0; i < rows; ++i) {
        uint64_t ones = 0;
        for (uint64_t j = 0; j < cols; ++j) ones += a[i * cols + j];
        if (ones != target) return false;
    }
    for (uint64_t j = 0; j < cols; ++j) {
        uint64_t ones = 0;
        for (uint64_t i = 0; i < rows; ++i) ones += a[i * cols + j];
        if (ones != target) return false;
    }
    return true;
}

bool is_conference(const Entry *a, uint64_t rows, uint64_t cols) {
    if (rows != cols || rows < 2) return false;
    for (uint64_t i = 0; i < rows; ++i)
        if (a[i * cols + i] != 0) return false;
    for (uint64_t i = 0; i < rows; ++i)
        for (uint64_t j = 0; j < i; ++j) {
            uint64_t different = 0;
            for (uint64_t k = 0; k < cols; ++k)
                if (k != i && k != j) different += a[i * cols + k] != a[j * cols + k];
            if (different * 2 != cols - 2) return false;
        }
    return true;
}

std::vector<Entry> canonical_form(const Entry *a, uint64_t rows, uint64_t cols) {
    std::vector<Entry> original(a, a + rows * cols);
    if (rows == 0 || cols == 0) return original;

    std::vector<uint64_t> row_order(rows);
    std::iota(row_order.begin(), row_order.end(), 0);
    std::vector<Entry> best;
    do {
        uint64_t first_row = row_order[0];
        for (uint64_t first_col = 0; first_col < cols; ++first_col) {
            std::vector<std::vector<Entry>> columns(cols, std::vector<Entry>(rows));
            Entry corner = a[first_row * cols + first_col];
            for (uint64_t c = 0; c < cols; ++c)
                for (uint64_t i = 0; i < rows; ++i) {
                    uint64_t r = row_order[i];
                    columns[c][i] = a[r * cols + c] ^ a[r * cols + first_col] ^
                                    a[first_row * cols + c] ^ corner;
                }
            std::sort(columns.begin(), columns.end());
            std::vector<Entry> candidate(rows * cols);
            for (uint64_t i = 0; i < rows; ++i)
                for (uint64_t c = 0; c < cols; ++c) candidate[i * cols + c] = columns[c][i];
            if (best.empty() || candidate < best) best = std::move(candidate);
        }
    } while (std::next_permutation(row_order.begin(), row_order.end()));
    return best;
}

R run_predicate(const Request &req) {
    const Family &fam = *req.family;
    auto size_r = fam.size();
    if (!size_r.ok) return R::failure(size_r.error.status, size_r.error.message);
    uint64_t size = size_r.value;
    Reduction reduction = parse_reduction(req.reduction);
    Shared shared;
    auto prepared = prepare_all(reduction, size, shared);
    if (!prepared.ok) return R::failure(prepared.error.status, prepared.error.message);

    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, size ? size : 1));
    std::vector<Accumulator> accs;
    for (uint32_t t = 0; t < threads; ++t) accs.emplace_back(reduction, &shared);
    auto statuses = parallel_ranges(size, threads, [&](uint32_t t, uint64_t begin, uint64_t end) -> Status {
        Matrix member;
        for (uint64_t i = begin; i < end; ++i) {
            if (accs[t].exhausted(i)) break;
            auto status = fam.member_into(i, member);
            if (!status.ok) return status;
            bool value;
            if (req.op == "is_hadamard") value = is_hadamard(member.entries.data(), member.rows, member.cols);
            else if (req.op == "is_skew") value = is_skew(member.entries.data(), member.rows, member.cols);
            else if (req.op == "is_regular") value = is_regular(member.entries.data(), member.rows, member.cols);
            else if (req.op == "is_conference") value = is_conference(member.entries.data(), member.rows, member.cols);
            else return fail(4, "unknown hadamard predicate " + req.op);
            accs[t].boolean(i, value);
        }
        return ok();
    });
    for (const auto &status : statuses)
        if (!status.ok) return R::failure(status.error.status, status.error.message);
    return assemble(req, reduction, accs, shared);
}

R run_canonical(const Request &req) {
    const Family &fam = *req.family;
    auto size_r = fam.size();
    if (!size_r.ok) return R::failure(size_r.error.status, size_r.error.message);
    uint64_t size = size_r.value, rows = fam.rows(), cols = fam.cols();
    if (size > (1ULL << 40)) return R::failure(INVALID, "family too large to materialise");

    auto output = std::make_shared<Matrix>();
    output->p = 2;
    output->count = size;
    output->rows = rows;
    output->cols = cols;
    output->entries.assign(size * rows * cols, 0);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, size ? size : 1));
    auto statuses = parallel_ranges(size, threads, [&](uint32_t, uint64_t begin, uint64_t end) -> Status {
        Matrix member;
        for (uint64_t i = begin; i < end; ++i) {
            auto status = fam.member_into(i, member);
            if (!status.ok) return status;
            auto form = canonical_form(member.entries.data(), rows, cols);
            std::copy(form.begin(), form.end(), output->entries.begin() + i * rows * cols);
        }
        return ok();
    });
    for (const auto &status : statuses)
        if (!status.ok) return R::failure(status.error.status, status.error.message);

    auto object = std::make_shared<Object>();
    object->kind = "gfp.matrix";
    object->matrix = std::move(output);
    return R::success(object);
}

R run(const Request &req) {
    if (req.op == "canonical_form") return run_canonical(req);
    return run_predicate(req);
}

BackendRegistration registration{Backend{
    "hadamard", "generic",
    [] { return true; },
    [](const Request &req) { return req.family->prime() == 2; },
    run,
    0}};

} // namespace
} // namespace lk::hadamard
