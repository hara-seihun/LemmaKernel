/* Portable counts and incidence tests for finite subspace lattices. */
#include "../../../../runtime/src/reduce.hpp"
#include "../../../gfp/backends/generic/field.hpp"

namespace lk::lattice_of_subspaces {
namespace {

using R = Result<std::shared_ptr<Object>>;
constexpr int INVALID = 1;

Result<uint64_t> checked_gaussian(uint64_t q, uint64_t n, uint64_t k) {
    using N = Result<uint64_t>;
    if (k > n) return N::success(0);
    k = std::min(k, n - k);
    std::vector<uint64_t> powers(k + 1, 1), row(k + 1, 0);
    row[0] = 1;
    for (uint64_t j = 1; j <= k; ++j) {
        unsigned __int128 v = (unsigned __int128)powers[j - 1] * q;
        if (v > UINT64_MAX) return N::failure(INVALID, "integer answer does not fit in 64 bits");
        powers[j] = (uint64_t)v;
    }
    for (uint64_t i = 1; i <= n; ++i) {
        for (uint64_t j = std::min(i, k); j > 0; --j) {
            unsigned __int128 v = (unsigned __int128)powers[j] * row[j] + row[j - 1];
            if (v > UINT64_MAX) return N::failure(INVALID, "integer answer does not fit in 64 bits");
            row[j] = (uint64_t)v;
        }
    }
    return N::success(row[k]);
}

Result<uint64_t> checked_flag_count(uint64_t q, uint64_t n, const Entry *dims, uint64_t count) {
    using N = Result<uint64_t>;
    for (uint64_t i = 0; i < count; ++i)
        if (dims[i] > n || (i && dims[i - 1] >= dims[i])) return N::success(0);
    uint64_t value = 1, previous = 0;
    for (uint64_t i = 0; i < count; ++i) {
        auto factor = checked_gaussian(q, n - previous, dims[i] - previous);
        if (!factor.ok) return factor;
        unsigned __int128 product = (unsigned __int128)value * factor.value;
        if (product > UINT64_MAX) return N::failure(INVALID, "integer answer does not fit in 64 bits");
        value = (uint64_t)product;
        previous = dims[i];
    }
    return N::success(value);
}

uint64_t matrix_rank(gfp::EchelonBasis &basis, const Entry *entries, uint64_t rows, uint64_t cols) {
    basis.clear();
    for (uint64_t r = 0; r < rows; ++r) basis.add(entries + r * cols);
    return basis.rank();
}

struct FixedSubspace {
    uint64_t p = 0, rows = 0, cols = 0;
    std::vector<Entry> entries;
};

Result<FixedSubspace> fixed_subspace(const Request &req) {
    using F = Result<FixedSubspace>;
    auto it = req.handle_args.find("subspace");
    if (it == req.handle_args.end() || !it->second->matrix)
        return F::failure(INVALID, "subspace must be a matrix of row vectors");
    const Matrix &source = *it->second->matrix;
    FixedSubspace out;
    out.p = source.p;
    out.cols = source.cols;
    out.rows = source.rows == 1 ? source.count : source.rows;
    uint64_t entries = out.rows * out.cols;
    if (source.entries.size() < entries)
        return F::failure(INVALID, "subspace payload is shorter than its shape");
    out.entries.assign(source.entries.begin(), source.entries.begin() + entries);
    if (out.p != req.family->prime() || out.cols != req.family->cols())
        return F::failure(INVALID, "subspace must be over the same prime with the same number of columns");
    return F::success(std::move(out));
}

R run(const Request &req) {
    const Family &family = *req.family;
    auto size_result = family.size();
    if (!size_result.ok) return R::failure(size_result.error.status, size_result.error.message);
    uint64_t size = size_result.value;
    Reduction reduction = parse_reduction(req.reduction);
    Shared shared;
    auto prepared = prepare_all(reduction, size, shared);
    if (!prepared.ok) return R::failure(prepared.error.status, prepared.error.message);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, size ? size : 1));
    std::vector<Accumulator> accumulators;
    for (uint32_t t = 0; t < threads; ++t) accumulators.emplace_back(reduction, &shared);

    if (req.op == "gaussian_binomial" || req.op == "flag_count") {
        uint64_t q = req.int_args.at("q"), n = req.int_args.at("n");
        if (q < 2) return R::failure(INVALID, "q must be at least 2");
        Family::Kind expected = req.op == "gaussian_binomial" ? Family::Kind::Range : Family::Kind::Words;
        if (family.kind != expected)
            return R::failure(INVALID, req.op + " is defined on " +
                (expected == Family::Kind::Range ? "range" : "words") + " families only");
        std::vector<Matrix> members(threads);
        auto statuses = parallel_ranges(size, threads, [&](uint32_t t, uint64_t begin, uint64_t end) -> Status {
            for (uint64_t i = begin; i < end; ++i) {
                auto st = family.member_into(i, members[t]);
                if (!st.ok) return st;
                Result<uint64_t> value = req.op == "gaussian_binomial"
                    ? checked_gaussian(q, n, members[t].entries[0])
                    : checked_flag_count(q, n, members[t].entries.data(), members[t].cols);
                if (!value.ok) return fail(value.error.status, value.error.message);
                accumulators[t].integer(i, value.value);
            }
            return ok();
        });
        for (const auto &st : statuses)
            if (!st.ok) return R::failure(st.error.status, st.error.message);
        return assemble(req, reduction, accumulators, shared);
    }

    if (family.prime() < 2 || family.prime() >= (1ULL << 32))
        return R::failure(INVALID, "row-space operations need a matrix family over p < 2^32");

    bool predicate = req.op == "contains" || req.op == "is_contained_in";
    Result<FixedSubspace> fixed;
    if (predicate) {
        fixed = fixed_subspace(req);
        if (!fixed.ok) return R::failure(fixed.error.status, fixed.error.message);
    }
    uint64_t h = predicate ? 0 : req.int_args.at("h");
    std::vector<Matrix> members(threads);
    std::vector<std::unique_ptr<gfp::EchelonBasis>> bases;
    for (uint32_t t = 0; t < threads; ++t)
        bases.push_back(std::make_unique<gfp::EchelonBasis>(family.prime(), family.cols()));

    auto statuses = parallel_ranges(size, threads, [&](uint32_t t, uint64_t begin, uint64_t end) -> Status {
        auto &basis = *bases[t];
        for (uint64_t i = begin; i < end; ++i) {
            if (accumulators[t].exhausted(i)) break;
            auto st = family.member_into(i, members[t]);
            if (!st.ok) return st;
            const Matrix &member = members[t];
            if (req.op == "contained_subspace_count" || req.op == "containing_subspace_count") {
                uint64_t rank = matrix_rank(basis, member.entries.data(), member.rows, member.cols);
                Result<uint64_t> value;
                if (req.op == "contained_subspace_count") value = checked_gaussian(family.prime(), rank, h);
                else if (rank > h) value = Result<uint64_t>::success(0);
                else value = checked_gaussian(family.prime(), member.cols - rank, h - rank);
                if (!value.ok) return fail(value.error.status, value.error.message);
                accumulators[t].integer(i, value.value);
                continue;
            }
            basis.clear();
            if (req.op == "contains") {
                for (uint64_t r = 0; r < member.rows; ++r) basis.add(member.at(0) + r * member.cols);
                uint64_t rank = basis.rank();
                for (uint64_t r = 0; r < fixed.value.rows; ++r)
                    basis.add(fixed.value.entries.data() + r * fixed.value.cols);
                accumulators[t].boolean(i, basis.rank() == rank);
            } else if (req.op == "is_contained_in") {
                for (uint64_t r = 0; r < fixed.value.rows; ++r)
                    basis.add(fixed.value.entries.data() + r * fixed.value.cols);
                uint64_t rank = basis.rank();
                for (uint64_t r = 0; r < member.rows; ++r) basis.add(member.at(0) + r * member.cols);
                accumulators[t].boolean(i, basis.rank() == rank);
            } else {
                return fail(4, "unknown lattice_of_subspaces operation " + req.op);
            }
        }
        return ok();
    });
    for (const auto &st : statuses)
        if (!st.ok) return R::failure(st.error.status, st.error.message);
    return assemble(req, reduction, accumulators, shared);
}

BackendRegistration registration{Backend{
    "lattice_of_subspaces", "generic",
    [] { return true; },
    [](const Request &) { return true; },
    run,
    0}};

} // namespace
} // namespace lk::lattice_of_subspaces
