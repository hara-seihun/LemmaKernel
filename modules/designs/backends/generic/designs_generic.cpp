#include "../../../../runtime/src/registry.hpp"

#include <algorithm>
#include <functional>
#include <limits>
#include <queue>

namespace lk::designs {
namespace {

using R = Result<std::shared_ptr<Object>>;
constexpr int INVALID = 1;

struct BlockFamily {
    uint64_t v = 0, k = 0;
    std::vector<std::vector<uint32_t>> blocks;
};

R matrix_result(const std::vector<std::vector<uint64_t>> &rows) {
    uint64_t cols = rows.empty() ? 0 : rows.front().size();
    for (const auto &row : rows)
        if (row.size() != cols) return R::failure(4, "designs backend constructed a ragged result");
    auto matrix = std::make_shared<U64Matrices>();
    matrix->count = 1;
    matrix->rows = rows.size();
    matrix->cols = cols;
    for (const auto &row : rows) matrix->entries.insert(matrix->entries.end(), row.begin(), row.end());
    auto out = std::make_shared<Object>();
    out->kind = "designs.matrix";
    out->u64_matrices = matrix;
    return R::success(out);
}

Result<BlockFamily> read_blocks(const Family &family, uint64_t limit = std::numeric_limits<uint64_t>::max()) {
    auto size = family.size();
    if (!size.ok) return Result<BlockFamily>::failure(size.error.status, size.error.message);
    if (size.value == 0) return Result<BlockFamily>::failure(INVALID, "a block family must contain at least one block");
    BlockFamily out;
    out.v = family.cols();
    out.k = family.rows();
    if (out.v == 0 || out.k == 0 || out.k > out.v)
        return Result<BlockFamily>::failure(INVALID, "blocks must have shape k x v with 1 <= k <= v");
    uint64_t count = size.value < limit ? (uint64_t)size.value : limit;
    out.blocks.reserve(count);
    Matrix member;
    for (uint64_t index = 0; index < count; ++index) {
        auto status = family.member_into(index, member);
        if (!status.ok) return Result<BlockFamily>::failure(status.error.status, status.error.message);
        std::vector<uint32_t> block;
        block.reserve(out.k);
        uint32_t previous = 0;
        for (uint64_t row = 0; row < out.k; ++row) {
            const Entry *entries = member.at(0) + row * out.v;
            uint64_t point = out.v;
            for (uint64_t col = 0; col < out.v; ++col) {
                if (entries[col] == 1 && point == out.v) point = col;
                else if (entries[col] != 0)
                    return Result<BlockFamily>::failure(INVALID, "block rows must be distinct standard basis vectors in increasing point order");
            }
            if (point == out.v || (row && point <= previous))
                return Result<BlockFamily>::failure(INVALID, "block rows must be distinct standard basis vectors in increasing point order");
            block.push_back((uint32_t)point);
            previous = (uint32_t)point;
        }
        out.blocks.push_back(std::move(block));
    }
    return Result<BlockFamily>::success(std::move(out));
}

void combinations_rec(uint32_t n, uint32_t left, uint32_t next, std::vector<uint32_t> &current,
                      std::vector<std::vector<uint32_t>> &out) {
    if (left == 0) {
        out.push_back(current);
        return;
    }
    for (uint32_t x = next; x <= n - left; ++x) {
        current.push_back(x);
        combinations_rec(n, left - 1, x + 1, current, out);
        current.pop_back();
    }
}

std::vector<std::vector<uint32_t>> combinations(uint64_t n, uint64_t k) {
    std::vector<std::vector<uint32_t>> out;
    if (n > UINT32_MAX || k > n) return out;
    std::vector<uint32_t> current;
    combinations_rec((uint32_t)n, (uint32_t)k, 0, current, out);
    return out;
}

bool contains(const std::vector<uint32_t> &block, const std::vector<uint32_t> &subset) {
    return std::includes(block.begin(), block.end(), subset.begin(), subset.end());
}

std::vector<uint64_t> lambda_vector(const BlockFamily &family, uint64_t t) {
    auto subsets = combinations(family.v, t);
    std::vector<uint64_t> counts(subsets.size(), 0);
    for (size_t i = 0; i < subsets.size(); ++i)
        for (const auto &block : family.blocks)
            counts[i] += contains(block, subsets[i]);
    return counts;
}

bool all_equal(const std::vector<uint64_t> &values) {
    return std::all_of(values.begin(), values.end(), [&](uint64_t x) { return x == values.front(); });
}

uint64_t intersection_size(const std::vector<uint32_t> &a, const std::vector<uint32_t> &b) {
    uint64_t count = 0;
    for (uint32_t x : a) count += std::binary_search(b.begin(), b.end(), x);
    return count;
}

std::vector<uint64_t> intersection_numbers(const BlockFamily &family) {
    std::vector<uint64_t> counts(family.k + 1, 0);
    for (size_t i = 0; i < family.blocks.size(); ++i)
        for (size_t j = i + 1; j < family.blocks.size(); ++j)
            ++counts[intersection_size(family.blocks[i], family.blocks[j])];
    return counts;
}

bool covers_points(const BlockFamily &family, const std::vector<uint32_t> &indices) {
    std::vector<uint8_t> seen(family.v, 0);
    for (uint32_t index : indices)
        for (uint32_t point : family.blocks[index])
            if (seen[point]++) return false;
    return std::all_of(seen.begin(), seen.end(), [](uint8_t x) { return x == 1; });
}

bool search_resolution(const std::vector<std::vector<uint32_t>> &classes, std::vector<uint8_t> &remaining,
                       std::vector<std::vector<uint64_t>> &answer) {
    auto first = std::find(remaining.begin(), remaining.end(), 1);
    if (first == remaining.end()) return true;
    uint32_t block = (uint32_t)(first - remaining.begin());
    for (const auto &candidate : classes) {
        if (!std::binary_search(candidate.begin(), candidate.end(), block)) continue;
        if (!std::all_of(candidate.begin(), candidate.end(), [&](uint32_t i) { return remaining[i] != 0; })) continue;
        for (uint32_t i : candidate) remaining[i] = 0;
        answer.emplace_back(candidate.begin(), candidate.end());
        if (search_resolution(classes, remaining, answer)) return true;
        answer.pop_back();
        for (uint32_t i : candidate) remaining[i] = 1;
    }
    return false;
}

std::vector<std::vector<uint64_t>> resolution(const BlockFamily &family) {
    if (family.v % family.k != 0) return {};
    uint64_t class_size = family.v / family.k;
    if (class_size == 0 || family.blocks.empty() || family.blocks.size() % class_size != 0) return {};
    auto candidates = combinations(family.blocks.size(), class_size);
    candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                                    [&](const auto &c) { return !covers_points(family, c); }),
                     candidates.end());
    std::vector<uint8_t> remaining(family.blocks.size(), 1);
    std::vector<std::vector<uint64_t>> answer;
    if (!search_resolution(candidates, remaining, answer)) return {};
    return answer;
}

R run_block_operation(const Request &request) {
    uint64_t prefix = std::numeric_limits<uint64_t>::max();
    auto blocks = read_blocks(*request.family, prefix);
    if (!blocks.ok) return R::failure(blocks.error.status, blocks.error.message);
    const BlockFamily &family = blocks.value;
    if (request.op == "is_design" || request.op == "lambda_vector") {
        auto arg = request.int_args.find("t");
        if (arg == request.int_args.end() || arg->second > family.k)
            return R::failure(INVALID, "t must satisfy 0 <= t <= k");
        auto counts = lambda_vector(family, arg->second);
        if (request.op == "lambda_vector") return matrix_result({std::move(counts)});
        return matrix_result({{all_equal(counts) ? 1ULL : 0ULL, counts.front()}});
    }
    if (request.op == "intersection_numbers") return matrix_result({intersection_numbers(family)});
    if (request.op == "dual_is_design") {
        std::vector<uint64_t> replications(family.v, 0);
        for (const auto &block : family.blocks)
            for (uint32_t point : block) ++replications[point];
        std::vector<uint64_t> intersections;
        for (size_t i = 0; i < family.blocks.size(); ++i)
            for (size_t j = i + 1; j < family.blocks.size(); ++j)
                intersections.push_back(intersection_size(family.blocks[i], family.blocks[j]));
        uint64_t lambda0 = intersections.empty() ? 0 : intersections.front();
        bool flag = family.blocks.size() >= 2 && replications.front() >= 2 && all_equal(replications) && all_equal(intersections);
        return matrix_result({{flag ? 1ULL : 0ULL, lambda0}});
    }
    if (request.op == "is_resolvable") return matrix_result(resolution(family));
    return R::failure(4, "unknown designs block operation " + request.op);
}

struct Binomials {
    uint64_t n, k;
    std::vector<uint64_t> values;

    Binomials(uint64_t nn, uint64_t kk) : n(nn), k(kk), values((nn + 1) * (kk + 1), 0) {
        for (uint64_t i = 0; i <= n; ++i) {
            at(i, 0) = 1;
            for (uint64_t j = 1; j <= std::min(i, k); ++j) {
                uint64_t a = get(i - 1, j - 1), b = get(i - 1, j);
                at(i, j) = a > UINT64_MAX - b ? UINT64_MAX : a + b;
            }
        }
    }
    uint64_t &at(uint64_t i, uint64_t j) { return values[i * (k + 1) + j]; }
    uint64_t get(uint64_t i, uint64_t j) const { return j > i || j > k ? 0 : values[i * (k + 1) + j]; }

    uint64_t rank(const std::vector<uint32_t> &subset) const {
        uint64_t index = 0, previous = 0;
        for (uint64_t j = 0; j < subset.size(); ++j) {
            for (uint64_t x = previous; x < subset[j]; ++x) index += get(n - 1 - x, subset.size() - 1 - j);
            previous = subset[j] + 1;
        }
        return index;
    }
};

std::vector<std::vector<uint64_t>> subset_orbits(uint64_t v, uint64_t size, const Matrix &generators,
                                                  std::vector<std::vector<uint32_t>> &subsets) {
    subsets = combinations(v, size);
    Binomials binom(v, size);
    std::vector<uint8_t> assigned(subsets.size(), 0);
    std::vector<std::vector<uint64_t>> orbits;
    std::vector<uint32_t> image(size);
    for (uint64_t start = 0; start < subsets.size(); ++start) {
        if (assigned[start]) continue;
        std::vector<uint64_t> orbit{start};
        assigned[start] = 1;
        for (size_t front = 0; front < orbit.size(); ++front) {
            const auto &subset = subsets[orbit[front]];
            for (uint64_t g = 0; g < generators.count; ++g) {
                const Entry *perm = generators.at(g);
                for (uint64_t j = 0; j < size; ++j) image[j] = perm[subset[j]];
                std::sort(image.begin(), image.end());
                uint64_t index = binom.rank(image);
                if (!assigned[index]) {
                    assigned[index] = 1;
                    orbit.push_back(index);
                }
            }
        }
        orbits.push_back(std::move(orbit));
    }
    return orbits;
}

R run_kramer_mesner(const Request &request) {
    const Family &family = *request.family;
    if (family.kind != Family::Kind::Subsets)
        return R::failure(INVALID, "kramer_mesner_matrix needs a subsets family");
    auto group_arg = request.handle_args.find("group");
    if (group_arg == request.handle_args.end() || !group_arg->second->matrix || group_arg->second->matrix->p != 0)
        return R::failure(INVALID, "group must be a permutation group on the dictionary positions");
    const Matrix &group = *group_arg->second->matrix;
    uint64_t v = family.data->count, k = family.k;
    auto t_arg = request.int_args.find("t");
    if (t_arg == request.int_args.end() || t_arg->second > k)
        return R::failure(INVALID, "t must satisfy 0 <= t <= k");
    if (group.count == 0 || group.cols != v)
        return R::failure(INVALID, "permutation generators must act on every dictionary position");
    uint64_t t = t_arg->second;
    std::vector<std::vector<uint32_t>> t_subsets, k_subsets;
    auto row_orbits = subset_orbits(v, t, group, t_subsets);
    auto col_orbits = subset_orbits(v, k, group, k_subsets);
    std::vector<std::vector<uint64_t>> answer(row_orbits.size(), std::vector<uint64_t>(col_orbits.size(), 0));
    for (size_t i = 0; i < row_orbits.size(); ++i) {
        const auto &representative = t_subsets[row_orbits[i][0]];
        for (size_t j = 0; j < col_orbits.size(); ++j)
            for (uint64_t index : col_orbits[j])
                answer[i][j] += contains(k_subsets[index], representative);
    }
    return matrix_result(answer);
}

R run(const Request &request) {
    if (request.op == "kramer_mesner_matrix") return run_kramer_mesner(request);
    return run_block_operation(request);
}

BackendRegistration registration{Backend{
    "designs", "generic",
    [] { return true; },
    [](const Request &) { return true; },
    run,
    0}};

} // namespace
} // namespace lk::designs
