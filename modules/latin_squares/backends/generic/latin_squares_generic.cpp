#include "../../../../runtime/src/reduce.hpp"

#include <functional>
#include <unordered_map>

namespace lk::latin_squares {
namespace {

using R = Result<std::shared_ptr<Object>>;
constexpr int INVALID = 1;
constexpr int INTERNAL = 4;

enum class Op { IsLatin, HasOrthogonalMate, TransversalCount, IsGroupTable, IsotopyCanonicalForm };

std::vector<std::vector<Entry>> permutations(uint64_t n) {
    std::vector<Entry> row(n);
    for (uint64_t i = 0; i < n; ++i) row[i] = (Entry)i;
    std::vector<std::vector<Entry>> out;
    do out.push_back(row); while (std::next_permutation(row.begin(), row.end()));
    return out;
}

bool is_latin(const std::vector<Entry> &square, uint64_t n) {
    std::vector<uint8_t> seen(n);
    for (uint64_t r = 0; r < n; ++r) {
        std::fill(seen.begin(), seen.end(), 0);
        for (uint64_t c = 0; c < n; ++c) {
            Entry x = square[r * n + c];
            if (x >= n || seen[x]) return false;
            seen[x] = 1;
        }
    }
    for (uint64_t c = 0; c < n; ++c) {
        std::fill(seen.begin(), seen.end(), 0);
        for (uint64_t r = 0; r < n; ++r) {
            Entry x = square[r * n + c];
            if (x >= n || seen[x]) return false;
            seen[x] = 1;
        }
    }
    return true;
}

bool has_orthogonal_mate(const std::vector<Entry> &square, uint64_t n) {
    if (!is_latin(square, n)) return false;
    auto rows = permutations(n);
    std::vector<uint32_t> columns(n, 0);
    std::vector<uint8_t> pairs(n * n, 0);
    std::function<bool(uint64_t)> search = [&](uint64_t r) {
        if (r == n) return true;
        uint64_t begin = 0, end = rows.size();
        if (r == 0) end = 1;
        for (uint64_t k = begin; k < end; ++k) {
            const auto &row = rows[k];
            bool fits = true;
            for (uint64_t c = 0; c < n; ++c) {
                uint64_t pair = square[r * n + c] * n + row[c];
                if ((columns[c] & (1u << row[c])) || pairs[pair]) { fits = false; break; }
            }
            if (!fits) continue;
            for (uint64_t c = 0; c < n; ++c) {
                columns[c] |= 1u << row[c];
                pairs[square[r * n + c] * n + row[c]] = 1;
            }
            if (search(r + 1)) return true;
            for (uint64_t c = 0; c < n; ++c) {
                columns[c] &= ~(1u << row[c]);
                pairs[square[r * n + c] * n + row[c]] = 0;
            }
        }
        return false;
    };
    return search(0);
}

uint64_t transversal_count(const std::vector<Entry> &square, uint64_t n) {
    std::vector<Entry> columns(n);
    for (uint64_t i = 0; i < n; ++i) columns[i] = (Entry)i;
    uint64_t count = 0;
    do {
        uint32_t symbols = 0;
        bool distinct = true;
        for (uint64_t i = 0; i < n; ++i) {
            Entry x = square[i * n + columns[i]];
            if (x >= n || (symbols & (1u << x))) { distinct = false; break; }
            symbols |= 1u << x;
        }
        count += distinct;
    } while (std::next_permutation(columns.begin(), columns.end()));
    return count;
}

bool is_group_table(const std::vector<Entry> &square, uint64_t n) {
    if (!is_latin(square, n)) return false;
    bool identity = false;
    for (uint64_t e = 0; e < n; ++e) {
        bool works = true;
        for (uint64_t x = 0; x < n; ++x)
            if (square[e * n + x] != x || square[x * n + e] != x) { works = false; break; }
        identity |= works;
    }
    if (!identity) return false;
    for (uint64_t x = 0; x < n; ++x)
        for (uint64_t y = 0; y < n; ++y)
            for (uint64_t z = 0; z < n; ++z)
                if (square[square[x * n + y] * n + z] != square[x * n + square[y * n + z]]) return false;
    return true;
}

std::vector<Entry> canonical_form(const std::vector<Entry> &square, uint64_t n) {
    auto orders = permutations(n);
    std::vector<Entry> best, candidate(n * n);
    for (const auto &rows : orders) {
        for (const auto &columns : orders) {
            std::unordered_map<Entry, Entry> labels;
            Entry next = 0;
            uint64_t q = 0;
            for (Entry r : rows) {
                for (Entry c : columns) {
                    Entry x = square[r * n + c];
                    auto [it, inserted] = labels.emplace(x, next);
                    if (inserted) ++next;
                    candidate[q++] = it->second;
                }
            }
            if (best.empty() || std::lexicographical_compare(candidate.begin(), candidate.end(), best.begin(), best.end()))
                best = candidate;
        }
    }
    return best;
}

struct Outputs {
    Shared shared;
    std::vector<Entry> canonical;
};

struct Walker : Family::Visitor {
    Op op;
    uint64_t n;
    Outputs *out;
    Accumulator acc;
    uint64_t visited = 0;
    std::vector<Entry> square;
    std::vector<uint8_t> appended;

    Walker(Op operation, Reduction reduction, uint64_t order, Outputs *outputs)
        : op(operation), n(order), out(outputs), acc(reduction, &outputs->shared) {
        square.reserve(n * n);
    }

    Step push(const Entry *row, Index first, Index) override {
        if (op != Op::TransversalCount && op != Op::IsotopyCanonicalForm && acc.exhausted(first)) {
            appended.push_back(0);
            return Step::Skip;
        }
        square.insert(square.end(), row, row + n);
        appended.push_back(1);
        return Step::Descend;
    }

    void pop() override {
        if (appended.back()) square.resize(square.size() - n);
        appended.pop_back();
    }

    void leaf(Index index) override {
        switch (op) {
        case Op::IsLatin: acc.boolean(index, is_latin(square, n)); break;
        case Op::HasOrthogonalMate: acc.boolean(index, has_orthogonal_mate(square, n)); break;
        case Op::TransversalCount: acc.integer(index, transversal_count(square, n)); break;
        case Op::IsGroupTable: acc.boolean(index, is_group_table(square, n)); break;
        case Op::IsotopyCanonicalForm: {
            auto form = canonical_form(square, n);
            std::copy(form.begin(), form.end(), out->canonical.begin() + index * n * n);
            ++visited;
            break;
        }
        }
    }

    void take_all(Index first, Index count) override {
        acc.booleans(first, count, true);
    }

    void skip_all(Index first, Index count) override {
        acc.booleans(first, count, false);
    }
};

R run(const Request &req) {
    Op op;
    if (req.op == "is_latin") op = Op::IsLatin;
    else if (req.op == "has_orthogonal_mate") op = Op::HasOrthogonalMate;
    else if (req.op == "transversal_count") op = Op::TransversalCount;
    else if (req.op == "is_group_table") op = Op::IsGroupTable;
    else if (req.op == "isotopy_canonical_form") op = Op::IsotopyCanonicalForm;
    else return R::failure(INTERNAL, "unknown latin_squares operation " + req.op);

    const Family &family = *req.family;
    uint64_t n = family.rows();
    if (n == 0 || family.cols() != n) return R::failure(INVALID, "Latin-square operations need nonempty square arrays");
    if (n > 5) return R::failure(INVALID, "latin_squares.generic supports order at most 5");
    auto size_result = family.size();
    if (!size_result.ok) return R::failure(size_result.error.status, size_result.error.message);
    auto tops_result = family.top_count();
    if (!tops_result.ok) return R::failure(tops_result.error.status, tops_result.error.message);
    uint64_t size = size_result.value, tops = tops_result.value;
    Reduction reduction = parse_reduction(req.reduction);
    Outputs outputs;
    if (op == Op::IsotopyCanonicalForm) {
        if (reduction != Reduction::All) return R::failure(INVALID, "isotopy_canonical_form only reduces with `all`");
        outputs.canonical.resize(size * n * n);
    } else {
        auto status = prepare_all(reduction, size, outputs.shared);
        if (!status.ok) return R::failure(status.error.status, status.error.message);
    }

    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, tops ? tops : 1));
    std::vector<Walker> walkers;
    walkers.reserve(threads);
    for (uint32_t t = 0; t < threads; ++t) walkers.emplace_back(op, reduction, n, &outputs);
    auto statuses = parallel_ranges(tops, threads, [&](uint32_t t, uint64_t begin, uint64_t end) {
        return family.enumerate(walkers[t], begin, end);
    });
    for (const auto &status : statuses)
        if (!status.ok) return R::failure(status.error.status, status.error.message);

    if (op == Op::IsotopyCanonicalForm) {
        uint64_t visited = 0;
        for (const auto &walker : walkers) visited += walker.visited;
        if (visited != size)
            return R::failure(INTERNAL, "enumeration visited " + std::to_string(visited) + " members of " + std::to_string(size));
        auto matrix = std::make_shared<Matrix>();
        matrix->p = NATURALS;
        matrix->count = size;
        matrix->rows = n;
        matrix->cols = n;
        matrix->entries = std::move(outputs.canonical);
        auto object = std::make_shared<Object>();
        object->kind = "lk.naturals";
        object->matrix = std::move(matrix);
        return R::success(object);
    }

    std::vector<Accumulator> accumulators;
    accumulators.reserve(walkers.size());
    for (auto &walker : walkers) accumulators.push_back(std::move(walker.acc));
    return assemble(req, reduction, accumulators, outputs.shared);
}

BackendRegistration registration{Backend{
    "latin_squares", "generic",
    [] { return true; },
    [](const Request &req) {
        return req.family->rows() >= 1 && req.family->rows() == req.family->cols() && req.family->rows() <= 5;
    },
    run,
    0}};

} // namespace
} // namespace lk::latin_squares
