#pragma once
#include "object.hpp"

namespace lk {

/* A family is a tree of descriptions. Every member is a matrix over one prime p with a fixed
 * shape; members have canonical indices 0..size-1 (order defined in the manifest). Enumeration
 * is depth-first over rows so that a consumer can share work along common prefixes: rows are
 * pushed one at a time, and the consumer may refuse to descend (Skip) or accept every leaf below
 * without visiting it (TakeAll). Leaves below one node have contiguous indices. */
struct Family {
    enum class Kind { Explicit, Subsets, Grassmannian, AllMatrices, Transform, Stack };
    Kind kind;
    std::shared_ptr<Matrix> data; /* batch, dictionary, C, or stacked rows */
    std::shared_ptr<Family> child;
    uint64_t p = 0, k = 0, n = 0, h = 0, m = 0;

    uint64_t prime() const;
    uint64_t rows() const; /* rows of one member */
    uint64_t cols() const;
    Result<uint64_t> size() const;
    bool is_explicit() const;
    Result<Matrix> member(uint64_t index) const;

    struct Visitor {
        enum class Step { Descend, Skip, TakeAll };
        virtual ~Visitor() = default;
        virtual Step push(const Entry *row) = 0;
        virtual void pop() = 0;
        virtual void leaf(uint64_t index) = 0;
        virtual void take_all(uint64_t first_index, uint64_t count) = 0;
        virtual void skip_all(uint64_t first_index, uint64_t count) = 0;
    };
    /* Number of top-level branches; enumerate(v, a, b) visits branches [a, b). */
    Result<uint64_t> top_count() const;
    Status enumerate(Visitor &v, uint64_t top_begin, uint64_t top_end) const;
};

Result<std::shared_ptr<Family>> make_explicit(std::shared_ptr<Matrix> batch);
Result<std::shared_ptr<Family>> make_subsets(std::shared_ptr<Matrix> dictionary, uint64_t k);
Result<std::shared_ptr<Family>> make_grassmannian(uint64_t p, uint64_t n, uint64_t h);
Result<std::shared_ptr<Family>> make_all_matrices(uint64_t p, uint64_t rows, uint64_t cols);
Result<std::shared_ptr<Family>> make_transform(std::shared_ptr<Family> inner, std::shared_ptr<Matrix> c);
Result<std::shared_ptr<Family>> make_stack(std::shared_ptr<Family> inner, std::shared_ptr<Matrix> rows);

const char *family_kind_name(Family::Kind k);

} // namespace lk
