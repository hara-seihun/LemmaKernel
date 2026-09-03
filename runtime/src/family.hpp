#pragma once
#include "object.hpp"
#include <atomic>

namespace lk {

/* A family is a tree of descriptions. Every member is a matrix over one prime p (or of
 * permutations, p == 0, or of naturals, p == NATURALS) with a fixed shape; members have canonical
 * indices 0..size-1 (order defined in the manifest). Enumeration is depth-first over rows so that
 * a consumer can share work along common prefixes: rows are pushed one at a time, and the
 * consumer may refuse to descend (Skip) or accept every leaf below without visiting it (TakeAll).
 * Leaves below one node have contiguous indices, and every push names that range. */
struct PivotTable;

struct Family {
    enum class Kind { Explicit, Subsets, Grassmannian, AllMatrices, Transform, Stack, GroupElements,
                      SubsetsOf, SymmetricMatrices, Range, Words };
    Kind kind;
    /* batch, dictionary (Subsets, and the materialised inner family for SubsetsOf), C, stacked
     * rows, or group generators */
    std::shared_ptr<Matrix> data;
    std::shared_ptr<Family> child;
    uint64_t p = 0, k = 0, n = 0, h = 0, m = 0; /* Words: p is the alphabet size, n the length */
    uint64_t a = 0, b = 0;                     /* Range: [a, b) */
    /* GroupElements: every element of the generated permutation group, sorted lexicographically,
     * computed on first use (count x n entries). Read through the atomic pointer so that the
     * per-member fast path takes no lock. */
    mutable std::shared_ptr<const std::vector<Entry>> elements;
    mutable std::atomic<const std::vector<Entry> *> elements_ready{nullptr};
    /* Grassmannian: pivot sets and offsets, computed on first use. */
    mutable std::shared_ptr<const PivotTable> pivots;
    mutable std::atomic<const PivotTable *> pivots_ready{nullptr};

    uint64_t prime() const; /* 0 for permutation members, NATURALS for integer members */
    uint64_t rows() const; /* rows of one member */
    uint64_t cols() const;
    Result<uint64_t> size() const;
    bool is_explicit() const;
    Result<Matrix> member(uint64_t index) const;
    /* member() into caller-owned storage; `out.entries` is reused when already large enough. */
    Status member_into(uint64_t index, Matrix &out) const;
    /* Inverse of member(): the canonical index of a member given as its rows. Only kinds with a
     * closed-form order support it (subsets, grassmannian, all_matrices, group_elements). */
    Result<uint64_t> index_of(const Matrix &member) const;
    /* Group order (GroupElements only); computes the closure. */
    Result<const std::vector<Entry> *> group_elements() const;
    Result<const PivotTable *> pivot_table() const;

    struct Visitor {
        enum class Step { Descend, Skip, TakeAll };
        virtual ~Visitor() = default;
        /* Enter the subtree whose leaves have indices [first, first + below). */
        virtual Step push(const Entry *row, uint64_t first, uint64_t below) = 0;
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
Result<std::shared_ptr<Family>> make_group_elements(std::shared_ptr<Matrix> generators);
/* k-subsets of another family's members, each member flattened to one row of rows*cols entries.
 * The inner family is materialised once; it must have at most 2^22 members. */
Result<std::shared_ptr<Family>> make_subsets_of(std::shared_ptr<Family> inner, uint64_t k);
Result<std::shared_ptr<Family>> make_symmetric_matrices(uint64_t p, uint64_t n);
Result<std::shared_ptr<Family>> make_range(uint64_t a, uint64_t b);
Result<std::shared_ptr<Family>> make_words(uint64_t alphabet, uint64_t length);

/* Closure of a set of permutations (count x n, p == 0) under composition: every element of the
 * generated group, sorted lexicographically. Fails above `limit` elements. */
Result<std::vector<Entry>> permutation_closure(const Matrix &generators, uint64_t limit);

const char *family_kind_name(Family::Kind k);

} // namespace lk
