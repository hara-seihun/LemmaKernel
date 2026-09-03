#pragma once
#include "object.hpp"
#include <atomic>

namespace lk {

/* A family is a tree of descriptions. Every member is a matrix with one field-size tag p, a
 * permutation with p == 0, naturals with p == NATURALS, or an integral Gram matrix with
 * p == GRAMS. Members have canonical
 * indices 0..size-1 (order defined in the manifest). Enumeration is depth-first over rows so that
 * a consumer can share work along common prefixes: rows are pushed one at a time, and the
 * consumer may refuse to descend (Skip) or accept every leaf below without visiting it (TakeAll).
 * Leaves below one node have contiguous indices, and every push names that range. */
struct PivotTable;
struct PartitionTable;

struct Family {
    enum class Kind { Explicit, Subsets, Grassmannian, AllMatrices, Transform, Stack, GroupElements,
                      GroupTables, SubsetsOf, SymmetricMatrices, AlternatingMatrices, Range, Words,
                      LatinSquares, Partitions, Compositions, StandardTableaux, AllGraphs,
                      EdgeSubgraphs, CayleyGraphs, Sublattices };
    Kind kind;
    /* Batch, group tables, dictionary (Subsets, and the materialised inner family for SubsetsOf),
     * C, stacked rows, group generators, or the shape of a StandardTableaux family. */
    std::shared_ptr<Matrix> data;
    std::shared_ptr<Family> child;
    uint64_t p = 0, k = 0, n = 0, h = 0, m = 0; /* Words: p is the alphabet size, n the length;
                                                  LatinSquares: n is the order */
    uint64_t a = 0, b = 0;                     /* Range: [a, b). Partitions: flags distinct, odd. */
    /* Expensive cardinalities are fixed when their family is constructed. */
    bool size_cached = false;
    uint64_t cached_size = 0;
    /* GroupElements: every element of the generated permutation or matrix group, sorted
     * lexicographically by flat entries and computed on first use. Read through the atomic
     * pointer so that the per-member fast path takes no lock. */
    mutable std::shared_ptr<const std::vector<Entry>> elements;
    mutable std::atomic<const std::vector<Entry> *> elements_ready{nullptr};
    /* CayleyGraphs: inverse classes of nonidentity group elements, as indices in `elements`. */
    std::shared_ptr<const std::vector<std::vector<uint64_t>>> cayley_classes;
    /* Grassmannian: pivot sets and offsets, computed on first use. */
    mutable std::shared_ptr<const PivotTable> pivots;
    mutable std::atomic<const PivotTable *> pivots_ready{nullptr};
    /* Partitions: the immutable sparse counting table used to unrank every member. */
    std::shared_ptr<const PartitionTable> partition_counts;
    uint64_t prime() const; /* 0 for permutations, NATURALS for naturals, GRAMS for integral Grams */
    uint64_t rows() const; /* rows of one member */
    uint64_t cols() const;
    Result<uint64_t> size() const;
    bool is_explicit() const;
    Result<Matrix> member(uint64_t index) const;
    /* member() into caller-owned storage; `out.entries` is reused when already large enough. */
    Status member_into(uint64_t index, Matrix &out) const;
    /* Inverse of member(): the canonical index of a member given as its rows. Only kinds with a
     * rank implementation support it (subsets, grassmannian, all_matrices, latin_squares,
     * group_elements). */
    Result<uint64_t> index_of(const Matrix &member) const;
    /* Group elements (GroupElements only); computes the closure. */
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
/* A batch of Cayley tables, or the Cayley table of a group given by permutation generators. */
Result<std::shared_ptr<Family>> make_group_tables(std::shared_ptr<Matrix> tables);
Result<std::shared_ptr<Family>> make_generated_group(std::shared_ptr<Matrix> generators);
/* k-subsets of another family's members, each member flattened to one row of rows*cols entries.
 * The inner family is materialised once; it must have at most 2^22 members. */
Result<std::shared_ptr<Family>> make_subsets_of(std::shared_ptr<Family> inner, uint64_t k);
Result<std::shared_ptr<Family>> make_symmetric_matrices(uint64_t p, uint64_t n);
Result<std::shared_ptr<Family>> make_alternating_matrices(uint64_t p, uint64_t n);
Result<std::shared_ptr<Family>> make_range(uint64_t a, uint64_t b);
Result<std::shared_ptr<Family>> make_words(uint64_t alphabet, uint64_t length);
Result<std::shared_ptr<Family>> make_latin_squares(uint64_t n);
/* Partitions of total, padded with trailing zeros to total entries. A zero bound means
 * unrestricted. `distinct` and `odd` are Boolean flags. */
Result<std::shared_ptr<Family>> make_partitions(uint64_t total, uint64_t max_part, uint64_t max_parts,
                                                uint64_t max_multiplicity, uint64_t distinct, uint64_t odd);
/* Positive compositions of total, padded to total entries. `parts == 0` allows every length;
 * `max_part == 0` does not bound a part. */
Result<std::shared_ptr<Family>> make_compositions(uint64_t total, uint64_t parts, uint64_t max_part);
Result<std::shared_ptr<Family>> make_standard_tableaux(std::shared_ptr<Matrix> shape);
Result<std::shared_ptr<Family>> make_all_graphs(uint64_t n);
Result<std::shared_ptr<Family>> make_edge_subgraphs(std::shared_ptr<Matrix> host, uint64_t k);
Result<std::shared_ptr<Family>> make_cayley_graphs(std::shared_ptr<Matrix> generators);
Result<std::shared_ptr<Family>> make_sublattices(std::shared_ptr<Matrix> gram, uint64_t index);

/* Closures under composition or multiplication, sorted lexicographically. */
Result<std::vector<Entry>> permutation_closure(const Matrix &generators, uint64_t limit);
Result<std::vector<Entry>> matrix_closure(const Matrix &generators, uint64_t limit);

const char *family_kind_name(Family::Kind k);

} // namespace lk
