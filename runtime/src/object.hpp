#pragma once
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace lk {

struct Error {
    int status;
    std::string message;
};

template <class T> struct Result {
    bool ok;
    T value;
    Error error;
    static Result success(T v) { return Result{true, std::move(v), {}}; }
    static Result failure(int status, std::string msg) { return Result{false, T{}, Error{status, std::move(msg)}}; }
};
struct Unit {};
using Status = Result<Unit>;
inline Status ok() { return Status::success(Unit{}); }
inline Status fail(int status, std::string msg) { return Status::failure(status, std::move(msg)); }

using Entry = uint32_t;

/* Matrix::p for a batch of natural-number matrices (entries < 2^32, no arithmetic meaning;
 * interchange kind "lk.naturals"). Members of `range` and `words` families are these. */
constexpr uint64_t NATURALS = UINT64_MAX;

/* A batch of count rows x cols matrices. p is the prime, 0 for a batch of permutations
 * (rows == 1, each entry a point index < cols; interchange kind "orbits.perms"), or NATURALS. */
struct Matrix {
    uint64_t p = 0, count = 0, rows = 0, cols = 0;
    std::vector<Entry> entries;
    const Entry *at(uint64_t i) const { return entries.data() + i * rows * cols; }
    Entry *at(uint64_t i) { return entries.data() + i * rows * cols; }
};

struct Basis {
    uint64_t p = 0, count = 0, cols = 0;
    std::vector<uint64_t> offsets;
    std::vector<Entry> entries;
};

struct Solutions {
    uint64_t p = 0, count = 0, length = 0;
    std::vector<uint8_t> solvable;
    std::vector<Entry> entries;
};

struct Inverses {
    uint64_t p = 0, count = 0, n = 0;
    std::vector<uint8_t> invertible;
    std::vector<Entry> entries;
};

struct Witness {
    uint64_t p = 0, count = 0, rows = 0, cols = 0;
    std::vector<Entry> r, t;
};

struct CycleIndex {
    uint64_t degree = 0, denominator = 0;
    std::vector<uint64_t> multiplicities;
    std::vector<uint64_t> cycles;
};

struct U64Matrices {
    uint64_t count = 0, rows = 0, cols = 0;
    std::vector<uint64_t> entries;
};

struct Integers {
    std::vector<uint64_t> values;
};

struct Count {
    uint64_t value = 0, visited = 0, family_size = 0;
};

struct Histogram {
    uint64_t visited = 0, family_size = 0;
    std::vector<uint64_t> bins;
};

struct Hits {
    uint64_t p = 0, rows = 0, cols = 0, total = 0, visited = 0, family_size = 0;
    std::vector<uint64_t> indices;
    std::vector<Entry> members;
};

/* `first`: the least index whose value is true, materialised; `found` 0 and no member when
 * there is none. `visited` is index + 1 when found (every member below was decided), else size. */
struct First {
    uint64_t p = 0, rows = 0, cols = 0, found = 0, index = 0, visited = 0, family_size = 0;
    std::vector<Entry> member;
};

/* `max` / `min`: the extreme value and the least index attaining it, materialised. */
struct Extremum {
    uint64_t p = 0, rows = 0, cols = 0, value = 0, index = 0, visited = 0, family_size = 0;
    std::vector<Entry> member;
};

struct Family;

struct Object {
    std::string kind;
    std::shared_ptr<Matrix> matrix;
    std::shared_ptr<Basis> basis;
    std::shared_ptr<Solutions> solutions;
    std::shared_ptr<Inverses> inverses;
    std::shared_ptr<Witness> witness;
    std::shared_ptr<CycleIndex> cycle_index;
    std::shared_ptr<U64Matrices> u64_matrices;
    std::shared_ptr<Integers> integers;
    std::shared_ptr<Count> count;
    std::shared_ptr<Histogram> histogram;
    std::shared_ptr<Hits> hits;
    std::shared_ptr<First> first;
    std::shared_ptr<Extremum> extremum;
    std::shared_ptr<Family> family;
    std::map<std::string, uint64_t> params() const;
};

unsigned entry_width(uint64_t p); /* 4 when p == 0 (permutations) or NATURALS */
inline const char *matrix_kind(const Matrix &m) {
    return m.p == 0 ? "orbits.perms" : m.p == NATURALS ? "lk.naturals" : "gfp.matrix";
}
bool is_prime(uint64_t p);

Result<std::shared_ptr<Object>> decode(const uint8_t *bytes, size_t len);
std::vector<uint8_t> encode(const Object &o);

} // namespace lk
