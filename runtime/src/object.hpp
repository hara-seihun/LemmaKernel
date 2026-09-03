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

/* A batch of count rows x cols matrices. p is the prime, or 0 for a batch of permutations
 * (rows == 1, each entry a point index < cols; interchange kind "orbits.perms"). */
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

struct Family;

struct Object {
    std::string kind;
    std::shared_ptr<Matrix> matrix;
    std::shared_ptr<Basis> basis;
    std::shared_ptr<Solutions> solutions;
    std::shared_ptr<Inverses> inverses;
    std::shared_ptr<Witness> witness;
    std::shared_ptr<Integers> integers;
    std::shared_ptr<Count> count;
    std::shared_ptr<Histogram> histogram;
    std::shared_ptr<Hits> hits;
    std::shared_ptr<Family> family;
    std::map<std::string, uint64_t> params() const;
};

unsigned entry_width(uint64_t p); /* 4 when p == 0 (permutations) */
inline const char *matrix_kind(const Matrix &m) { return m.p == 0 ? "orbits.perms" : "gfp.matrix"; }
bool is_prime(uint64_t p);

Result<std::shared_ptr<Object>> decode(const uint8_t *bytes, size_t len);
std::vector<uint8_t> encode(const Object &o);

} // namespace lk
