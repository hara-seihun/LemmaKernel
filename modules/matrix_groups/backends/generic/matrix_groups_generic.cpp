/* Portable backend for structural queries on matrix groups.
 *
 * A subsets member has one flattened n x n generator in each row. Order uses a Schreier
 * stabilizer chain for the right action on the standard basis vectors. The other queries reduce
 * their defining finite-field systems with the shared gfp elimination code. */
#include "../../../../runtime/src/reduce.hpp"
#include "../../../gfp/backends/generic/field.hpp"

#include <cmath>
#include <functional>
#include <unordered_map>
#include <unordered_set>

namespace lk::matrix_groups {
namespace {

using R = Result<std::shared_ptr<Object>>;
using Flat = std::vector<Entry>;
using Generators = std::vector<Flat>;
constexpr int INVALID = 1;

struct FlatHash {
    size_t operator()(const Flat &v) const {
        uint64_t h = 1469598103934665603ULL;
        for (Entry x : v) {
            h ^= x;
            h *= 1099511628211ULL;
        }
        return (size_t)h;
    }
};

Flat identity(uint64_t n) {
    Flat out(n * n, 0);
    for (uint64_t i = 0; i < n; ++i) out[i * n + i] = 1;
    return out;
}

Flat multiply(const Flat &a, const Flat &b, uint64_t n, const gfp::Field &f) {
    Flat out(n * n, 0);
    for (uint64_t i = 0; i < n; ++i)
        for (uint64_t k = 0; k < n; ++k) {
            Entry x = a[i * n + k];
            if (!x) continue;
            for (uint64_t j = 0; j < n; ++j)
                out[i * n + j] = (Entry)f.reduce((uint64_t)out[i * n + j] + (uint64_t)x * b[k * n + j]);
        }
    return out;
}

Flat vector_multiply(const Flat &v, const Flat &a, uint64_t n, const gfp::Field &f) {
    Flat out(n, 0);
    for (uint64_t k = 0; k < n; ++k) {
        if (!v[k]) continue;
        for (uint64_t j = 0; j < n; ++j)
            out[j] = (Entry)f.reduce((uint64_t)out[j] + (uint64_t)v[k] * a[k * n + j]);
    }
    return out;
}

uint64_t matrix_rank(const Entry *entries, uint64_t rows, uint64_t cols, uint64_t p) {
    gfp::EchelonBasis basis(p, cols);
    for (uint64_t i = 0; i < rows; ++i) basis.add(entries + i * cols);
    return basis.rank();
}

uint64_t matrix_rank(const Flat &entries, uint64_t rows, uint64_t cols, uint64_t p) {
    return matrix_rank(entries.data(), rows, cols, p);
}

bool invert(const Flat &a, uint64_t n, const gfp::Field &f, Flat &out) {
    uint64_t width = 2 * n;
    Flat aug(n * width, 0);
    for (uint64_t i = 0; i < n; ++i) {
        std::copy(a.begin() + i * n, a.begin() + (i + 1) * n, aug.begin() + i * width);
        aug[i * width + n + i] = 1;
    }
    uint64_t row = 0;
    for (uint64_t col = 0; col < n; ++col) {
        uint64_t pivot = row;
        while (pivot < n && aug[pivot * width + col] == 0) ++pivot;
        if (pivot == n) return false;
        if (pivot != row)
            std::swap_ranges(aug.begin() + pivot * width, aug.begin() + (pivot + 1) * width,
                             aug.begin() + row * width);
        Entry *prow = aug.data() + row * width;
        if (prow[col] != 1) f.scale(prow, f.inverse(prow[col]), width);
        for (uint64_t i = 0; i < n; ++i) {
            if (i == row) continue;
            Entry c = aug[i * width + col];
            if (c) f.subtract_multiple(aug.data() + i * width, prow, c, width);
        }
        ++row;
    }
    out.assign(n * n, 0);
    for (uint64_t i = 0; i < n; ++i)
        std::copy(aug.begin() + i * width + n, aug.begin() + (i + 1) * width, out.begin() + i * n);
    return true;
}

struct Setup {
    const Family *family = nullptr;
    uint64_t p = 0;
    uint64_t n = 0;
};

Result<Setup> setup(const Request &req) {
    if (req.family->kind != Family::Kind::Subsets)
        return Result<Setup>::failure(INVALID, "matrix_groups operations are defined on subsets families only");
    const Family &family = *req.family;
    uint64_t width = family.data->cols;
    uint64_t n = (uint64_t)std::sqrt((long double)width);
    while (n && n < width / n) ++n;
    while (n && n > width / n) --n;
    if (n == 0 || width % n != 0 || width / n != n)
        return Result<Setup>::failure(INVALID, "generator rows must have positive perfect square length n*n");
    for (uint64_t i = 0; i < family.data->count; ++i)
        if (matrix_rank(family.data->at(i), n, n, family.prime()) != n)
            return Result<Setup>::failure(INVALID, "generator dictionary row " + std::to_string(i) +
                " is not an invertible n x n matrix");
    return Result<Setup>::success(Setup{&family, family.prime(), n});
}

Result<Generators> member_generators(const Setup &s, uint64_t index) {
    Matrix member;
    auto st = s.family->member_into(index, member);
    if (!st.ok) return Result<Generators>::failure(st.error.status, st.error.message);
    Generators out(member.rows);
    uint64_t nn = s.n * s.n;
    for (uint64_t i = 0; i < member.rows; ++i)
        out[i].assign(member.entries.begin() + i * nn, member.entries.begin() + (i + 1) * nn);
    return Result<Generators>::success(std::move(out));
}

void insert_with_inverse(Generators &into, std::unordered_set<Flat, FlatHash> &seen,
                         const Flat &a, const Flat &one, uint64_t n, const gfp::Field &f) {
    if (a == one || !seen.insert(a).second) return;
    into.push_back(a);
    Flat inv;
    if (invert(a, n, f, inv) && inv != one && seen.insert(inv).second) into.push_back(std::move(inv));
}

/* Schreier stabilizer chain on e_0, ..., e_{n-1}. Transversals are matrices t_y with e_i t_y = y. */
Result<uint64_t> schreier_order(const Generators &input, uint64_t n, uint64_t p) {
    gfp::Field f(p);
    Flat one = identity(n);
    Generators gens;
    std::unordered_set<Flat, FlatHash> initial;
    for (const Flat &a : input) insert_with_inverse(gens, initial, a, one, n, f);

    uint64_t order = 1;
    for (uint64_t base = 0; base < n && !gens.empty(); ++base) {
        Flat e(n, 0);
        e[base] = 1;
        std::vector<Flat> orbit{e};
        std::vector<Flat> transversal{one};
        std::unordered_map<Flat, uint64_t, FlatHash> index;
        index.emplace(e, 0);
        for (size_t front = 0; front < orbit.size(); ++front) {
            Flat t = transversal[front];
            for (const Flat &s : gens) {
                Flat y = vector_multiply(orbit[front], s, n, f);
                if (index.count(y)) continue;
                uint64_t j = orbit.size();
                index.emplace(y, j);
                orbit.push_back(std::move(y));
                transversal.push_back(multiply(t, s, n, f));
            }
        }
        if (orbit.size() && order > UINT64_MAX / orbit.size())
            return Result<uint64_t>::failure(INVALID, "group order does not fit in 64 bits");
        order *= orbit.size();

        std::vector<Flat> inverse_transversal(transversal.size());
        for (size_t i = 0; i < transversal.size(); ++i)
            if (!invert(transversal[i], n, f, inverse_transversal[i]))
                return Result<uint64_t>::failure(4, "internal Schreier transversal is singular");

        Generators stabilizer;
        std::unordered_set<Flat, FlatHash> seen;
        for (size_t x = 0; x < orbit.size(); ++x) {
            for (const Flat &s : gens) {
                Flat y = vector_multiply(orbit[x], s, n, f);
                auto found = index.find(y);
                if (found == index.end())
                    return Result<uint64_t>::failure(4, "internal Schreier orbit is not closed");
                Flat h = multiply(multiply(transversal[x], s, n, f), inverse_transversal[found->second], n, f);
                insert_with_inverse(stabilizer, seen, h, one, n, f);
            }
        }
        gens = std::move(stabilizer);
    }
    return Result<uint64_t>::success(order);
}

bool orbit_span_is_full(const Flat &v, const Generators &gens, uint64_t n, uint64_t p) {
    gfp::Field f(p);
    gfp::EchelonBasis basis(p, n);
    basis.add(v.data());
    for (uint64_t front = 0; front < basis.rank(); ++front) {
        Flat w(basis.row(front), basis.row(front) + n);
        for (const Flat &a : gens) {
            Flat image = vector_multiply(w, a, n, f);
            basis.add(image.data());
            if (basis.rank() == n) return true;
        }
    }
    return basis.rank() == n;
}

template <class Fn>
bool every_projective_vector(uint64_t p, uint64_t n, Fn fn) {
    Flat v(n, 0);
    std::function<bool(uint64_t)> tail = [&](uint64_t pos) {
        if (pos == n) return fn(v);
        for (uint64_t x = 0; x < p; ++x) {
            v[pos] = (Entry)x;
            if (!tail(pos + 1)) return false;
        }
        return true;
    };
    for (uint64_t lead = 0; lead < n; ++lead) {
        std::fill(v.begin(), v.end(), 0);
        v[lead] = 1;
        if (!tail(lead + 1)) return false;
    }
    return true;
}

bool is_irreducible(const Generators &gens, uint64_t n, uint64_t p) {
    return every_projective_vector(p, n, [&](const Flat &v) { return orbit_span_is_full(v, gens, n, p); });
}

uint64_t centralizer_dimension(const Generators &gens, uint64_t n, uint64_t p) {
    uint64_t vars = n * n;
    gfp::EchelonBasis equations(p, vars);
    Flat row(vars);
    for (const Flat &a : gens)
        for (uint64_t i = 0; i < n; ++i)
            for (uint64_t j = 0; j < n; ++j) {
                for (uint64_t x = 0; x < vars; ++x) {
                    uint64_t r = x / n, c = x % n;
                    Entry left = r == i ? a[c * n + j] : 0;
                    Entry right = c == j ? a[i * n + r] : 0;
                    row[x] = (Entry)((left + p - right) % p);
                }
                equations.add(row.data());
            }
    return vars - equations.rank();
}

std::vector<Flat> invariant_form_basis(const Generators &gens, uint64_t n, uint64_t p) {
    uint64_t vars = n * n;
    gfp::EchelonBasis equations(p, vars);
    Flat row(vars);
    for (const Flat &a : gens)
        for (uint64_t i = 0; i < n; ++i)
            for (uint64_t j = 0; j < n; ++j) {
                for (uint64_t x = 0; x < vars; ++x) {
                    uint64_t r = x / n, c = x % n;
                    uint64_t moved = (uint64_t)a[i * n + r] * a[j * n + c] % p;
                    uint64_t original = i == r && j == c;
                    row[x] = (Entry)((moved + p - original) % p);
                }
                equations.add(row.data());
            }
    Flat rref;
    std::vector<uint32_t> pivots;
    equations.rref(rref, pivots);
    std::vector<uint8_t> is_pivot(vars, 0);
    for (uint32_t c : pivots) is_pivot[c] = 1;
    std::vector<Flat> basis;
    for (uint64_t free = 0; free < vars; ++free) {
        if (is_pivot[free]) continue;
        Flat v(vars, 0);
        v[free] = 1;
        for (uint64_t i = 0; i < pivots.size(); ++i) {
            Entry x = rref[i * vars + free];
            v[pivots[i]] = x ? (Entry)(p - x) : 0;
        }
        basis.push_back(std::move(v));
    }
    return basis;
}

bool preserves_form(const Generators &gens, uint64_t n, uint64_t p) {
    std::vector<Flat> basis = invariant_form_basis(gens, n, p);
    if (basis.empty()) return false;
    Flat candidate(n * n, 0);
    std::vector<Entry> coeffs(basis.size(), 0);
    std::function<bool(size_t, bool)> search = [&](size_t pos, bool nonzero) {
        if (pos == basis.size()) {
            if (!nonzero) return false;
            std::fill(candidate.begin(), candidate.end(), 0);
            for (size_t i = 0; i < basis.size(); ++i) {
                if (!coeffs[i]) continue;
                for (uint64_t j = 0; j < n * n; ++j)
                    candidate[j] = (Entry)(((uint64_t)candidate[j] + (uint64_t)coeffs[i] * basis[i][j]) % p);
            }
            return matrix_rank(candidate, n, n, p) == n;
        }
        for (uint64_t x = 0; x < p; ++x) {
            coeffs[pos] = (Entry)x;
            if (search(pos + 1, nonzero || x != 0)) return true;
        }
        return false;
    };
    return search(0, false);
}

Result<Flat> subspace_image(const Flat &w, uint64_t d, const Flat &a, uint64_t n, uint64_t p) {
    gfp::Field f(p);
    Flat product(d * n, 0);
    for (uint64_t i = 0; i < d; ++i)
        for (uint64_t k = 0; k < n; ++k) {
            Entry x = w[i * n + k];
            if (!x) continue;
            for (uint64_t j = 0; j < n; ++j)
                product[i * n + j] = (Entry)f.reduce((uint64_t)product[i * n + j] + (uint64_t)x * a[k * n + j]);
        }
    gfp::EchelonBasis basis(p, n);
    for (uint64_t i = 0; i < d; ++i) basis.add(product.data() + i * n);
    if (basis.rank() != d) return Result<Flat>::failure(4, "invertible generator lowered a subspace dimension");
    Flat out;
    std::vector<uint32_t> pivots;
    basis.rref(out, pivots);
    return Result<Flat>::success(std::move(out));
}

Result<bool> has_block_of_dimension(const Generators &gens, uint64_t n, uint64_t p, uint64_t d) {
    auto family_r = make_grassmannian(p, n, d);
    if (!family_r.ok) return Result<bool>::failure(family_r.error.status, family_r.error.message);
    std::shared_ptr<Family> family = family_r.value;
    auto size_r = family->size();
    if (!size_r.ok) return Result<bool>::failure(size_r.error.status, size_r.error.message);
    uint64_t wanted = n / d;
    Matrix member;
    for (uint64_t start = 0; start < size_r.value; ++start) {
        std::vector<uint64_t> queue{start};
        std::unordered_set<uint64_t> seen{start};
        bool too_many = false;
        for (size_t front = 0; front < queue.size() && !too_many; ++front) {
            auto st = family->member_into(queue[front], member);
            if (!st.ok) return Result<bool>::failure(st.error.status, st.error.message);
            for (const Flat &a : gens) {
                auto image = subspace_image(member.entries, d, a, n, p);
                if (!image.ok) return Result<bool>::failure(image.error.status, image.error.message);
                Matrix canonical;
                canonical.p = p; canonical.count = 1; canonical.rows = d; canonical.cols = n;
                canonical.entries = std::move(image.value);
                auto index = family->index_of(canonical);
                if (!index.ok) return Result<bool>::failure(index.error.status, index.error.message);
                if (seen.insert(index.value).second) {
                    queue.push_back(index.value);
                    if (queue.size() > wanted) { too_many = true; break; }
                }
            }
        }
        if (too_many || queue.size() != wanted) continue;
        gfp::EchelonBasis sum(p, n);
        for (uint64_t index : queue) {
            auto st = family->member_into(index, member);
            if (!st.ok) return Result<bool>::failure(st.error.status, st.error.message);
            for (uint64_t i = 0; i < d; ++i) sum.add(member.entries.data() + i * n);
        }
        if (sum.rank() == n) return Result<bool>::success(true);
    }
    return Result<bool>::success(false);
}

Result<bool> is_imprimitive(const Generators &gens, uint64_t n, uint64_t p) {
    if (!is_irreducible(gens, n, p)) return Result<bool>::success(false);
    for (uint64_t d = 1; d < n; ++d) {
        if (n % d) continue;
        auto found = has_block_of_dimension(gens, n, p, d);
        if (!found.ok || found.value) return found;
    }
    return Result<bool>::success(false);
}

enum class Op { Order, Irreducible, AbsolutelyIrreducible, PreservesForm, Imprimitive };

R run(const Request &req) {
    auto configured = setup(req);
    if (!configured.ok) return R::failure(configured.error.status, configured.error.message);
    Setup s = configured.value;
    Op op;
    bool boolean = true;
    if (req.op == "order") { op = Op::Order; boolean = false; }
    else if (req.op == "is_irreducible") op = Op::Irreducible;
    else if (req.op == "is_absolutely_irreducible") op = Op::AbsolutelyIrreducible;
    else if (req.op == "preserves_form") op = Op::PreservesForm;
    else if (req.op == "is_imprimitive") op = Op::Imprimitive;
    else return R::failure(4, "unknown matrix_groups operation " + req.op);

    auto size_r = s.family->size();
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
        for (uint64_t i = begin; i < end; ++i) {
            if (accs[t].exhausted(i)) break;
            auto group = member_generators(s, i);
            if (!group.ok) return fail(group.error.status, group.error.message);
            if (!boolean) {
                auto value = schreier_order(group.value, s.n, s.p);
                if (!value.ok) return fail(value.error.status, value.error.message);
                accs[t].integer(i, value.value);
                continue;
            }
            Result<bool> value = Result<bool>::success(false);
            switch (op) {
            case Op::Irreducible:
                value.value = is_irreducible(group.value, s.n, s.p); break;
            case Op::AbsolutelyIrreducible:
                value.value = is_irreducible(group.value, s.n, s.p) &&
                              centralizer_dimension(group.value, s.n, s.p) == 1; break;
            case Op::PreservesForm:
                value.value = preserves_form(group.value, s.n, s.p); break;
            case Op::Imprimitive:
                value = is_imprimitive(group.value, s.n, s.p); break;
            case Op::Order: break;
            }
            if (!value.ok) return fail(value.error.status, value.error.message);
            accs[t].boolean(i, value.value);
        }
        return ok();
    });
    for (const Status &status : statuses)
        if (!status.ok) return R::failure(status.error.status, status.error.message);
    return assemble(req, reduction, accs, shared);
}

BackendRegistration registration{Backend{
    "matrix_groups", "generic",
    [] { return true; },
    [](const Request &) { return true; },
    run,
    0}};

} // namespace
} // namespace lk::matrix_groups
