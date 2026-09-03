/* orbits generic backend: portable C++.
 *
 * An action is a permutation of a family's member indices, one per generator. Every operation
 * here is "start at a member, walk its orbit by breadth-first search over the generators, look
 * at the indices you meet". The ranking through the family's canonical order (unrank the index,
 * act, canonicalise, rank) is what makes "least index" meaningful and identical across backends.
 *
 * Costs, so the next backend knows what to beat: one BFS per member, so a family of N members
 * with orbits of size s costs N * s * |generators| index maps. Orderly generation (refusing to
 * extend a prefix that is not minimal under the point stabiliser) would remove the s factor for
 * is_canonical on subsets; nothing of the kind is done here. */
#include "../../../../runtime/src/reduce.hpp"
#include "../../../gfp/backends/generic/field.hpp"

#include <unordered_map>
#include <unordered_set>

namespace lk::orbits {
namespace {

using R = Result<std::shared_ptr<Object>>;
constexpr int INVALID = 1;

/* Binomial table C(i, j) for 0 <= i <= D, 0 <= j <= k, saturating at UINT64_MAX. */
struct Binomials {
    uint64_t D, k;
    std::vector<uint64_t> c;
    Binomials(uint64_t d, uint64_t kk) : D(d), k(kk), c((d + 1) * (kk + 1), 0) {
        for (uint64_t i = 0; i <= D; ++i) {
            at(i, 0) = 1;
            for (uint64_t j = 1; j <= std::min(i, k); ++j) {
                uint64_t a = at(i - 1, j - 1), b = j <= i - 1 ? at(i - 1, j) : 0;
                at(i, j) = a + b < a ? UINT64_MAX : a + b;
            }
        }
    }
    uint64_t &at(uint64_t i, uint64_t j) { return c[i * (k + 1) + j]; }
    uint64_t get(uint64_t i, uint64_t j) const { return j > i ? 0 : c[i * (k + 1) + j]; }
};

/* The action of a list of permutations of D points on the k-subsets of the dictionary. */
struct PermOnSubsets {
    uint64_t D, k;
    const Entry *perms; /* count x D */
    uint64_t count;
    Binomials binom;
    std::vector<uint64_t> tuple, image;

    PermOnSubsets(uint64_t d, uint64_t kk, const Entry *p, uint64_t c) : D(d), k(kk), perms(p), count(c), binom(d, kk), tuple(kk), image(kk) {}

    void unrank(uint64_t index) {
        uint64_t prev = 0;
        for (uint64_t j = 0; j < k; ++j) {
            uint64_t c = prev;
            for (;; ++c) {
                uint64_t below = binom.get(D - 1 - c, k - 1 - j);
                if (index < below) break;
                index -= below;
            }
            tuple[j] = c;
            prev = c + 1;
        }
    }
    uint64_t rank(const std::vector<uint64_t> &t) const {
        uint64_t index = 0, prev = 0;
        for (uint64_t j = 0; j < k; ++j) {
            for (uint64_t c = prev; c < t[j]; ++c) index += binom.get(D - 1 - c, k - 1 - j);
            prev = t[j] + 1;
        }
        return index;
    }
    uint64_t apply(uint64_t g, uint64_t index) {
        unrank(index);
        const Entry *perm = perms + g * D;
        for (uint64_t j = 0; j < k; ++j) image[j] = perm[tuple[j]];
        std::sort(image.begin(), image.end());
        return rank(image);
    }
};

/* Matrices acting on the right of a matrix family: M -> canonical(M A). */
struct MatOnFamily {
    const Family &fam;
    const Matrix &mats; /* count x n x n */
    gfp::Field field;
    gfp::EchelonBasis basis;
    Matrix member, scratch;
    std::vector<Entry> product;
    std::vector<uint32_t> piv;

    MatOnFamily(const Family &f, const Matrix &m) : fam(f), mats(m), field(f.prime()), basis(f.prime(), f.cols()) {
        scratch.p = f.prime(); scratch.count = 1; scratch.rows = f.rows(); scratch.cols = f.cols();
    }
    Result<uint64_t> apply(uint64_t g, uint64_t index) {
        auto st = fam.member_into(index, member);
        if (!st.ok) return Result<uint64_t>::failure(st.error.status, st.error.message);
        uint64_t rows = member.rows, n = member.cols;
        const Entry *a = mats.at(g);
        product.assign(rows * n, 0);
        for (uint64_t r = 0; r < rows; ++r)
            for (uint64_t i = 0; i < n; ++i) {
                uint64_t v = member.entries[r * n + i];
                if (!v) continue;
                for (uint64_t j = 0; j < n; ++j)
                    product[r * n + j] = (Entry)field.reduce(product[r * n + j] + v * a[i * n + j]);
            }
        if (fam.kind == Family::Kind::Grassmannian) {
            basis.clear();
            for (uint64_t r = 0; r < rows; ++r) basis.add(product.data() + r * n);
            if (basis.rank() != rows) return Result<uint64_t>::failure(INVALID, "matrix " + std::to_string(g) + " is singular on this subspace: not a group action on the Grassmannian");
            basis.rref(scratch.entries, piv);
        } else {
            scratch.entries.swap(product);
        }
        return fam.index_of(scratch);
    }
};

/* Closure of n x n matrices over F_p under multiplication, up to `limit` elements. */
Result<uint64_t> matrix_group_order(const Matrix &gens, uint64_t limit) {
    uint64_t n = gens.cols, nn = n * n, p = gens.p;
    gfp::Field field(p);
    std::vector<Entry> store;
    struct Hash {
        const std::vector<Entry> *store; uint64_t nn;
        size_t operator()(uint64_t i) const {
            uint64_t h = 1469598103934665603ULL;
            for (uint64_t j = 0; j < nn; ++j) { h ^= (*store)[i * nn + j]; h *= 1099511628211ULL; }
            return (size_t)h;
        }
    };
    struct Eq {
        const std::vector<Entry> *store; uint64_t nn;
        bool operator()(uint64_t a, uint64_t b) const { return std::equal(store->begin() + a * nn, store->begin() + (a + 1) * nn, store->begin() + b * nn); }
    };
    std::unordered_set<uint64_t, Hash, Eq> seen(64, Hash{&store, nn}, Eq{&store, nn});
    for (uint64_t i = 0; i < n; ++i)
        for (uint64_t j = 0; j < n; ++j) store.push_back(i == j);
    seen.insert(0);
    std::vector<Entry> tmp(nn);
    for (uint64_t front = 0; front < store.size() / nn; ++front) {
        for (uint64_t g = 0; g < gens.count; ++g) {
            const Entry *a = store.data() + front * nn, *b = gens.at(g);
            for (uint64_t i = 0; i < n; ++i)
                for (uint64_t j = 0; j < n; ++j) {
                    uint64_t acc = 0;
                    for (uint64_t l = 0; l < n; ++l) acc = field.reduce(acc + (uint64_t)a[i * n + l] * b[l * n + j]);
                    tmp[i * n + j] = (Entry)acc;
                }
            uint64_t idx = store.size() / nn;
            store.insert(store.end(), tmp.begin(), tmp.end());
            if (!seen.insert(idx).second) store.resize(idx * nn);
            else if (idx + 1 > limit) return Result<uint64_t>::failure(INVALID, "group has more than " + std::to_string(limit) + " elements");
        }
    }
    return Result<uint64_t>::success(store.size() / nn);
}

struct Setup {
    const Family *fam = nullptr;
    std::shared_ptr<Matrix> group; /* perms (p == 0) or matrices */
    bool perms = false;
};

Result<Setup> setup_action(const Request &req) {
    Setup s;
    s.fam = req.family.get();
    auto it = req.handle_args.find("group");
    if (it == req.handle_args.end() || !it->second->matrix) return Result<Setup>::failure(INVALID, "`group` must be an orbits.perms or gfp.matrix batch of generators");
    s.group = it->second->matrix;
    s.perms = s.group->p == 0;
    const Family &f = *s.fam;
    if (s.perms) {
        if (f.kind != Family::Kind::Subsets && f.kind != Family::Kind::SubsetsOf)
            return Result<Setup>::failure(INVALID, "permutation generators act on `subsets` and `subsets_of` families only");
        if (s.group->cols != f.data->count)
            return Result<Setup>::failure(INVALID, "permutations of " + std::to_string(s.group->cols) + " points cannot act on a dictionary of " + std::to_string(f.data->count) + " rows");
    } else {
        if (f.kind != Family::Kind::Grassmannian && f.kind != Family::Kind::AllMatrices)
            return Result<Setup>::failure(INVALID, "matrix generators act on `grassmannian` and `all_matrices` families only");
        if (s.group->rows != s.group->cols || s.group->cols != f.cols() || s.group->p != f.prime())
            return Result<Setup>::failure(INVALID, "matrix generators must be n x n over the family's prime with n = member columns");
    }
    if (s.group->count == 0) return Result<Setup>::failure(INVALID, "need at least one generator");
    return Result<Setup>::success(s);
}

/* One thread's orbit machinery. */
struct Walker {
    Setup s;
    std::unique_ptr<PermOnSubsets> perm_action;
    std::unique_ptr<MatOnFamily> mat_action;
    std::vector<uint64_t> queue;
    std::unordered_set<uint64_t> seen;

    explicit Walker(const Setup &setup) : s(setup) {
        if (s.perms) perm_action = std::make_unique<PermOnSubsets>(s.fam->data->count, s.fam->k, s.group->entries.data(), s.group->count);
        else mat_action = std::make_unique<MatOnFamily>(*s.fam, *s.group);
    }
    Result<uint64_t> apply(uint64_t g, uint64_t index) {
        if (perm_action) return Result<uint64_t>::success(perm_action->apply(g, index));
        return mat_action->apply(g, index);
    }
    /* Orbit of `start`. Returns (least index, orbit size); with `stop_below` the search ends as
     * soon as an index below `start` appears (enough for is_canonical). */
    Result<std::pair<uint64_t, uint64_t>> orbit(uint64_t start, bool stop_below) {
        using P = Result<std::pair<uint64_t, uint64_t>>;
        queue.clear();
        seen.clear();
        queue.push_back(start);
        seen.insert(start);
        uint64_t least = start;
        for (size_t front = 0; front < queue.size(); ++front) {
            uint64_t x = queue[front];
            for (uint64_t g = 0; g < s.group->count; ++g) {
                auto y = apply(g, x);
                if (!y.ok) return P::failure(y.error.status, y.error.message);
                if (seen.insert(y.value).second) {
                    queue.push_back(y.value);
                    if (y.value < least) {
                        least = y.value;
                        if (stop_below) return P::success({least, 0});
                    }
                }
            }
        }
        return P::success({least, queue.size()});
    }
};

enum class Op { IsCanonical, CanonicalIndex, OrbitSize, StabilizerOrder };

R run_orbit_op(const Request &req, Op op) {
    auto setup = setup_action(req);
    if (!setup.ok) return R::failure(setup.error.status, setup.error.message);
    const Family &fam = *req.family;
    auto size_r = fam.size();
    if (!size_r.ok) return R::failure(size_r.error.status, size_r.error.message);
    uint64_t size = size_r.value;
    uint64_t order = 0;
    if (op == Op::StabilizerOrder) {
        if (setup.value.perms) {
            auto c = permutation_closure(*setup.value.group, 1ULL << 26);
            if (!c.ok) return R::failure(c.error.status, c.error.message);
            order = c.value.size() / setup.value.group->cols;
        } else {
            auto c = matrix_group_order(*setup.value.group, 1ULL << 26);
            if (!c.ok) return R::failure(c.error.status, c.error.message);
            order = c.value;
        }
    }
    Reduction reduction = parse_reduction(req.reduction);
    Shared shared;
    auto st = prepare_all(reduction, size, shared);
    if (!st.ok) return R::failure(st.error.status, st.error.message);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, size ? size : 1));
    std::vector<Walker> walkers;
    std::vector<Accumulator> accs;
    for (uint32_t t = 0; t < threads; ++t) {
        walkers.emplace_back(setup.value);
        accs.emplace_back(reduction, &shared);
    }
    auto statuses = parallel_ranges(size, threads, [&](uint32_t t, uint64_t begin, uint64_t end) -> Status {
        for (uint64_t i = begin; i < end; ++i) {
            if (accs[t].exhausted(i)) break;
            auto o = walkers[t].orbit(i, op == Op::IsCanonical);
            if (!o.ok) return fail(o.error.status, o.error.message);
            switch (op) {
            case Op::IsCanonical: accs[t].boolean(i, o.value.first == i); break;
            case Op::CanonicalIndex: accs[t].integer(i, o.value.first); break;
            case Op::OrbitSize: accs[t].integer(i, o.value.second); break;
            case Op::StabilizerOrder: accs[t].integer(i, order / o.value.second); break;
            }
        }
        return ok();
    });
    for (const auto &s : statuses)
        if (!s.ok) return R::failure(s.error.status, s.error.message);
    return assemble(req, reduction, accs, shared);
}

/* Fixed k-subsets of a permutation: a subset is fixed iff it is a union of cycles, so the count is
 * the coefficient of x^k in the product over cycles of (1 + x^len). O(n k) per element. */
uint64_t fixed_subsets(const Entry *perm, uint64_t n, uint64_t k, std::vector<uint8_t> &seen, std::vector<uint64_t> &poly) {
    seen.assign(n, 0);
    poly.assign(k + 1, 0);
    poly[0] = 1;
    for (uint64_t i = 0; i < n; ++i) {
        if (seen[i]) continue;
        uint64_t len = 0;
        for (uint64_t j = i; !seen[j]; j = perm[j]) { seen[j] = 1; ++len; }
        if (len > k) continue;
        for (uint64_t d = k; d >= len; --d) poly[d] += poly[d - len];
    }
    return poly[k];
}

R run_fixed_points(const Request &req) {
    const Family &group = *req.family;
    auto it = req.handle_args.find("on");
    if (it == req.handle_args.end() || !it->second->family) return R::failure(INVALID, "`on` must be a family");
    const Family &on = *it->second->family;
    if (on.kind != Family::Kind::Subsets && on.kind != Family::Kind::SubsetsOf)
        return R::failure(INVALID, "fixed_points: permutations act on `subsets` and `subsets_of` families only");
    if (group.data->cols != on.data->count) return R::failure(INVALID, "fixed_points: the group's points must be the dictionary rows of `on`");
    auto elems = group.group_elements();
    if (!elems.ok) return R::failure(elems.error.status, elems.error.message);
    uint64_t n = group.data->cols, order = elems.value->size() / n;
    Reduction reduction = parse_reduction(req.reduction);
    Shared shared;
    auto st = prepare_all(reduction, order, shared);
    if (!st.ok) return R::failure(st.error.status, st.error.message);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, order));
    std::vector<Accumulator> accs;
    for (uint32_t t = 0; t < threads; ++t) accs.emplace_back(reduction, &shared);
    auto statuses = parallel_ranges(order, threads, [&](uint32_t t, uint64_t begin, uint64_t end) -> Status {
        std::vector<uint8_t> seen;
        std::vector<uint64_t> poly;
        for (uint64_t g = begin; g < end; ++g)
            accs[t].integer(g, fixed_subsets(elems.value->data() + g * n, n, on.k, seen, poly));
        return ok();
    });
    for (const auto &s : statuses)
        if (!s.ok) return R::failure(s.error.status, s.error.message);
    return assemble(req, reduction, accs, shared);
}

R run_projective_action(const Request &req) {
    const Matrix &batch = *req.family->data;
    auto it = req.handle_args.find("points");
    if (it == req.handle_args.end() || !it->second->matrix) return R::failure(INVALID, "`points` must be a gfp.matrix dictionary of 1 x n rows");
    Matrix pts = *it->second->matrix;
    if (pts.count == 1 && pts.rows != 1) { pts.count = pts.rows; pts.rows = 1; }
    uint64_t n = batch.cols;
    if (batch.rows != n) return R::failure(INVALID, "projective_action: members must be square n x n matrices");
    if (pts.p != batch.p || pts.rows != 1 || pts.cols != n) return R::failure(INVALID, "projective_action: points must be 1 x n rows over the same prime");
    gfp::Field field(batch.p);
    auto normalise = [&](std::vector<Entry> &v) -> bool {
        for (uint64_t i = 0; i < n; ++i)
            if (v[i]) {
                if (v[i] != 1) field.scale(v.data(), field.inverse(v[i]), n);
                return true;
            }
        return false;
    };
    std::unordered_map<std::string, uint64_t> index;
    for (uint64_t i = 0; i < pts.count; ++i) {
        std::vector<Entry> v(pts.at(i), pts.at(i) + n);
        std::vector<Entry> w = v;
        if (!normalise(w) || w != v) return R::failure(INVALID, "projective_action: point " + std::to_string(i) + " is zero or not scaled to leading entry 1");
        std::string key((const char *)v.data(), n * sizeof(Entry));
        if (!index.emplace(key, i).second) return R::failure(INVALID, "projective_action: point " + std::to_string(i) + " is a duplicate");
    }
    auto out = std::make_shared<Matrix>();
    out->p = 0; out->count = batch.count; out->rows = 1; out->cols = pts.count;
    out->entries.assign(batch.count * pts.count, 0);
    std::vector<Entry> w(n);
    for (uint64_t g = 0; g < batch.count; ++g) {
        const Entry *a = batch.at(g);
        for (uint64_t i = 0; i < pts.count; ++i) {
            const Entry *v = pts.at(i);
            std::fill(w.begin(), w.end(), 0);
            for (uint64_t l = 0; l < n; ++l) {
                if (!v[l]) continue;
                for (uint64_t j = 0; j < n; ++j) w[j] = (Entry)field.reduce(w[j] + (uint64_t)v[l] * a[l * n + j]);
            }
            if (!normalise(w)) return R::failure(INVALID, "projective_action: matrix " + std::to_string(g) + " sends point " + std::to_string(i) + " to zero");
            auto f = index.find(std::string((const char *)w.data(), n * sizeof(Entry)));
            if (f == index.end()) return R::failure(INVALID, "projective_action: matrix " + std::to_string(g) + " sends point " + std::to_string(i) + " outside the dictionary");
            out->entries[g * pts.count + i] = (Entry)f->second;
        }
    }
    auto o = std::make_shared<Object>();
    o->kind = "orbits.perms";
    o->matrix = out;
    return R::success(o);
}

R run(const Request &req) {
    if (req.op == "is_canonical") return run_orbit_op(req, Op::IsCanonical);
    if (req.op == "canonical_index") return run_orbit_op(req, Op::CanonicalIndex);
    if (req.op == "orbit_size") return run_orbit_op(req, Op::OrbitSize);
    if (req.op == "stabilizer_order") return run_orbit_op(req, Op::StabilizerOrder);
    if (req.op == "fixed_points") return run_fixed_points(req);
    if (req.op == "projective_action") return run_projective_action(req);
    return R::failure(4, "unknown orbits operation " + req.op);
}

BackendRegistration registration{Backend{
    "orbits", "generic",
    [] { return true; },
    [](const Request &) { return true; },
    run,
    0}};

} // namespace
} // namespace lk::orbits
