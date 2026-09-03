/* Portable backend for matrix-group orbits on row spaces.
 *
 * A family member is canonicalised to an rref basis with zero rows removed. Orbit comparison
 * uses the index of that basis in the runtime's Grassmannian order. This still gives a unique
 * answer when transform or stack changes rank or maps several family members to one subspace. */
#include "../../../../runtime/src/reduce.hpp"
#include "../../../gfp/backends/generic/field.hpp"

#include <unordered_map>
#include <unordered_set>

namespace lk::subspace_orbits {
namespace {

using R = Result<std::shared_ptr<Object>>;
constexpr int INVALID = 1;
constexpr uint64_t GROUP_LIMIT = 1ULL << 26;

bool grassmannian_derived(const Family &f) {
    if (f.kind == Family::Kind::Grassmannian) return true;
    if (f.kind == Family::Kind::Transform || f.kind == Family::Kind::Stack)
        return f.child && grassmannian_derived(*f.child);
    return false;
}

struct Setup {
    const Family *family = nullptr;
    std::shared_ptr<Matrix> group;
    uint64_t projective = 0;
};

Status validate_generators(const Matrix &group, const Family &family) {
    if (group.p == 0 || group.p == NATURALS)
        return fail(INVALID, "`group` must be a gfp.matrix batch of generators");
    if (group.count == 0) return fail(INVALID, "subspace_orbits needs at least one generator");
    uint64_t n = family.cols();
    if (group.p != family.prime() || group.rows != n || group.cols != n)
        return fail(INVALID, "group generators must be n x n over the family's prime, with n equal to the member columns");
    gfp::EchelonBasis basis(group.p, n);
    for (uint64_t g = 0; g < group.count; ++g) {
        basis.clear();
        for (uint64_t r = 0; r < n; ++r) basis.add(group.at(g) + r * n);
        if (basis.rank() != n)
            return fail(INVALID, "group generator " + std::to_string(g) + " is not invertible");
    }
    return ok();
}

Result<Setup> setup_action(const Request &req) {
    Setup setup;
    setup.family = req.family.get();
    if (!grassmannian_derived(*setup.family))
        return Result<Setup>::failure(INVALID, "subspace_orbits accepts a Grassmannian and its transform/stack derivatives only");
    auto group = req.handle_args.find("group");
    if (group == req.handle_args.end() || !group->second->matrix)
        return Result<Setup>::failure(INVALID, "`group` must be a gfp.matrix batch of generators");
    setup.group = group->second->matrix;
    auto mode = req.int_args.find("projective");
    if (mode == req.int_args.end() || mode->second > 1)
        return Result<Setup>::failure(INVALID, "`projective` must be 0 for GL or 1 for PGL/PGammaL over F_p");
    setup.projective = mode->second;
    auto valid = validate_generators(*setup.group, *setup.family);
    if (!valid.ok) return Result<Setup>::failure(valid.error.status, valid.error.message);
    return Result<Setup>::success(std::move(setup));
}

void multiply(const std::vector<Entry> &a, const Entry *b, uint64_t rows, uint64_t n,
              const gfp::Field &field, std::vector<Entry> &out) {
    out.assign(rows * n, 0);
    for (uint64_t r = 0; r < rows; ++r)
        for (uint64_t k = 0; k < n; ++k) {
            Entry x = a[r * n + k];
            if (!x) continue;
            for (uint64_t c = 0; c < n; ++c)
                out[r * n + c] = (Entry)field.reduce((uint64_t)out[r * n + c] + (uint64_t)x * b[k * n + c]);
        }
}

void multiply_square(const std::vector<Entry> &a, const Entry *b, uint64_t n,
                     const gfp::Field &field, std::vector<Entry> &out) {
    multiply(a, b, n, n, field, out);
}

void normalise_projective(std::vector<Entry> &a, const gfp::Field &field) {
    auto first = std::find_if(a.begin(), a.end(), [](Entry x) { return x != 0; });
    if (first != a.end() && *first != 1)
        field.scale(a.data(), field.inverse(*first), a.size());
}

struct VectorHash {
    size_t operator()(const std::vector<Entry> &v) const {
        uint64_t h = 1469598103934665603ULL;
        for (Entry x : v) {
            h ^= x;
            h *= 1099511628211ULL;
        }
        return (size_t)h;
    }
};

Result<uint64_t> matrix_group_order(const Matrix &gens, bool projective) {
    uint64_t n = gens.cols;
    gfp::Field field(gens.p);
    std::vector<Entry> identity(n * n, 0);
    for (uint64_t i = 0; i < n; ++i) identity[i * n + i] = 1;
    std::vector<std::vector<Entry>> queue{identity};
    std::unordered_set<std::vector<Entry>, VectorHash> seen;
    seen.insert(identity);
    std::vector<Entry> product;
    for (size_t front = 0; front < queue.size(); ++front) {
        std::vector<Entry> current = queue[front];
        for (uint64_t g = 0; g < gens.count; ++g) {
            multiply_square(current, gens.at(g), n, field, product);
            if (projective) normalise_projective(product, field);
            if (seen.insert(product).second) {
                queue.push_back(product);
                if (queue.size() > GROUP_LIMIT)
                    return Result<uint64_t>::failure(INVALID, "group has more than " + std::to_string(GROUP_LIMIT) + " elements");
            }
        }
    }
    return Result<uint64_t>::success(queue.size());
}

struct OrbitAnswer {
    uint64_t current = 0;
    uint64_t least = 0;
    uint64_t size = 0;
};

struct Walker {
    const Setup &setup;
    uint64_t p, n;
    gfp::Field field;
    gfp::EchelonBasis basis;
    Matrix member;
    std::vector<Entry> product;
    std::vector<uint32_t> pivots;
    std::unordered_map<uint64_t, std::shared_ptr<Family>> grassmannians;

    explicit Walker(const Setup &s)
        : setup(s), p(s.family->prime()), n(s.family->cols()), field(p), basis(p, n) {}

    std::vector<Entry> canonicalise(const Entry *rows, uint64_t count) {
        basis.clear();
        for (uint64_t r = 0; r < count; ++r) basis.add(rows + r * n);
        std::vector<Entry> out;
        basis.rref(out, pivots);
        return out;
    }

    Result<uint64_t> subspace_index(const std::vector<Entry> &rref) {
        uint64_t rank = n ? rref.size() / n : 0;
        if (rank == 0) return Result<uint64_t>::success(0);
        auto found = grassmannians.find(rank);
        if (found == grassmannians.end()) {
            auto made = make_grassmannian(p, n, rank);
            if (!made.ok) return Result<uint64_t>::failure(made.error.status, made.error.message);
            found = grassmannians.emplace(rank, std::move(made.value)).first;
        }
        Matrix m;
        m.p = p;
        m.count = 1;
        m.rows = rank;
        m.cols = n;
        m.entries = rref;
        return found->second->index_of(m);
    }

    Result<std::vector<Entry>> image(const std::vector<Entry> &rref, uint64_t g) {
        uint64_t rank = n ? rref.size() / n : 0;
        multiply(rref, setup.group->at(g), rank, n, field, product);
        auto out = canonicalise(product.data(), rank);
        if (out.size() != rref.size())
            return Result<std::vector<Entry>>::failure(INVALID, "an invertible generator changed the subspace rank");
        return Result<std::vector<Entry>>::success(std::move(out));
    }

    Result<OrbitAnswer> orbit(uint64_t member_index, bool stop_below) {
        auto loaded = setup.family->member_into(member_index, member);
        if (!loaded.ok) return Result<OrbitAnswer>::failure(loaded.error.status, loaded.error.message);
        auto start = canonicalise(member.entries.data(), member.rows);
        auto start_index = subspace_index(start);
        if (!start_index.ok) return Result<OrbitAnswer>::failure(start_index.error.status, start_index.error.message);

        OrbitAnswer answer{start_index.value, start_index.value, 0};
        std::vector<std::vector<Entry>> queue{start};
        std::unordered_set<std::vector<Entry>, VectorHash> seen;
        seen.insert(start);
        for (size_t front = 0; front < queue.size(); ++front) {
            std::vector<Entry> current = queue[front];
            for (uint64_t g = 0; g < setup.group->count; ++g) {
                auto next = image(current, g);
                if (!next.ok) return Result<OrbitAnswer>::failure(next.error.status, next.error.message);
                if (!seen.insert(next.value).second) continue;
                auto index = subspace_index(next.value);
                if (!index.ok) return Result<OrbitAnswer>::failure(index.error.status, index.error.message);
                answer.least = std::min(answer.least, index.value);
                if (stop_below && answer.least < answer.current)
                    return Result<OrbitAnswer>::success(answer);
                queue.push_back(std::move(next.value));
            }
        }
        answer.size = queue.size();
        return Result<OrbitAnswer>::success(answer);
    }
};

enum class Op { IsCanonical, CanonicalIndex, OrbitSize, StabilizerOrder };

R run_orbit_op(const Request &req, Op op) {
    auto setup = setup_action(req);
    if (!setup.ok) return R::failure(setup.error.status, setup.error.message);
    auto size = req.family->size();
    if (!size.ok) return R::failure(size.error.status, size.error.message);

    uint64_t order = 0;
    if (op == Op::StabilizerOrder) {
        auto closed = matrix_group_order(*setup.value.group, setup.value.projective == 1);
        if (!closed.ok) return R::failure(closed.error.status, closed.error.message);
        order = closed.value;
    }

    Reduction reduction = parse_reduction(req.reduction);
    Shared shared;
    auto prepared = prepare_all(reduction, size.value, shared);
    if (!prepared.ok) return R::failure(prepared.error.status, prepared.error.message);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, size.value ? size.value : 1));
    std::vector<std::unique_ptr<Walker>> walkers;
    std::vector<Accumulator> accumulators;
    for (uint32_t t = 0; t < threads; ++t) {
        walkers.push_back(std::make_unique<Walker>(setup.value));
        accumulators.emplace_back(reduction, &shared);
    }

    auto statuses = parallel_ranges(size.value, threads, [&](uint32_t t, uint64_t begin, uint64_t end) -> Status {
        for (uint64_t i = begin; i < end; ++i) {
            if (accumulators[t].exhausted(i)) break;
            auto orbit = walkers[t]->orbit(i, op == Op::IsCanonical);
            if (!orbit.ok) return fail(orbit.error.status, orbit.error.message);
            switch (op) {
            case Op::IsCanonical:
                accumulators[t].boolean(i, orbit.value.current == orbit.value.least);
                break;
            case Op::CanonicalIndex:
                accumulators[t].integer(i, orbit.value.least);
                break;
            case Op::OrbitSize:
                accumulators[t].integer(i, orbit.value.size);
                break;
            case Op::StabilizerOrder:
                accumulators[t].integer(i, order / orbit.value.size);
                break;
            }
        }
        return ok();
    });
    for (const auto &status : statuses)
        if (!status.ok) return R::failure(status.error.status, status.error.message);
    return assemble(req, reduction, accumulators, shared);
}

R run(const Request &req) {
    if (req.op == "is_canonical") return run_orbit_op(req, Op::IsCanonical);
    if (req.op == "canonical_index") return run_orbit_op(req, Op::CanonicalIndex);
    if (req.op == "orbit_size") return run_orbit_op(req, Op::OrbitSize);
    if (req.op == "stabilizer_order") return run_orbit_op(req, Op::StabilizerOrder);
    return R::failure(4, "unknown subspace_orbits operation " + req.op);
}

BackendRegistration registration{Backend{
    "subspace_orbits", "generic",
    [] { return true; },
    [](const Request &) { return true; },
    run,
    0}};

} // namespace
} // namespace lk::subspace_orbits
