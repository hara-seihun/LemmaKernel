#include "../../../../runtime/src/reduce.hpp"

#include <boost/multiprecision/cpp_int.hpp>
#include <functional>
#include <limits>
#include <numeric>
#include <set>

namespace lk::polytopes_small {
namespace {

using Big = boost::multiprecision::cpp_int;
using Point = std::vector<Big>;
using BigMatrix = std::vector<Point>;
using R = Result<std::shared_ptr<Object>>;
constexpr int INVALID = 1;
constexpr uint64_t MAX_LATTICE_CANDIDATES = 100000000;

template <class Fn>
bool combinations_rec(uint64_t n, uint64_t k, uint64_t next, std::vector<uint64_t> &choice, Fn &fn) {
    if (choice.size() == k) return fn(choice);
    uint64_t need = k - choice.size();
    for (uint64_t i = next; i + need <= n; ++i) {
        choice.push_back(i);
        if (combinations_rec(n, k, i + 1, choice, fn)) return true;
        choice.pop_back();
    }
    return false;
}

template <class Fn>
bool combinations(uint64_t n, uint64_t k, Fn fn) {
    if (k > n) return false;
    std::vector<uint64_t> choice;
    return combinations_rec(n, k, 0, choice, fn);
}

Big determinant(const BigMatrix &matrix) {
    if (matrix.empty()) return 1;
    Big total = 0;
    for (uint64_t j = 0; j < matrix.front().size(); ++j) {
        BigMatrix minor;
        for (uint64_t i = 1; i < matrix.size(); ++i) {
            Point row;
            for (uint64_t c = 0; c < matrix[i].size(); ++c)
                if (c != j) row.push_back(matrix[i][c]);
            minor.push_back(std::move(row));
        }
        Big term = matrix[0][j] * determinant(minor);
        total += (j & 1) ? -term : term;
    }
    return total;
}

uint64_t matrix_rank(const BigMatrix &matrix) {
    if (matrix.empty() || matrix.front().empty()) return 0;
    uint64_t bound = std::min<uint64_t>(matrix.size(), matrix.front().size());
    for (uint64_t k = bound; k > 0; --k) {
        bool found = combinations(matrix.size(), k, [&](const std::vector<uint64_t> &rows) {
            return combinations(matrix.front().size(), k, [&](const std::vector<uint64_t> &cols) {
                BigMatrix minor;
                for (uint64_t i : rows) {
                    Point row;
                    for (uint64_t j : cols) row.push_back(matrix[i][j]);
                    minor.push_back(std::move(row));
                }
                return determinant(minor) != 0;
            });
        });
        if (found) return k;
    }
    return 0;
}

uint64_t affine_rank(const std::vector<Point> &points) {
    if (points.empty()) return 0;
    BigMatrix differences;
    for (uint64_t i = 1; i < points.size(); ++i) {
        Point row(points[i].size());
        for (uint64_t j = 0; j < row.size(); ++j) row[j] = points[i][j] - points[0][j];
        differences.push_back(std::move(row));
    }
    return matrix_rank(differences);
}

Big dot(const Point &a, const Point &b) {
    Big total = 0;
    for (uint64_t i = 0; i < a.size(); ++i) total += a[i] * b[i];
    return total;
}

Point project(const Point &point, const std::vector<uint64_t> &columns) {
    Point out;
    for (uint64_t j : columns) out.push_back(point[j]);
    return out;
}

struct Facet {
    std::vector<uint64_t> indices;
    Point normal;
    Big constant;
};

struct Analysis {
    std::vector<Point> points;
    uint64_t dimension = 0;
    uint64_t rank = 0;
    std::vector<uint64_t> projection;
    std::vector<Point> projected;
    std::vector<Facet> facets;
    std::set<std::vector<uint64_t>> faces;
};

Analysis analyse(const Matrix &member) {
    Analysis a;
    a.dimension = member.cols;
    for (uint64_t i = 0; i < member.rows; ++i) {
        Point point;
        for (uint64_t j = 0; j < member.cols; ++j) point.push_back(member.entries[i * member.cols + j]);
        if (std::find(a.points.begin(), a.points.end(), point) == a.points.end()) a.points.push_back(std::move(point));
    }
    a.rank = affine_rank(a.points);
    combinations(a.dimension, a.rank, [&](const std::vector<uint64_t> &columns) {
        std::vector<Point> projected;
        for (const auto &point : a.points) projected.push_back(project(point, columns));
        if (affine_rank(projected) != a.rank) return false;
        a.projection = columns;
        a.projected = std::move(projected);
        return true;
    });
    if (a.projected.empty()) {
        a.projection.resize(a.rank);
        std::iota(a.projection.begin(), a.projection.end(), 0);
        for (const auto &point : a.points) a.projected.push_back(project(point, a.projection));
    }
    std::set<std::vector<uint64_t>> facet_sets;
    if (a.rank > 0) combinations(a.points.size(), a.rank, [&](const std::vector<uint64_t> &choice) {
        std::vector<Point> selected;
        for (uint64_t i : choice) selected.push_back(a.projected[i]);
        if (affine_rank(selected) != a.rank - 1) return false;
        BigMatrix differences;
        for (uint64_t i = 1; i < selected.size(); ++i) {
            Point row(a.rank);
            for (uint64_t j = 0; j < a.rank; ++j) row[j] = selected[i][j] - selected[0][j];
            differences.push_back(std::move(row));
        }
        Point normal(a.rank);
        for (uint64_t j = 0; j < a.rank; ++j) {
            BigMatrix minor;
            for (const auto &source : differences) {
                Point row;
                for (uint64_t c = 0; c < a.rank; ++c) if (c != j) row.push_back(source[c]);
                minor.push_back(std::move(row));
            }
            normal[j] = determinant(minor);
            if (j & 1) normal[j] = -normal[j];
        }
        Big constant = dot(normal, selected[0]);
        std::vector<Big> values;
        bool all_nonpositive = true, all_nonnegative = true, nonzero = false;
        for (const auto &point : a.projected) {
            Big value = dot(normal, point) - constant;
            values.push_back(value);
            all_nonpositive = all_nonpositive && value <= 0;
            all_nonnegative = all_nonnegative && value >= 0;
            nonzero = nonzero || value != 0;
        }
        if (!nonzero || (!all_nonpositive && !all_nonnegative)) return false;
        if (all_nonnegative) {
            for (Big &x : normal) x = -x;
            constant = -constant;
            for (Big &x : values) x = -x;
        }
        std::vector<uint64_t> indices;
        for (uint64_t i = 0; i < values.size(); ++i) if (values[i] == 0) indices.push_back(i);
        if (facet_sets.insert(indices).second) a.facets.push_back(Facet{std::move(indices), std::move(normal), std::move(constant)});
        return false;
    });
    std::vector<uint64_t> full(a.points.size());
    std::iota(full.begin(), full.end(), 0);
    a.faces.insert(full);
    for (const Facet &facet : a.facets) {
        std::vector<std::vector<uint64_t>> current(a.faces.begin(), a.faces.end());
        for (const auto &face : current) {
            std::vector<uint64_t> intersection;
            std::set_intersection(face.begin(), face.end(), facet.indices.begin(), facet.indices.end(), std::back_inserter(intersection));
            if (!intersection.empty()) a.faces.insert(std::move(intersection));
        }
    }
    return a;
}

uint64_t face_dimension(const Analysis &a, const std::vector<uint64_t> &face) {
    std::vector<Point> points;
    for (uint64_t i : face) points.push_back(a.projected[i]);
    return affine_rank(points);
}

std::vector<uint64_t> f_vector(const Analysis &a) {
    std::vector<uint64_t> out(a.dimension + 1, 0);
    for (const auto &face : a.faces) ++out[face_dimension(a, face)];
    return out;
}

bool is_simplicial(const Analysis &a) {
    if (a.rank <= 1) return true;
    std::set<uint64_t> vertices;
    for (const auto &face : a.faces) if (face_dimension(a, face) == 0) vertices.insert(face.front());
    for (const Facet &facet : a.facets) {
        uint64_t count = 0;
        for (uint64_t i : facet.indices) count += vertices.contains(i);
        if (count != a.rank) return false;
    }
    return true;
}

bool in_dilate(const Analysis &a, uint64_t t, const std::vector<uint64_t> &candidate) {
    std::vector<Point> scaled;
    for (const auto &point : a.points) {
        Point x;
        for (const Big &coordinate : point) x.push_back(t * coordinate);
        scaled.push_back(std::move(x));
    }
    Point x;
    for (uint64_t coordinate : candidate) x.push_back(coordinate);
    scaled.push_back(x);
    if (affine_rank(scaled) != a.rank) return false;
    Point projected = project(x, a.projection);
    for (const Facet &facet : a.facets)
        if (dot(facet.normal, projected) > t * facet.constant) return false;
    return true;
}

Result<uint64_t> lattice_count(const Analysis &a, uint64_t t) {
    if (t == 0) return Result<uint64_t>::success(1);
    std::vector<uint64_t> lower(a.dimension, UINT64_MAX), upper(a.dimension, 0);
    unsigned __int128 candidates = 1;
    for (uint64_t j = 0; j < a.dimension; ++j) {
        for (const auto &point : a.points) {
            uint64_t coordinate = point[j].convert_to<uint64_t>();
            lower[j] = std::min(lower[j], t * coordinate);
            upper[j] = std::max(upper[j], t * coordinate);
        }
        candidates *= upper[j] - lower[j] + 1;
        if (candidates > MAX_LATTICE_CANDIDATES)
            return Result<uint64_t>::failure(INVALID, "dilated bounding box has more than 100000000 lattice points");
    }
    uint64_t count = 0;
    std::vector<uint64_t> point(a.dimension);
    std::function<void(uint64_t)> visit = [&](uint64_t j) {
        if (j == a.dimension) {
            count += in_dilate(a, t, point);
            return;
        }
        for (uint64_t x = lower[j]; x <= upper[j]; ++x) {
            point[j] = x;
            visit(j + 1);
            if (x == UINT64_MAX) break;
        }
    };
    visit(0);
    return Result<uint64_t>::success(count);
}

uint64_t choose(uint64_t n, uint64_t k) {
    if (k > n) return 0;
    k = std::min(k, n - k);
    unsigned __int128 value = 1;
    for (uint64_t i = 1; i <= k; ++i) value = value * (n - k + i) / i;
    return (uint64_t)value;
}

Result<std::vector<uint64_t>> h_star(const Analysis &a) {
    std::vector<uint64_t> counts;
    for (uint64_t t = 0; t <= a.rank; ++t) {
        auto count = lattice_count(a, t);
        if (!count.ok) return Result<std::vector<uint64_t>>::failure(count.error.status, count.error.message);
        counts.push_back(count.value);
    }
    std::vector<uint64_t> hs(a.dimension + 1, 0);
    for (uint64_t j = 0; j <= a.rank; ++j) {
        Big value = 0;
        for (uint64_t i = 0; i <= j; ++i) {
            Big term = Big(choose(a.rank + 1, j - i)) * counts[i];
            value += ((j - i) & 1) ? -term : term;
        }
        if (value < 0 || value > std::numeric_limits<uint64_t>::max())
            return Result<std::vector<uint64_t>>::failure(INVALID, "h-star coefficient does not fit u64");
        hs[j] = value.convert_to<uint64_t>();
    }
    return Result<std::vector<uint64_t>>::success(std::move(hs));
}

R run(const Request &req) {
    const Family &family = *req.family;
    if (family.kind != Family::Kind::Subsets)
        return R::failure(INVALID, "polytopes_small operations are defined on subsets families only");
    if (family.k == 0 || family.data->count == 0)
        return R::failure(INVALID, "polytope members must be nonempty");
    if (family.cols() > 6) return R::failure(INVALID, "ambient dimension must be at most 6");
    auto size_result = family.size();
    if (!size_result.ok) return R::failure(size_result.error.status, size_result.error.message);
    uint64_t size = size_result.value;
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, size ? size : 1));

    if (req.op == "f_vector" || req.op == "ehrhart_polynomial") {
        auto vectors = std::make_shared<U64Vectors>();
        vectors->count = size;
        vectors->length = family.cols() + 1;
        vectors->entries.assign(size * vectors->length, 0);
        auto statuses = parallel_ranges(size, threads, [&](uint32_t, uint64_t begin, uint64_t end) -> Status {
            Matrix member;
            for (uint64_t i = begin; i < end; ++i) {
                auto materialised = family.member_into(i, member);
                if (!materialised.ok) return materialised;
                Analysis analysis = analyse(member);
                std::vector<uint64_t> value;
                if (req.op == "f_vector") value = f_vector(analysis);
                else {
                    auto hs = h_star(analysis);
                    if (!hs.ok) return fail(hs.error.status, hs.error.message);
                    value = std::move(hs.value);
                }
                std::copy(value.begin(), value.end(), vectors->entries.begin() + i * vectors->length);
            }
            return ok();
        });
        for (const auto &status : statuses)
            if (!status.ok) return R::failure(status.error.status, status.error.message);
        auto out = std::make_shared<Object>();
        out->kind = "polytopes_small.vectors";
        out->u64_vectors = vectors;
        return R::success(out);
    }

    Reduction reduction = parse_reduction(req.reduction);
    Shared shared;
    auto prepared = prepare_all(reduction, size, shared);
    if (!prepared.ok) return R::failure(prepared.error.status, prepared.error.message);
    std::vector<Accumulator> accumulators;
    for (uint32_t t = 0; t < threads; ++t) accumulators.emplace_back(reduction, &shared);
    auto statuses = parallel_ranges(size, threads, [&](uint32_t thread, uint64_t begin, uint64_t end) -> Status {
        Matrix member;
        for (uint64_t i = begin; i < end; ++i) {
            if (accumulators[thread].exhausted(i)) break;
            auto materialised = family.member_into(i, member);
            if (!materialised.ok) return materialised;
            if (req.op == "vertex_count") {
                bool binary = std::all_of(member.entries.begin(), member.entries.end(), [](Entry x) { return x <= 1; });
                if (binary) {
                    std::set<std::vector<Entry>> points;
                    for (uint64_t row = 0; row < member.rows; ++row)
                        points.emplace(member.entries.begin() + row * member.cols,
                                       member.entries.begin() + (row + 1) * member.cols);
                    accumulators[thread].integer(i, points.size());
                } else {
                    accumulators[thread].integer(i, f_vector(analyse(member))[0]);
                }
            } else if (req.op == "is_simplicial") {
                accumulators[thread].boolean(i, is_simplicial(analyse(member)));
            } else return fail(INVALID, "unknown polytopes_small operation " + req.op);
        }
        return ok();
    });
    for (const auto &status : statuses)
        if (!status.ok) return R::failure(status.error.status, status.error.message);
    return assemble(req, reduction, accumulators, shared);
}

BackendRegistration registration{Backend{
    "polytopes_small", "generic",
    [] { return true; },
    [](const Request &) { return true; },
    run,
    0}};

} // namespace
} // namespace lk::polytopes_small
