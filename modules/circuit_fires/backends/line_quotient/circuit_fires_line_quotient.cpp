#include "../circuit_fires_common.hpp"
#include "../../../../runtime/src/reduce.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace lk::circuit_fires {
namespace {

/* Choose rows whose covectors form a basis of the fibre dual. The corresponding scalar
 * potential components are already flat in their selected directions, so modulo their harmless
 * affine slopes each is an arbitrary function on V/<z>. Remaining flatness equations and the
 * total defect then act on h*p^(g-1) quotient values. This is an exact coordinate change on the
 * full function space, not a polynomial-degree truncation. */
using R = Result<std::shared_ptr<Object>>;
constexpr int INVALID = 1;
constexpr uint64_t MAX_LINE_WIDTH = 8192;
constexpr uint64_t DENSE_LINE_WIDTH = 2048;
constexpr uint64_t SHARED_GEOMETRY_BYTES = 256ULL << 20;

uint64_t checked_power(uint64_t base, uint64_t exponent) {
    uint64_t value = 1;
    for (uint64_t i = 0; i < exponent; ++i) {
        if (value > UINT64_MAX / base) return 0;
        value *= base;
    }
    return value;
}

class Field {
public:
    explicit Field(uint32_t prime) : p_(prime) {
        for (uint32_t coefficient = 0; coefficient < p_; ++coefficient)
            for (uint32_t left = 0; left < p_; ++left)
                for (uint32_t right = 0; right < p_; ++right) {
                    addmul_[coefficient][left][right] =
                        (uint8_t)((left + coefficient * right) % p_);
                    submul_[coefficient][left][right] =
                        (uint8_t)((left + p_ - coefficient * right % p_) % p_);
                }
        for (uint32_t left = 0; left < p_; ++left)
            for (uint32_t right = 0; right < p_; ++right)
                mul_[left][right] = (uint8_t)(left * right % p_);
    }

    uint32_t prime() const { return p_; }

    uint8_t normalize(int64_t value) const {
        value %= (int64_t)p_;
        if (value < 0) value += p_;
        return (uint8_t)value;
    }

    uint8_t inverse(uint8_t value) const {
        uint32_t result = 1, base = value, exponent = p_ - 2;
        while (exponent) {
            if (exponent & 1) result = result * base % p_;
            base = base * base % p_;
            exponent >>= 1;
        }
        return (uint8_t)result;
    }

    uint8_t multiply(uint8_t left, uint8_t right) const { return mul_[left][right]; }
    uint8_t add_product(uint8_t left, uint8_t coefficient, uint8_t right) const {
        return addmul_[coefficient][left][right];
    }
    uint8_t subtract_product(uint8_t left, uint8_t coefficient, uint8_t right) const {
        return submul_[coefficient][left][right];
    }

private:
    uint32_t p_;
    uint8_t mul_[8][8]{};
    uint8_t addmul_[8][8][8]{};
    uint8_t submul_[8][8][8]{};
};

class DenseBasis {
public:
    DenseBasis(const Field &field, uint64_t width) : field_(field), width_(width) {}

    uint64_t rank() const { return pivots_.size(); }
    uint64_t width() const { return width_; }
    const uint8_t *row(uint64_t index) const { return rows_.data() + index * width_; }

    bool add(std::vector<uint8_t> &candidate) {
        for (uint64_t i = 0; i < rank(); ++i) {
            uint8_t coefficient = candidate[pivots_[i]];
            if (!coefficient) continue;
            const uint8_t *pivot = row(i);
            for (uint64_t column = 0; column < width_; ++column)
                candidate[column] = field_.subtract_product(
                    candidate[column], coefficient, pivot[column]);
        }
        auto found = std::find_if(candidate.begin(), candidate.end(),
                                  [](uint8_t value) { return value != 0; });
        if (found == candidate.end()) return false;
        uint32_t pivot = (uint32_t)(found - candidate.begin());
        uint8_t scale = field_.inverse(*found);
        if (scale != 1)
            for (uint8_t &value : candidate) value = field_.multiply(value, scale);
        pivots_.push_back(pivot);
        rows_.insert(rows_.end(), candidate.begin(), candidate.end());
        return true;
    }

    void rref() {
        uint64_t output = 0;
        for (uint64_t column = 0; column < width_ && output < rank(); ++column) {
            uint64_t pivot = output;
            while (pivot < rank() && rows_[pivot * width_ + column] == 0) ++pivot;
            if (pivot == rank()) continue;
            if (pivot != output)
                for (uint64_t j = 0; j < width_; ++j)
                    std::swap(rows_[output * width_ + j], rows_[pivot * width_ + j]);
            uint8_t *selected = rows_.data() + output * width_;
            uint8_t scale = field_.inverse(selected[column]);
            if (scale != 1)
                for (uint64_t j = 0; j < width_; ++j)
                    selected[j] = field_.multiply(selected[j], scale);
            for (uint64_t other = 0; other < rank(); ++other) {
                if (other == output) continue;
                uint8_t *target = rows_.data() + other * width_;
                uint8_t coefficient = target[column];
                if (!coefficient) continue;
                for (uint64_t j = 0; j < width_; ++j)
                    target[j] = field_.subtract_product(target[j], coefficient, selected[j]);
            }
            pivots_[output] = (uint32_t)column;
            ++output;
        }
    }

    uint32_t pivot(uint64_t index) const { return pivots_[index]; }

private:
    const Field &field_;
    uint64_t width_;
    std::vector<uint8_t> rows_;
    std::vector<uint32_t> pivots_;
};

class DenseSpace {
public:
    DenseSpace(const Field &field, uint64_t width)
        : field_(field), width_(width), dimension_(width), identity_(true) {}

    uint64_t dimension() const { return dimension_; }

    void add_event(std::vector<uint8_t> &row, uint64_t variable, int64_t coefficient) const {
        uint8_t scalar = field_.normalize(coefficient);
        if (!scalar) return;
        if (identity_) {
            row[variable] = field_.add_product(row[variable], scalar, 1);
            return;
        }
        const uint8_t *source = values_.data() + variable * dimension_;
        for (uint64_t column = 0; column < dimension_; ++column)
            row[column] = field_.add_product(row[column], scalar, source[column]);
    }

    void restrict_to_nullspace(DenseBasis &equations) {
        if (equations.rank() == 0) return;
        equations.rref();
        uint64_t next_dimension = dimension_ - equations.rank();
        std::vector<uint8_t> is_pivot(dimension_, 0);
        for (uint64_t row = 0; row < equations.rank(); ++row)
            is_pivot[equations.pivot(row)] = 1;
        std::vector<uint8_t> kernel(dimension_ * next_dimension, 0);
        uint64_t output = 0;
        for (uint64_t free_column = 0; free_column < dimension_; ++free_column) {
            if (is_pivot[free_column]) continue;
            kernel[free_column * next_dimension + output] = 1;
            for (uint64_t row = 0; row < equations.rank(); ++row) {
                uint8_t value = equations.row(row)[free_column];
                kernel[(uint64_t)equations.pivot(row) * next_dimension + output] =
                    value ? (uint8_t)(field_.prime() - value) : 0;
            }
            ++output;
        }
        if (identity_) {
            values_ = std::move(kernel);
        } else {
            std::vector<uint8_t> next(width_ * next_dimension, 0);
            for (uint64_t original = 0; original < width_; ++original)
                for (uint64_t middle = 0; middle < dimension_; ++middle) {
                    uint8_t left = values_[original * dimension_ + middle];
                    if (!left) continue;
                    const uint8_t *right = kernel.data() + middle * next_dimension;
                    uint8_t *target = next.data() + original * next_dimension;
                    for (uint64_t column = 0; column < next_dimension; ++column)
                        target[column] = field_.add_product(target[column], left, right[column]);
                }
            values_ = std::move(next);
        }
        dimension_ = next_dimension;
        identity_ = false;
    }

private:
    const Field &field_;
    uint64_t width_, dimension_;
    bool identity_;
    std::vector<uint8_t> values_;
};

using SparseRow = std::unordered_map<uint32_t, uint8_t>;

void put(const Field &field, SparseRow &row, uint32_t column, int64_t coefficient) {
    uint8_t scalar = field.normalize(coefficient);
    if (!scalar) return;
    auto found = row.find(column);
    uint8_t value = found == row.end() ? scalar
                                       : field.add_product(found->second, scalar, 1);
    if (value) {
        if (found == row.end()) row.emplace(column, value);
        else found->second = value;
    } else if (found != row.end()) {
        row.erase(found);
    }
}

class SparseBasis {
public:
    SparseBasis(const Field &field, uint64_t width) : field_(field), rows_(width) {}

    bool add(SparseRow candidate) {
        while (!candidate.empty()) {
            auto pivot_entry = std::max_element(
                candidate.begin(), candidate.end(),
                [](const auto &left, const auto &right) { return left.first < right.first; });
            uint32_t pivot = pivot_entry->first;
            if (!rows_[pivot]) {
                uint8_t scale = field_.inverse(pivot_entry->second);
                if (scale != 1)
                    for (auto &[column, value] : candidate) {
                        (void)column;
                        value = field_.multiply(value, scale);
                    }
                rows_[pivot] = std::make_unique<SparseRow>(std::move(candidate));
                return true;
            }
            uint8_t coefficient = pivot_entry->second;
            for (const auto &[column, value] : *rows_[pivot])
                put(field_, candidate, column, -(int64_t)coefficient * value);
        }
        return false;
    }

private:
    const Field &field_;
    std::vector<std::unique_ptr<SparseRow>> rows_;
};

struct DirectionData {
    std::vector<uint32_t> translation;
    std::vector<uint32_t> coset;
    std::vector<uint32_t> representative;
    uint32_t zero_coset = 0;
};

class Geometry {
public:
    Geometry(const Matrix &family, uint32_t prime, uint64_t base_dim)
        : p_(prime), g_(base_dim), points_(checked_power(prime, base_dim)),
          quotient_(points_ / prime), powers_(base_dim), cached_(points_) {
        uint64_t place = 1;
        for (uint64_t coordinate = 0; coordinate < g_; ++coordinate) {
            powers_[coordinate] = place;
            place *= p_;
        }
        std::vector<uint8_t> seen(points_, 0);
        uint64_t unique = 0;
        for (uint64_t member = 0; member < family.count; ++member)
            for (uint64_t row = 0; row < family.rows; ++row) {
                uint64_t direction = encode(family.at(member) + row * family.cols);
                if (direction && !seen[direction]) {
                    seen[direction] = 1;
                    ++unique;
                }
            }
        unsigned __int128 bytes = (unsigned __int128)unique *
                                  (2 * points_ + quotient_) * sizeof(uint32_t);
        if (bytes <= SHARED_GEOMETRY_BYTES)
            for (uint64_t direction = 1; direction < points_; ++direction)
                if (seen[direction]) cached_[direction] = make(direction);
    }

    uint64_t points() const { return points_; }
    uint64_t quotient() const { return quotient_; }

    uint64_t encode(const Entry *direction) const {
        uint64_t code = 0;
        for (uint64_t coordinate = 0; coordinate < g_; ++coordinate)
            code += direction[coordinate] * powers_[coordinate];
        return code;
    }

    const DirectionData *cached(uint64_t direction) const {
        return cached_[direction].get();
    }

    std::unique_ptr<DirectionData> make(uint64_t direction) const {
        auto result = std::make_unique<DirectionData>();
        result->translation.resize(points_);
        result->coset.assign(points_, UINT32_MAX);
        for (uint64_t point = 0; point < points_; ++point)
            result->translation[point] = add(point, direction);
        uint32_t next = 0;
        for (uint64_t point = 0; point < points_; ++point) {
            if (result->coset[point] != UINT32_MAX) continue;
            result->representative.push_back((uint32_t)point);
            uint64_t current = point;
            for (uint32_t step = 0; step < p_; ++step) {
                result->coset[current] = next;
                current = result->translation[current];
            }
            ++next;
        }
        result->zero_coset = result->coset[0];
        return result;
    }

private:
    uint64_t add(uint64_t point, uint64_t direction) const {
        uint64_t result = 0;
        for (uint64_t coordinate = 0; coordinate < g_; ++coordinate) {
            uint64_t place = powers_[coordinate];
            uint64_t digit = point / place % p_;
            uint64_t step = direction / place % p_;
            result += ((digit + step) % p_) * place;
        }
        return result;
    }

    uint32_t p_;
    uint64_t g_, points_, quotient_;
    std::vector<uint64_t> powers_;
    std::vector<std::unique_ptr<DirectionData>> cached_;
};

class MemberGeometry {
public:
    explicit MemberGeometry(const Geometry &geometry) : geometry_(geometry) {}

    const DirectionData &line(uint64_t direction) {
        if (const DirectionData *shared = geometry_.cached(direction)) return *shared;
        auto found = local_.find(direction);
        if (found != local_.end()) return *found->second;
        auto [inserted, unused] = local_.emplace(direction, geometry_.make(direction));
        (void)unused;
        return *inserted->second;
    }

private:
    const Geometry &geometry_;
    std::unordered_map<uint64_t, std::unique_ptr<DirectionData>> local_;
};

bool fibre_coordinates(const Entry *configuration, uint64_t rows, uint64_t base_dim,
                       uint64_t fibre_dim, const Field &field,
                       std::vector<uint32_t> &pivots, std::vector<uint8_t> &coordinates) {
    DenseBasis span(field, fibre_dim);
    std::vector<uint8_t> candidate(fibre_dim);
    for (uint64_t row = 0; row < rows && pivots.size() < fibre_dim; ++row) {
        const Entry *covector = configuration + row * (base_dim + fibre_dim) + base_dim;
        for (uint64_t coordinate = 0; coordinate < fibre_dim; ++coordinate)
            candidate[coordinate] = (uint8_t)covector[coordinate];
        if (span.add(candidate)) pivots.push_back((uint32_t)row);
    }
    if (pivots.size() != fibre_dim) return false;

    uint64_t augmented_width = 2 * fibre_dim;
    std::vector<uint8_t> augmented(fibre_dim * augmented_width, 0);
    for (uint64_t row = 0; row < fibre_dim; ++row) {
        const Entry *covector = configuration + pivots[row] * (base_dim + fibre_dim) + base_dim;
        for (uint64_t column = 0; column < fibre_dim; ++column)
            augmented[row * augmented_width + column] = (uint8_t)covector[column];
        augmented[row * augmented_width + fibre_dim + row] = 1;
    }
    for (uint64_t column = 0; column < fibre_dim; ++column) {
        uint64_t pivot = column;
        while (pivot < fibre_dim && augmented[pivot * augmented_width + column] == 0) ++pivot;
        if (pivot == fibre_dim) return false;
        if (pivot != column)
            for (uint64_t j = 0; j < augmented_width; ++j)
                std::swap(augmented[column * augmented_width + j],
                          augmented[pivot * augmented_width + j]);
        uint8_t scale = field.inverse(augmented[column * augmented_width + column]);
        for (uint64_t j = 0; j < augmented_width; ++j)
            augmented[column * augmented_width + j] =
                field.multiply(augmented[column * augmented_width + j], scale);
        for (uint64_t other = 0; other < fibre_dim; ++other) {
            if (other == column) continue;
            uint8_t coefficient = augmented[other * augmented_width + column];
            if (!coefficient) continue;
            for (uint64_t j = 0; j < augmented_width; ++j)
                augmented[other * augmented_width + j] = field.subtract_product(
                    augmented[other * augmented_width + j], coefficient,
                    augmented[column * augmented_width + j]);
        }
    }

    coordinates.assign(rows * fibre_dim, 0);
    for (uint64_t row = 0; row < rows; ++row) {
        const Entry *covector = configuration + row * (base_dim + fibre_dim) + base_dim;
        for (uint64_t coordinate = 0; coordinate < fibre_dim; ++coordinate) {
            uint64_t value = 0;
            for (uint64_t source = 0; source < fibre_dim; ++source)
                value += covector[source] *
                         augmented[source * augmented_width + fibre_dim + coordinate];
            coordinates[row * fibre_dim + coordinate] = (uint8_t)(value % field.prime());
        }
    }
    return true;
}

class LineQuotient {
public:
    LineQuotient(const Matrix &family, uint32_t prime, uint64_t base_dim, uint64_t fibre_dim)
        : g_(base_dim), h_(fibre_dim), rows_(family.rows), field_(prime),
          geometry_(family, prime, base_dim) {}

    bool fire(const Entry *configuration) const {
        std::vector<uint32_t> pivots;
        std::vector<uint8_t> coordinates;
        if (!fibre_coordinates(configuration, rows_, g_, h_, field_, pivots, coordinates))
            return false;
        uint64_t width = h_ * geometry_.quotient();
        return width <= DENSE_LINE_WIDTH
            ? dense_fire(configuration, pivots, coordinates)
            : sparse_fire(configuration, pivots, coordinates);
    }

private:
    std::vector<uint64_t> direction_codes(const Entry *configuration) const {
        std::vector<uint64_t> result(rows_);
        for (uint64_t row = 0; row < rows_; ++row)
            result[row] = geometry_.encode(configuration + row * (g_ + h_));
        return result;
    }

    bool dense_fire(const Entry *configuration, const std::vector<uint32_t> &pivots,
                    const std::vector<uint8_t> &coordinates) const {
        (void)configuration;
        auto directions = direction_codes(configuration);
        MemberGeometry member_geometry(geometry_);
        std::vector<const DirectionData *> component_lines(h_);
        std::vector<uint8_t> is_pivot(rows_, 0);
        for (uint64_t coordinate = 0; coordinate < h_; ++coordinate) {
            is_pivot[pivots[coordinate]] = 1;
            component_lines[coordinate] = &member_geometry.line(directions[pivots[coordinate]]);
        }

        uint64_t width = h_ * geometry_.quotient();
        DenseSpace space(field_, width);
        for (uint64_t row_index = 0; row_index < rows_; ++row_index) {
            if (is_pivot[row_index]) continue;
            const DirectionData &target = member_geometry.line(directions[row_index]);
            DenseBasis equations(field_, space.dimension());
            std::vector<uint8_t> row(space.dimension(), 0);
            for (uint64_t coset = 0; coset < geometry_.quotient(); ++coset) {
                uint64_t x0 = target.representative[coset];
                uint64_t x1 = target.translation[x0];
                uint64_t x2 = target.translation[x1];
                for (uint32_t step = 0; step + 2 < field_.prime(); ++step) {
                    std::fill(row.begin(), row.end(), 0);
                    for (uint64_t coordinate = 0; coordinate < h_; ++coordinate) {
                        uint8_t scalar = coordinates[row_index * h_ + coordinate];
                        if (!scalar) continue;
                        uint64_t offset = coordinate * geometry_.quotient();
                        const auto &line = *component_lines[coordinate];
                        space.add_event(row, offset + line.coset[x2], scalar);
                        space.add_event(row, offset + line.coset[x1], -2 * (int64_t)scalar);
                        space.add_event(row, offset + line.coset[x0], scalar);
                    }
                    equations.add(row);
                    x0 = x1;
                    x1 = x2;
                    x2 = target.translation[x2];
                }
            }
            for (uint64_t coset = 0; coset < geometry_.quotient(); ++coset) {
                if (coset == target.zero_coset) continue;
                uint64_t x = target.representative[coset];
                uint64_t shifted = target.translation[x];
                std::fill(row.begin(), row.end(), 0);
                for (uint64_t coordinate = 0; coordinate < h_; ++coordinate) {
                    uint8_t scalar = coordinates[row_index * h_ + coordinate];
                    if (!scalar) continue;
                    uint64_t offset = coordinate * geometry_.quotient();
                    const auto &line = *component_lines[coordinate];
                    space.add_event(row, offset + line.coset[shifted], scalar);
                    space.add_event(row, offset + line.coset[x], -(int64_t)scalar);
                    space.add_event(row, offset + line.coset[directions[row_index]],
                                    -(int64_t)scalar);
                    space.add_event(row, offset + line.coset[0], scalar);
                }
                equations.add(row);
            }
            space.restrict_to_nullspace(equations);
            if (space.dimension() == 0) return false;
        }

        std::vector<uint8_t> defect(space.dimension(), 0);
        for (uint64_t row_index = 0; row_index < rows_; ++row_index)
            for (uint64_t coordinate = 0; coordinate < h_; ++coordinate) {
                uint8_t scalar = coordinates[row_index * h_ + coordinate];
                if (!scalar) continue;
                uint64_t offset = coordinate * geometry_.quotient();
                const auto &line = *component_lines[coordinate];
                space.add_event(defect, offset + line.coset[directions[row_index]], scalar);
                space.add_event(defect, offset + line.coset[0], -(int64_t)scalar);
            }
        return std::any_of(defect.begin(), defect.end(), [](uint8_t value) { return value != 0; });
    }

    bool sparse_fire(const Entry *configuration, const std::vector<uint32_t> &pivots,
                     const std::vector<uint8_t> &coordinates) const {
        auto directions = direction_codes(configuration);
        MemberGeometry member_geometry(geometry_);
        std::vector<const DirectionData *> component_lines(h_);
        std::vector<uint8_t> is_pivot(rows_, 0);
        for (uint64_t coordinate = 0; coordinate < h_; ++coordinate) {
            is_pivot[pivots[coordinate]] = 1;
            component_lines[coordinate] = &member_geometry.line(directions[pivots[coordinate]]);
        }

        uint64_t width = h_ * geometry_.quotient();
        SparseBasis equations(field_, width);
        for (uint64_t row_index = 0; row_index < rows_; ++row_index) {
            if (is_pivot[row_index]) continue;
            const DirectionData &target = member_geometry.line(directions[row_index]);
            for (uint64_t point = 1; point < geometry_.points(); ++point) {
                SparseRow row;
                row.reserve(4 * h_);
                uint64_t shifted = target.translation[point];
                for (uint64_t coordinate = 0; coordinate < h_; ++coordinate) {
                    uint8_t scalar = coordinates[row_index * h_ + coordinate];
                    if (!scalar) continue;
                    uint64_t offset = coordinate * geometry_.quotient();
                    const auto &line = *component_lines[coordinate];
                    put(field_, row, (uint32_t)(offset + line.coset[shifted]), scalar);
                    put(field_, row, (uint32_t)(offset + line.coset[point]), -(int64_t)scalar);
                    put(field_, row, (uint32_t)(offset + line.coset[directions[row_index]]),
                        -(int64_t)scalar);
                    put(field_, row, (uint32_t)(offset + line.coset[0]), scalar);
                }
                equations.add(std::move(row));
            }
        }

        SparseRow defect;
        defect.reserve(2 * rows_ * h_);
        for (uint64_t row_index = 0; row_index < rows_; ++row_index)
            for (uint64_t coordinate = 0; coordinate < h_; ++coordinate) {
                uint8_t scalar = coordinates[row_index * h_ + coordinate];
                if (!scalar) continue;
                uint64_t offset = coordinate * geometry_.quotient();
                const auto &line = *component_lines[coordinate];
                put(field_, defect, (uint32_t)(offset + line.coset[directions[row_index]]), scalar);
                put(field_, defect, (uint32_t)(offset + line.coset[0]), -(int64_t)scalar);
            }
        return equations.add(std::move(defect));
    }

    uint64_t g_, h_, rows_;
    Field field_;
    Geometry geometry_;
};

bool accepts(const Request &request) {
    if (request.family->kind != Family::Kind::Explicit) return false;
    if (request.op != "is_fire") return true;
    const Matrix &family = *request.family->data;
    auto base = request.int_args.find("base_dim");
    if (base == request.int_args.end()) return true;
    uint64_t p = family.p, g = base->second;
    if (!is_prime(p) || p > 7 || g == 0 || g >= family.cols) return true;
    uint64_t h = family.cols - g;
    if (g > 5 || h > 8 || family.rows > 40) return true;
    uint64_t points = checked_power(p, g);
    if (!points || h > UINT64_MAX / (points / p)) return false;
    return h * (points / p) <= MAX_LINE_WIDTH;
}

R run(const Request &request) {
    if (request.op != "is_fire") return run_reduced_polynomial(request);
    if (request.family->kind != Family::Kind::Explicit)
        return R::failure(INVALID, "circuit_fires operations need an explicit family");
    const Matrix &family = *request.family->data;
    uint64_t p = family.p;
    uint64_t g = request.int_args.at("base_dim");
    if (!is_prime(p) || p > 7)
        return R::failure(INVALID, "circuit_fires needs a prime field of order at most 7");
    if (g == 0 || g >= family.cols)
        return R::failure(INVALID, "base_dim must be positive and smaller than the member column count");
    uint64_t h = family.cols - g;
    if (g > 5 || h > 8 || family.rows > 40)
        return R::failure(INVALID, "configuration exceeds the line_quotient backend limits");
    uint64_t points = checked_power(p, g);
    if (!points || h > UINT64_MAX / (points / p) || h * (points / p) > MAX_LINE_WIDTH)
        return R::failure(INVALID, "configuration exceeds the line_quotient backend limits");

    LineQuotient line_quotient(family, (uint32_t)p, g, h);
    Reduction reduction = parse_reduction(request.reduction);
    Shared shared;
    auto prepared = prepare_all(reduction, family.count, shared);
    if (!prepared.ok) return R::failure(prepared.error.status, prepared.error.message);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(request.threads, family.count));
    std::vector<Accumulator> accumulators;
    for (uint32_t thread = 0; thread < threads; ++thread)
        accumulators.emplace_back(reduction, &shared);
    auto statuses = parallel_ranges(family.count, threads,
        [&](uint32_t thread, uint64_t begin, uint64_t end) -> Status {
            for (uint64_t index = begin; index < end; ++index) {
                if (accumulators[thread].exhausted(index)) break;
                const Entry *configuration = family.at(index);
                bool value = configuration_is_circuit(
                    configuration, family.rows, g, h, (uint32_t)p);
                if (value) value = line_quotient.fire(configuration);
                accumulators[thread].boolean(index, value);
            }
            return ok();
        });
    for (const auto &status : statuses)
        if (!status.ok) return R::failure(status.error.status, status.error.message);
    return assemble(request, reduction, accumulators, shared);
}

BackendRegistration registration{Backend{
    "circuit_fires", "line_quotient",
    [] { return true; },
    accepts,
    run,
    100}};

} // namespace
} // namespace lk::circuit_fires
