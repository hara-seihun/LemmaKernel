#include "../../../../runtime/src/reduce.hpp"

#include <algorithm>
#include <functional>
#include <unordered_map>
#include <unordered_set>

namespace lk::circuit_fires {
namespace {

using R = Result<std::shared_ptr<Object>>;
constexpr int INVALID = 1;
constexpr int INTERNAL = 4;

uint32_t inverse(uint32_t value, uint32_t p) {
    uint32_t result = 1;
    uint32_t exponent = p - 2;
    while (exponent) {
        if (exponent & 1) result = result * value % p;
        value = value * value % p;
        exponent >>= 1;
    }
    return result;
}

struct Basis {
    uint32_t p;
    uint64_t width;
    std::vector<Entry> rows;
    std::vector<uint32_t> pivots;

    uint64_t rank() const { return pivots.size(); }

    void reduce(std::vector<Entry> &row) const {
        for (size_t i = 0; i < pivots.size(); ++i) {
            Entry coefficient = row[pivots[i]];
            if (!coefficient) continue;
            const Entry *pivot = rows.data() + i * width;
            for (uint64_t column = 0; column < width; ++column)
                row[column] = (row[column] + p - coefficient * pivot[column] % p) % p;
        }
    }

    bool add(std::vector<Entry> row) {
        reduce(row);
        auto found = std::find_if(row.begin(), row.end(), [](Entry value) { return value != 0; });
        if (found == row.end()) return false;
        uint32_t pivot = (uint32_t)(found - row.begin());
        Entry scale = inverse(*found, p);
        if (scale != 1)
            for (Entry &value : row) value = value * scale % p;
        pivots.push_back(pivot);
        rows.insert(rows.end(), row.begin(), row.end());
        return true;
    }
};

uint64_t matrix_rank(std::vector<Entry> rows, uint64_t count, uint64_t width, uint32_t p) {
    Basis basis{p, width, {}, {}};
    for (uint64_t row = 0; row < count; ++row)
        basis.add(std::vector<Entry>(rows.begin() + row * width, rows.begin() + (row + 1) * width));
    return basis.rank();
}

bool circuit(const Entry *configuration, uint64_t k, uint64_t g, uint64_t h, uint32_t p) {
    if (k == 0 || g == 0 || h == 0) return false;
    uint64_t columns = g + h;
    std::vector<Entry> directions(k * g), covectors(k * h), tensors(k * g * h);
    std::unordered_set<uint64_t> projective_directions;
    for (uint64_t row = 0; row < k; ++row) {
        const Entry *source = configuration + row * columns;
        bool nonzero_direction = false, nonzero_covector = false;
        uint64_t first = g;
        for (uint64_t column = 0; column < g; ++column) {
            directions[row * g + column] = source[column];
            if (source[column] && first == g) first = column;
            nonzero_direction |= source[column] != 0;
        }
        for (uint64_t column = 0; column < h; ++column) {
            Entry value = source[g + column];
            covectors[row * h + column] = value;
            nonzero_covector |= value != 0;
            for (uint64_t base = 0; base < g; ++base)
                tensors[(row * h + column) * g + base] = value * source[base] % p;
        }
        if (!nonzero_direction || !nonzero_covector) return false;
        Entry scale = inverse(source[first], p);
        uint64_t code = 0, place = 1;
        for (uint64_t column = 0; column < g; ++column) {
            code += (source[column] * scale % p) * place;
            place *= p;
        }
        if (!projective_directions.insert(code).second) return false;
    }
    if (matrix_rank(directions, k, g, p) != g || matrix_rank(covectors, k, h, p) != h)
        return false;
    uint64_t tensor_width = g * h;
    for (uint64_t column = 0; column < tensor_width; ++column) {
        uint64_t sum = 0;
        for (uint64_t row = 0; row < k; ++row) sum += tensors[row * tensor_width + column];
        if (sum % p) return false;
    }
    return matrix_rank(std::move(tensors), k, tensor_width, p) + 1 == k;
}

uint64_t checked_power(uint64_t p, uint64_t exponent) {
    uint64_t value = 1;
    for (uint64_t i = 0; i < exponent; ++i) {
        if (value > UINT64_MAX / p) return 0;
        value *= p;
    }
    return value;
}

uint64_t add_direction(uint64_t point, const Entry *direction, uint64_t g, uint32_t p) {
    uint64_t result = 0, place = 1;
    for (uint64_t coordinate = 0; coordinate < g; ++coordinate) {
        uint64_t digit = point / place % p;
        result += ((digit + direction[coordinate]) % p) * place;
        place *= p;
    }
    return result;
}

bool verifies(const Entry *configuration, uint64_t k, uint64_t g, uint64_t h, uint32_t p,
              const Matrix &defects, const Matrix &potential) {
    if (!circuit(configuration, k, g, h, p)) return false;
    uint64_t points = checked_power(p, g);
    if (!points || defects.p != p || defects.count != 1 || defects.rows != 1 || defects.cols != k ||
        potential.p != p || potential.count != 1 || potential.rows != points || potential.cols != h)
        return false;
    const Entry *lambda = defects.at(0);
    const Entry *values = potential.at(0);
    for (uint64_t coordinate = 0; coordinate < h; ++coordinate)
        if (values[coordinate] != 0) return false;
    uint64_t carry = 0;
    for (uint64_t row = 0; row < k; ++row) {
        const Entry *source = configuration + row * (g + h);
        carry += lambda[row];
        for (uint64_t point = 0; point < points; ++point) {
            uint64_t shifted = add_direction(point, source, g, p);
            uint64_t defect = 0;
            for (uint64_t coordinate = 0; coordinate < h; ++coordinate) {
                Entry difference = (values[shifted * h + coordinate] + p -
                                    values[point * h + coordinate]) % p;
                defect += source[g + coordinate] * difference;
            }
            if (defect % p != lambda[row]) return false;
        }
    }
    return carry % p != 0;
}

bool find_potential(const Entry *configuration, uint64_t k, uint64_t g, uint64_t h,
                    uint32_t p, std::vector<Entry> &solution) {
    if (!circuit(configuration, k, g, h, p)) return false;
    uint64_t points = checked_power(p, g);
    uint64_t variables = points * h + k;
    uint64_t equations = k * points + h + 1;
    if (!points || variables > 1024 || equations > (1ULL << 20) ||
        variables + 1 > (1ULL << 24) / equations)
        return false;

    uint64_t width = variables + 1;
    std::vector<Entry> system(equations * width, 0);
    uint64_t equation = 0;
    for (uint64_t row = 0; row < k; ++row) {
        const Entry *source = configuration + row * (g + h);
        for (uint64_t point = 0; point < points; ++point, ++equation) {
            Entry *target = system.data() + equation * width;
            uint64_t shifted = add_direction(point, source, g, p);
            for (uint64_t coordinate = 0; coordinate < h; ++coordinate) {
                target[shifted * h + coordinate] =
                    (target[shifted * h + coordinate] + source[g + coordinate]) % p;
                target[point * h + coordinate] =
                    (target[point * h + coordinate] + p - source[g + coordinate]) % p;
            }
            target[points * h + row] = p - 1;
        }
    }
    for (uint64_t coordinate = 0; coordinate < h; ++coordinate, ++equation)
        system[equation * width + coordinate] = 1;
    Entry *carry = system.data() + equation * width;
    for (uint64_t row = 0; row < k; ++row) carry[points * h + row] = 1;
    carry[variables] = 1;

    uint64_t rank = 0;
    std::vector<uint64_t> pivots;
    for (uint64_t column = 0; column < variables && rank < equations; ++column) {
        uint64_t pivot = rank;
        while (pivot < equations && system[pivot * width + column] == 0) ++pivot;
        if (pivot == equations) continue;
        for (uint64_t j = column; j < width; ++j)
            std::swap(system[rank * width + j], system[pivot * width + j]);
        Entry scale = inverse(system[rank * width + column], p);
        for (uint64_t j = column; j < width; ++j)
            system[rank * width + j] = system[rank * width + j] * scale % p;
        for (uint64_t other = 0; other < equations; ++other) {
            if (other == rank) continue;
            Entry factor = system[other * width + column];
            if (!factor) continue;
            for (uint64_t j = column; j < width; ++j)
                system[other * width + j] =
                    (system[other * width + j] + p - factor * system[rank * width + j] % p) % p;
        }
        pivots.push_back(column);
        ++rank;
    }
    for (uint64_t row = rank; row < equations; ++row) {
        bool zero = true;
        for (uint64_t column = 0; column < variables; ++column)
            zero &= system[row * width + column] == 0;
        if (zero && system[row * width + variables] != 0) return false;
    }
    solution.assign(variables, 0);
    for (uint64_t row = 0; row < rank; ++row)
        solution[pivots[row]] = system[row * width + variables];
    return true;
}

void exponents_of_degree(uint32_t p, uint64_t variables, uint64_t coordinate,
                         uint32_t remaining, std::vector<uint32_t> &current,
                         std::vector<std::vector<uint32_t>> &output) {
    if (coordinate == variables) {
        if (remaining == 0) output.push_back(current);
        return;
    }
    uint32_t largest = std::min<uint32_t>(p - 1, remaining);
    for (uint32_t value = 0; value <= largest; ++value) {
        current[coordinate] = value;
        exponents_of_degree(p, variables, coordinate + 1, remaining - value, current, output);
    }
}

std::vector<std::vector<uint32_t>> exponents_of_degree(uint32_t p, uint64_t variables,
                                                        uint32_t degree) {
    std::vector<std::vector<uint32_t>> output;
    std::vector<uint32_t> current(variables, 0);
    exponents_of_degree(p, variables, 0, degree, current, output);
    return output;
}

uint64_t exponent_code(const std::vector<uint32_t> &exponent, uint32_t p) {
    uint64_t code = 0, place = 1;
    for (uint32_t value : exponent) {
        code += value * place;
        place *= p;
    }
    return code;
}

bool degree_p_fire(const Entry *configuration, uint64_t k, uint64_t g, uint64_t h, uint32_t p) {
    auto monomials = exponents_of_degree(p, g, p);
    auto derivatives = exponents_of_degree(p, g, p - 1);
    uint64_t width = monomials.size() * h;
    std::unordered_map<uint64_t, uint64_t> monomial_index;
    for (uint64_t index = 0; index < monomials.size(); ++index)
        monomial_index.emplace(exponent_code(monomials[index], p), index);

    Basis constraints{p, width, {}, {}};
    std::vector<Entry> row(width, 0);
    for (uint64_t circuit_row = 0; circuit_row < k; ++circuit_row) {
        const Entry *source = configuration + circuit_row * (g + h);
        for (const auto &beta : derivatives) {
            std::fill(row.begin(), row.end(), 0);
            for (uint64_t direction_coordinate = 0; direction_coordinate < g;
                 ++direction_coordinate) {
                if (beta[direction_coordinate] == p - 1 || source[direction_coordinate] == 0)
                    continue;
                auto alpha = beta;
                ++alpha[direction_coordinate];
                uint64_t monomial = monomial_index.at(exponent_code(alpha, p));
                Entry derivative = (beta[direction_coordinate] + 1) *
                                   source[direction_coordinate] % p;
                for (uint64_t fibre = 0; fibre < h; ++fibre)
                    row[monomial * h + fibre] =
                        (row[monomial * h + fibre] +
                         derivative * source[g + fibre]) % p;
            }
            constraints.add(row);
        }
    }

    std::fill(row.begin(), row.end(), 0);
    for (uint64_t circuit_row = 0; circuit_row < k; ++circuit_row) {
        const Entry *source = configuration + circuit_row * (g + h);
        for (uint64_t monomial = 0; monomial < monomials.size(); ++monomial) {
            Entry value = 1;
            for (uint64_t coordinate = 0; coordinate < g; ++coordinate)
                for (uint32_t power = 0; power < monomials[monomial][coordinate]; ++power)
                    value = value * source[coordinate] % p;
            for (uint64_t fibre = 0; fibre < h; ++fibre)
                row[monomial * h + fibre] =
                    (row[monomial * h + fibre] + value * source[g + fibre]) % p;
        }
    }
    constraints.reduce(row);
    return std::any_of(row.begin(), row.end(), [](Entry value) { return value != 0; });
}

struct VectorHash {
    size_t operator()(const std::vector<Entry> &values) const {
        uint64_t hash = 1469598103934665603ULL;
        for (Entry value : values) {
            hash ^= value;
            hash *= 1099511628211ULL;
        }
        return (size_t)hash;
    }
};

class PolynomialDual {
public:
    PolynomialDual(uint32_t p, uint64_t g) : p_(p), g_(g), points_(checked_power(p, g)) {
        binomial_.assign(p_, std::vector<Entry>(p_, 0));
        for (uint32_t n = 0; n < p_; ++n) {
            binomial_[n][0] = binomial_[n][n] = 1;
            for (uint32_t k = 1; k < n; ++k)
                binomial_[n][k] = (binomial_[n - 1][k - 1] + binomial_[n - 1][k]) % p_;
        }
        for (uint64_t code = 0; code < points_; ++code) {
            uint64_t value = code, degree = 0;
            std::vector<uint32_t> exponent(g_);
            for (uint64_t coordinate = 0; coordinate < g_; ++coordinate) {
                exponent[coordinate] = value % p_;
                value /= p_;
                degree += exponent[coordinate];
            }
            if (degree >= 2) monomials_.push_back(std::move(exponent));
        }
    }

    void prepare(const Matrix &family) {
        std::unordered_set<std::vector<Entry>, VectorHash> directions;
        for (uint64_t member = 0; member < family.count; ++member)
            for (uint64_t row = 0; row < family.rows; ++row) {
                const Entry *source = family.at(member) + row * family.cols;
                directions.emplace(source, source + g_);
            }
        for (const auto &direction : directions) build(direction);
    }

    bool fire(const Entry *configuration, uint64_t k, uint64_t h) const {
        uint64_t width = monomials_.size() * h;
        Basis full{p_, width, {}, {}};
        std::vector<Entry> row(width);
        for (uint64_t index = 0; index < k; ++index) {
            const Entry *source = configuration + index * (g_ + h);
            std::vector<Entry> direction(source, source + g_);
            const Basis &scalar = cache_.at(direction);
            for (uint64_t basis_row = 0; basis_row < scalar.rank(); ++basis_row) {
                std::fill(row.begin(), row.end(), 0);
                const Entry *coefficients = scalar.rows.data() + basis_row * monomials_.size();
                for (uint64_t monomial = 0; monomial < monomials_.size(); ++monomial)
                    if (coefficients[monomial])
                        for (uint64_t fibre = 0; fibre < h; ++fibre)
                            row[monomial * h + fibre] =
                                coefficients[monomial] * source[g_ + fibre] % p_;
                full.add(row);
            }
        }

        std::fill(row.begin(), row.end(), 0);
        for (uint64_t index = 0; index < k; ++index) {
            const Entry *source = configuration + index * (g_ + h);
            for (uint64_t monomial = 0; monomial < monomials_.size(); ++monomial) {
                Entry weight = 1;
                for (uint64_t coordinate = 0; coordinate < g_; ++coordinate)
                    for (uint32_t power = 0; power < monomials_[monomial][coordinate]; ++power)
                        weight = weight * source[coordinate] % p_;
                for (uint64_t fibre = 0; fibre < h; ++fibre)
                    row[monomial * h + fibre] =
                        (row[monomial * h + fibre] + weight * source[g_ + fibre]) % p_;
            }
        }
        full.reduce(row);
        return std::any_of(row.begin(), row.end(), [](Entry value) { return value != 0; });
    }

private:
    uint32_t p_;
    uint64_t g_, points_;
    std::vector<std::vector<Entry>> binomial_;
    std::vector<std::vector<uint32_t>> monomials_;
    std::unordered_map<std::vector<Entry>, Basis, VectorHash> cache_;

    uint64_t encode(const std::vector<uint32_t> &digits) const {
        uint64_t code = 0, place = 1;
        for (uint32_t digit : digits) {
            code += digit * place;
            place *= p_;
        }
        return code;
    }

    void build(const std::vector<Entry> &direction) {
        if (cache_.contains(direction)) return;
        std::unordered_map<uint64_t, std::vector<Entry>> raw;
        for (uint64_t monomial = 0; monomial < monomials_.size(); ++monomial) {
            const auto &powers = monomials_[monomial];
            uint64_t subset_count = 1;
            for (uint32_t power : powers) subset_count *= power + 1;
            for (uint64_t subset_code = 1; subset_code < subset_count; ++subset_code) {
                uint64_t value = subset_code;
                std::vector<uint32_t> selected(g_), remainder(g_);
                Entry direction_power = 1;
                for (uint64_t coordinate = 0; coordinate < g_; ++coordinate) {
                    selected[coordinate] = value % (powers[coordinate] + 1);
                    value /= powers[coordinate] + 1;
                    remainder[coordinate] = powers[coordinate] - selected[coordinate];
                    for (uint32_t power = 0; power < selected[coordinate]; ++power)
                        direction_power = direction_power * direction[coordinate] % p_;
                }
                if (!direction_power) continue;
                uint64_t split_count = 1;
                for (uint32_t power : remainder) split_count *= power + 1;
                for (uint64_t split_code = 1; split_code < split_count; ++split_code) {
                    value = split_code;
                    std::vector<uint32_t> t(g_), q(g_);
                    Entry coefficient = direction_power;
                    for (uint64_t coordinate = 0; coordinate < g_; ++coordinate) {
                        t[coordinate] = value % (remainder[coordinate] + 1);
                        value /= remainder[coordinate] + 1;
                        q[coordinate] = remainder[coordinate] - t[coordinate];
                        coefficient = coefficient * binomial_[powers[coordinate]][selected[coordinate]] % p_;
                        coefficient = coefficient * binomial_[remainder[coordinate]][t[coordinate]] % p_;
                    }
                    if (!coefficient) continue;
                    uint64_t key = encode(q) * points_ + encode(t);
                    auto [it, inserted] = raw.try_emplace(key, monomials_.size(), 0);
                    it->second[monomial] = (it->second[monomial] + coefficient) % p_;
                }
            }
        }
        Basis basis{p_, monomials_.size(), {}, {}};
        for (auto &[key, row] : raw) {
            (void)key;
            basis.add(std::move(row));
        }
        cache_.emplace(direction, std::move(basis));
    }
};

R run(const Request &req) {
    if (req.family->kind != Family::Kind::Explicit)
        return R::failure(INVALID, "circuit_fires operations need an explicit family");
    const Matrix &family = *req.family->data;
    uint64_t p = family.p;
    uint64_t g = req.int_args.at("base_dim");
    if (!is_prime(p) || p > 7)
        return R::failure(INVALID, "circuit_fires needs a prime field of order at most 7");
    if (g == 0 || g >= family.cols)
        return R::failure(INVALID, "base_dim must be positive and smaller than the member column count");
    uint64_t h = family.cols - g;
    if (g > 5 || h > 8 || family.rows > 40)
        return R::failure(INVALID, "configuration exceeds the reduced_polynomial backend limits");

    const Matrix *defects = nullptr, *potential = nullptr;
    if (req.op == "verifies_potential") {
        auto d = req.handle_args.find("defects"), c = req.handle_args.find("potential");
        if (d == req.handle_args.end() || !d->second->matrix)
            return R::failure(INVALID, "defects must be one field vector");
        if (c == req.handle_args.end() || !c->second->matrix)
            return R::failure(INVALID, "potential must be one field matrix");
        defects = d->second->matrix.get();
        potential = c->second->matrix.get();
        uint64_t points = checked_power(p, g);
        if (defects->p != p || defects->count != 1 || defects->rows != 1 || defects->cols != family.rows)
            return R::failure(INVALID, "defects must have one entry per configuration row over the same prime");
        if (!points || potential->p != p || potential->count != 1 ||
            potential->rows != points || potential->cols != h)
            return R::failure(INVALID, "potential must have p^base_dim rows and fibre_dim columns over the same prime");
    } else if (req.op != "is_circuit" && req.op != "is_fire" && req.op != "find_potential") {
        return R::failure(INTERNAL, "unknown circuit_fires operation " + req.op);
    }

    if (req.op == "find_potential") {
        uint64_t points = checked_power(p, g);
        uint64_t variables = points * h + family.rows;
        uint64_t equations = family.rows * points + h + 1;
        if (!points || variables > 1024 || equations > (1ULL << 20) ||
            variables + 1 > (1ULL << 24) / equations)
            return R::failure(INVALID, "configuration is too large for explicit potential construction");
        auto solutions = std::make_shared<Solutions>();
        solutions->p = p;
        solutions->count = family.count;
        solutions->length = variables;
        solutions->solvable.assign(family.count, 0);
        solutions->entries.assign(family.count * variables, 0);
        uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, family.count));
        auto statuses = parallel_ranges(family.count, threads,
            [&](uint32_t, uint64_t begin, uint64_t end) -> Status {
                for (uint64_t index = begin; index < end; ++index) {
                    std::vector<Entry> answer;
                    if (!find_potential(family.at(index), family.rows, g, h, (uint32_t)p, answer))
                        continue;
                    solutions->solvable[index] = 1;
                    std::copy(answer.begin(), answer.end(),
                              solutions->entries.begin() + index * variables);
                }
                return ok();
            });
        for (const auto &status : statuses)
            if (!status.ok) return R::failure(status.error.status, status.error.message);
        auto object = std::make_shared<Object>();
        object->kind = "gfp.solutions";
        object->solutions = std::move(solutions);
        return R::success(std::move(object));
    }

    std::unique_ptr<PolynomialDual> dual;
    std::vector<uint8_t> fire_state;
    if (req.op == "is_fire") {
        fire_state.assign(family.count, 0);
        uint32_t preparation_threads =
            std::max<uint32_t>(1, std::min<uint64_t>(req.threads, family.count));
        auto statuses = parallel_ranges(family.count, preparation_threads,
            [&](uint32_t, uint64_t begin, uint64_t end) -> Status {
                for (uint64_t index = begin; index < end; ++index) {
                    const Entry *configuration = family.at(index);
                    if (!circuit(configuration, family.rows, g, h, (uint32_t)p)) continue;
                    fire_state[index] = degree_p_fire(
                        configuration, family.rows, g, h, (uint32_t)p) ? 1 : 2;
                }
                return ok();
            });
        for (const auto &status : statuses)
            if (!status.ok) return R::failure(status.error.status, status.error.message);
        if (std::find(fire_state.begin(), fire_state.end(), 2) != fire_state.end()) {
            dual = std::make_unique<PolynomialDual>((uint32_t)p, g);
            dual->prepare(family);
        }
    }

    Reduction reduction = parse_reduction(req.reduction);
    Shared shared;
    auto prepared = prepare_all(reduction, family.count, shared);
    if (!prepared.ok) return R::failure(prepared.error.status, prepared.error.message);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, family.count));
    std::vector<Accumulator> accumulators;
    for (uint32_t thread = 0; thread < threads; ++thread)
        accumulators.emplace_back(reduction, &shared);
    auto statuses = parallel_ranges(family.count, threads,
        [&](uint32_t thread, uint64_t begin, uint64_t end) -> Status {
            for (uint64_t index = begin; index < end; ++index) {
                if (accumulators[thread].exhausted(index)) break;
                const Entry *configuration = family.at(index);
                bool value;
                if (req.op == "is_fire") {
                    value = fire_state[index] == 1 ||
                            (fire_state[index] == 2 && dual->fire(configuration, family.rows, h));
                } else {
                    value = circuit(configuration, family.rows, g, h, (uint32_t)p);
                }
                if (value && req.op == "verifies_potential")
                    value = verifies(configuration, family.rows, g, h, (uint32_t)p,
                                     *defects, *potential);
                accumulators[thread].boolean(index, value);
            }
            return ok();
        });
    for (const auto &status : statuses)
        if (!status.ok) return R::failure(status.error.status, status.error.message);
    return assemble(req, reduction, accumulators, shared);
}

BackendRegistration registration{Backend{
    "circuit_fires", "reduced_polynomial",
    [] { return true; },
    [](const Request &req) { return req.family->kind == Family::Kind::Explicit; },
    run,
    0}};

} // namespace
} // namespace lk::circuit_fires
