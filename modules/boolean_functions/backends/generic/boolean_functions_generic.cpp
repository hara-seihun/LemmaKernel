/* Portable spectral and algebraic operations on binary truth tables. */
#include "../../../../runtime/src/reduce.hpp"

#include <bit>
#include <numeric>
#include <unordered_map>

namespace lk::boolean_functions {
namespace {

using R = Result<std::shared_ptr<Object>>;
constexpr int INVALID = 1;
constexpr int INTERNAL = 4;

bool is_power_of_two(uint64_t n) { return n != 0 && (n & (n - 1)) == 0; }

Status validate(const Request &req, bool scalar) {
    const Family &family = *req.family;
    if (family.prime() != 2) return fail(INVALID, "boolean_functions needs an F_2 matrix family");
    if (!is_power_of_two(family.cols())) return fail(INVALID, "truth-table column count must be a positive power of two");
    if (scalar && family.rows() != 1) return fail(INVALID, "operation needs a scalar truth table with one output row");
    return ok();
}

std::vector<int64_t> walsh_spectrum(const Entry *row, uint64_t size) {
    std::vector<int64_t> spectrum(size);
    for (uint64_t x = 0; x < size; ++x) spectrum[x] = row[x] ? -1 : 1;
    for (uint64_t half = 1; half < size; half *= 2)
        for (uint64_t block = 0; block < size; block += 2 * half)
            for (uint64_t i = 0; i < half; ++i) {
                int64_t a = spectrum[block + i], b = spectrum[block + half + i];
                spectrum[block + i] = a + b;
                spectrum[block + half + i] = a - b;
            }
    return spectrum;
}

uint64_t magnitude(int64_t value) {
    return value < 0 ? (uint64_t)(-value) : (uint64_t)value;
}

uint64_t nonlinearity(const Entry *row, uint64_t size) {
    auto spectrum = walsh_spectrum(row, size);
    uint64_t largest = 0;
    for (int64_t coefficient : spectrum) largest = std::max(largest, magnitude(coefficient));
    return (size - largest) / 2;
}

uint64_t algebraic_degree(const Entry *table, uint64_t outputs, uint64_t size) {
    uint64_t degree = 0;
    std::vector<Entry> coefficients(size);
    for (uint64_t output = 0; output < outputs; ++output) {
        std::copy(table + output * size, table + (output + 1) * size, coefficients.begin());
        for (uint64_t bit = 1; bit < size; bit *= 2)
            for (uint64_t mask = 0; mask < size; ++mask)
                if (mask & bit) coefficients[mask] ^= coefficients[mask ^ bit];
        for (uint64_t mask = 0; mask < size; ++mask)
            if (coefficients[mask]) degree = std::max<uint64_t>(degree, std::popcount(mask));
    }
    return degree;
}

bool is_bent(const Entry *row, uint64_t size) {
    unsigned n = std::countr_zero(size);
    if (n % 2 != 0) return false;
    uint64_t target = uint64_t{1} << (n / 2);
    auto spectrum = walsh_spectrum(row, size);
    return std::all_of(spectrum.begin(), spectrum.end(), [target](int64_t x) { return magnitude(x) == target; });
}

bool is_apn_packed(const Entry *table, uint64_t outputs, uint64_t size) {
    uint64_t output_count = uint64_t{1} << outputs;
    std::vector<uint8_t> counts(output_count);
    for (uint64_t a = 1; a < size; ++a) {
        std::fill(counts.begin(), counts.end(), 0);
        for (uint64_t x = 0; x < size; ++x) {
            uint64_t difference = 0;
            for (uint64_t output = 0; output < outputs; ++output)
                difference |= uint64_t(table[output * size + x] ^ table[output * size + (x ^ a)]) << output;
            if (++counts[difference] > 2) return false;
        }
    }
    return true;
}

bool is_apn_wide(const Entry *table, uint64_t outputs, uint64_t size) {
    std::unordered_map<std::string, uint8_t> counts;
    std::string difference(outputs, '\0');
    for (uint64_t a = 1; a < size; ++a) {
        counts.clear();
        for (uint64_t x = 0; x < size; ++x) {
            for (uint64_t output = 0; output < outputs; ++output)
                difference[output] = (char)(table[output * size + x] ^ table[output * size + (x ^ a)]);
            if (++counts[difference] > 2) return false;
        }
    }
    return true;
}

bool is_apn(const Entry *table, uint64_t outputs, uint64_t size) {
    return outputs <= 20 ? is_apn_packed(table, outputs, size) : is_apn_wide(table, outputs, size);
}

std::vector<Entry> affine_class(const Entry *row, uint64_t size) {
    std::vector<Entry> best(row, row + size);
    uint64_t weight = std::accumulate(best.begin(), best.end(), uint64_t{0});
    auto is_global_minimum = [&] {
        uint64_t zeros = size - weight;
        for (uint64_t i = 0; i < size; ++i)
            if (best[i] != (i < zeros ? 0 : 1)) return false;
        return true;
    };
    bool done = is_global_minimum();
    std::vector<uint64_t> linear{0};

    auto search = [&](auto &&self, uint64_t origin) -> void {
        if (done) return;
        if (linear.size() == size) {
            std::vector<Entry> candidate(size);
            for (uint64_t x = 0; x < size; ++x) candidate[x] = row[origin ^ linear[x]];
            if (candidate < best) {
                best = std::move(candidate);
                done = is_global_minimum();
            }
            return;
        }

        uint64_t old_size = linear.size();
        for (uint64_t image = 1; image < size && !done; ++image) {
            if (std::find(linear.begin(), linear.end(), image) != linear.end()) continue;
            for (uint64_t x = 0; x < old_size; ++x) linear.push_back(linear[x] ^ image);

            bool can_improve = true;
            for (uint64_t x = 0; x < linear.size(); ++x) {
                Entry value = row[origin ^ linear[x]];
                if (value < best[x]) break;
                if (value > best[x]) {
                    can_improve = false;
                    break;
                }
            }
            if (can_improve) self(self, origin);
            linear.resize(old_size);
        }
    };

    for (uint64_t origin = 0; origin < size && !done; ++origin) search(search, origin);
    return best;
}

R run_spectrum(const Request &req) {
    auto valid = validate(req, true);
    if (!valid.ok) return R::failure(valid.error.status, valid.error.message);
    const Family &family = *req.family;
    auto size_result = family.size();
    if (!size_result.ok) return R::failure(size_result.error.status, size_result.error.message);
    uint64_t count = size_result.value, length = family.cols();
    if ((unsigned __int128)count * length > SIZE_MAX / sizeof(int64_t))
        return R::failure(INVALID, "Walsh spectra are too large to materialise");

    auto output = std::make_shared<I64Matrices>();
    output->count = count;
    output->rows = 1;
    output->cols = length;
    output->entries.resize(count * length);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, count ? count : 1));
    auto statuses = parallel_ranges(count, threads, [&](uint32_t, uint64_t begin, uint64_t end) -> Status {
        Matrix member;
        for (uint64_t i = begin; i < end; ++i) {
            auto status = family.member_into(i, member);
            if (!status.ok) return status;
            auto spectrum = walsh_spectrum(member.entries.data(), length);
            std::copy(spectrum.begin(), spectrum.end(), output->entries.begin() + i * length);
        }
        return ok();
    });
    for (const auto &status : statuses)
        if (!status.ok) return R::failure(status.error.status, status.error.message);

    auto object = std::make_shared<Object>();
    object->kind = "lk.signed_matrices";
    object->i64_matrices = std::move(output);
    return R::success(object);
}

R run_affine_class(const Request &req) {
    auto valid = validate(req, true);
    if (!valid.ok) return R::failure(valid.error.status, valid.error.message);
    const Family &family = *req.family;
    auto size_result = family.size();
    if (!size_result.ok) return R::failure(size_result.error.status, size_result.error.message);
    uint64_t count = size_result.value, length = family.cols();
    if ((unsigned __int128)count * length > SIZE_MAX / sizeof(Entry))
        return R::failure(INVALID, "affine canonical forms are too large to materialise");

    auto output = std::make_shared<Matrix>();
    output->p = 2;
    output->count = count;
    output->rows = 1;
    output->cols = length;
    output->entries.resize(count * length);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, count ? count : 1));
    auto statuses = parallel_ranges(count, threads, [&](uint32_t, uint64_t begin, uint64_t end) -> Status {
        Matrix member;
        for (uint64_t i = begin; i < end; ++i) {
            auto status = family.member_into(i, member);
            if (!status.ok) return status;
            auto canonical = affine_class(member.entries.data(), length);
            std::copy(canonical.begin(), canonical.end(), output->entries.begin() + i * length);
        }
        return ok();
    });
    for (const auto &status : statuses)
        if (!status.ok) return R::failure(status.error.status, status.error.message);

    auto object = std::make_shared<Object>();
    object->kind = "gfp.matrix";
    object->matrix = std::move(output);
    return R::success(object);
}

R run_reduced(const Request &req) {
    bool scalar = req.op == "nonlinearity" || req.op == "is_bent";
    auto valid = validate(req, scalar);
    if (!valid.ok) return R::failure(valid.error.status, valid.error.message);
    const Family &family = *req.family;
    auto size_result = family.size();
    if (!size_result.ok) return R::failure(size_result.error.status, size_result.error.message);
    uint64_t count = size_result.value, outputs = family.rows(), length = family.cols();
    Reduction reduction = parse_reduction(req.reduction);
    Shared shared;
    auto prepared = prepare_all(reduction, count, shared);
    if (!prepared.ok) return R::failure(prepared.error.status, prepared.error.message);

    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, count ? count : 1));
    std::vector<Accumulator> accumulators;
    for (uint32_t thread = 0; thread < threads; ++thread) accumulators.emplace_back(reduction, &shared);
    auto statuses = parallel_ranges(count, threads, [&](uint32_t thread, uint64_t begin, uint64_t end) -> Status {
        Matrix member;
        for (uint64_t i = begin; i < end; ++i) {
            if (accumulators[thread].exhausted(i)) break;
            auto status = family.member_into(i, member);
            if (!status.ok) return status;
            const Entry *table = member.entries.data();
            if (req.op == "nonlinearity") accumulators[thread].integer(i, nonlinearity(table, length));
            else if (req.op == "algebraic_degree") accumulators[thread].integer(i, algebraic_degree(table, outputs, length));
            else if (req.op == "is_bent") accumulators[thread].boolean(i, is_bent(table, length));
            else if (req.op == "is_apn") accumulators[thread].boolean(i, is_apn(table, outputs, length));
            else return fail(INTERNAL, "unknown boolean_functions operation " + req.op);
        }
        return ok();
    });
    for (const auto &status : statuses)
        if (!status.ok) return R::failure(status.error.status, status.error.message);
    return assemble(req, reduction, accumulators, shared);
}

R run(const Request &req) {
    if (req.op == "walsh_spectrum") return run_spectrum(req);
    if (req.op == "affine_class") return run_affine_class(req);
    return run_reduced(req);
}

BackendRegistration registration{Backend{
    "boolean_functions", "generic",
    [] { return true; },
    [](const Request &req) { return req.family->prime() == 2; },
    run,
    0}};

} // namespace
} // namespace lk::boolean_functions
