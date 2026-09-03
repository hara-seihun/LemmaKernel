/* Portable point-set computations in PG(d-1,p).
 *
 * A request fixes the point dictionary. Incidence with every ambient line or hyperplane is
 * computed once, then each selected subset is reduced to table lookups. */
#include "../../../../runtime/src/reduce.hpp"
#include "../../../gfp/backends/generic/field.hpp"

#include <unordered_map>
#include <unordered_set>

namespace lk::projective_sets {
namespace {

using R = Result<std::shared_ptr<Object>>;
constexpr int INVALID = 1;
constexpr int INTERNAL = 4;
constexpr uint64_t MAX_INCIDENCES = 1ULL << 31;

enum class Op {
    IsArc, IsCap, IsBlockingSet, IsHyperoval, IsOvoid,
    MaxCollinear, SpannedRank, SecantCount, TangentCount, PassantCount
};

Result<Op> parse_op(const std::string &name) {
    static const std::map<std::string, Op> names{
        {"is_arc", Op::IsArc}, {"is_cap", Op::IsCap}, {"is_blocking_set", Op::IsBlockingSet},
        {"is_hyperoval", Op::IsHyperoval}, {"is_ovoid", Op::IsOvoid},
        {"max_collinear", Op::MaxCollinear}, {"spanned_rank", Op::SpannedRank},
        {"secant_count", Op::SecantCount}, {"tangent_count", Op::TangentCount},
        {"passant_count", Op::PassantCount}};
    auto it = names.find(name);
    if (it == names.end()) return Result<Op>::failure(INTERNAL, "unknown projective_sets operation " + name);
    return Result<Op>::success(it->second);
}

bool needs_lines(Op op) {
    return op == Op::IsCap || op == Op::IsHyperoval || op == Op::IsOvoid ||
           op == Op::MaxCollinear || op == Op::SecantCount ||
           op == Op::TangentCount || op == Op::PassantCount;
}

bool needs_hyperplanes(Op op) { return op == Op::IsArc || op == Op::IsBlockingSet; }

Result<uint64_t> checked_power(uint64_t base, uint64_t exponent) {
    uint64_t value = 1;
    for (uint64_t i = 0; i < exponent; ++i) {
        if (base && value > UINT64_MAX / base)
            return Result<uint64_t>::failure(INVALID, "ambient projective space is too large");
        value *= base;
    }
    return Result<uint64_t>::success(value);
}

Status validate_dictionary(const Family &family) {
    const Matrix &dictionary = *family.data;
    std::unordered_set<std::string> seen;
    seen.reserve(dictionary.count * 2);
    for (uint64_t i = 0; i < dictionary.count; ++i) {
        const Entry *point = dictionary.at(i);
        uint64_t lead = dictionary.cols;
        for (uint64_t j = 0; j < dictionary.cols; ++j)
            if (point[j]) { lead = j; break; }
        if (lead == dictionary.cols)
            return fail(INVALID, "projective point dictionary contains zero");
        if (point[lead] != 1)
            return fail(INVALID, "projective point dictionary is not normalised");
        std::string key(reinterpret_cast<const char *>(point), dictionary.cols * sizeof(Entry));
        if (!seen.insert(std::move(key)).second)
            return fail(INVALID, "projective point dictionary contains a duplicate");
    }
    return ok();
}

struct Geometry {
    const Family &family;
    uint64_t p, d, dictionary_size;
    std::unordered_map<std::string, uint64_t> point_indices;
    uint64_t line_count = 0, hyperplane_count = 0;
    std::vector<uint8_t> line_incidence, hyperplane_incidence;

    explicit Geometry(const Family &f)
        : family(f), p(f.prime()), d(f.cols()), dictionary_size(f.data->count) {
        point_indices.reserve(dictionary_size * 2);
        for (uint64_t i = 0; i < dictionary_size; ++i)
            point_indices.emplace(key(f.data->at(i)), i);
    }

    std::string key(const Entry *row) const {
        return std::string(reinterpret_cast<const char *>(row), d * sizeof(Entry));
    }

    Status member_points(const Matrix &member, std::vector<uint64_t> &out) const {
        out.resize(member.rows);
        for (uint64_t i = 0; i < member.rows; ++i) {
            auto found = point_indices.find(key(member.entries.data() + i * d));
            if (found == point_indices.end()) return fail(INTERNAL, "subset member contains a row outside its dictionary");
            out[i] = found->second;
        }
        return ok();
    }

    Status build_hyperplanes() {
        std::vector<uint64_t> tails(d);
        uint64_t total = 0;
        for (uint64_t lead = 0; lead < d; ++lead) {
            auto count = checked_power(p, d - lead - 1);
            if (!count.ok) return fail(count.error.status, count.error.message);
            tails[lead] = count.value;
            if (total > UINT64_MAX - count.value) return fail(INVALID, "ambient projective space is too large");
            total += count.value;
        }
        if (dictionary_size && total > MAX_INCIDENCES / dictionary_size)
            return fail(INVALID, "hyperplane incidence table would be too large");
        hyperplane_count = total;
        hyperplane_incidence.reserve(total * dictionary_size);
        gfp::Field field(p);
        std::vector<Entry> form(d, 0);
        for (uint64_t lead = 0; lead < d; ++lead) {
            for (uint64_t code = 0; code < tails[lead]; ++code) {
                std::fill(form.begin(), form.end(), 0);
                form[lead] = 1;
                uint64_t digits = code;
                for (uint64_t j = d; j-- > lead + 1;) {
                    form[j] = static_cast<Entry>(digits % p);
                    digits /= p;
                }
                for (uint64_t point = 0; point < dictionary_size; ++point) {
                    const Entry *v = family.data->at(point);
                    uint64_t product = 0;
                    for (uint64_t j = 0; j < d; ++j)
                        product = field.reduce(product + static_cast<uint64_t>(form[j]) * v[j]);
                    hyperplane_incidence.push_back(product == 0);
                }
            }
        }
        return ok();
    }

    Status build_lines() {
        auto lines = make_grassmannian(p, d, 2);
        if (!lines.ok) return fail(lines.error.status, lines.error.message);
        auto size = lines.value->size();
        if (!size.ok) return fail(size.error.status, size.error.message);
        if (dictionary_size && size.value > MAX_INCIDENCES / dictionary_size)
            return fail(INVALID, "line incidence table would be too large");
        line_count = size.value;
        line_incidence.reserve(line_count * dictionary_size);
        Matrix line;
        std::vector<Entry> reduced(d);
        for (uint64_t i = 0; i < line_count; ++i) {
            auto st = lines.value->member_into(i, line);
            if (!st.ok) return st;
            gfp::EchelonBasis basis(p, d);
            for (uint64_t row = 0; row < line.rows; ++row) basis.add(line.entries.data() + row * d);
            for (uint64_t point = 0; point < dictionary_size; ++point) {
                std::copy(family.data->at(point), family.data->at(point) + d, reduced.begin());
                basis.reduce_into(reduced.data());
                line_incidence.push_back(std::all_of(reduced.begin(), reduced.end(), [](Entry x) { return x == 0; }));
            }
        }
        return ok();
    }

    uint64_t intersection(const std::vector<uint8_t> &table, uint64_t object,
                          const std::vector<uint64_t> &points) const {
        uint64_t count = 0;
        const uint8_t *row = table.data() + object * dictionary_size;
        for (uint64_t point : points) count += row[point];
        return count;
    }

    bool cap(const std::vector<uint64_t> &points) const {
        for (uint64_t line = 0; line < line_count; ++line)
            if (intersection(line_incidence, line, points) > 2) return false;
        return true;
    }

    bool arc(const std::vector<uint64_t> &points) const {
        for (uint64_t hyperplane = 0; hyperplane < hyperplane_count; ++hyperplane)
            if (intersection(hyperplane_incidence, hyperplane, points) >= d) return false;
        return true;
    }

    bool blocking(const std::vector<uint64_t> &points) const {
        for (uint64_t hyperplane = 0; hyperplane < hyperplane_count; ++hyperplane)
            if (intersection(hyperplane_incidence, hyperplane, points) == 0) return false;
        return true;
    }

    uint64_t max_collinear(const std::vector<uint64_t> &points) const {
        uint64_t best = 0;
        for (uint64_t line = 0; line < line_count; ++line)
            best = std::max(best, intersection(line_incidence, line, points));
        return best;
    }

    uint64_t lines_with(const std::vector<uint64_t> &points, uint64_t target) const {
        uint64_t count = 0;
        for (uint64_t line = 0; line < line_count; ++line)
            count += intersection(line_incidence, line, points) == target;
        return count;
    }
};

R run(const Request &req) {
    if (req.family->kind != Family::Kind::Subsets)
        return R::failure(INVALID, "projective_sets operations are defined on `subsets` families only");
    if (req.family->prime() < 2 || req.family->prime() >= (1ULL << 32))
        return R::failure(INVALID, "projective_sets needs a prime p < 2^32");
    auto valid = validate_dictionary(*req.family);
    if (!valid.ok) return R::failure(valid.error.status, valid.error.message);
    auto parsed = parse_op(req.op);
    if (!parsed.ok) return R::failure(parsed.error.status, parsed.error.message);
    Op op = parsed.value;

    Geometry geometry(*req.family);
    if (needs_hyperplanes(op)) {
        auto st = geometry.build_hyperplanes();
        if (!st.ok) return R::failure(st.error.status, st.error.message);
    }
    if (needs_lines(op)) {
        auto st = geometry.build_lines();
        if (!st.ok) return R::failure(st.error.status, st.error.message);
    }

    auto size = req.family->size();
    if (!size.ok) return R::failure(size.error.status, size.error.message);
    Reduction reduction = parse_reduction(req.reduction);
    Shared shared;
    auto prepared = prepare_all(reduction, size.value, shared);
    if (!prepared.ok) return R::failure(prepared.error.status, prepared.error.message);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, size.value ? size.value : 1));
    std::vector<Accumulator> accumulators;
    for (uint32_t t = 0; t < threads; ++t) accumulators.emplace_back(reduction, &shared);

    auto statuses = parallel_ranges(size.value, threads, [&](uint32_t thread, uint64_t begin, uint64_t end) -> Status {
        Matrix member;
        std::vector<uint64_t> points;
        for (uint64_t index = begin; index < end; ++index) {
            if (accumulators[thread].exhausted(index)) break;
            auto st = req.family->member_into(index, member);
            if (!st.ok) return st;
            st = geometry.member_points(member, points);
            if (!st.ok) return st;
            switch (op) {
            case Op::IsArc:
                accumulators[thread].boolean(index, geometry.arc(points));
                break;
            case Op::IsCap:
                accumulators[thread].boolean(index, geometry.cap(points));
                break;
            case Op::IsBlockingSet:
                accumulators[thread].boolean(index, geometry.blocking(points));
                break;
            case Op::IsHyperoval:
                accumulators[thread].boolean(index, geometry.p == 2 && geometry.d == 3 && member.rows == 4 && geometry.cap(points));
                break;
            case Op::IsOvoid: {
                bool right_size = geometry.p <= (UINT64_MAX - 1) / geometry.p && member.rows == geometry.p * geometry.p + 1;
                accumulators[thread].boolean(index, geometry.d == 4 && right_size && geometry.cap(points));
                break;
            }
            case Op::MaxCollinear:
                accumulators[thread].integer(index, geometry.max_collinear(points));
                break;
            case Op::SpannedRank: {
                gfp::EchelonBasis basis(geometry.p, geometry.d);
                for (uint64_t row = 0; row < member.rows; ++row) basis.add(member.entries.data() + row * geometry.d);
                accumulators[thread].integer(index, basis.rank());
                break;
            }
            case Op::SecantCount:
                accumulators[thread].integer(index, geometry.lines_with(points, 2));
                break;
            case Op::TangentCount:
                accumulators[thread].integer(index, geometry.lines_with(points, 1));
                break;
            case Op::PassantCount:
                accumulators[thread].integer(index, geometry.lines_with(points, 0));
                break;
            }
        }
        return ok();
    });
    for (const auto &status : statuses)
        if (!status.ok) return R::failure(status.error.status, status.error.message);
    return assemble(req, reduction, accumulators, shared);
}

BackendRegistration registration{Backend{
    "projective_sets", "generic",
    [] { return true; },
    [](const Request &req) {
        return req.family->kind == Family::Kind::Subsets && req.family->prime() >= 2 && req.family->prime() < (1ULL << 32);
    },
    run,
    0}};

} // namespace
} // namespace lk::projective_sets
