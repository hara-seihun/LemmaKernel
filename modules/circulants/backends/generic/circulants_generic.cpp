/* circulants generic backend: exact spectra and corrected-Adam multiplier tests. */
#include "../../../../runtime/src/reduce.hpp"

#include <numeric>

namespace lk::circulants {
namespace {

using R = Result<std::shared_ptr<Object>>;
constexpr int INVALID = 1;

bool valid_mode(uint64_t directed) { return directed <= 1; }

bool adam_order(uint64_t n) {
    if (n == 0 || n % 8 == 0) return false;
    for (uint64_t d = 3; d <= n / d; d += 2)
        if (n % (d * d) == 0) return false;
    return true;
}

bool ci_order(uint64_t n, uint64_t directed) {
    return adam_order(n) || (directed == 0 && (n == 8 || n == 9 || n == 18));
}

Status connection(std::vector<uint64_t> &out, const Matrix &member, uint64_t n) {
    out.assign(member.entries.begin(), member.entries.end());
    std::sort(out.begin(), out.end());
    if (n == 0 || n >= (1ULL << 32)) return fail(INVALID, "n must satisfy 1 <= n < 2^32");
    if (std::find(out.begin(), out.end(), 0) != out.end())
        return fail(INVALID, "connection sets must be identity-free");
    if (!out.empty() && out.back() >= n)
        return fail(INVALID, "connection set contains a residue outside Z_n");
    if (std::adjacent_find(out.begin(), out.end()) != out.end())
        return fail(INVALID, "connection set contains a repeated residue");
    return ok();
}

std::vector<uint64_t> effective_set(uint64_t n, uint64_t directed, const std::vector<uint64_t> &raw) {
    if (directed) return raw;
    std::vector<uint64_t> out = raw;
    out.reserve(raw.size() * 2);
    for (uint64_t x : raw) out.push_back((n - x) % n);
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    if (!out.empty() && out.front() == 0) out.erase(out.begin());
    return out;
}

std::vector<uint64_t> units(uint64_t n) {
    std::vector<uint64_t> out;
    for (uint64_t a = 0; a < n; ++a)
        if (std::gcd(a, n) == 1) out.push_back(a);
    return out;
}

std::vector<uint64_t> multiplier_image(uint64_t n, uint64_t a, const std::vector<uint64_t> &set) {
    std::vector<uint64_t> out;
    out.reserve(set.size());
    for (uint64_t x : set) out.push_back((uint64_t)((unsigned __int128)a * x % n));
    std::sort(out.begin(), out.end());
    return out;
}

bool equivalent(uint64_t n, const std::vector<uint64_t> &left, const std::vector<uint64_t> &right,
                const std::vector<uint64_t> &us) {
    if (left.size() != right.size()) return false;
    for (uint64_t a : us)
        if (multiplier_image(n, a, left) == right) return true;
    return false;
}

bool canonical(uint64_t n, const std::vector<uint64_t> &set, const std::vector<uint64_t> &us) {
    for (uint64_t a : us)
        if (multiplier_image(n, a, set) < set) return false;
    return true;
}

Result<uint64_t> request_mode(const Request &req) {
    auto it = req.int_args.find("directed");
    if (it == req.int_args.end() || !valid_mode(it->second))
        return Result<uint64_t>::failure(INVALID, "directed must be 0 or 1");
    return Result<uint64_t>::success(it->second);
}

Result<uint64_t> request_order(const Request &req) {
    auto it = req.int_args.find("n");
    if (it == req.int_args.end() || it->second == 0 || it->second >= (1ULL << 32))
        return Result<uint64_t>::failure(INVALID, "n must satisfy 1 <= n < 2^32");
    return Result<uint64_t>::success(it->second);
}

Status validate_family(const Family &family, uint64_t n) {
    if (family.prime() != NATURALS)
        return fail(INVALID, "connection sets must contain natural-number residues");
    auto size = family.size();
    if (!size.ok) return fail(size.error.status, size.error.message);
    Matrix member;
    std::vector<uint64_t> set;
    for (uint64_t i = 0; i < size.value; ++i) {
        auto status = family.member_into(i, member);
        if (!status.ok) return status;
        status = connection(set, member, n);
        if (!status.ok) return status;
    }
    return ok();
}

template <class Fn>
R reduce_connections(const Request &req, Fn fn) {
    auto size = req.family->size();
    if (!size.ok) return R::failure(size.error.status, size.error.message);
    Reduction reduction = parse_reduction(req.reduction);
    Shared shared;
    auto status = prepare_all(reduction, size.value, shared);
    if (!status.ok) return R::failure(status.error.status, status.error.message);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, size.value ? size.value : 1));
    std::vector<Accumulator> accumulators;
    for (uint32_t t = 0; t < threads; ++t) accumulators.emplace_back(reduction, &shared);
    auto statuses = parallel_ranges(size.value, threads, [&](uint32_t thread, uint64_t begin, uint64_t end) -> Status {
        Matrix member;
        std::vector<uint64_t> raw;
        for (uint64_t i = begin; i < end; ++i) {
            if (accumulators[thread].exhausted(i)) break;
            auto st = req.family->member_into(i, member);
            if (!st.ok) return st;
            st = connection(raw, member, fn.n);
            if (!st.ok) return st;
            accumulators[thread].boolean(i, fn(raw));
        }
        return ok();
    });
    for (const auto &st : statuses)
        if (!st.ok) return R::failure(st.error.status, st.error.message);
    return assemble(req, reduction, accumulators, shared);
}

struct Isomorphic {
    uint64_t n, directed;
    std::vector<uint64_t> target, us;
    bool operator()(const std::vector<uint64_t> &raw) const {
        return equivalent(n, effective_set(n, directed, raw), target, us);
    }
};

struct IsCanonical {
    uint64_t n, directed;
    std::vector<uint64_t> us;
    bool operator()(const std::vector<uint64_t> &raw) const {
        auto effective = effective_set(n, directed, raw);
        return (directed || raw == effective) && canonical(n, effective, us);
    }
};

R run_isomorphic(const Request &req, uint64_t n, uint64_t directed) {
    if (!ci_order(n, directed))
        return R::failure(INVALID, "Adam's multiplier criterion is not complete for this order and graph type");
    auto it = req.handle_args.find("target");
    if (it == req.handle_args.end() || !it->second->matrix) return R::failure(INVALID, "target must be one natural-number vector");
    const Matrix &target_obj = *it->second->matrix;
    if (target_obj.p != NATURALS || target_obj.count != 1 || target_obj.rows != 1)
        return R::failure(INVALID, "target must be one natural-number vector");
    std::vector<uint64_t> target;
    auto status = connection(target, target_obj, n);
    if (!status.ok) return R::failure(status.error.status, "target " + status.error.message);
    status = validate_family(*req.family, n);
    if (!status.ok) return R::failure(status.error.status, status.error.message);
    return reduce_connections(req, Isomorphic{n, directed, effective_set(n, directed, target), units(n)});
}

R run_is_canonical(const Request &req, uint64_t n, uint64_t directed) {
    if (!ci_order(n, directed))
        return R::failure(INVALID, "Adam's multiplier criterion is not complete for this order and graph type");
    auto status = validate_family(*req.family, n);
    if (!status.ok) return R::failure(status.error.status, status.error.message);
    return reduce_connections(req, IsCanonical{n, directed, units(n)});
}

R run_spectrum(const Request &req, uint64_t n, uint64_t directed) {
    if (req.reduction != "all") return R::failure(INVALID, "spectrum values only reduce with `all`");
    if (req.family->prime() != NATURALS)
        return R::failure(INVALID, "connection sets must contain natural-number residues");
    auto size = req.family->size();
    if (!size.ok) return R::failure(size.error.status, size.error.message);
    if (size.value > SIZE_MAX / sizeof(uint64_t) - 1)
        return R::failure(3, "spectrum output is too large");

    std::vector<uint64_t> set_offsets((size_t)size.value + 1);
    std::vector<Entry> sets;
    Matrix member;
    std::vector<uint64_t> raw;
    for (uint64_t i = 0; i < size.value; ++i) {
        auto status = req.family->member_into(i, member);
        if (!status.ok) return R::failure(status.error.status, status.error.message);
        status = connection(raw, member, n);
        if (!status.ok) return R::failure(status.error.status, status.error.message);
        auto set = effective_set(n, directed, raw);
        if (set.size() > SIZE_MAX - sets.size()) return R::failure(3, "spectrum output is too large");
        set_offsets[i] = sets.size();
        for (uint64_t x : set) sets.push_back((Entry)x);
    }
    set_offsets[size.value] = sets.size();

    unsigned __int128 rows = (unsigned __int128)size.value * n;
    unsigned __int128 terms = (unsigned __int128)sets.size() * n;
    unsigned __int128 offset_bytes = (rows + 1) * sizeof(uint64_t);
    unsigned __int128 exponent_bytes = terms * sizeof(Entry);
    if (rows + 1 > SIZE_MAX / sizeof(uint64_t) || terms > SIZE_MAX / sizeof(Entry) ||
        offset_bytes + exponent_bytes > SIZE_MAX)
        return R::failure(3, "spectrum output is too large");

    auto spectra = std::make_shared<Spectra>();
    spectra->n = n;
    spectra->count = size.value;
    spectra->offsets.resize((size_t)rows + 1);
    uint64_t row = 0, position = 0;
    for (uint64_t i = 0; i < size.value; ++i) {
        uint64_t width = set_offsets[i + 1] - set_offsets[i];
        for (uint64_t j = 0; j < n; ++j) {
            spectra->offsets[row++] = position;
            position += width;
        }
    }
    spectra->offsets[row] = position;
    spectra->exponents.resize((size_t)terms);

    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, size.value ? size.value : 1));
    auto statuses = parallel_ranges(size.value, threads, [&](uint32_t, uint64_t begin, uint64_t end) -> Status {
        for (uint64_t i = begin; i < end; ++i) {
            uint64_t set_begin = set_offsets[i], set_end = set_offsets[i + 1];
            for (uint64_t j = 0; j < n; ++j) {
                Entry *row_begin = spectra->exponents.data() + spectra->offsets[i * n + j];
                Entry *out = row_begin;
                for (uint64_t k = set_begin; k < set_end; ++k)
                    *out++ = (Entry)((unsigned __int128)j * sets[k] % n);
                std::sort(row_begin, out);
            }
        }
        return ok();
    });
    for (const auto &status : statuses)
        if (!status.ok) return R::failure(status.error.status, status.error.message);
    auto out = std::make_shared<Object>();
    out->kind = "circulants.spectra";
    out->spectra = spectra;
    return R::success(out);
}

R run_is_ci(const Request &req, uint64_t directed) {
    auto size = req.family->size();
    if (!size.ok) return R::failure(size.error.status, size.error.message);
    for (uint64_t i = 0; i < size.value; ++i) {
        auto member = req.family->member(i);
        if (!member.ok) return R::failure(member.error.status, member.error.message);
        if (member.value.entries.empty() || member.value.entries[0] == 0)
            return R::failure(INVALID, "is_ci needs positive orders");
    }
    Reduction reduction = parse_reduction(req.reduction);
    Shared shared;
    auto status = prepare_all(reduction, size.value, shared);
    if (!status.ok) return R::failure(status.error.status, status.error.message);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, size.value ? size.value : 1));
    std::vector<Accumulator> accumulators;
    for (uint32_t t = 0; t < threads; ++t) accumulators.emplace_back(reduction, &shared);
    auto statuses = parallel_ranges(size.value, threads, [&](uint32_t thread, uint64_t begin, uint64_t end) -> Status {
        for (uint64_t i = begin; i < end; ++i) {
            if (accumulators[thread].exhausted(i)) break;
            auto member = req.family->member(i);
            if (!member.ok) return fail(member.error.status, member.error.message);
            accumulators[thread].boolean(i, ci_order(member.value.entries[0], directed));
        }
        return ok();
    });
    for (const auto &st : statuses)
        if (!st.ok) return R::failure(st.error.status, st.error.message);
    return assemble(req, reduction, accumulators, shared);
}

R run(const Request &req) {
    auto mode = request_mode(req);
    if (!mode.ok) return R::failure(mode.error.status, mode.error.message);
    if (req.op == "is_ci") return run_is_ci(req, mode.value);
    auto order = request_order(req);
    if (!order.ok) return R::failure(order.error.status, order.error.message);
    if (req.op == "spectrum") return run_spectrum(req, order.value, mode.value);
    if (req.op == "isomorphic") return run_isomorphic(req, order.value, mode.value);
    if (req.op == "is_canonical") return run_is_canonical(req, order.value, mode.value);
    return R::failure(4, "unknown circulants operation " + req.op);
}

BackendRegistration registration{Backend{
    "circulants", "generic",
    [] { return true; },
    [](const Request &) { return true; },
    run,
    0}};

} // namespace
} // namespace lk::circulants
