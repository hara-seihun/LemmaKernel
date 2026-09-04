/* heat_dirichlet generic backend: the portable CPU runner over the shared setup and arithmetic
 * of heat_dirichlet_common.hpp; members are split across threads. */
#include "../heat_dirichlet_common.hpp"

namespace lk::heat_dirichlet {
namespace {

using namespace lk::heat_dirichlet::detail;

Status validate(const Family &family, bool boxes) {
    if (family.prime() != NATURALS)
        return fail(INVALID, "heat_dirichlet operations need members that are natural numbers (lk.naturals), not elements of a field");
    if (boxes) {
        if (family.rows() != 1 || (family.cols() != 1 && family.cols() != 4))
            return fail(INVALID, "phase_bound needs 1 x 1 or 1 x 4 members (a box index or its four grid indices)");
        return ok();
    }
    if (family.rows() != 1 || family.cols() != 1)
        return fail(INVALID, "heat_dirichlet operations need 1 x 1 members, one natural number each; these are " +
                                 std::to_string(family.rows()) + " x " + std::to_string(family.cols()));
    return ok();
}

inline uint64_t member(const Family &family, uint64_t i) {
    return family.kind == Family::Kind::Range ? family.a + i : (uint64_t)family.data->entries[i];
}

R run(const Request &req) {
    auto parsed = parse_op(req.op);
    if (!parsed.ok) return R::failure(parsed.error.status, parsed.error.message);
    Op op = parsed.value;
    auto valid = validate(*req.family, op == Op::PhaseBound);
    if (!valid.ok) return R::failure(valid.error.status, valid.error.message);
    Params P;
    PhaseParams PP;
    Status st;
    if (op == Op::PhaseBound) {
        st = PP.init_phase(req);
        if (!st.ok) return R::failure(st.error.status, st.error.message);
        st = PP.precompute_phase(std::max<uint32_t>(1, req.threads));
        if (!st.ok) return R::failure(st.error.status, st.error.message);
    } else {
        st = P.init(req);
        if (!st.ok) return R::failure(st.error.status, st.error.message);
    }
    if (op == Op::MollifiedTermUpper || op == Op::BlockTermUpper) {
        st = P.precompute();
        if (!st.ok) return R::failure(st.error.status, st.error.message);
    }
    auto size_r = req.family->size();
    if (!size_r.ok) return R::failure(size_r.error.status, size_r.error.message);
    uint64_t size = size_r.value;
    if (op != Op::BlockTermUpper && op != Op::PhaseBound)
        for (uint64_t i = 0; i < size; ++i)
            if (member(*req.family, i) < (op == Op::SigmaLower ? 2u : 1u))
                return R::failure(INVALID, op == Op::SigmaLower ? "sigma_lower needs a cutoff of at least 2"
                                                                 : std::string(req.op) + " needs members of at least 1");

    Reduction reduction = parse_reduction(req.reduction);
    Shared shared;
    auto prepared = prepare_all(reduction, size, shared);
    if (!prepared.ok) return R::failure(prepared.error.status, prepared.error.message);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, size ? size : 1));
    std::vector<Accumulator> accumulators;
    for (uint32_t t = 0; t < threads; ++t) accumulators.emplace_back(reduction, &shared);

    auto statuses = parallel_ranges(size, threads, [&](uint32_t thread, uint64_t begin, uint64_t end) -> Status {
        Accumulator &acc = accumulators[thread];
        std::vector<Parts> scratch;
        std::vector<I> firsts;
        for (uint64_t i = begin; i < end; ++i) {
            if (op == Op::PhaseBound) {
                auto js = PP.box_of(*req.family, i);
                if (!js) return fail(INVALID, "box index beyond the grid at member " + std::to_string(i));
                auto v = PP.phase_bound(*js, scratch, firsts);
                if (!v) return fail(INVALID, "value does not fit in 64 bits at member " + std::to_string(i));
                acc.integer(i, *v);
                continue;
            }
            uint64_t n = member(*req.family, i);
            std::optional<uint64_t> v;
            switch (op) {
            case Op::WeightUpper: v = P.out(P.g_upper(n)); break;
            case Op::MollifiedTermUpper: v = P.out(P.term_upper(n)); break;
            case Op::BlockTermUpper: v = P.out(P.block_upper(n)); break;
            case Op::SigmaLower: {
                I s = P.sigma_lower(n);
                v = (uint64_t)((s < 0 ? 0 : s) >> (int)(K - P.scale));
                break;
            }
            case Op::PhaseBound: break; /* handled above */
            }
            if (!v) return fail(INVALID, "exponent exceeds 7 or value does not fit in 64 bits at member " + std::to_string(n));
            acc.integer(i, *v);
        }
        return ok();
    });
    for (const auto &status : statuses)
        if (!status.ok) return R::failure(status.error.status, status.error.message);
    return assemble(req, reduction, accumulators, shared);
}

BackendRegistration registration{Backend{
    "heat_dirichlet", "generic",
    [] { return true; },
    [](const Request &) { return true; },
    run,
    0}};

} // namespace
} // namespace lk::heat_dirichlet
