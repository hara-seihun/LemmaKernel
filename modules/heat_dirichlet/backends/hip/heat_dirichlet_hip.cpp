/* heat_dirichlet hip backend: phase_bound with the boxes on the GPU.
 *
 * The request's setup (parameters, smooth numbers, coefficient enclosures, the cosine table) is
 * the shared CPU code of heat_dirichlet_common.hpp; the boxes go to liblemmakernel_hip.so
 * (phase_kernel.hip, built by hipcc), which returns the same integers as the CPU backend. That
 * library is loaded with dlopen from the directory of liblemmakernel.so, or from
 * LEMMAKERNEL_HIP_LIB; without it, or without a device, this backend is unavailable and the
 * generic backend serves every request. */
#include "../heat_dirichlet_common.hpp"
#include "phase_gpu_abi.hpp"

#include <dlfcn.h>

#include <cstdlib>
#include <mutex>

namespace lk::heat_dirichlet {
namespace {

using namespace lk::heat_dirichlet::detail;

struct Device {
    void *handle = nullptr;
    gpu::PhaseFn phase = nullptr;
    gpu::AvailableFn available = nullptr;
    std::string error;
};

std::string library_path() {
    if (const char *env = getenv("LEMMAKERNEL_HIP_LIB")) return env;
    Dl_info info;
    if (dladdr((void *)&library_path, &info) && info.dli_fname) {
        std::string path = info.dli_fname;
        auto slash = path.rfind('/');
        return (slash == std::string::npos ? std::string() : path.substr(0, slash + 1)) + "liblemmakernel_hip.so";
    }
    return "liblemmakernel_hip.so";
}

const Device &device() {
    static Device dev;
    static std::once_flag once;
    std::call_once(once, [] {
        std::string path = library_path();
        dev.handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!dev.handle) {
            dev.error = std::string("cannot load ") + path + ": " + (dlerror() ? dlerror() : "");
            return;
        }
        dev.phase = (gpu::PhaseFn)dlsym(dev.handle, "lk_heat_dirichlet_phase_gpu");
        dev.available = (gpu::AvailableFn)dlsym(dev.handle, "lk_heat_dirichlet_phase_available");
        if (!dev.phase || !dev.available) {
            dev.error = path + " lacks the phase_bound entry points";
            dev.phase = nullptr;
            dev.available = nullptr;
        }
    });
    return dev;
}

bool available() {
    const Device &dev = device();
    return dev.available && dev.available();
}

bool accepts(const Request &req) {
    if (req.op != "phase_bound") return false;
    /* the psi accumulators live in shared memory: 64 threads x npsi x 8 bytes up to 64 samples,
     * 32 threads beyond, to 128 */
    auto it = req.int_args.find("npsi");
    if (it != req.int_args.end() && it->second > 128) return false;
    return true;
}

R run(const Request &req) {
    auto valid = [&]() -> Status {
        if (req.family->prime() != NATURALS)
            return fail(INVALID, "heat_dirichlet operations need members that are natural numbers (lk.naturals), not elements of a field");
        if (req.family->rows() != 1 || (req.family->cols() != 1 && req.family->cols() != 4))
            return fail(INVALID, "phase_bound needs 1 x 1 or 1 x 4 members (a box index or its four grid indices)");
        return ok();
    }();
    if (!valid.ok) return R::failure(valid.error.status, valid.error.message);
    PhaseParams PP;
    Status st = PP.init_phase(req);
    if (!st.ok) return R::failure(st.error.status, st.error.message);
    st = PP.precompute_phase(std::max<uint32_t>(1, req.threads));
    if (!st.ok) return R::failure(st.error.status, st.error.message);
    auto size_r = req.family->size();
    if (!size_r.ok) return R::failure(size_r.error.status, size_r.error.message);
    uint64_t size = size_r.value;

    /* pack */
    gpu::Params P{};
    P.M = PP.M;
    P.npsi = (uint32_t)PP.npsi;
    for (int p = 0; p < 4; ++p) P.g[p] = (uint32_t)PP.g[p];
    for (int p = 0; p < 5; ++p) P.h[p] = (int64_t)PP.h[p];
    P.ncomp = (uint32_t)PP.components.size();
    P.ncircle = PP.M;
    P.invM = (uint64_t)(((unsigned __int128)1 << 64) / PP.M);
    P.loss = (int64_t)PP.loss;
    P.offset = (int64_t)PP.offset;
    P.shift = (uint32_t)(K - PP.scale);
    std::vector<gpu::Component> comps;
    std::vector<gpu::Term> terms;
    for (const Poly &poly : PP.components) {
        gpu::Component c{(int64_t)poly.W, (int64_t)poly.Q, (int64_t)poly.mass, (uint32_t)terms.size(), (uint32_t)poly.terms.size()};
        comps.push_back(c);
        for (const Term &t : poly.terms) {
            gpu::Term g{};
            g.cS_lo = (int64_t)t.cS.lo;
            g.cS_hi = (int64_t)t.cS.hi;
            g.cA_lo = (int64_t)t.cA.lo;
            g.cA_hi = (int64_t)t.cA.hi;
            g.X = (int32_t)t.X;
            const Smooth &sm = PP.smooth[t.idx];
            for (int p = 0; p < 4; ++p) g.v[p] = (uint8_t)sm.v[p];
            g.taylorS = t.vh <= S;
            g.taylorA = t.vh + PP.h[4] <= S;
            terms.push_back(g);
        }
    }
    P.nterms = (uint32_t)terms.size();
    std::vector<gpu::Circle> circle(PP.M);
    for (uint64_t J = 0; J < PP.M; ++J)
        circle[J] = {(int64_t)PP.circle[J].re.lo, (int64_t)PP.circle[J].re.hi, (int64_t)PP.circle[J].im.lo,
                     (int64_t)PP.circle[J].im.hi};
    std::vector<uint32_t> boxes(size * 4);
    for (uint64_t i = 0; i < size; ++i) {
        auto js = PP.box_of(*req.family, i);
        if (!js) return R::failure(INVALID, "box index beyond the grid at member " + std::to_string(i));
        for (int p = 0; p < 4; ++p) boxes[4 * i + p] = (uint32_t)(*js)[p];
    }

    std::vector<uint64_t> values(size);
    const Device &dev = device();
    if (!dev.phase) return R::failure(INTERNAL, dev.error);
    char err[512] = {0};
    if (size) {
        int rc = dev.phase(&P, comps.data(), terms.data(), circle.data(), boxes.data(), size, values.data(), err, sizeof err);
        if (rc) return R::failure(INTERNAL, std::string("hip backend: ") + err);
    }

    Reduction reduction = parse_reduction(req.reduction);
    Shared shared;
    auto prepared = prepare_all(reduction, size, shared);
    if (!prepared.ok) return R::failure(prepared.error.status, prepared.error.message);
    std::vector<Accumulator> accumulators;
    accumulators.emplace_back(reduction, &shared);
    for (uint64_t i = 0; i < size; ++i) {
        if (values[i] == UINT64_MAX) return R::failure(INVALID, "value does not fit in 64 bits at member " + std::to_string(i));
        accumulators[0].integer(i, values[i]);
    }
    return assemble(req, reduction, accumulators, shared);
}

BackendRegistration registration{Backend{
    "heat_dirichlet", "hip",
    available,
    accepts,
    run,
    1}};

} // namespace
} // namespace lk::heat_dirichlet
