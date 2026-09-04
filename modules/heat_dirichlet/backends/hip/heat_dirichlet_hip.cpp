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
#include <memory>
#include <mutex>

namespace lk::heat_dirichlet {
namespace {

using namespace lk::heat_dirichlet::detail;

struct Device {
    void *handle = nullptr;
    gpu::PrepareFn prepare = nullptr;
    gpu::RunFn run = nullptr;
    gpu::ReleaseFn release = nullptr;
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
        dev.prepare = (gpu::PrepareFn)dlsym(dev.handle, "lk_heat_dirichlet_phase_prepare");
        dev.run = (gpu::RunFn)dlsym(dev.handle, "lk_heat_dirichlet_phase_run");
        dev.release = (gpu::ReleaseFn)dlsym(dev.handle, "lk_heat_dirichlet_phase_release");
        dev.available = (gpu::AvailableFn)dlsym(dev.handle, "lk_heat_dirichlet_phase_available");
        if (!dev.prepare || !dev.run || !dev.release || !dev.available) {
            dev.error = path + " lacks the phase_bound entry points";
            dev.prepare = nullptr;
            dev.run = nullptr;
            dev.release = nullptr;
            dev.available = nullptr;
        }
    });
    return dev;
}

bool available() {
    const Device &dev = device();
    return dev.available && dev.available();
}

/* The arguments of a phase_bound request except the family, as bytes: `grid` selects the grid
 * arguments (g2, g3, g5, g7, npsi), which only the second stage of the setup reads, or the rest. */
std::string request_key(const Request &req, bool grid) {
    static const std::set<std::string> grid_args{"g2", "g3", "g5", "g7", "npsi"};
    std::string key = req.op;
    for (const auto &[name, v] : req.int_args) {
        if (grid_args.count(name) != grid) continue;
        key += '|' + name + '=';
        key.append((const char *)&v, sizeof v);
    }
    if (grid) return key;
    for (const auto &[name, h] : req.handle_args) {
        key += '|' + name + ':';
        if (!h->matrix) return std::string(); /* not a matrix argument: never cached */
        const Matrix &m = *h->matrix;
        uint64_t dims[4] = {m.p, m.count, m.rows, m.cols};
        key.append((const char *)dims, sizeof dims);
        key.append((const char *)m.entries.data(), m.entries.size() * sizeof(Entry));
    }
    return key;
}

/* The first stage of a request's setup (everything but the grid). */
struct Coefficients {
    std::string key;
    PhaseParams PP;
};
/* A request ready to run: the regridded setup and its tables in device memory. */
struct Prepared {
    std::string key, grid_key;
    PhaseParams PP;
    void *handle = nullptr;
    ~Prepared() {
        if (handle) device().release(handle);
    }
};

std::mutex cache_mutex;
/* Deliberately leaked: a static destructor would free device memory after the HIP runtime's own
 * teardown at exit, which crashes; the process's exit reclaims the device memory anyway. */
std::shared_ptr<Coefficients> &coefficients_cache = *new std::shared_ptr<Coefficients>();
std::shared_ptr<Prepared> &cache = *new std::shared_ptr<Prepared>();

/* The request's setup and device tables: from the cache when the arguments repeat, from the
 * cached first stage when only the grid changed, otherwise computed. */
Result<std::shared_ptr<Prepared>> prepare(const Request &req) {
    using RP = Result<std::shared_ptr<Prepared>>;
    std::string key = request_key(req, false), grid_key = request_key(req, true);
    std::shared_ptr<Coefficients> coeff;
    if (!key.empty()) {
        std::lock_guard<std::mutex> lock(cache_mutex);
        if (cache && cache->key == key && cache->grid_key == grid_key) return RP::success(cache);
        if (coefficients_cache && coefficients_cache->key == key) coeff = coefficients_cache;
    }
    /* the request is parsed and validated in full either way */
    PhaseParams parsed;
    Status st = parsed.init_phase(req);
    if (!st.ok) return RP::failure(st.error.status, st.error.message);
    if (!coeff) {
        coeff = std::make_shared<Coefficients>();
        coeff->key = key;
        coeff->PP = parsed;
        st = coeff->PP.precompute_phase(std::max<uint32_t>(1, req.threads));
        if (!st.ok) return RP::failure(st.error.status, st.error.message);
        if (!key.empty()) {
            std::lock_guard<std::mutex> lock(cache_mutex);
            coefficients_cache = coeff;
        }
    }
    auto prep = std::make_shared<Prepared>();
    prep->key = key;
    prep->grid_key = grid_key;
    PhaseParams &PP = prep->PP;
    PP = coeff->PP;
    for (int p = 0; p < 4; ++p) PP.g[p] = parsed.g[p];
    PP.npsi = parsed.npsi;
    st = PP.regrid();
    if (!st.ok) return RP::failure(st.error.status, st.error.message);

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
    const Device &dev = device();
    if (!dev.prepare) return RP::failure(INTERNAL, dev.error);
    char err[512] = {0};
    if (dev.prepare(&P, comps.data(), terms.data(), circle.data(), &prep->handle, err, sizeof err))
        return RP::failure(INTERNAL, std::string("hip backend: ") + err);
    /* the setup's tables are on the device now; the host copies are not needed for the boxes */
    PP.circle.clear();
    PP.circle.shrink_to_fit();
    if (!key.empty()) {
        std::lock_guard<std::mutex> lock(cache_mutex);
        cache = prep;
    }
    return RP::success(prep);
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
    auto prepared = prepare(req);
    if (!prepared.ok) return R::failure(prepared.error.status, prepared.error.message);
    const PhaseParams &PP = prepared.value->PP;
    auto size_r = req.family->size();
    if (!size_r.ok) return R::failure(size_r.error.status, size_r.error.message);
    uint64_t size = size_r.value;
    std::vector<uint32_t> boxes(size * 4);
    for (uint64_t i = 0; i < size; ++i) {
        auto js = PP.box_of(*req.family, i);
        if (!js) return R::failure(INVALID, "box index beyond the grid at member " + std::to_string(i));
        for (int p = 0; p < 4; ++p) boxes[4 * i + p] = (uint32_t)(*js)[p];
    }
    std::vector<uint64_t> values(size);
    char err[512] = {0};
    if (size && device().run(prepared.value->handle, boxes.data(), size, values.data(), err, sizeof err))
        return R::failure(INTERNAL, std::string("hip backend: ") + err);

    Reduction reduction = parse_reduction(req.reduction);
    Shared shared;
    auto ready = prepare_all(reduction, size, shared);
    if (!ready.ok) return R::failure(ready.error.status, ready.error.message);
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
