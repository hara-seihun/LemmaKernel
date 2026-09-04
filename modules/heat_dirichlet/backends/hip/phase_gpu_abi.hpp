/* The C ABI between the heat_dirichlet hip backend (in liblemmakernel) and the device library
 * liblemmakernel_hip.so built by hipcc: the phase-aware bound's per-request data, packed.
 *
 * Every value is at scale 2^48 as in heat_dirichlet_common.hpp; the sums a box computes fit
 * 64-bit words (a component's coefficient mass is below 16, so centre values are below 2^52 and
 * gradients below 2^57) while products are formed in 128 bits, exactly as on the CPU, so the
 * device returns the same integers. */
#pragma once
#include <cstddef>
#include <cstdint>

namespace lk::heat_dirichlet::gpu {

struct Term {
    int64_t cS_lo, cS_hi, cA_lo, cA_hi;
    int32_t X;              /* half-angle over a box in table steps */
    uint8_t v[4];           /* exponent vector */
    uint8_t taylorS, taylorA; /* Taylor enclosure (else the arc's rectangle) for each part */
    uint8_t pad[2];
};
struct Component {
    int64_t W, Q, mass;
    uint32_t begin, count;  /* its terms */
};
struct Circle {
    int64_t c_lo, c_hi, s_lo, s_hi;
};
struct Params {
    uint64_t M;
    uint32_t npsi;
    uint32_t g[4];
    int64_t h[5];
    uint32_t ncomp;          /* the head first */
    uint32_t nterms;
    uint64_t ncircle;        /* = M */
    uint64_t invM;           /* floor(2^64 / M), for the reduction of indices modulo M */
    int64_t loss;            /* at scale 2^48 */
    int64_t offset;          /* the plain integer offset */
    uint32_t shift;          /* 48 - scale */
};

/* Values are written to out[i]; UINT64_MAX marks a value that does not fit 64 bits. Returns 0
 * on success, otherwise a nonzero code with a message in err. */
using PhaseFn = int (*)(const Params *P, const Component *comps, const Term *terms, const Circle *circle,
                        const uint32_t *boxes, uint64_t nboxes, uint64_t *out, char *err, size_t errlen);
/* Nonzero when a device is usable. */
using AvailableFn = int (*)();

} // namespace lk::heat_dirichlet::gpu
