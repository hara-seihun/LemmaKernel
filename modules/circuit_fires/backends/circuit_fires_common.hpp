#pragma once

#include "../../../runtime/src/registry.hpp"

namespace lk::circuit_fires {

using CircuitFireResult = Result<std::shared_ptr<Object>>;

bool configuration_is_circuit(const Entry *configuration, uint64_t rows, uint64_t base_dim,
                              uint64_t fibre_dim, uint32_t prime);
CircuitFireResult run_reduced_polynomial(const Request &request);

} // namespace lk::circuit_fires
