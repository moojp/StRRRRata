#pragma once

#include <cstdlib>

namespace strata::kernels {

inline bool iq4nl_hidden_residual_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("STRATA_IQ4NL_H_RESIDUAL");
        return value && value[0] == '1' && value[1] == 0;
    }();
    return enabled;
}

} // namespace strata::kernels
