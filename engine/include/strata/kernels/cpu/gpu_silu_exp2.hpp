// Only the existing aligned GU paths call this helper. Default is unchanged.
// exp2f tracks HIP __expf more closely, but is NOT claimed bit-exact to it.
#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
namespace strata::kernels::cpu::gpu_silu_exp2_detail {
inline bool enabled() {
    static const bool value = [] {
        const char* p = std::getenv("STRATA_CPU_GPU_SILU_EXP2");
        if (p == nullptr || p[0] == '\0' || (p[0] == '0' && p[1] == '\0')) return false;
        if (p[0] != '1' || p[1] != '\0')
            throw std::runtime_error("STRATA_CPU_GPU_SILU_EXP2 expects 0 or 1");
        return true;
    }();
    return value;
}
inline float apply(bool use_exp2, float gate, float up) {
    if (use_exp2) return (gate / (1.f + std::exp2(-gate * 0x1.715476p+0f))) * up;
    return (gate / (1.f + std::exp(-gate))) * up;
}
inline float apply(float gate, float up) { return apply(enabled(), gate, up); }
}
