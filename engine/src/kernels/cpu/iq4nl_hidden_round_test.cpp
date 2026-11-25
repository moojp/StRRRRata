#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

int main() {
    namespace cpu = strata::kernels::cpu;
    if (!cpu::cpu_avx2_ok()) return 77;
    cpu::NativeFmt fmt;
    std::string error;
    if (!cpu::native_fmt(18, 20, 2560, 640, fmt, error)) return 1;
    std::array<float, 640> hidden{};
    std::array<uint8_t, 680> packed{};
    const float inf = std::numeric_limits<float>::infinity();
    for (int n = 0; n < 127; ++n) {
        for (int b = 0; b < 20; ++b) {
            const float scale = std::ldexp(1.f, b % 9 - 4);
            float* block = hidden.data() + 32 * b;
            block[0] = 127.f * scale;
            for (int j = 1; j < 32; ++j) {
                const float tie = (j % 2 ? 1.f : -1.f) * (n + 0.5f) * scale;
                block[j] = j % 3 == 0 ? tie : std::nextafter(tie, j % 3 == 1 ? inf : -inf);
            }
        }
        cpu::native_quant_h(fmt, hidden.data(), packed.data());
        for (int b = 0; b < 20; ++b) {
            uint16_t scale_bits;
            std::memcpy(&scale_bits, packed.data() + 34 * b, sizeof(scale_bits));
            if (scale_bits != (15 + b % 9 - 4) * 1024) return 1;
            const float d = hidden[32 * b] / 127.f;
            for (int j = 0; j < 32; ++j) {
                const int expected = static_cast<int>(std::round(hidden[32 * b + j] / d));
                if (static_cast<int8_t>(packed[34 * b + 2 + j]) != expected) {
                    std::fprintf(stderr, "IQ4_NL hidden code mismatch at tie %d, block %d, lane %d\n", n, b, j);
                    return 1;
                }
            }
        }
    }
    return 0;
}
