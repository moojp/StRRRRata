// IQ4_NL consumes only the Q8_1 scale and signed codes. Emit them in the CPU Q8_0 layout.
#include "strata/kernels/cpu/q81_gu_align.hpp"

#include <immintrin.h>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace strata::kernels::cpu {
namespace {
constexpr int block_size = 32;
constexpr int block_bytes = sizeof(uint16_t) + block_size;
constexpr float max_scale = 65504.f;

__m256 magnitude(__m256 x) {
    return _mm256_andnot_ps(_mm256_set1_ps(-0.f), x);
}

__m128 column_maxima(const float* x) {
    const __m256 a = _mm256_max_ps(magnitude(_mm256_loadu_ps(x)), magnitude(_mm256_loadu_ps(x + 8)));
    const __m256 b = _mm256_max_ps(magnitude(_mm256_loadu_ps(x + 16)), magnitude(_mm256_loadu_ps(x + 24)));
    const __m256 m = _mm256_max_ps(a, b);
    return _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
}

__m128 block_maxima(const float* x) {
    __m128 a = column_maxima(x), b = column_maxima(x + 32);
    __m128 c = column_maxima(x + 64), d = column_maxima(x + 96);
    _MM_TRANSPOSE4_PS(a, b, c, d);
    return _mm_max_ps(_mm_max_ps(a, b), _mm_max_ps(c, d));
}

__m256i exact_codes(__m256 x, __m256 scale) {
    __m256 q = _mm256_div_ps(x, scale);
    q = _mm256_min_ps(_mm256_max_ps(q, _mm256_set1_ps(-127.f)), _mm256_set1_ps(127.f));
    // Bias by the predecessor of 0.5 then truncate. With round-to-nearest,
    // exact half ties go away from zero and the preceding floats stay below.
    const __m256 signed_bias = _mm256_or_ps(_mm256_and_ps(q, _mm256_set1_ps(-0.f)),
                                            _mm256_set1_ps(0x1.fffffep-2f));
    return _mm256_cvttps_epi32(_mm256_add_ps(q, signed_bias));
}

void write_codes(const float* x, uint8_t* out, float scale, float inverse, bool use_grid) {
    __m256i q[4];
    if (use_grid) {
        __m256i ties = _mm256_setzero_si256();
        for (int i = 0; i < 4; ++i) {
            // The offset maps [-127,127] into one FP32 binade with spacing 2^-13.
            // A grid cell centered on a half-integer covers +/-2^-14, including
            // the FP32 rounding error from replacing GPU division with multiplication.
            const __m256i bits = _mm256_castps_si256(_mm256_fmadd_ps(_mm256_loadu_ps(x + 8 * i),
                                              _mm256_set1_ps(inverse), _mm256_set1_ps(1536.f)));
            const __m256i fraction = _mm256_and_si256(bits, _mm256_set1_epi32(8191));
            ties = _mm256_or_si256(ties, _mm256_cmpeq_epi32(fraction, _mm256_set1_epi32(4096)));
            // 0x44c00000 encodes the offset 1536.f; 4096 is half a grid unit of 8192.
            q[i] = _mm256_srai_epi32(_mm256_add_epi32(bits, _mm256_set1_epi32(4096 - 0x44c00000)), 13);
        }
        use_grid = _mm256_testz_si256(ties, ties);
    }
    if (!use_grid) {
        for (int i = 0; i < 4; ++i)
            q[i] = exact_codes(_mm256_loadu_ps(x + 8 * i), _mm256_set1_ps(scale));
    }
    const __m256i lo = _mm256_packs_epi32(q[0], q[1]), hi = _mm256_packs_epi32(q[2], q[3]);
    const __m256i packed = _mm256_permutevar8x32_epi32(_mm256_packs_epi16(lo, hi),
                                                       _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out), packed);
}
}  // namespace

bool iq4nl_gpu_hidden_quant(const float* x, void* output) {
    if (_MM_GET_ROUNDING_MODE() != _MM_ROUND_NEAREST) return false;
    auto* dst = static_cast<uint8_t*>(output);
    for (int group = 0; group < 5; ++group, x += 4 * block_size, dst += 4 * block_bytes) {
        const __m128 maxima = block_maxima(x);
        const __m128 scales = _mm_min_ps(_mm_div_ps(maxima, _mm_set1_ps(127.f)), _mm_set1_ps(max_scale));
        alignas(16) float am[4], scale[4], inverse[4];
        _mm_store_ps(am, maxima);
        _mm_store_ps(scale, scales);
        _mm_store_ps(inverse, _mm_div_ps(_mm_set1_ps(127.f), _mm_max_ps(maxima, _mm_set1_ps(0x1p-100f))));
        uint16_t half[4];
        _mm_storel_epi64(reinterpret_cast<__m128i*>(half),
                         _mm_cvtps_ph(scales, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        for (int j = 0; j < 4; ++j) {
            uint8_t* block = dst + block_bytes * j;
            std::memcpy(block, half + j, sizeof(uint16_t));
            if (am[j] == 0.f) {
                std::memset(block + sizeof(uint16_t), 0, block_size);
                continue;
            }
            if (scale[j] == 0.f || !std::isfinite(am[j])) return false;
            const bool use_grid = am[j] >= 0x1p-100f && am[j] <= max_scale * 127.f;
            write_codes(x + block_size * j, block + sizeof(uint16_t), scale[j], inverse[j], use_grid);
        }
    }
    return true;
}
}  // namespace strata::kernels::cpu
