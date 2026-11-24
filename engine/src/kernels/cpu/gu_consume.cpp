// Default math untouched. Only the existing opted-in exact GU dispatch calls
// this AVX2 TU. Keep the qualified GU TU for NT1 and NT4..8 and all fallbacks.
#include "strata/kernels/cpu/q81_gu_align.hpp"
#include "strata/kernels/cpu/gpu_silu_exp2.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#include <immintrin.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <stdexcept>
namespace {
#include "q81_gu_private/exact/guard_memo.hpp"
namespace consumeimpl {
namespace C=strata::kernels::cpu;
namespace q81math {
inline void require(bool v,const char* why){if(!v)throw std::runtime_error(why);}
inline float h2f(uint16_t h){return _cvtsh_ss(h);}
inline uint16_t f2h(float v){return _cvtss_sh(v,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);}
inline float reduce8(__m256 x){
    x=_mm256_add_ps(x,_mm256_permute2f128_ps(x,x,1));
    x=_mm256_add_ps(x,_mm256_permute_ps(x,0x4e));
    x=_mm256_add_ps(x,_mm256_permute_ps(x,0xb1));return _mm_cvtss_f32(_mm256_castps256_ps128(x));
}
}
#include "q81_gu_private/stock_style.hpp"
namespace A=q81math;
struct InputView {std::span<const int8_t,2560> code;std::span<const float,80> scales;uint64_t generation;};
static_assert(sizeof(InputView)==3*sizeof(void*));
static_assert(offsetof(stockstyle::Input,scales)==2560);
#include "q81_gu_private/exact/gu_consume.hpp"
template<int NT> void batch(const C::NativeFmt& f,const uint8_t* blob,const void* const* raw,float* const* out,int r0,int r1){
    auto view=[&](int t){
        const auto address=(reinterpret_cast<uintptr_t>(raw[t])+31)&~uintptr_t(31);
        uint64_t generation;std::memcpy(&generation,(const uint8_t*)raw[t]+f.act_bytes-8,8);
        A::require(generation!=0,"exact GU consume needs quantized generation");
        return InputView{std::span<const int8_t,2560>(reinterpret_cast<const int8_t*>(address),2560),
            std::span<const float,80>(reinterpret_cast<const float*>(address+2560),80),generation};
    };
    auto make=[&]<size_t... I>(std::index_sequence<I...>){return std::array<InputView,NT>{view(I)...};};
    const auto x=make(std::make_index_sequence<NT>{});
    switch(f.gu_type){
        case 18:return exact8::rows<18,NT,true>(f,blob,x.data(),out,r0,r1);
        case 21:return exact8::rows<21,NT,true>(f,blob,x.data(),out,r0,r1);
        case 22:return exact8::rows<22,NT,true>(f,blob,x.data(),out,r0,r1);
        case 23:return exact8::rows<23,NT,true>(f,blob,x.data(),out,r0,r1);
        default:throw std::runtime_error("unreachable GU consume format");
    }
}
}
}
namespace strata::kernels::cpu {
bool q81_gu_consume_rows(const NativeFmt& f,const uint8_t* blob,const void* const* act,int nt,float* const* out,int r0,int r1){
    if((nt!=2&&nt!=3)||f.n_embd!=2560||f.n_ff!=640||f.had_gu||f.had_d||f.act_bytes<2919||
       (f.gu_type!=18&&f.gu_type!=21&&f.gu_type!=22&&f.gu_type!=23))return false;
    consumeimpl::A::require(r0>=0&&r1>=r0&&r1<=640,"exact GU consume row range");
    if(nt==2)consumeimpl::batch<2>(f,blob,act,out,r0,r1);
    else consumeimpl::batch<3>(f,blob,act,out,r0,r1);
    return true;
}
}
