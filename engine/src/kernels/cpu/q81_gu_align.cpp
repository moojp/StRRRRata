// Opt-in exact CPU implementation. GPU, weights, routing and default math stay unchanged.
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
#include <new>
#include <atomic>
#include <type_traits>
namespace {
#include "q81_gu_private/exact/guard_memo.hpp"
namespace viewimpl {
namespace C=strata::kernels::cpu;
namespace q81math {
inline void require(bool b,const char* why){if(!b)throw std::runtime_error(why);}
inline float h2f(uint16_t h){return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h)));}
inline uint16_t f2h(float f){return (uint16_t)_cvtss_sh(f,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);}
inline float reduce8(__m256 x){
    x=_mm256_add_ps(x,_mm256_permute2f128_ps(x,x,1));
    x=_mm256_add_ps(x,_mm256_permute_ps(x,0x4e));
    x=_mm256_add_ps(x,_mm256_permute_ps(x,0xb1));return _mm_cvtss_f32(_mm256_castps256_ps128(x));
}

}
#include "q81_gu_private/stock_style.hpp"
namespace A=q81math;
struct InputView {
    std::span<const int8_t,2560> code;
    std::span<const float,80> scales;
    uint64_t generation;
};
static_assert(sizeof(InputView)==3*sizeof(void*));
#include "q81_gu_private/exact/exact8_view.hpp"

template<int Type,int NT> void ranged(const C::NativeFmt& f,const uint8_t* b,const InputView* x,float* const* out,int r0,int r1){
    if constexpr(NT!=1||Type==21||Type==22){exact8::rows<Type,NT,true>(f,b,x,out,r0,r1);return;}
    const bool use_exp2=r0<r1&&strata::kernels::cpu::gpu_silu_exp2_detail::enabled();
    for(int r=r0;r<r1;++r){
        float g[NT],u[NT];
        exact8::dot<Type,NT,true>(b+r*f.gu_row,x,g);
        exact8::dot<Type,NT,true>(b+f.up_off+r*f.gu_row,x,u);
        for(int t=0;t<NT;++t)out[t][r]=strata::kernels::cpu::gpu_silu_exp2_detail::apply(use_exp2,g[t],u[t]);
    }
}
template<int NT> void batch(const C::NativeFmt& f,const uint8_t* b,const void* const* raw,float* const* out,int r0,int r1){
    auto view=[&](int t){
        const auto address=(reinterpret_cast<uintptr_t>(raw[t])+31)&~uintptr_t(31);
        const auto* q=reinterpret_cast<const stockstyle::Input*>(address);
        uint64_t generation=0;std::memcpy(&generation,(const uint8_t*)raw[t]+f.act_bytes-sizeof(generation),sizeof(generation));
        A::require(generation!=0,"exact8 view needs quantized generation");
        return InputView{q->code,q->scales,generation};
    };
    auto make=[&]<size_t... I>(std::index_sequence<I...>){return std::array<InputView,NT>{view(I)...};};
    const auto x=make(std::make_index_sequence<NT>{});
    switch(f.gu_type){
        case 18:ranged<18,NT>(f,b,x.data(),out,r0,r1);return;
        case 21:ranged<21,NT>(f,b,x.data(),out,r0,r1);return;
        case 22:ranged<22,NT>(f,b,x.data(),out,r0,r1);return;
        case 23:ranged<23,NT>(f,b,x.data(),out,r0,r1);return;
        default:throw std::runtime_error("exact8 view unsupported format");
    }
}
} // namespace viewimpl: avoid type/inline ODR collisions with frozen oracle.
namespace q81math=viewimpl::q81math;
namespace stockstyle=viewimpl::stockstyle;
#include "q81_gu_private/exact/prepare.hpp"
#include "q81_gu_private/exact/down.hpp"
} // namespace: private helpers cannot collide with test or other engine TUs.
namespace strata::kernels::cpu {
void q81_gu_quant(const NativeFmt& f,const float* x,void* dst){
    quant5::wire<3>(f, x, dst);
}
void q81_gu_rows(const NativeFmt& f,const uint8_t* blob,const void* const* act,int nt,float* const* out,int r0,int r1){
    viewimpl::A::require(f.n_embd==2560&&f.n_ff==640&&!f.had_gu&&!f.had_d&&f.act_bytes>=2880+31+8,"exact8 view geometry");
    viewimpl::A::require(nt>=0&&r0>=0&&r1>=r0&&r1<=640,"exact8 view row/token range");
    for(int off=0;off<nt;){int n=std::min(nt-off,8);
        switch(n){
            case 1:viewimpl::batch<1>(f,blob,act+off,out+off,r0,r1);break;
            case 2:viewimpl::batch<2>(f,blob,act+off,out+off,r0,r1);break;
            case 3:viewimpl::batch<3>(f,blob,act+off,out+off,r0,r1);break;
            case 4:viewimpl::batch<4>(f,blob,act+off,out+off,r0,r1);break;
            case 5:viewimpl::batch<5>(f,blob,act+off,out+off,r0,r1);break;
            case 6:viewimpl::batch<6>(f,blob,act+off,out+off,r0,r1);break;
            case 7:viewimpl::batch<7>(f,blob,act+off,out+off,r0,r1);break;
            case 8:viewimpl::batch<8>(f,blob,act+off,out+off,r0,r1);break;
        }off+=n;
    }
}

// Called only after the generic dispatch's runtime ISA/opt-in checks. A false
// return preserves the ORIGINAL whole-call down path, including large tails.
bool q81_down_reuse_rows(const NativeFmt& f,const uint8_t* blob,const void* const* hq,
                        int nt,float* const* out,int r0,int r1) {
    if(nt<1||nt>8||f.d_type!=20||f.n_embd!=2560||f.n_ff!=640||f.tail_off||f.had_d)
        return false;
    viewimpl::A::require(r0>=0&&r1>=r0&&r1<=int(f.n_embd),"down reuse row range");
    if(r0==r1)return true;
    auto call=[&]<int N>() {
        downreuse::Prepared<N> y(hq);
        if constexpr(N==1)downreuse::singleton_rows(blob+f.down_off,f.d_row,y,out,r0,r1);
        else if constexpr(N==2||N==3)downreuse::rowlane_rows<N>(blob+f.down_off,f.d_row,y,out,r0,r1);
        else if constexpr(N<=4)downreuse::paired_rows<N>(blob+f.down_off,f.d_row,y,out,r0,r1);
        else downreuse::rows<N,0,N>(blob+f.down_off,f.d_row,y,out,r0,r1);
    };
    switch(nt){
        case 1:call.template operator()<1>();return true;
        case 2:call.template operator()<2>();return true;
        case 3:call.template operator()<3>();return true;
        case 4:call.template operator()<4>();return true;
        case 5:call.template operator()<5>();return true;
        case 6:call.template operator()<6>();return true;
        case 7:call.template operator()<7>();return true;
        case 8:call.template operator()<8>();return true;
    }
    return false;
}
bool q81_down_prepare(const NativeFmt& f,const void* const* hq,int nt,Q81DownPrepared& prepared) {
    prepared.ready=false;prepared.nt=0;
    if(nt<1||nt>8||f.d_type!=20||f.n_embd!=2560||f.n_ff!=640||f.tail_off||f.had_d)
        return false;
    auto create=[&]<int N>() {
        using T=downreuse::Prepared<N>;
        static_assert(sizeof(T)<=sizeof(prepared.storage));
        static_assert(alignof(Q81DownPrepared)>=alignof(T));
        static_assert(std::is_trivially_destructible_v<T>);
        // Replacing a trivially destructible previous snapshot ends its
        // lifetime. The storage is never reused while a consumer is active.
        ::new(static_cast<void*>(prepared.storage)) T(hq);
    };
    switch(nt) {
        case 1:create.template operator()<1>();break;
        case 2:create.template operator()<2>();break;
        case 3:create.template operator()<3>();break;
        case 4:create.template operator()<4>();break;
        case 5:create.template operator()<5>();break;
        case 6:create.template operator()<6>();break;
        case 7:create.template operator()<7>();break;
        case 8:create.template operator()<8>();break;
    }
    prepared.nt=nt;prepared.ready=true;return true;
}

bool q81_down_prepared_rows(const NativeFmt& f,const uint8_t* blob,
                           const Q81DownPrepared& prepared,int nt,float* const* out,int r0,int r1) {
    if(!prepared.ready||nt!=prepared.nt||nt<1||nt>8||f.d_type!=20||
       f.n_embd!=2560||f.n_ff!=640||f.tail_off||f.had_d)return false;
    viewimpl::A::require(r0>=0&&r1>=r0&&r1<=int(f.n_embd),"prepared down row range");
    if(r0==r1)return true;
    auto call=[&]<int N>() {
        const auto& y=*std::launder(reinterpret_cast<const downreuse::Prepared<N>*>(prepared.storage));
        if constexpr(N==1)downreuse::singleton_rows(blob+f.down_off,f.d_row,y,out,r0,r1);
        else if constexpr(N==2||N==3)downreuse::rowlane_rows<N>(blob+f.down_off,f.d_row,y,out,r0,r1);
        else if constexpr(N<=4)downreuse::paired_rows<N>(blob+f.down_off,f.d_row,y,out,r0,r1);
        else downreuse::rows<N,0,N>(blob+f.down_off,f.d_row,y,out,r0,r1);
    };
    switch(nt) {
        case 1:call.template operator()<1>();break;
        case 2:call.template operator()<2>();break;
        case 3:call.template operator()<3>();break;
        case 4:call.template operator()<4>();break;
        case 5:call.template operator()<5>();break;
        case 6:call.template operator()<6>();break;
        case 7:call.template operator()<7>();break;
        case 8:call.template operator()<8>();break;
    }
    return true;
}
} // namespace strata::kernels::cpu
