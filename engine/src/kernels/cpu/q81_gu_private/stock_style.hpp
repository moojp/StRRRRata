// CPU-only experiment: reuse the release's SIMD weight decoder verbatim.
// Keep block32 Q8_1 inputs, but use normal CPU float-vector accumulation.
// IQ2_S/IQ3_XXS use their rational scale rather than GPU integer truncation.
#pragma once
#include "strata/kernels/cpu/expert_layout.hpp"
#ifndef ALIGN_GATHER
#define ALIGN_GATHER 0
#endif
namespace stockdecode {
#include "stock_decode.generated.hpp"
}
namespace stockstyle {
namespace A=q81math;
namespace C=strata::kernels::cpu;
// Compact planar layout: aligned code loads, no unused int16 image or s field.
// Fits the release activation scratch without changing its public ABI.
struct alignas(32) Input {
    alignas(32)std::array<int8_t,2560> code;
    alignas(32)std::array<float,80> scales;
};
static_assert(sizeof(Input)<=C::kNativeActBytes);
// Same GPU-compatible division and half-away rounding, vectorized without
// reciprocal approximation. Integer rounding uses fractional comparison, not
// add(0.5): that addition can misround values immediately below a half.
inline void prepare(Input& a,const float* x,int n){
    A::require(n==640||n==2560,"stockstyle activation geometry");const int blocks=n/32;
    const __m256 mask=_mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    for(int k=0;k<blocks;++k){
        __m256 v[4],av[4];
        for(int j=0;j<4;++j){
            v[j]=_mm256_loadu_ps(x+32*k+8*j);av[j]=_mm256_and_ps(v[j],mask);
            A::require(_mm256_movemask_ps(_mm256_cmp_ps(av[j],_mm256_set1_ps(0x1.fffffep127f),_CMP_LE_OQ))==255,"nonfinite activation");
        }
        __m256 mx=_mm256_max_ps(_mm256_max_ps(av[0],av[1]),_mm256_max_ps(av[2],av[3]));
        // All lanes are finite nonnegative magnitudes. Reassociation of max
        // is exact here; it does not reassociate any floating point sum.
        __m128 m=_mm_max_ps(_mm256_castps256_ps128(mx),_mm256_extractf128_ps(mx,1));
        m=_mm_max_ps(m,_mm_movehl_ps(m,m));
        m=_mm_max_ss(m,_mm_shuffle_ps(m,m,_MM_SHUFFLE(1,1,1,1)));
        float am=_mm_cvtss_f32(m);
        float d=std::min(am/127.f,65504.f);
        A::require(am==0.f||d>0.f,"activation scale underflow");
        auto half=A::f2h(d);a.scales[k]=A::h2f(half);
        __m128i p[4];
        for(int j=0;j<4;++j){
            __m256 mag=_mm256_min_ps(_mm256_div_ps(av[j],_mm256_set1_ps(am==0.f?1.f:d)),_mm256_set1_ps(127.f));
            __m256 fl=_mm256_round_ps(mag,_MM_FROUND_TO_ZERO|_MM_FROUND_NO_EXC);
            __m256i inc=_mm256_and_si256(_mm256_castps_si256(_mm256_cmp_ps(_mm256_sub_ps(mag,fl),_mm256_set1_ps(.5f),_CMP_GE_OQ)),_mm256_set1_epi32(1));
            __m256i q=_mm256_add_epi32(_mm256_cvttps_epi32(fl),inc);
            __m256i sign=_mm256_srai_epi32(_mm256_castps_si256(v[j]),31);
            q=_mm256_sub_epi32(_mm256_xor_si256(q,sign),sign);
            p[j]=_mm_packs_epi32(_mm256_castsi256_si128(q),_mm256_extracti128_si256(q,1));
        }
        int8_t* dst=a.code.data()+32*k;
        _mm_storeu_si128((__m128i*)dst,_mm_packs_epi16(p[0],p[1]));
        _mm_storeu_si128((__m128i*)(dst+16),_mm_packs_epi16(p[2],p[3]));
    }
}
template<int Type> inline float dot(const uint8_t* row,const Input& x) {
    const int pf=stockdecode::prefetch_distance();
    __m256 accum0=_mm256_setzero_ps();
    for(int b=0;b<10;++b){
        const uint8_t* blk=row+b*stockdecode::Fmt32<Type>::bytes;
        stockdecode::rows_ahead(blk,pf);
        const float dw=A::h2f(stockdecode::u16(blk))*stockdecode::Fmt32<Type>::K;
        for(int j=0;j<4;++j)for(int half=0;half<2;++half){
            __m256i g,sgn,sc;stockdecode::Fmt32<(ALIGN_GATHER&&Type!=23)?Type+100:Type>::decode(blk,j,half,g,sgn,sc);
            const int k=8*b+2*j+half;
            __m256i v=_mm256_sign_epi8(_mm256_load_si256((const __m256i*)(x.code.data()+32*k)),sgn);
            __m256i sums=_mm256_madd_epi16(_mm256_maddubs_epi16(g,v),sc);
            __m256 scale=_mm256_set1_ps(dw*x.scales[k]);
            accum0=_mm256_fmadd_ps(scale,_mm256_cvtepi32_ps(sums),accum0);
        }
    }
    return stockdecode::hsum8(accum0);
}
template<int Type> inline void gu(const C::NativeFmt& f,const uint8_t* blob,const Input& x,float* h){
    const bool use_exp2=strata::kernels::cpu::gpu_silu_exp2_detail::enabled();
    for(int r=0;r<640;++r){
        float g=dot<Type>(blob+r*f.gu_row,x),u=dot<Type>(blob+f.up_off+r*f.gu_row,x);
        h[r]=strata::kernels::cpu::gpu_silu_exp2_detail::apply(use_exp2,g,u);
    }
}
inline void gu(const C::NativeFmt& f,const uint8_t* b,const Input& x,float* h){
    switch(f.gu_type){
        case 18:gu<18>(f,b,x,h);return;
        case 21:gu<21>(f,b,x,h);return;
        case 22:gu<22>(f,b,x,h);return;
        case 23:gu<23>(f,b,x,h);return;
        default:throw std::runtime_error("stockstyle unsupported gate type");
    }
}
// Existing release down kernels accept GPU-rounded codes/scales in their own
// storage. Only storage changes here; no weight copy or expanded weight cache.
struct Down {
    std::array<block_q8_0,20> iq4;
    C::ActQ q2;
    void prepare(const C::NativeFmt& f,const Input& x){
        if(f.d_type==20){
            A::require(f.d_act==8,"IQ4_NL activation format");
            for(int k=0;k<20;++k){iq4[k].d=A::f2h(x.scales[k]);std::memcpy(iq4[k].qs,x.code.data()+32*k,32);}
        }else{
            A::require(f.d_type==42&&!C::q2_bitplane_enabled(),"Q2 down experiment requires default bitplane off");
            q2.nchunks=20;q2.bp_pairs=0;
            for(int k=0;k<20;++k){
                std::memcpy(q2.q+32*k,x.code.data()+32*k,32);q2.scale[k]=x.scales[k];
                int sum=0;for(int j=0;j<32;++j)sum+=x.code[32*k+j];
                q2.sum[k]=sum;q2.hx[k]=sum*x.scales[k];
            }
        }
    }
    void run(const C::NativeFmt& f,const uint8_t* b,float* out){
        float* op[]={out};
        if(f.d_type==20){const void* p[]={iq4.data()};C::native_down_rows(f,b,p,1,op,0,2560);}
        else{const C::ActQ* p[]={&q2};C::q2_rows_any(b+f.down_off,f.d_row,10,p,1,op,0,2560);}
    }
};
}
