// Offline only. Preserve release IQ4_NL FP32 accumulation/reduction, including
// maddubs saturation/sign semantics. Repack activations once per row call.
#pragma once
#include <immintrin.h>
#include <cstdint>
#include <cstring>
namespace downreuse {
inline float half(uint16_t x) {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(x)));
}
inline uint16_t u16(const void* p) { uint16_t x;std::memcpy(&x,p,2);return x; }
inline __m256 scale_pair(const uint8_t* p0,const uint8_t* p1){
    // Read first two half scales in one F16C conversion. The second half of
    // the initial32bit load (weight code bytes) is overwritten before use.
    const auto packed=_mm_insert_epi16(_mm_loadu_si32(p0),int(u16(p1)),1);
    const auto values=_mm_cvtph_ps(packed);
    const auto duplicated=_mm_shuffle_ps(values,values,0x50); // d0,d0,d1,d1
    // Select only the two defined64bit lanes; undefined upper128 are unread.
    return _mm256_castsi256_ps(_mm256_permute4x64_epi64(
        _mm256_castsi128_si256(_mm_castps_si128(duplicated)),0x50));
}
inline float sum(__m256 v) {
    __m128 x=_mm_add_ps(_mm256_castps256_ps128(v),_mm256_extractf128_ps(v,1));
    x=_mm_hadd_ps(x,x);x=_mm_hadd_ps(x,x);return _mm_cvtss_f32(x);
}
template<int NT> struct Prepared {
    alignas(32) int8_t codes[20][NT][32];
    float scales[20][NT];
    explicit Prepared(const void* const* hq) {
        for(int b=0;b<20;++b)for(int t=0;t<NT;++t) {
            const auto* p=static_cast<const uint8_t*>(hq[t])+b*34;
            _mm256_store_si256(reinterpret_cast<__m256i*>(codes[b][t]),
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p+2)));
            scales[b][t]=half(u16(p));
        }
    }
};
template<int NT,int First,int Count> __attribute__((noinline))
void rows(const uint8_t* w,size_t stride,const Prepared<NT>& y,
          float* const* out,int r0,int r1) {
    const __m128i values=_mm_loadu_si128(reinterpret_cast<const __m128i*>(kvalues_iq4nl));
    const __m128i mask=_mm_set1_epi8(15);
    for(int r=r0;r<r1;++r) {
        const uint8_t* row=w+size_t(r)*stride;
        __m256 acc[Count];for(int t=0;t<Count;++t)acc[t]=_mm256_setzero_ps();
        for(int b=0;b<20;++b) {
            const uint8_t* p=row+b*18;
            _mm_prefetch(reinterpret_cast<const char*>(p)+2048,_MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(p)+2112,_MM_HINT_T0);
            const __m128i bits=_mm_loadu_si128(reinterpret_cast<const __m128i*>(p+2));
            const __m128i lo=_mm_shuffle_epi8(values,_mm_and_si128(bits,mask));
            const __m128i hi=_mm_shuffle_epi8(values,_mm_and_si128(_mm_srli_epi16(bits,4),mask));
            const __m256i q=_mm256_inserti128_si256(_mm256_castsi128_si256(lo),hi,1);
            const __m256i aq=_mm256_sign_epi8(q,q);
            const float dx=half(u16(p));
            for(int t=0;t<Count;++t) {
                const __m256i a=_mm256_load_si256(reinterpret_cast<const __m256i*>(y.codes[b][First+t]));
                const __m256i products=_mm256_madd_epi16(
                    _mm256_maddubs_epi16(aq,_mm256_sign_epi8(a,q)),_mm256_set1_epi16(1));
                acc[t]=_mm256_fmadd_ps(_mm256_set1_ps(dx*y.scales[b][First+t]),
                                     _mm256_cvtepi32_ps(products),acc[t]);
            }
        }
        for(int t=0;t<Count;++t)out[First+t][r]=sum(acc[t]);
    }
}
inline void decode_lanes(const uint8_t* p0,const uint8_t* p1,__m256i& lo,__m256i& hi){
    const auto values=_mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(kvalues_iq4nl)));
    const auto mask=_mm256_set1_epi8(15);
    const auto bits=_mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p0+2))),
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(p1+2)),1);
    lo=_mm256_shuffle_epi8(values,_mm256_and_si256(bits,mask));
    hi=_mm256_shuffle_epi8(values,_mm256_and_si256(_mm256_srli_epi16(bits,4),mask));
}
inline void decode_two(const uint8_t* p0,const uint8_t* p1,__m256i& q0,__m256i& q1){
    __m256i lo,hi;decode_lanes(p0,p1,lo,hi);
    q0=_mm256_permute2x128_si256(lo,hi,0x20);
    q1=_mm256_permute2x128_si256(lo,hi,0x31);
}
inline void decode_pair(const uint8_t* p,__m256i& q0,__m256i& q1){decode_two(p,p+18,q0,q1);}
template<int NT> __attribute__((noinline)) void paired_rows(const uint8_t* w,size_t stride,const Prepared<NT>& y,
    float* const* out,int r0,int r1){
    static_assert(NT>=2&&NT<=4);
    const auto ones=_mm256_set1_epi16(1);
    for(int r=r0;r<r1;++r){
        const auto* row=w+size_t(r)*stride;__m256 acc[NT];for(auto& a:acc)a=_mm256_setzero_ps();
        for(int b=0;b<20;b+=2){
            const auto* p=row+b*18;
            // Keep001's exact prefetch addresses/count: only decode changes.
            _mm_prefetch(reinterpret_cast<const char*>(p)+2048,_MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(p)+2112,_MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(p)+2066,_MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(p)+2130,_MM_HINT_T0);
            __m256i q0,q1;decode_pair(p,q0,q1);
            const auto abs0=_mm256_sign_epi8(q0,q0),abs1=_mm256_sign_epi8(q1,q1);
            const float d0=half(u16(p)),d1=half(u16(p+18));
            for(int t=0;t<NT;++t){
                const auto a0=_mm256_load_si256(reinterpret_cast<const __m256i*>(y.codes[b][t]));
                const auto a1=_mm256_load_si256(reinterpret_cast<const __m256i*>(y.codes[b+1][t]));
                const auto z0=_mm256_madd_epi16(_mm256_maddubs_epi16(abs0,_mm256_sign_epi8(a0,q0)),ones);
                const auto z1=_mm256_madd_epi16(_mm256_maddubs_epi16(abs1,_mm256_sign_epi8(a1,q1)),ones);
                // A single stream, with precisely the original20 block order.
                acc[t]=_mm256_fmadd_ps(_mm256_set1_ps(d0*y.scales[b][t]),_mm256_cvtepi32_ps(z0),acc[t]);
                acc[t]=_mm256_fmadd_ps(_mm256_set1_ps(d1*y.scales[b+1][t]),_mm256_cvtepi32_ps(z1),acc[t]);
            }
        }
        for(int t=0;t<NT;++t)out[t][r]=sum(acc[t]);
    }
}
// Share each activation load across two output rows. Six live accumulators
// suffice atNT3; leave NT4+ on004's lower-register-pressure row kernel.
template<int NT> __attribute__((noinline)) void rowpair_rows(const uint8_t* w,size_t stride,const Prepared<NT>& y,
    float* const* out,int r0,int r1){
    static_assert(NT==2||NT==3);const auto ones=_mm256_set1_epi16(1);int r=r0;
    for(;r+1<r1;r+=2){
        const auto* row0=w+size_t(r)*stride;const auto* row1=row0+stride;
        __m256 acc[2][NT];for(auto& row:acc)for(auto& a:row)a=_mm256_setzero_ps();
        for(int b=0;b<20;++b){
            const auto* p0=row0+b*18;const auto* p1=row1+b*18;
            _mm_prefetch(reinterpret_cast<const char*>(p0)+2048,_MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(p0)+2112,_MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(p1)+2048,_MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(p1)+2112,_MM_HINT_T0);
            __m256i q0,q1;decode_two(p0,p1,q0,q1);
            const auto abs0=_mm256_sign_epi8(q0,q0),abs1=_mm256_sign_epi8(q1,q1);
            const float d0=half(u16(p0)),d1=half(u16(p1));
            for(int t=0;t<NT;++t){
                const auto a=_mm256_load_si256(reinterpret_cast<const __m256i*>(y.codes[b][t]));
                const auto z0=_mm256_madd_epi16(_mm256_maddubs_epi16(abs0,_mm256_sign_epi8(a,q0)),ones);
                const auto z1=_mm256_madd_epi16(_mm256_maddubs_epi16(abs1,_mm256_sign_epi8(a,q1)),ones);
                const float dy=y.scales[b][t];
                acc[0][t]=_mm256_fmadd_ps(_mm256_set1_ps(d0*dy),_mm256_cvtepi32_ps(z0),acc[0][t]);
                acc[1][t]=_mm256_fmadd_ps(_mm256_set1_ps(d1*dy),_mm256_cvtepi32_ps(z1),acc[1][t]);
            }
        }
        for(int t=0;t<NT;++t){out[t][r]=sum(acc[0][t]);out[t][r+1]=sum(acc[1][t]);}
    }
    if(r<r1)paired_rows<NT>(w,stride,y,out,r,r1);
}
// Keep each weight row in its128bit lane through integer dots/FMA. This
// removes two per-block weight lane permutations; final reconstruction uses
// precisely the original low+high/hadd/hadd tree for each output row.
template<int NT> __attribute__((noinline)) void rowlane_rows(const uint8_t* w,size_t stride,const Prepared<NT>& y,
    float* const* out,int r0,int r1){
    static_assert(NT==2||NT==3);const auto ones=_mm256_set1_epi16(1);int r=r0;
    for(;r+1<r1;r+=2){
        __m256 low_acc[NT],high_acc[NT];for(int t=0;t<NT;++t)low_acc[t]=high_acc[t]=_mm256_setzero_ps();
        for(int b=0;b<20;++b){
            const auto* p0=w+size_t(r)*stride+b*18;const auto* p1=p0+stride;
            _mm_prefetch(reinterpret_cast<const char*>(p0)+2048,_MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(p0)+2112,_MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(p1)+2048,_MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(p1)+2112,_MM_HINT_T0);
            __m256i lo,hi;decode_lanes(p0,p1,lo,hi);
            const auto abslo=_mm256_sign_epi8(lo,lo),abshi=_mm256_sign_epi8(hi,hi);
            const auto dx=scale_pair(p0,p1);
            for(int t=0;t<NT;++t){
                const auto a=_mm256_broadcastsi128_si256(_mm_load_si128(reinterpret_cast<const __m128i*>(y.codes[b][t])));
                const auto c=_mm256_broadcastsi128_si256(_mm_load_si128(reinterpret_cast<const __m128i*>(y.codes[b][t]+16)));
                const auto low_dot=_mm256_madd_epi16(_mm256_maddubs_epi16(abslo,_mm256_sign_epi8(a,lo)),ones);
                const auto high_dot=_mm256_madd_epi16(_mm256_maddubs_epi16(abshi,_mm256_sign_epi8(c,hi)),ones);
                const auto scale=_mm256_mul_ps(dx,_mm256_set1_ps(y.scales[b][t]));
                low_acc[t]=_mm256_fmadd_ps(scale,_mm256_cvtepi32_ps(low_dot),low_acc[t]);
                high_acc[t]=_mm256_fmadd_ps(scale,_mm256_cvtepi32_ps(high_dot),high_acc[t]);
            }
        }
        for(int t=0;t<NT;++t){
            auto a=_mm_add_ps(_mm256_castps256_ps128(low_acc[t]),_mm256_castps256_ps128(high_acc[t]));
            auto b=_mm_add_ps(_mm256_extractf128_ps(low_acc[t],1),_mm256_extractf128_ps(high_acc[t],1));
            a=_mm_hadd_ps(a,a);a=_mm_hadd_ps(a,a);b=_mm_hadd_ps(b,b);b=_mm_hadd_ps(b,b);
            out[t][r]=_mm_cvtss_f32(a);out[t][r+1]=_mm_cvtss_f32(b);
        }
    }
    if(r<r1)rowpair_rows<NT>(w,stride,y,out,r,r1);
}
// The default singleton uses ggml AVX2's TWO streams, unlike the release's
// multi-token IQ4_NL kernel above. Preserve both block order and final tree.
inline float singleton_sum(__m256 v){
    __m128 x=_mm256_extractf128_ps(v,1);
    x=_mm_add_ps(x,_mm256_castps256_ps128(v));
    x=_mm_add_ps(x,_mm_movehl_ps(x,x));
    x=_mm_add_ss(x,_mm_movehdup_ps(x));return _mm_cvtss_f32(x);
}
__attribute__((noinline)) inline void singleton_rows(const uint8_t* w,size_t stride,const Prepared<1>& y,
    float* const* out,int r0,int r1){
    const auto values=_mm_loadu_si128(reinterpret_cast<const __m128i*>(kvalues_iq4nl));
    const auto mask=_mm_set1_epi8(15);
    const auto ones=_mm256_set1_epi16(1);
    for(int r=r0;r<r1;++r){
        const uint8_t* row=w+size_t(r)*stride;__m256 acc[2]={_mm256_setzero_ps(),_mm256_setzero_ps()};
        for(int b=0;b<20;b+=2){
#pragma GCC unroll 2
            for(int k=0;k<2;++k){
                const uint8_t* p=row+(b+k)*18;
                const auto bits=_mm_loadu_si128(reinterpret_cast<const __m128i*>(p+2));
                const auto lo=_mm_shuffle_epi8(values,_mm_and_si128(bits,mask));
                const auto hi=_mm_shuffle_epi8(values,_mm_and_si128(_mm_srli_epi16(bits,4),mask));
                const auto q=_mm256_inserti128_si256(_mm256_castsi128_si256(lo),hi,1);
                const auto a=_mm256_load_si256(reinterpret_cast<const __m256i*>(y.codes[b+k][0]));
                const auto products=_mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_sign_epi8(q,q),_mm256_sign_epi8(a,q)),ones);
                acc[k]=_mm256_fmadd_ps(_mm256_set1_ps(y.scales[b+k][0]*half(u16(p))),_mm256_cvtepi32_ps(products),acc[k]);
            }
        }
        out[0][r]=singleton_sum(_mm256_add_ps(acc[0],acc[1]));
    }
}
template<int NT,bool Fused> void run(const strata::kernels::cpu::NativeFmt& f,
    const uint8_t* blob,const void* const* hq,float* const* out,int r0,int r1) {
    namespace C=strata::kernels::cpu;
    if(f.d_type!=20) { C::native_down_rows(f,blob,hq,NT,out,r0,r1);return; }
    if(f.n_ff!=640 || f.tail_off)throw std::runtime_error("down prototype geometry");
    Prepared<NT> y(hq); // Counted in timing. No persistent cache or stale-pointer risk.
    if constexpr(NT==1){singleton_rows(blob+f.down_off,f.d_row,y,out,r0,r1);return;}
    if constexpr(NT==2||NT==3){rowlane_rows<NT>(blob+f.down_off,f.d_row,y,out,r0,r1);return;}
    if constexpr(NT>=2&&NT<=4){paired_rows<NT>(blob+f.down_off,f.d_row,y,out,r0,r1);return;}
    if constexpr(Fused || NT<=4)rows<NT,0,NT>(blob+f.down_off,f.d_row,y,out,r0,r1);
    else { rows<NT,0,4>(blob+f.down_off,f.d_row,y,out,r0,r1);
           rows<NT,4,NT-4>(blob+f.down_off,f.d_row,y,out,r0,r1); }
}
} // namespace downreuse
