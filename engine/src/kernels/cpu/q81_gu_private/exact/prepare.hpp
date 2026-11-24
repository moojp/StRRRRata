// Exact half-away rounding, not an approximate division or reciprocal.
#pragma once
#include <new>
#include <atomic>
namespace quant5 {
inline __m256i round_half_away(__m256 mag){
    const auto nearest=_mm256_round_ps(mag,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    // Round-to-even differs from half-away only at positive half ties that
    // rounded down. In that case mag-nearest is exactly+.5, never-.5.
    const auto tie=_mm256_cmp_ps(_mm256_sub_ps(mag,nearest),_mm256_set1_ps(.5f),_CMP_EQ_OQ);
    return _mm256_sub_epi32(_mm256_cvttps_epi32(nearest),_mm256_castps_si256(tie));
}
inline __m256i original_round(__m256 mag){
    const auto fl=_mm256_round_ps(mag,_MM_FROUND_TO_ZERO|_MM_FROUND_NO_EXC);
    const auto inc=_mm256_and_si256(_mm256_castps_si256(_mm256_cmp_ps(
        _mm256_sub_ps(mag,fl),_mm256_set1_ps(.5f),_CMP_GE_OQ)),_mm256_set1_epi32(1));
    return _mm256_add_epi32(_mm256_cvttps_epi32(fl),inc);
}
template<class Input> inline void prepare(Input& a,const float* x,int n){
    q81math::require(n==640||n==2560,"stockstyle activation geometry");
    const auto mask=_mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    for(int k=0;k<n/32;++k){
        __m256 v[4],av[4];
        for(int j=0;j<4;++j){
            v[j]=_mm256_loadu_ps(x+32*k+8*j);av[j]=_mm256_and_ps(v[j],mask);
            q81math::require(_mm256_movemask_ps(_mm256_cmp_ps(av[j],
                _mm256_set1_ps(0x1.fffffep127f),_CMP_LE_OQ))==255,"nonfinite activation");
        }
        const auto mx=_mm256_max_ps(_mm256_max_ps(av[0],av[1]),_mm256_max_ps(av[2],av[3]));
        auto m=_mm_max_ps(_mm256_castps256_ps128(mx),_mm256_extractf128_ps(mx,1));
        m=_mm_max_ps(m,_mm_movehl_ps(m,m));
        m=_mm_max_ss(m,_mm_shuffle_ps(m,m,_MM_SHUFFLE(1,1,1,1)));
        const float am=_mm_cvtss_f32(m),d=std::min(am/127.f,65504.f);
        q81math::require(am==0.f||d>0.f,"activation scale underflow");
        a.scales[k]=q81math::h2f(q81math::f2h(d));
        __m128i p[4];
        for(int j=0;j<4;++j){
            const auto mag=_mm256_min_ps(_mm256_div_ps(av[j],
                _mm256_set1_ps(am==0.f?1.f:d)),_mm256_set1_ps(127.f));
            auto q=round_half_away(mag);
            const auto sign=_mm256_srai_epi32(_mm256_castps_si256(v[j]),31);
            q=_mm256_sub_epi32(_mm256_xor_si256(q,sign),sign);
            p[j]=_mm_packs_epi32(_mm256_castsi256_si128(q),_mm256_extracti128_si256(q,1));
        }
        auto* dst=a.code.data()+32*k;
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst),_mm_packs_epi16(p[0],p[1]));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst+16),_mm_packs_epi16(p[2],p[3]));
    }
}
inline std::atomic<uint64_t> generation{1};
template<int Mode> __attribute__((noinline)) void wire(
    const strata::kernels::cpu::NativeFmt& f,const float* x,void* dst){
    static_assert(Mode>=1&&Mode<=3);
    constexpr size_t Wire=2880;
    q81math::require(f.n_embd==2560&&f.n_ff==640&&!f.had_gu&&!f.had_d&&
        f.act_bytes>=Wire+31+8,"Q81 gate/up: unsupported geometry or scratch size");
    auto* raw=static_cast<uint8_t*>(dst);
    auto* aligned=reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(raw)+31)&~uintptr_t(31));
    const auto src_address=reinterpret_cast<uintptr_t>(x),dst_address=reinterpret_cast<uintptr_t>(dst);
    // Avoid endpoint overflow. The original stack-temporary implementation
    // also supports overlapping source/output, so preserve that behavior.
    const bool overlap=dst_address>=src_address?dst_address-src_address<2560*sizeof(float)
        :src_address-dst_address<f.act_bytes;
    if(Mode==3&&!overlap){
        // Start a fresh implicit-lifetime aggregate in the aligned wire. No
        // persistent pointer/cache; all code/scale elements are assigned.
        auto* q=::new(static_cast<void*>(aligned)) stockstyle::Input;
        prepare(*q,x,2560);
        std::memset(raw,0,size_t(aligned-raw));
        std::memset(aligned+Wire,0,f.act_bytes-size_t(aligned-raw)-Wire);
    }else{
        stockstyle::Input q;
        if constexpr(Mode==1)stockstyle::prepare(q,x,2560);else prepare(q,x,2560);
        std::memset(raw,0,f.act_bytes);std::memcpy(aligned,&q,Wire);
    }
    const auto g=generation.fetch_add(1,std::memory_order_relaxed);
    q81math::require(g!=0,"Q81 generation counter exhausted");
    std::memcpy(raw+f.act_bytes-8,&g,8);
}
}
