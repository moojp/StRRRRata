// Multi-token research extension. Decode each four-block group once per row.
#pragma once
#include "gpu_down_order_single.hpp"
namespace down_gpu_tree_multi {
namespace D=down_gpu_tree_proto;
template<int NT> struct Prepared {
    alignas(32) int8_t codes[20][NT][32];float scales[20][NT];
    explicit Prepared(const void* const* input){
        for(int b=0;b<20;++b)for(int t=0;t<NT;++t){
            const auto* q=static_cast<const uint8_t*>(input[t])+34*b;
            auto values=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(q+2));
            if(_mm256_movemask_epi8(_mm256_cmpeq_epi8(values,_mm256_set1_epi8(-128))))throw std::runtime_error("Unsupported GPU activation code");
            const auto lo=_mm256_castsi256_si128(values),hi=_mm256_extracti128_si256(values,1);
            // Each SIMD half now contains exactly one GPU half-block's dot.
            const auto interleaved=_mm256_inserti128_si256(_mm256_castsi128_si256(_mm_unpacklo_epi8(lo,hi)),_mm_unpackhi_epi8(lo,hi),1);
            _mm256_store_si256(reinterpret_cast<__m256i*>(codes[b][t]),interleaved);scales[b][t]=D::half(q);
            if(!std::isfinite(scales[b][t]))throw std::runtime_error("Nonfinite prototype scale");
        }
    }
};
struct Group {
    __m256i weights[4],magnitudes[4];__m128 dw;
    Group()=default;
    explicit Group(const uint8_t* p){
        const auto table=_mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(D::values)));
        const auto mask=_mm256_set1_epi8(15);
        for(int j=0;j<4;++j){
            const auto bits=_mm_loadu_si128(reinterpret_cast<const __m128i*>(p+18*j+2));
            const auto wide=_mm256_cvtepu8_epi16(bits);
            const auto indices=_mm256_and_si256(_mm256_or_si256(wide,_mm256_slli_epi16(wide,4)),mask);
            weights[j]=_mm256_shuffle_epi8(table,indices);magnitudes[j]=_mm256_abs_epi8(weights[j]);
        }
        auto half=_mm_cvtsi32_si128(D::u16(p));half=_mm_insert_epi16(half,D::u16(p+18),1);
        half=_mm_insert_epi16(half,D::u16(p+36),2);half=_mm_insert_epi16(half,D::u16(p+54),3);dw=_mm_cvtph_ps(half);
    }
    template<int NT> inline __m256 apply(const Prepared<NT>& x,int block,int t,__m256 acc) const {
        __m256i p[4];const auto one=_mm256_set1_epi16(1);
        for(int j=0;j<4;++j){
            const auto a=_mm256_load_si256(reinterpret_cast<const __m256i*>(x.codes[block+j][t]));
            p[j]=_mm256_madd_epi16(_mm256_maddubs_epi16(magnitudes[j],_mm256_sign_epi8(a,weights[j])),one);
        }
        const auto a=_mm256_hadd_epi32(p[0],p[1]),b=_mm256_hadd_epi32(p[2],p[3]);
        const auto sums=_mm256_hadd_epi32(a,b);
        const auto dx=_mm_setr_ps(x.scales[block][t],x.scales[block+1][t],x.scales[block+2][t],x.scales[block+3][t]);
        const auto ds=_mm_mul_ps(dw,dx);
        const auto scale=_mm256_insertf128_ps(_mm256_castps128_ps256(ds),ds,1);
        return _mm256_fmadd_ps(scale,_mm256_cvtepi32_ps(sums),acc);
    }
};
// Two blocks from each of two output rows. Each activation load is shared.
// Keep only signed weights live: abs(activation)*sign(weight,activation)
// gives identical integer partial dots for codes in [-127,127], without
// retaining four extra weight-magnitude vectors across all token accumulators.
struct PairGroup {
    __m256i weights[4];__m128 dw;
    PairGroup(const uint8_t* p,size_t stride){
        const uint8_t* source[4]={p,p+18,p+stride,p+stride+18};
        const auto table=_mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(D::values)));
        const auto mask=_mm256_set1_epi8(15);
        for(int j=0;j<4;++j){
            const auto bits=_mm_loadu_si128(reinterpret_cast<const __m128i*>(source[j]+2));
            const auto wide=_mm256_cvtepu8_epi16(bits);
            const auto indices=_mm256_and_si256(_mm256_or_si256(wide,_mm256_slli_epi16(wide,4)),mask);
            weights[j]=_mm256_shuffle_epi8(table,indices);
        }
        auto half=_mm_cvtsi32_si128(D::u16(source[0]));half=_mm_insert_epi16(half,D::u16(source[1]),1);
        half=_mm_insert_epi16(half,D::u16(source[2]),2);half=_mm_insert_epi16(half,D::u16(source[3]),3);dw=_mm_cvtph_ps(half);
    }
    template<int NT> inline __m256 apply(const Prepared<NT>& x,int block,int t,__m256 acc) const {
        const auto q0=_mm256_load_si256(reinterpret_cast<const __m256i*>(x.codes[block][t]));
        const auto q1=_mm256_load_si256(reinterpret_cast<const __m256i*>(x.codes[block+1][t]));
        const auto m0=_mm256_abs_epi8(q0),m1=_mm256_abs_epi8(q1);
        const auto one=_mm256_set1_epi16(1);__m256i p[4];
        for(int j=0;j<4;++j)p[j]=_mm256_madd_epi16(_mm256_maddubs_epi16(j%2?m1:m0,_mm256_sign_epi8(weights[j],j%2?q1:q0)),one);
        const auto a=_mm256_hadd_epi32(p[0],p[1]),b=_mm256_hadd_epi32(p[2],p[3]);
        const auto sums=_mm256_hadd_epi32(a,b);
        const auto ds=_mm_mul_ps(dw,_mm_setr_ps(x.scales[block][t],x.scales[block+1][t],x.scales[block][t],x.scales[block+1][t]));
        const auto scale=_mm256_insertf128_ps(_mm256_castps128_ps256(ds),ds,1);
        return _mm256_fmadd_ps(scale,_mm256_cvtepi32_ps(sums),acc);
    }
};
inline float reduce(__m256 x){
    // Lanes are [block0..3/half0, block0..3/half1]. This is a
    // representation permutation, not an FP reassociation: XOR2,1,4 here
    // corresponds exactly to GPU XOR4,2,1 in the original lane numbering.
    x=_mm256_add_ps(x,_mm256_permute_ps(x,0x4e));x=_mm256_add_ps(x,_mm256_permute_ps(x,0xb1));
    x=_mm256_add_ps(x,_mm256_permute2f128_ps(x,x,1));return _mm_cvtss_f32(_mm256_castps256_ps128(x));
}
template<int NT> __attribute__((noinline)) void rows(const uint8_t* weight,size_t stride,const Prepared<NT>& x,float* const* output,int begin,int end){
    static_assert(NT>=1&&NT<=8);const auto zero=_mm256_setzero_ps();
    for(int r=begin;r<end;++r){
        const auto* row=weight+size_t(r)*stride;__m256 ab[NT],c[NT];
        {const Group g0(row);for(int t=0;t<NT;++t)ab[t]=g0.apply(x,0,t,zero);}
        {const Group g4(row+16*18);for(int t=0;t<NT;++t)ab[t]=g4.apply(x,16,t,ab[t]);}
        {
            const Group g2(row+8*18);
            for(int t=0;t<NT;++t)ab[t]=_mm256_add_ps(ab[t],g2.apply(x,8,t,zero));
        }
        {
            const Group g1(row+4*18);
            for(int t=0;t<NT;++t)c[t]=g1.apply(x,4,t,zero);
        }
        {
            const Group g3(row+12*18);
            for(int t=0;t<NT;++t)output[t][r]=reduce(_mm256_add_ps(ab[t],_mm256_add_ps(c[t],g3.apply(x,12,t,zero))));
        }
    }
}
template<int NT> __attribute__((noinline)) void pair_rows(const uint8_t* weight,size_t stride,const Prepared<NT>& x,float* const* output,int begin,int end){
    static_assert(NT==2||NT==3);const auto zero=_mm256_setzero_ps();int r=begin;
    for(;r+1<end;r+=2){
        const auto* row=weight+size_t(r)*stride;__m256 ab0[NT],ab1[NT];
        {const PairGroup g0(row,stride);for(int t=0;t<NT;++t)ab0[t]=g0.apply(x,0,t,zero);}
        {const PairGroup g16(row+16*18,stride);for(int t=0;t<NT;++t)ab0[t]=g16.apply(x,16,t,ab0[t]);}
        {const PairGroup g8(row+8*18,stride);for(int t=0;t<NT;++t)ab0[t]=_mm256_add_ps(ab0[t],g8.apply(x,8,t,zero));}
        {
            __m256 c0[NT];
            {const PairGroup g4(row+4*18,stride);for(int t=0;t<NT;++t)c0[t]=g4.apply(x,4,t,zero);}
            {const PairGroup g12(row+12*18,stride);for(int t=0;t<NT;++t)ab0[t]=_mm256_add_ps(ab0[t],_mm256_add_ps(c0[t],g12.apply(x,12,t,zero)));}
        }
        {const PairGroup g2(row+2*18,stride);for(int t=0;t<NT;++t)ab1[t]=g2.apply(x,2,t,zero);}
        {const PairGroup g18(row+18*18,stride);for(int t=0;t<NT;++t)ab1[t]=g18.apply(x,18,t,ab1[t]);}
        {const PairGroup g10(row+10*18,stride);for(int t=0;t<NT;++t)ab1[t]=_mm256_add_ps(ab1[t],g10.apply(x,10,t,zero));}
        __m256 c1[NT];
        {const PairGroup g6(row+6*18,stride);for(int t=0;t<NT;++t)c1[t]=g6.apply(x,6,t,zero);}
        {const PairGroup g14(row+14*18,stride);for(int t=0;t<NT;++t){
            auto sum=_mm256_add_ps(ab0[t],_mm256_add_ps(ab1[t],_mm256_add_ps(c1[t],g14.apply(x,14,t,zero))));
            sum=_mm256_add_ps(sum,_mm256_permute_ps(sum,0xb1));sum=_mm256_add_ps(sum,_mm256_permute2f128_ps(sum,sum,1));
            const auto low=_mm256_castps256_ps128(sum);
            output[t][r]=_mm_cvtss_f32(low);output[t][r+1]=_mm_cvtss_f32(_mm_shuffle_ps(low,low,0xaa));
        }}
    }
    if(r<end)rows<NT>(weight,stride,x,output,r,end);
}
template<int NT> inline void run(const uint8_t* weight,size_t stride,const void* const* input,float* const* output,int begin,int end){
    if constexpr(NT==1){D::Prepared x(input[0]);D::rows(weight,stride,x,output[0],begin,end);}
    else {Prepared<NT> x(input);if constexpr(NT==2||NT==3)pair_rows<NT>(weight,stride,x,output,begin,end);else rows<NT>(weight,stride,x,output,begin,end);}
}
inline void dispatch(int nt,const uint8_t* weight,size_t stride,const void* const* input,float* const* output,int begin,int end){
    switch(nt){
        case 1:return run<1>(weight,stride,input,output,begin,end);
        case 2:return run<2>(weight,stride,input,output,begin,end);
        case 3:return run<3>(weight,stride,input,output,begin,end);
        case 4:return run<4>(weight,stride,input,output,begin,end);
        case 5:return run<5>(weight,stride,input,output,begin,end);
        case 6:return run<6>(weight,stride,input,output,begin,end);
        case 7:return run<7>(weight,stride,input,output,begin,end);
        case 8:return run<8>(weight,stride,input,output,begin,end);
        default:throw std::runtime_error("Prototype width outside1..8");
    }
}
}
