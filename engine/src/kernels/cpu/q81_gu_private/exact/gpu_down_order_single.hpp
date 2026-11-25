// Research-only IQ4_NL down: integer half-block dots, then the GPU FP32 tree.
#pragma once
#include <immintrin.h>
#include <array>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <stdexcept>
namespace down_gpu_tree_proto {
inline constexpr int8_t values[16]={-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
inline float half(const void* p){uint16_t h;std::memcpy(&h,p,2);return _cvtsh_ss(h);}
inline uint16_t u16(const void* p){uint16_t h;std::memcpy(&h,p,2);return h;}
struct Prepared {
    alignas(32) int8_t code[20][32];float scale[20];
    explicit Prepared(const void* input){
        const auto* raw=static_cast<const uint8_t*>(input);
        for(int b=0;b<20;++b){
            std::memcpy(code[b],raw+34*b+2,32);scale[b]=half(raw+34*b);
            if(!std::isfinite(scale[b]))throw std::runtime_error("Nonfinite prototype scale");
            for(int j=0;j<32;++j)if(code[b][j]==-128)throw std::runtime_error("Prototype requires GPU code range[-127,127]");
        }
    }
};
inline __m256 group4(const uint8_t* row,const Prepared& x,int block,__m256 accumulator){
    // Duplicate the packed bytes into both lanes, then select low nibbles
    // in the first lane and high nibbles in the second for one table lookup.
    const auto table = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(values)));
    const auto shifts = _mm256_setr_epi32(0, 0, 0, 0, 4, 4, 4, 4);
    const auto mask = _mm256_set1_epi8(15);
    const auto one = _mm256_set1_epi16(1);
    __m256i p[4];
    for(int j=0;j<4;++j){
        const auto* weight = row + 18 * (block + j);
        const auto bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(weight + 2));
        const auto packed = _mm256_broadcastsi128_si256(bytes);
        const auto indices = _mm256_and_si256(_mm256_srlv_epi32(packed, shifts), mask);
        const auto w = _mm256_shuffle_epi8(table, indices);
        const auto q=_mm256_load_si256(reinterpret_cast<const __m256i*>(x.code[block+j]));
        p[j]=_mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(w),_mm256_sign_epi8(q,w)),one);
    }
    // Four weight halves in one F16C conversion, then duplicate each product
    // for its two GPU half-block lanes. Same separately rounded FP32 multiply.
    const auto* start=row+18*block;
    auto packed_scales=_mm_cvtsi32_si128(u16(start));
    packed_scales=_mm_insert_epi16(packed_scales,u16(start+18),1);
    packed_scales=_mm_insert_epi16(packed_scales,u16(start+36),2);
    packed_scales=_mm_insert_epi16(packed_scales,u16(start+54),3);
    const auto ds=_mm_mul_ps(_mm_cvtph_ps(packed_scales),_mm_loadu_ps(x.scale+block));
    const auto scales=_mm256_insertf128_ps(_mm256_castps128_ps256(_mm_unpacklo_ps(ds,ds)),_mm_unpackhi_ps(ds,ds),1);
    const auto a=_mm256_hadd_epi32(p[0],p[1]),b=_mm256_hadd_epi32(p[2],p[3]);
    const auto lo=_mm_add_epi32(_mm256_castsi256_si128(a),_mm256_extracti128_si256(a,1));
    const auto hi=_mm_add_epi32(_mm256_castsi256_si128(b),_mm256_extracti128_si256(b,1));
    const auto sums=_mm256_inserti128_si256(_mm256_castsi128_si256(lo),hi,1);
    return _mm256_fmadd_ps(scales,_mm256_cvtepi32_ps(sums),accumulator);
}
inline float dot(const uint8_t* row,const Prepared& x){
    const auto zero=_mm256_setzero_ps();
    auto a=group4(row,x,0,zero);a=group4(row,x,16,a);
    const auto b=group4(row,x,8,zero),c=group4(row,x,4,zero),d=group4(row,x,12,zero);
    auto sum=_mm256_add_ps(_mm256_add_ps(a,b),_mm256_add_ps(c,d));
    sum=_mm256_add_ps(sum,_mm256_permute2f128_ps(sum,sum,1));
    sum=_mm256_add_ps(sum,_mm256_permute_ps(sum,0x4e));
    sum=_mm256_add_ps(sum,_mm256_permute_ps(sum,0xb1));return _mm_cvtss_f32(_mm256_castps256_ps128(sum));
}
__attribute__((noinline)) inline void rows(const uint8_t* weight,size_t stride,const Prepared& x,float* output,int begin,int end){
    for(int r=begin;r<end;++r)output[r]=dot(weight+size_t(r)*stride,x);
}
// Independent scalar transcription of the half-block/FMA/XOR GPU contract.
inline float scalar(const uint8_t* row,const void* input){
    const auto* q=static_cast<const uint8_t*>(input);
    auto step=[&](int k,float accumulator){
        int block=k/2,part=k%2,sum=0;const auto* w=row+18*block;
        for(int j=0;j<8;++j){
            int8_t lo,hi;std::memcpy(&lo,q+34*block+2+8*part+j,1);std::memcpy(&hi,q+34*block+2+16+8*part+j,1);
            uint8_t packed=w[2+8*part+j];sum+=int(values[packed&15])*int(lo)+int(values[packed>>4])*int(hi);
        }
        return std::fma(half(w)*half(q+34*block),float(sum),accumulator);
    };
    std::array<float,8> lane{};
    for(int t=0;t<8;++t){float a=step(t,0.f);a=step(t+32,a);lane[t]=(a+step(t+16,0.f))+(step(t+8,0.f)+step(t+24,0.f));}
    for(int offset=4;offset;offset>>=1){auto previous=lane;for(int t=0;t<8;++t)lane[t]=previous[t]+previous[t^offset];}
    return lane[0];
}
}
