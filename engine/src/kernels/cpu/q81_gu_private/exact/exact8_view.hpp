// Offline025: replace four IQ3XXS sign-table loads with exact SIMD parity.
// No FP64, expanded weights, persistent cache, reciprocal or GPU change.
#pragma once
namespace exact8 {
#include "half_codebook.generated.hpp"
template<int Shift> inline __m256i truncdiv(__m256i x) {
    static_assert(Shift>0 && Shift<32);
    // Unsigned abs(INT_MIN) is2^31; after a positive logical shift its
    // magnitude fits positive int32. Restore sign, giving division toward0.
    return _mm256_sign_epi32(_mm256_srli_epi32(_mm256_abs_epi32(x),Shift),x);
}
inline void halves(const __m256i* p,__m256i& lo,__m256i& hi) {
    auto a=_mm256_hadd_epi32(_mm256_hadd_epi32(p[0],p[1]),_mm256_hadd_epi32(p[2],p[3]));
    auto b=_mm256_hadd_epi32(_mm256_hadd_epi32(p[4],p[5]),_mm256_hadd_epi32(p[6],p[7]));
    lo=_mm256_permute2x128_si256(a,b,0x20);
    hi=_mm256_permute2x128_si256(a,b,0x31);
}
// IQ2_S/IQ3_XXS/IQ3_S grid magnitudes are <=64 (checked exhaustively
// against all three frozen tables before the experiment). Four products
// fit signed16: 4*64*127=32512. IQ4_XS may contain128; do NOT use this there.
inline void halves16(const __m256i* p,__m256i& lo,__m256i& hi) {
    const auto one=_mm256_set1_epi16(1);
    auto a=_mm256_hadd_epi32(_mm256_madd_epi16(_mm256_hadd_epi16(p[0],p[1]),one),_mm256_madd_epi16(_mm256_hadd_epi16(p[2],p[3]),one));
    auto b=_mm256_hadd_epi32(_mm256_madd_epi16(_mm256_hadd_epi16(p[4],p[5]),one),_mm256_madd_epi16(_mm256_hadd_epi16(p[6],p[7]),one));
    lo=_mm256_permute2x128_si256(a,b,0x20);hi=_mm256_permute2x128_si256(a,b,0x31);
}
template<int Type> struct Block {
    __m256i g[8],sign[8],low,high;
    float d;
    __attribute__((always_inline)) explicit Block(const uint8_t* p):d(A::h2f(stockdecode::u16(p))) {
        alignas(32) int l[8],h[8];
        for(int j=0;j<8;++j) {
            __m256i unused;stockdecode::Fmt32<Type>::decode(p,j/2,j%2,g[j],sign[j],unused);
            if constexpr(Type==18) l[j]=2*int(stockdecode::u32(p+66+4*j)>>28)+1,h[j]=0;
            if constexpr(Type==22) l[j]=2*(p[74+j]&15)+1,h[j]=2*(p[74+j]>>4)+1;
            if constexpr(Type==21) l[j]=1+2*((p[106+j/2]>>(4*(j%2)))&15),h[j]=0;
            if constexpr(Type==23) l[j]=(((p[4+j/2]>>(4*(j%2)))&15)|(((stockdecode::u16(p+2)>>(2*j))&3)<<4))-32,h[j]=0;
        }
        low=_mm256_load_si256((const __m256i*)l);high=_mm256_load_si256((const __m256i*)h);
    }
    __attribute__((always_inline)) __m256i dot(const int8_t* codes) const {
        __m256i partial[8];
        for(int j=0;j<8;++j) {
            auto x=_mm256_sign_epi8(_mm256_load_si256((const __m256i*)(codes+32*j)),sign[j]);
            if constexpr(Type==23)partial[j]=_mm256_madd_epi16(_mm256_maddubs_epi16(g[j],x),_mm256_set1_epi16(1));
            else partial[j]=_mm256_maddubs_epi16(g[j],x);
        }
        __m256i lo,hi;if constexpr(Type==23)halves(partial,lo,hi);else halves16(partial,lo,hi);auto sum=_mm256_add_epi32(lo,hi);
        if constexpr(Type==18) return truncdiv<2>(_mm256_mullo_epi32(sum,low)); // grid even => sum even
        if constexpr(Type==22){
            auto odd=_mm256_and_si256(sum,_mm256_set1_epi32(1));
            auto correction=_mm256_sign_epi32(odd,sum);
            return truncdiv<3>(_mm256_sub_epi32(_mm256_add_epi32(_mm256_mullo_epi32(lo,low),_mm256_mullo_epi32(hi,high)),correction));
        }
        return _mm256_mullo_epi32(sum,low);
    }
};
template<int Type,int NT,bool Ordered> void dot(const uint8_t* row,const InputView* x,float* result) {
    __m256 acc[NT][4];for(int t=0;t<NT;++t)for(auto& a:acc[t])a=_mm256_setzero_ps();
    for(int b=0;b<10;++b) {
        Block<Type> block(row+b*stockdecode::Fmt32<Type>::bytes);
        for(int t=0;t<NT;++t) {
            auto sums=block.dot(x[t].code.data()+256*b);
            auto ds=_mm256_mul_ps(_mm256_set1_ps(block.d),_mm256_load_ps(x[t].scales.data()+8*b));
            const int lane=Ordered?b%4:0;
            acc[t][lane]=_mm256_fmadd_ps(ds,_mm256_cvtepi32_ps(sums),acc[t][lane]);
        }
    }
    for(int t=0;t<NT;++t) {
        auto v=Ordered?_mm256_add_ps(_mm256_add_ps(acc[t][0],acc[t][2]),_mm256_add_ps(acc[t][1],acc[t][3])):acc[t][0];
        result[t]=A::reduce8(v);
    }
}
template<int Type,int NT,bool Ordered> void unfused_rows(const C::NativeFmt& f,const uint8_t* blob,const InputView* x,float* const* out) {
    const bool use_exp2=strata::kernels::cpu::gpu_silu_exp2_detail::enabled();
    for(int r=0;r<640;++r) {
        float g[NT],u[NT];dot<Type,NT,Ordered>(blob+r*f.gu_row,x,g);dot<Type,NT,Ordered>(blob+f.up_off+r*f.gu_row,x,u);
        for(int t=0;t<NT;++t)out[t][r]=strata::kernels::cpu::gpu_silu_exp2_detail::apply(use_exp2,g[t],u[t]);
    }
}
template<int N> inline bool narrow22_codes(const int8_t* codes){
    static_assert(N>0 && N%32==0);
    const auto zero=_mm256_setzero_si256(),limit=_mm256_set1_epi64x(762);
    for(int k=0;k<N;k+=32){
        const auto raw=_mm256_loadu_si256((const __m256i*)(codes+k));
        // abs(-128) has unsigned byte value128: sad_epu8 handles it correctly.
        const auto norm=_mm256_sad_epu8(_mm256_abs_epi8(raw),zero);
        if(_mm256_movemask_epi8(_mm256_cmpgt_epi64(norm,limit)))return false;
    }
    return true;
}
template<int NT,class Input> inline bool can_narrow22(const Input* x){
    for(int t=0;t<NT;++t){
        if constexpr(requires { x[t].narrow22; }){
            if(!x[t].narrow22)return false;
        }else if(!narrow22_codes<2560>(x[t].code.data()))return false;
    }
    return true;
}
template<int Type,bool Narrow22=false> inline __m256i pair_sum(__m256i a,__m256i b){
    if constexpr(Type==23)return _mm256_hadd_epi32(a,b);
    else if constexpr(Type==18||Type==21||(Type==22&&Narrow22))return _mm256_hadd_epi16(a,b);
    else return _mm256_madd_epi16(_mm256_hadd_epi16(a,b),_mm256_set1_epi16(1));
}
template<int Type,bool Narrow22=false> inline __m256i quartet_sum(__m256i a,__m256i b){
    if constexpr(Type==18||Type==21||(Type==22&&Narrow22))return _mm256_madd_epi16(_mm256_hadd_epi16(a,b),_mm256_set1_epi16(1));
    else return _mm256_hadd_epi32(a,b);
}
inline __m256i sign18(uint32_t w){
    const auto x=_mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32(int(w)),_mm_setr_epi32(0,7,14,21)),_mm_set1_epi32(127));
    // Low nibble parity of x^(x>>4) equals parity of all seven index bits.
    // Upper bytes in each dword stay zero (table[0]==0).
    const auto nibble=_mm_and_si128(_mm_xor_si128(x,_mm_srli_epi32(x,4)),_mm_set1_epi32(15));
    const auto parity=_mm_shuffle_epi8(_mm_setr_epi8(0,-128,-128,0,-128,0,0,-128,-128,0,0,-128,0,-128,-128,0),nibble);
    const auto masks=_mm256_cvtepu32_epi64(_mm_or_si128(x,parity));
    const auto bytes=_mm256_shuffle_epi8(masks,_mm256_setr_epi8(0,0,0,0,0,0,0,0,8,8,8,8,8,8,8,8,0,0,0,0,0,0,0,0,8,8,8,8,8,8,8,8));
    const auto selectors=_mm256_set1_epi64x(0x8040201008040201ULL);
    return _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(bytes,selectors),selectors),_mm256_set1_epi8(1));
}
template<int Type> inline void decode_group(const uint8_t* p,int j,int half,__m256i& g,__m256i& sign,__m256i& sc){
    if constexpr(Type==18){
        const uint8_t* q=p+2+16*j+8*half;
        g=_mm256_set_epi32(half_iq3xxs_grid[q[7]],half_iq3xxs_grid[q[6]],half_iq3xxs_grid[q[5]],half_iq3xxs_grid[q[4]],
                          half_iq3xxs_grid[q[3]],half_iq3xxs_grid[q[2]],half_iq3xxs_grid[q[1]],half_iq3xxs_grid[q[0]]);
        const auto w=stockdecode::u32(p+66+8*j+4*half);
        sign=sign18(w);sc=stockdecode::sc32(2*int(w>>28)+1);
    }else if constexpr(Type==22){
        const int index=2*j+half;
        const auto low=_mm_cvtsi32_si128(int32_t(stockdecode::u32(p+2+4*index)));
        const auto high=_mm_cvtsi64_si128(int64_t(stockdecode::hi_spread.iq2s[p[66+index]]));
        // Transfer all four indices together, then extract on the integer side.
        const uint64_t grid_ids=uint64_t(_mm_cvtsi128_si64(_mm_unpacklo_epi8(low,high)));
        g=_mm256_set_epi64x(int64_t(iq2s_grid[grid_ids>>48]),int64_t(iq2s_grid[uint16_t(grid_ids>>32)]),
                           int64_t(iq2s_grid[uint16_t(grid_ids>>16)]),int64_t(iq2s_grid[uint16_t(grid_ids)]));
        sign=stockdecode::sgn_vec_at(p+34+4*index);
        sc=stockdecode::scale_vec(stockdecode::scale_vecs.s16[p[74+index]]);
    }else stockdecode::Fmt32<Type>::decode(p,j,half,g,sign,sc);
}
template<int Type> struct Coefficients {
    __m256i low,high,first_scales,last_scales;float d;
    __attribute__((always_inline)) explicit Coefficients(const uint8_t* p):d(A::h2f(stockdecode::u16(p))){
        const auto one=_mm256_set1_epi32(1),mask=_mm256_set1_epi32(15);
        high=_mm256_setzero_si256();
        if constexpr(Type==18){
            auto v=_mm256_loadu_si256((const __m256i*)(p+66));
            low=_mm256_or_si256(_mm256_slli_epi32(_mm256_srli_epi32(v,28),1),one);
            const auto paired=_mm256_or_si256(low,_mm256_slli_epi32(low,16));
            first_scales=_mm256_permute2x128_si256(paired,paired,0x00);
            last_scales=_mm256_permute2x128_si256(paired,paired,0x11);
        }
        if constexpr(Type==22){
            auto v=_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(p+74)));
            low=_mm256_or_si256(_mm256_slli_epi32(_mm256_and_si256(v,mask),1),one);
            high=_mm256_or_si256(_mm256_slli_epi32(_mm256_srli_epi32(v,4),1),one);
        }
        if constexpr(Type==21||Type==23){
            const auto s=_mm_cvtsi32_si128(stockdecode::u32(p+(Type==21?106:4)));
            auto v=_mm256_cvtepu8_epi32(_mm_unpacklo_epi8(s,s));
            auto nibbles=_mm256_and_si256(_mm256_blend_epi32(v,_mm256_srli_epi32(v,4),0xaa),mask);
            if constexpr(Type==21)low=_mm256_or_si256(_mm256_slli_epi32(nibbles,1),one);
            else {
                auto upper=_mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(stockdecode::u16(p+2)),
                    _mm256_setr_epi32(0,2,4,6,8,10,12,14)),_mm256_set1_epi32(3));
                low=_mm256_sub_epi32(_mm256_or_si256(nibbles,_mm256_slli_epi32(upper,4)),_mm256_set1_epi32(32));
            }
        }
    }
    __attribute__((always_inline)) __m256i finish(const __m256i* partial) const {
        __m256i lo,hi;if constexpr(Type==23)halves(partial,lo,hi);else halves16(partial,lo,hi);
        return finish_halves<false>(lo,hi);
    }
    template<bool Narrow22>
    __attribute__((always_inline)) __m256i weighted_quartet(__m256i a,__m256i b,bool last) const {
        if constexpr(Type==18){
            // Each int16 lane is an eight-code partial sum and is bounded
            // by 8*31*127. The duplicated scale multiplies and widens two
            // such lanes into a single int32, with no narrowing of their sum.
            const auto scales=last ? last_scales : first_scales;
            return _mm256_madd_epi16(_mm256_hadd_epi16(a,b),scales);
        }else{
            return quartet_sum<Type,Narrow22>(a,b);
        }
    }
    __attribute__((always_inline)) __m256i finish_packed(__m256i a,__m256i b) const {
        const auto lo=_mm256_permute2x128_si256(a,b,0x20);
        const auto hi=_mm256_permute2x128_si256(a,b,0x31);
        if constexpr(Type==18)return truncdiv<1>(_mm256_add_epi32(lo,hi));
        else return finish_halves<true>(lo,hi);
    }
    template<bool Packed> __attribute__((always_inline)) __m256i finish_halves(__m256i lo,__m256i hi) const {
        auto sum=_mm256_add_epi32(lo,hi);
        // Fused IQ3_XXS codes are exactly half the even source grid. The
        // original trunc((2*l+1)*S/4) becomes trunc((2*l+1)*(S/2)/2).
        if constexpr(Type==18)return truncdiv<Packed?1:2>(_mm256_mullo_epi32(sum,low));
        if constexpr(Type==22){auto correction=_mm256_sign_epi32(_mm256_and_si256(sum,_mm256_set1_epi32(1)),sum);return truncdiv<3>(_mm256_sub_epi32(_mm256_add_epi32(_mm256_mullo_epi32(lo,low),_mm256_mullo_epi32(hi,high)),correction));}
        return _mm256_mullo_epi32(sum,low);
    }
};
// Singleton specialization: only four FP32 streams are live per projection.
// Gate output is a roundedFP32 temporary in the requested output rows. Keep
// the multi-token018 fused algorithm unchanged;019 two-pass wide loop did
// not improve the fixed smoke and is retained as a separate failed direction.
template<int Type,bool Ordered,bool Narrow22,bool Up>
__attribute__((noinline)) void projection1_rows(const C::NativeFmt& f,const uint8_t* blob,const InputView* x,float* const* out,int r0,int r1){
    const bool use_exp2=Up&&r0<r1&&strata::kernels::cpu::gpu_silu_exp2_detail::enabled();
    const auto* base=blob+(Up?f.up_off:0);
    for(int r=r0;r<r1;++r){
        __m256 acc[4];for(auto& a:acc)a=_mm256_setzero_ps();
        for(int b=0;b<10;++b){
            const auto* weight=base+r*f.gu_row+b*stockdecode::Fmt32<Type>::bytes;
            const Coefficients<Type> scale(weight);__m256i quad[2];
            for(int q=0;q<2;++q){
                __m256i pairs[2];
#pragma GCC unroll 2
                for(int pair=0;pair<2;++pair){
                    __m256i first;
#pragma GCC unroll 2
                    for(int which=0;which<2;++which){
                        const int j=4*q+2*pair+which;__m256i grid,sign,unused;
                        decode_group<Type>(weight,j/2,j%2,grid,sign,unused);
                        auto raw=_mm256_load_si256((const __m256i*)(x[0].code.data()+256*b+32*j));
                        auto value=_mm256_maddubs_epi16(grid,_mm256_sign_epi8(raw,sign));
                        if constexpr(Type==23)value=_mm256_madd_epi16(value,_mm256_set1_epi16(1));
                        if(which==0)first=value;else pairs[pair]=pair_sum<Type,Narrow22>(first,value);
                    }
                }
                quad[q]=scale.template weighted_quartet<Narrow22>(pairs[0],pairs[1],q==1);
            }
            const int k=Ordered?b%4:0;auto dx=_mm256_load_ps(x[0].scales.data()+8*b);
            acc[k]=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_set1_ps(scale.d),dx),_mm256_cvtepi32_ps(scale.finish_packed(quad[0],quad[1])),acc[k]);
        }
        const float value=A::reduce8(Ordered?_mm256_add_ps(_mm256_add_ps(acc[0],acc[2]),_mm256_add_ps(acc[1],acc[3])):acc[0]);
        if constexpr(Up){const float g=out[0][r];out[0][r]=strata::kernels::cpu::gpu_silu_exp2_detail::apply(use_exp2,g,value);}else out[0][r]=value;
    }
}
// Share each raw activation load between gate/up. Keep only four decoded
// grid/sign vectors live; no decoded-weight cache, no FP64 or approximation.
template<int Type,int NT,bool Ordered,bool Narrow22> void rows_impl(const C::NativeFmt& f,const uint8_t* blob,const InputView* x,float* const* out,int r0,int r1){
    if constexpr(NT==1){
        projection1_rows<Type,Ordered,Narrow22,false>(f,blob,x,out,r0,r1);
        projection1_rows<Type,Ordered,Narrow22,true>(f,blob,x,out,r0,r1);return;
    }
    const bool use_exp2=r0<r1&&strata::kernels::cpu::gpu_silu_exp2_detail::enabled();
    for(int r=r0;r<r1;++r){
        __m256 ga[NT][4],ua[NT][4];for(int t=0;t<NT;++t)for(int k=0;k<4;++k)ga[t][k]=ua[t][k]=_mm256_setzero_ps();
        for(int b=0;b<10;++b){
            const auto* gb=blob+r*f.gu_row+b*stockdecode::Fmt32<Type>::bytes;
            const auto* ub=blob+f.up_off+r*f.gu_row+b*stockdecode::Fmt32<Type>::bytes;
            const Coefficients<Type> gs(gb),us(ub);
            __m256i gquad[NT][2],uquad[NT][2];
            for(int q=0;q<2;++q){
                __m256i gp[NT][2],up[NT][2];
#pragma GCC unroll 2
                for(int pair=0;pair<2;++pair){
                    __m256i gfirst[NT],ufirst[NT];
#pragma GCC unroll 2
                    for(int which=0;which<2;++which){
                        const int j=4*q+2*pair+which;
                        __m256i g,sg,u,su,unused;
                        decode_group<Type>(gb,j/2,j%2,g,sg,unused);
                        decode_group<Type>(ub,j/2,j%2,u,su,unused);
                        for(int t=0;t<NT;++t){
                            auto raw=_mm256_load_si256((const __m256i*)(x[t].code.data()+256*b+32*j));
                            auto gv=_mm256_maddubs_epi16(g,_mm256_sign_epi8(raw,sg));
                            auto uv=_mm256_maddubs_epi16(u,_mm256_sign_epi8(raw,su));
                            if constexpr(Type==23){gv=_mm256_madd_epi16(gv,_mm256_set1_epi16(1));uv=_mm256_madd_epi16(uv,_mm256_set1_epi16(1));}
                            if(which==0){gfirst[t]=gv;ufirst[t]=uv;}
                            else{gp[t][pair]=pair_sum<Type,Narrow22>(gfirst[t],gv);up[t][pair]=pair_sum<Type,Narrow22>(ufirst[t],uv);}
                        }
                    }
                }
                for(int t=0;t<NT;++t){gquad[t][q]=gs.template weighted_quartet<Narrow22>(gp[t][0],gp[t][1],q==1);uquad[t][q]=us.template weighted_quartet<Narrow22>(up[t][0],up[t][1],q==1);}
            }
            for(int t=0;t<NT;++t){
                auto dx=_mm256_load_ps(x[t].scales.data()+8*b);const int k=Ordered?b%4:0;
                ga[t][k]=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_set1_ps(gs.d),dx),_mm256_cvtepi32_ps(gs.finish_packed(gquad[t][0],gquad[t][1])),ga[t][k]);
                ua[t][k]=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_set1_ps(us.d),dx),_mm256_cvtepi32_ps(us.finish_packed(uquad[t][0],uquad[t][1])),ua[t][k]);
            }
        }
        for(int t=0;t<NT;++t){
            float g=A::reduce8(Ordered?_mm256_add_ps(_mm256_add_ps(ga[t][0],ga[t][2]),_mm256_add_ps(ga[t][1],ga[t][3])):ga[t][0]);
            float u=A::reduce8(Ordered?_mm256_add_ps(_mm256_add_ps(ua[t][0],ua[t][2]),_mm256_add_ps(ua[t][1],ua[t][3])):ua[t][0]);
            out[t][r]=strata::kernels::cpu::gpu_silu_exp2_detail::apply(use_exp2,g,u);
        }
    }
}
template<int Type,int NT,bool Ordered> void rows(const C::NativeFmt& f,const uint8_t* blob,const InputView* x,float* const* out,int r0=0,int r1=640){
    if constexpr(Type==22){
        // For every8-code group: |dot|<=43*L1<=43*762=32766.
        // Preparation is percall/timed, not a persistent or weight cache.
        if(can_narrow22<NT>(x)){rows_impl<Type,NT,Ordered,true>(f,blob,x,out,r0,r1);return;}
    }
    rows_impl<Type,NT,Ordered,false>(f,blob,x,out,r0,r1);
}
template<int NT,bool Ordered> void dispatch(const C::NativeFmt& f,const uint8_t* b,const InputView* x,float* const* out) {
    switch(f.gu_type) {
        case 18:if constexpr(NT==1)unfused_rows<18,NT,Ordered>(f,b,x,out);else rows<18,NT,Ordered>(f,b,x,out);return;
        case 21:rows<21,NT,Ordered>(f,b,x,out);return;
        case 22:rows<22,NT,Ordered>(f,b,x,out);return;
        case 23:if constexpr(NT==1)unfused_rows<23,NT,Ordered>(f,b,x,out);else rows<23,NT,Ordered>(f,b,x,out);return;
    }
    throw std::runtime_error("exact8 unsupported type");
}
}
