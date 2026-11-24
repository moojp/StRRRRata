// Default-OFF CPU GPU-order down implementation. Qualified GU object reused.
#include "strata/kernels/cpu/q81_gu_align.hpp"
#include "q81_gu_private/exact/down_pair_scales.hpp"
#include <new>
#include <type_traits>
namespace strata::kernels::cpu {
namespace {
namespace D=down_gpu_tree_proto;namespace M=down_pair_scales_proto;
bool eligible(const NativeFmt& f,int nt){
    return nt>=1&&nt<=8&&f.d_type==20&&f.n_embd==2560&&f.n_ff==640&&!f.tail_off&&!f.had_d;
}
void range(int begin,int end){if(begin<0||end<begin||end>2560)throw std::runtime_error("GPU-order down row range");}
template<int NT>bool construct(const void* const* input,Q81DownPrepared& out){
    using T=std::conditional_t<NT==1,D::Prepared,M::Prepared<NT>>;
    static_assert(sizeof(T)<=sizeof(out.storage)&&alignof(T)<=alignof(Q81DownPrepared));
    static_assert(std::is_trivially_destructible_v<T>);
    try{
        if constexpr(NT==1)::new(static_cast<void*>(out.storage)) T(input[0]);
        else ::new(static_cast<void*>(out.storage)) T(input);
    }catch(const std::runtime_error&){return false;}
    out.nt=-NT;out.ready=true;return true;
}
template<int NT>void consume(const NativeFmt& f,const uint8_t* blob,const Q81DownPrepared& p,float* const* out,int begin,int end){
    using T=std::conditional_t<NT==1,D::Prepared,M::Prepared<NT>>;
    const auto& x=*std::launder(reinterpret_cast<const T*>(p.storage));
    if constexpr(NT==1)D::rows(blob+f.down_off,f.d_row,x,out[0],begin,end);
    else if constexpr(NT==2||NT==3)M::pair_rows<NT>(blob+f.down_off,f.d_row,x,out,begin,end);
    else M::rows<NT>(blob+f.down_off,f.d_row,x,out,begin,end);
}
}
bool q81_gpu_down_prepare(const NativeFmt& f,const void* const* input,int nt,Q81DownPrepared& out){
    out.ready=false;out.nt=0;if(!eligible(f,nt))return false;
    switch(nt){
        case 1:return construct<1>(input,out);case 2:return construct<2>(input,out);
        case 3:return construct<3>(input,out);case 4:return construct<4>(input,out);
        case 5:return construct<5>(input,out);case 6:return construct<6>(input,out);
        case 7:return construct<7>(input,out);case 8:return construct<8>(input,out);
    }return false;
}
bool q81_gpu_down_prepared_rows(const NativeFmt& f,const uint8_t* blob,const Q81DownPrepared& p,int nt,float* const* out,int begin,int end){
    if(!p.ready||!eligible(f,nt)||p.nt!=-nt)return false;range(begin,end);if(begin==end)return true;
    switch(nt){
        case 1:consume<1>(f,blob,p,out,begin,end);break;case 2:consume<2>(f,blob,p,out,begin,end);break;
        case 3:consume<3>(f,blob,p,out,begin,end);break;case 4:consume<4>(f,blob,p,out,begin,end);break;
        case 5:consume<5>(f,blob,p,out,begin,end);break;case 6:consume<6>(f,blob,p,out,begin,end);break;
        case 7:consume<7>(f,blob,p,out,begin,end);break;case 8:consume<8>(f,blob,p,out,begin,end);break;
    }return true;
}
bool q81_gpu_down_rows(const NativeFmt& f,const uint8_t* blob,const void* const* input,int nt,float* const* out,int begin,int end){
    if(!eligible(f,nt))return false;range(begin,end);if(begin==end)return true;
    Q81DownPrepared prepared;
    if(!q81_gpu_down_prepare(f,input,nt,prepared))return false;
    return q81_gpu_down_prepared_rows(f,blob,prepared,nt,out,begin,end);
}
}
