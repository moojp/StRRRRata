// Research-only correction of the Q2_0 down activation's scale precision.
// Preserve codes, row kernels, allocation and integer/float reduction order.
#pragma once
#include "strata/kernels/cpu/expert.hpp"
#include <immintrin.h>
#include <algorithm>
#include <stdexcept>
namespace q81_q2_detail {
inline void align_scales(strata::kernels::cpu::ActQ& a){
    if(a.nchunks!=20)throw std::runtime_error("Prototype expects640 down values");
    for(int k=0;k<a.nchunks;++k){
        a.scale[k]=_cvtsh_ss(_cvtss_sh(std::min(a.scale[k],65504.f),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC));
        a.hx[k]=a.scale[k]*float(a.sum[k]);
    }
    if(a.bp_pairs){
        if(a.bp_pairs!=5)throw std::runtime_error("Unexpected bit-plane geometry");
        for(int p=0;p<a.bp_pairs;++p)for(int c=0;c<4;++c)
            a.pscale[8*p+2*c]=a.pscale[8*p+2*c+1]=a.scale[4*p+c];
    }
}
}
