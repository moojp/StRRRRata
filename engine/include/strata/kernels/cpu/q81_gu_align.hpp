// Experimental, opt-in GPU-style block32 activation contract for IQ gate/up.
// Public native signatures and defaultOFF arithmetic remain unchanged.
#pragma once
#include "strata/kernels/cpu/native_expert.hpp"
namespace strata::kernels::cpu {
struct ActQ;
//640-value Q2_0 down image only; caller checks opt-in and AVX2/F16C.
void q81_q2_down_scales(ActQ& a);
// These functions are AVX2/FMA/F16C code. Call only after a runtime ISA check.
void q81_gu_quant(const NativeFmt& f,const float* x,void* dst);
void q81_gu_rows(const NativeFmt& f,const uint8_t* blob,const void* const* act,int nt,float* const* ff,int r0,int r1);
// Caller must check AVX2/FMA/F16C and opt-in first. False means use the
// original whole-call down path; in particular never split a large tail
// through the different two-stream singleton arithmetic.
bool q81_down_reuse_rows(const NativeFmt& f,const uint8_t* blob,const void* const* hq,
                        int nt,float* const* out,int r0,int r1);

// Pool-owned immutable snapshot, not an address-keyed or weight cache. Each
// successful hidden quantization must replace it before a new batch publishes.
// Its owner retires/reuses it only after every consumer has completed+parked.
struct alignas(32) Q81DownPrepared {
    alignas(32) unsigned char storage[5760];
    int nt=0;
    bool ready=false;
};
bool native_down_batch_reuse_enabled();
bool native_prepare_down_batch(const NativeFmt& f,const void* const* hq,int nt,
                               NativeArithmetic arithmetic,Q81DownPrepared& prepared);
// Raw helpers require the existing AVX2/FMA/F16C runtime/opt-in checks above.
bool q81_down_prepare(const NativeFmt& f,const void* const* hq,int nt,Q81DownPrepared& prepared);
bool q81_down_prepared_rows(const NativeFmt& f,const uint8_t* blob,
                           const Q81DownPrepared& prepared,int nt,float* const* out,int r0,int r1);
// New GPU-order snapshots use nt=-width as a type tag. Legacy readers reject
// this tag before touching storage; only the producing TU reads the new type.
bool native_down_gpu_order_enabled();
bool q81_gpu_down_prepare(const NativeFmt&,const void* const*,int,Q81DownPrepared&);
bool q81_gpu_down_rows(const NativeFmt&,const uint8_t*,const void* const*,int,float* const*,int,int);
bool q81_gpu_down_prepared_rows(const NativeFmt&,const uint8_t*,const Q81DownPrepared&,int,float* const*,int,int);
bool native_down_prepared_rows(const NativeFmt&,const uint8_t*,const Q81DownPrepared&,int,float* const*,int,int);
// Opted-in exact GU only; NT2/3 optimized, other widths keep the qualified TU.
bool q81_gu_consume_rows(const NativeFmt&,const uint8_t*,const void* const*,int,float* const*,int,int);
}
