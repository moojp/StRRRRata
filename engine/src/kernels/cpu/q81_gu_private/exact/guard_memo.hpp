// Only activation eligibility, never weights/expert selection. No copied codes.
#pragma once
#include <array>
#include <cstdint>
namespace guardmemo {
template<int NT,class Input,class Scan> inline bool eligible(const Input* x,Scan&& scan){
    static_assert(NT>0);
    // Producer contract: every quantization has a globally fresh, nonzero
    // stamp, and its wire remains immutable until all consumers finish.
    // This is the existing Q81 packed-batch cache contract, not a new one.
    struct Cache {
        std::array<uintptr_t,NT> address{};
        std::array<uint64_t,NT> generation{};
        bool valid=false,value=false;
    };
    static thread_local Cache cache;
    bool hit=cache.valid;
    for(int t=0;t<NT;++t)
        hit=hit&&cache.address[t]==reinterpret_cast<uintptr_t>(x[t].code.data())
               &&cache.generation[t]==x[t].generation;
    if(hit)return cache.value;
    const bool value=scan();
    for(int t=0;t<NT;++t){
        cache.address[t]=reinterpret_cast<uintptr_t>(x[t].code.data());
        cache.generation[t]=x[t].generation;
    }
    cache.value=value;cache.valid=true;return value;
}
}
