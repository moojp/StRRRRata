// Separate TU: keep the qualified GU object's instructions/layout unchanged.
#include "strata/kernels/cpu/q81_gu_align.hpp"
#include "q81_gu_private/exact/q2_down_scales.hpp"
namespace strata::kernels::cpu {
void q81_q2_down_scales(ActQ& a){q81_q2_detail::align_scales(a);}
}
