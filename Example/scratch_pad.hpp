//
// Created by USCCACS02 on 8/5/20.
//

#ifndef DSN_MD_SCRATCH_PAD_HPP
#define DSN_MD_SCRATCH_PAD_HPP

#include <torch/torch.h>
#include "../MD_Engine/pmd.cpp"

double ComputeAccel(SubSystem &);

std::tuple<float, torch::Tensor> ComputeAccelPredicted(SubSystem &);

void SingleStep(SubSystem &subsystem, bool shouldTrack);

///
/// Similar to LJ2D_M function but works in a sing;e time splice to return the PE
std::tuple<float, torch::Tensor> LJ3D(SubSystem &, torch::Tensor, int Np);

///
/// Same as LJ2D_M but calls LJ3D_M in each time splice
std::tuple<torch::Tensor, torch::Tensor> LJ3D_M(SubSystem checkPointState, torch::Tensor, int Np);

#endif //DSN_MD_SCRATCH_PAD_HPP
