//
// Created by USCCACS02 on 8/5/20.
//

#ifndef DSN_MD_SCRATCH_PAD_HPP
#define DSN_MD_SCRATCH_PAD_HPP

#include <torch/torch.h>
#include "../MD_Engine/pmd.hpp"

double ComputeAccel(SubSystem&);
std::tuple<float, torch::Tensor> ComputeAccelPredicted(SubSystem&);
std::vector<int> SingleStep(SubSystem &subsystem);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> UpdateParamsNADAM(SubSystem checkPointState, torch::Tensor t_seq,
                                                                                         torch::Tensor params, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> icfs,
                                                                                         torch::Tensor velocities, torch::Tensor S, torch::Tensor totalEnergy,
                                                                                         int epoch, int n, int Np, int d, double alpha = 0.001, double epsilon = pow(10,-7),
                                                                                         torch::Tensor beta = torch::tensor({0.999, 0.999}));
std::pair<torch::Tensor, torch::Tensor> mainTrain(SubSystem checkPointState, torch::Tensor params, torch::Tensor t_seq,
                                                  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> icfs,
                                                  int num_epochs, torch::Tensor totalEnergy, int n, int Np, int d, double learn_rate = 0.0001, double momentum = 0.99);

// Similar to LJ2D_M function but works in a sing;e time splice to return the PE
std::tuple<float, torch::Tensor> LJ3D(SubSystem&, torch::Tensor, int Np);

// Same as LJ2D_M but calls LJ3D_M in each time splice
std::tuple<torch::Tensor, torch::Tensor> LJ3D_M(SubSystem checkPointState , torch::Tensor, int Np);

#endif //DSN_MD_SCRATCH_PAD_HPP
