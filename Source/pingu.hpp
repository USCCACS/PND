//
// Created by USCCACS02 on 1/12/20.
//

#ifndef DSN_MD_PINGU_HPP
#define DSN_MD_PINGU_HPP

#include <torch/torch.h>

class Pingu {
public:
    int PreTrainEpochs, MainTrainEpochs, StepTrain, nodes;
    torch::Tensor params;

    Pingu();
    void defineParams(int numberOfAtoms);
    std::pair<torch::Tensor, torch::Tensor> LossPreTrain(torch::Tensor t_seq,
                                                         std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> icfs,
                                                         int n , int Np, int d);

    virtual std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> Loss(torch::Tensor t_seq,
            std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> icfs,
            torch::Tensor totalEnergy, int n, int Np, int d);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> UpdatePreParamsNADAM(torch::Tensor t_seq,
            torch::Tensor params, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> icfs,
            torch::Tensor velocities,
            torch::Tensor S, int epoch, int n, int Np, int d, double alpha = 0.001,
            double epsilon = pow(10,-7), torch::Tensor beta = torch::tensor({0.999, 0.999}));
    std::pair<torch::Tensor, torch::Tensor> PreTrain(torch::Tensor params, torch::Tensor t_seq,
            std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> icfs,
            int num_epochs, int n, int Np, int d, double learn_rate = 0.001, double momentum = 0.99);
};


//torch::Tensor LJ3D_M(int stepsToState, torch::Tensor, int Np);
#endif //DSN_MD_PINGU_HPP

