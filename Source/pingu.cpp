//
// Created by USCCACS02 on 1/12/20.
//

#include <iostream>
#include <fstream>      // std::fstream
#include <numeric>
#include <vector>
#include <random>
#include "pingu.hpp"
#include "mpi.h"
#include <chrono>
#include <set>

using namespace std;

Pingu::Pingu() {
    ifstream ifs;
    ifs.open("Source/pingu.in", ifstream::in);
    if (!ifs.is_open()) {
        cerr << "failed to open pinny input file" << endl;
        terminate();
    }

    ifs >> StepTrain;
    ifs >> PreTrainEpochs >> MainTrainEpochs;
    ifs >> nodes;

    ifs.close();

    cout << "Pingu: Successfully completed reading input file" << endl;
}

void Pingu::defineParams(int numberOfAtoms) {
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(0, 1);

    vector<float> random_normal_container;
    for (int i = 0; i < (2 * nodes + 2 * numberOfAtoms * 3 * nodes + 2 * 3 * numberOfAtoms); ++i) {
        float number = distribution(generator);
        random_normal_container.push_back(number);
    }
    params = 0.7 * torch::from_blob(&random_normal_container[0],
                                    {2 * nodes + 2 * numberOfAtoms * 3 * nodes + 2 * 3 * numberOfAtoms, 1});
}

std::pair<torch::Tensor, torch::Tensor> Pingu::LossPreTrain(torch::Tensor t_seq,
                                                            std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> icfs,
                                                            int n, int Np, int d) {
    params.set_requires_grad(true);

    //Beginning of QT part
    torch::Tensor w0 = params.narrow(0, 0, n);
    torch::Tensor b0 = params.narrow(0, n, n);
    torch::Tensor w1 = params.narrow(0, 2 * n, 2 * n * 3 * Np).reshape({2 * 3 * Np, n});
    torch::Tensor b1 = params.narrow(0, 2 * n + (2 * n * 3 * Np), 2 * 3 * Np);
    torch::Tensor tmp = torch::ger(torch::squeeze(w0), torch::squeeze(t_seq)) + b0;
    torch::Tensor tmp1 = torch::sigmoid(tmp);
    torch::Tensor q = torch::matmul(w1, tmp1) + b1;
    torch::Tensor qt = q.transpose(0, 1);

    torch::Tensor trackedAtomsPositions = qt.narrow(1, 0, 3*Np);
    torch::Tensor trackedAtomsVelocities = qt.narrow(1, 3 * Np, 3*Np);

    //loss from initial and final states
    torch::Tensor icfsLoss = (trackedAtomsPositions[0] - std::get<0>(icfs)).pow(2).sum();
    icfsLoss += (trackedAtomsPositions[StepTrain - 1] - std::get<1>(icfs)).pow(2).sum();
    icfsLoss += (trackedAtomsVelocities[0] - std::get<2>(icfs)).pow(2).sum();
    icfsLoss += (trackedAtomsVelocities[StepTrain - 1] - std::get<3>(icfs)).pow(2).sum();

    torch::Tensor momentumLoss = torch::mean(qt.narrow(1, 3 * Np, Np).sum(1).pow(2) +
                                             qt.narrow(1, (3 * Np) + Np, Np).sum(1).pow(2) +
                                             qt.narrow(1, (3 * Np) + 2 * Np, Np).sum(1).pow(2));

    torch::Tensor loss = icfsLoss + momentumLoss;

    loss.backward();
    torch::Tensor grads = params.grad();
    params.set_requires_grad(false);

    return std::make_pair(grads, loss);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> Pingu::Loss(
        torch::Tensor t_seq, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> icfs,
        torch::Tensor totalEnergy, torch::Tensor kineticEnergy, torch::Tensor potentialEnergy, int n, int Np, int d) {}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> Pingu::UpdatePreParamsNADAM(torch::Tensor t_seq,
                                                                                                   std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> icfs,
                                                                                                   torch::Tensor velocities,
                                                                                                   torch::Tensor S,
                                                                                                   int epoch, int n,
                                                                                                   int Np, int d,
                                                                                                   double alpha,
                                                                                                   double epsilon,
                                                                                                   torch::Tensor beta) {
    epoch += 1;
    torch::Tensor tmp_beta = torch::pow(beta, epoch);
    std::pair<torch::Tensor, torch::Tensor> fromLossPreTrain = LossPreTrain(t_seq, icfs, n, Np, d);
    torch::Tensor grads = fromLossPreTrain.first;
    torch::Tensor meanLoss = fromLossPreTrain.second;
    torch::Tensor v1 = beta[0].item<double>() * velocities;
    torch::Tensor v2 = (1 - beta[1].item<double>()) * grads;
    velocities = v1 + v2;
    torch::Tensor velocities_t = velocities / (1 - tmp_beta[0]);
    S = beta[1] * S + (1 - beta[1]) * (torch::pow(grads, 2));
    torch::Tensor S_t = alpha / (torch::sqrt(S / (1 - tmp_beta[1])) + epsilon);
    params = params - (S_t * (beta[0] * velocities_t + (1 - beta[0]) / (1 - tmp_beta[0]) * grads));

    return std::make_tuple(params, velocities, S, meanLoss);
}

std::pair<torch::Tensor, torch::Tensor> Pingu::PreTrain(torch::Tensor t_seq,
                                                        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> icfs,
                                                        int num_epochs, int n, int Np, int d, double learn_rate,
                                                        double momentum) {
    torch::Tensor velocities = torch::zeros(params.sizes());
    torch::Tensor S = torch::zeros(params.sizes());
    torch::Tensor meanLoss;
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        auto values = UpdatePreParamsNADAM(t_seq, icfs, velocities, S, epoch, n, Np, d);
        params = std::get<0>(values);
        velocities = std::get<1>(values);
        S = std::get<2>(values);
        meanLoss = std::get<3>(values);
    }
    return std::make_pair(params, meanLoss);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
Pingu::UpdateParamsNADAM(torch::Tensor t_seq,
                         std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> icfs,
                         torch::Tensor velocities, torch::Tensor S, torch::Tensor totalEnergy, torch::Tensor kineticEnergy,
                         torch::Tensor potentialEnergy, int epoch, int n, int Np,
                         int d, double alpha, double epsilon, torch::Tensor beta) {}

std::pair<torch::Tensor, torch::Tensor>
Pingu::mainTrain(torch::Tensor params, torch::Tensor t_seq,
                 std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> icfs, int num_epochs,
                 torch::Tensor totalEnergy, torch::Tensor kineticEnergy, torch::Tensor potentialEnergy,
                 int n, int Np, int d, double learn_rate, double momentum) {
}



