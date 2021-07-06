/**
 * Source for function definitions that are meant to train a NN
 * with the purpose of solving DE’s closely resembling an MD system
 */

#include <iostream>
#include <fstream>
#include <numeric>
#include <vector>
#include <random>
#include "pnd.hpp"
#include "mpi.h"
#include <chrono>
#include <set>

using namespace std;

/**
 * Constructor definition - prompts objects to read input file
 * which specifies the number of time grids to predict,
 * epochs for training and other network properties such as
 * number of nodes.
 *
 * Default input file /Source/pnd.in
 */
PND::PND() {
    ifstream ifs;
    ifs.open("./Source/pnd.in", ifstream::in);
    if (!ifs.is_open()) {
        cerr << "failed to open pinny input file" << endl;
        terminate();
    }

    ifs >> StepTrain;
    ifs >> PreTrainEpochs >> MainTrainEpochs;
    ifs >> nodes;

    ifs.close();

    cout << "PND: Successfully completed reading input file" << endl;
}

/**
 * Method to define the number of atoms that the system comprises of. Used while reading the atom
 * trajectories from the simulation data
 *
 * @param numberOfAtoms This is the number of atoms in the molecular
 *                      dynamics simulation
 */
void PND::defineParams(int numberOfAtoms) {
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

/**
 * Define a Loss function for the network and before starting the main training task. Method can be overridden when class
 * inherits from PND
 *
 * @param t_seq time sequences for which the pre-training is carried out
 * @param icfs initial and final positions and velocites that are to be used for the pre-training task
 * @param n number of neurons in the NN
 * @param Np number of atoms in the MD system
 * @param d dimensionality of the space (d = 3 when data in x,y,z co-ordinates is available)
 * @return returns the the tuple containing sum of gradients of outputs with respect to the input as specified in the DE as well and the root mean square(RMS) loss value
 */
std::pair<torch::Tensor, torch::Tensor> PND::LossPreTrain(torch::Tensor t_seq,
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

/**
 * Define a Loss function for the network which is a set of DEs governing the evolution of the system.
 * Only method declaration provided, method must be defined when user's working class
 * inherits from PND
 *
 * @param t_seq time sequences for which the pre-training is carried out
 * @param icfs initial and final positions and velocites that are to be used for the pre-training task
 * @param totalEnergy system total energy across time steps
 * @param kineticEnergy system kinetic energy across time steps
 * @param potentialEnergy system potential energy across time steps
 * @param n Number of neurons in the NN
 * @param Np Number of atoms in the MD system
 * @param d dimensionality of the space (d = 3 when data in x,y,z co-ordinates is available)
 * @return returns a tuple of sum of gradients, mean squared error(MSE) loss, boundary loss, Least Action Loss, Hamiltonian Loss, momentum Loss
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> PND::Loss(
        torch::Tensor t_seq, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> icfs,
        torch::Tensor totalEnergy, torch::Tensor kineticEnergy, torch::Tensor potentialEnergy, int n, int Np, int d) {}

/** Defines an optimizer for adjusting the parameters in each pre-training iteration
 *
 * @param t_seq time sequences for which the pre-training is carried out
 * @param icfs initial and final positions and velocites that are to be used for the pre-training task
 * @param velocities
 * @param S
 * @param epoch defines the number of steps
 * @param n number of neurons in the NN
 * @param Np number of atoms in the MD system
 * @param d dimensionality of the space (d = 3 when data in x,y,z co-ordinates is available)
 * @param alpha a parameter for optimizer
 * @param epsilon constant for stability
 * @param beta tuple with values for the exponential decay rate for the 1st moment estimates and exponential decay rate for the exponentially weighted infinity norm
 * @return tuple of optimized parameters, optimizer values and RMS loss value
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> PND::UpdatePreParamsNADAM(torch::Tensor t_seq,
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

/**
 * Routine to follow for pre-training the NN, i.e., training with a data set.
 * Method definition trains the NN over a number of epochs with the UpdatePreParams(...) optimizer
 *
 * @param t_seq time sequences for which the pre-training is carried out
 * @param icfs initial and final positions and velocites that are to be used for the pre-training task
 * @param num_epochs number of epochs for which pre-trainig is to be carried out
 * @param n number of neurons in the NN
 * @param Np number of atoms in the MD system
 * @param d dimensionality of the space (d = 3 when data in x,y,z co-ordinates is available)
 * @param learn_rate parameter to pass to the optimizer
 * @param momentum parameter to pass to the optimizer that uses momentum
 * @return the tuple of trained weights and mean loss value at the end of training
 */
std::pair<torch::Tensor, torch::Tensor> PND::PreTrain(torch::Tensor t_seq,
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

/**
 * Defines an optimizer for adjusting the parameters in each main training iteration.
 * We provide the declaration and method must be defined when user's working class
 * inherits from PND
 *
 * @param t_seq time sequences for which the pre-training is carried out
 * @param icfs initial and final positions and velocites that are to be used for the pre-training task
 * @param velocities
 * @param S
 * @param totalEnergy system total energy across time steps
 * @param kineticEnergy system kinetic energy across time steps
 * @param potentialEnergy system potential energy across time steps
 * @param epoch defines the number of steps
 * @param n number of neurons in the NN
 * @param Np number of atoms in the MD system
 * @param d dimensionality of the space (d = 3 when data in x,y,z co-ordinates is available)
 * @param alpha parameters for optimizers
 * @param epsilon constant for stability
 * @param beta parameters for optimizers
 * @return the tuple of trained weights and mean loss value at the end of training
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
PND::UpdateParamsNADAM(torch::Tensor t_seq,
                       std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> icfs,
                       torch::Tensor velocities, torch::Tensor S, torch::Tensor totalEnergy, torch::Tensor kineticEnergy,
                       torch::Tensor potentialEnergy, int epoch, int n, int Np,
                       int d, double alpha, double epsilon, torch::Tensor beta) {}

/**
 * Routine to follow for main-training the NN, i.e., training for the target data.
 * Users may decide to call their parameter updating algorithm implementation in each training iteration.
 *
 *
 * @param params inital parameters of the NN which will be trained
 * @param t_seq time sequences for which the pre-training is carried out
 * @param icfs initial and final positions and velocites that are to be used for the pre-training task
 * @param num_epochs number of epochs for which training is to be carried out
 * @param totalEnergy system total energy across time steps
 * @param kineticEnergy system kinetic energy across time steps
 * @param potentialEnergy system potential energy across time steps
 * @param n number of neurons in the NN
 * @param Np number of atoms in the MD system
 * @param d number of atoms in the MD system
 * @param learn_rate learning rate for optimizer
 * @param momentum parameter to pass to the optimizer that uses momentum
 * @return the tuple of trained weights and mean loss value at the end of training
 */
std::pair<torch::Tensor, torch::Tensor>
PND::mainTrain(torch::Tensor params, torch::Tensor t_seq,
               std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> icfs, int num_epochs,
               torch::Tensor totalEnergy, torch::Tensor kineticEnergy, torch::Tensor potentialEnergy,
               int n, int Np, int d, double learn_rate, double momentum) {
}



