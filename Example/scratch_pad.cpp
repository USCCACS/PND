/**
 * scratch_pad.cpp is an examlpe to demonstrates how
 * users implement loss functions, specify intial and boundary conditions,
 * and conservation laws for their systems
 */

#include "../Source/pnd.cpp"
#include "scratch_pad.hpp"


class ScratchPad : public PND {

public:
    SubSystem checkPointState;

    void setCheckPoint(SubSystem checkPointState) {
        this->checkPointState = checkPointState;
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    Loss(torch::Tensor t_seq, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> icfs,
         torch::Tensor H0, torch::Tensor kineticEnergy, torch::Tensor potentialEnergy, int n, int Np, int d) {
        params.set_requires_grad(true);

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

        //loss from intial and final states
        torch::Tensor icfsLoss = (trackedAtomsPositions[0] - std::get<0>(icfs)).pow(2).sum();
        icfsLoss += (trackedAtomsPositions[StepTrain - 1] - std::get<1>(icfs)).pow(2).sum();
        icfsLoss += (trackedAtomsVelocities[0] - std::get<2>(icfs)).pow(2).sum();
        icfsLoss += (trackedAtomsVelocities[StepTrain - 1] - std::get<3>(icfs)).pow(2).sum();

        auto PEs = LJ3D_M(checkPointState, qt.narrow(1, 0, d * Np), Np); // Send the initial X,Y,Z
        torch::Tensor PE = std::get<0>(PEs).div(Np);
        torch::Tensor maxPE = PE.abs().max();
        icfsLoss = torch::div(icfsLoss, maxPE);

        torch::Tensor KE = 0.5 * (qt.narrow(1, d * Np, d * Np).pow(2).sum(1)).div(Np).reshape_as(PE);
        torch::Tensor Hm = PE + KE;
        torch::Tensor forces = std::get<1>(PEs);

        torch::Tensor eq =
               torch::sum(forces.narrow(1, 0, Np)) +
               torch::sum(forces.narrow(1, Np, Np)) +
               torch::sum(forces.narrow(1, 2*Np, Np));

        torch::Tensor energyLoss = torch::mean(torch::pow(Hm - H0, 2));

        torch::Tensor momentumLoss = torch::mean(qt.narrow(1, 3 * Np, Np).sum(1).pow(2) +
                                                 qt.narrow(1, (3 * Np) + Np, Np).sum(1).pow(2) +
                                                 qt.narrow(1, (3 * Np) + 2 * Np, Np).sum(1).pow(2));

        torch::Tensor loss = 20*icfsLoss + 80*energyLoss + momentumLoss + eq;

        loss.backward();
        torch::Tensor grads = params.grad();
        params.set_requires_grad(false);

        return std::make_tuple(grads, loss, icfsLoss, eq, energyLoss, momentumLoss);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> UpdateParamsNADAM(torch::Tensor t_seq,
                                                                                             std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> icfs,
                                                                                             torch::Tensor velocities,
                                                                                             torch::Tensor S,
                                                                                             torch::Tensor totalEnergy, torch::Tensor kineticEnergy, torch::Tensor potentialEnergy,
                                                                                             int epoch, int n, int Np,
                                                                                             int d, double alpha,
                                                                                             double epsilon,
                                                                                             torch::Tensor beta) {

        epoch += 1;
        torch::Tensor tmp_beta = torch::pow(beta, epoch);
        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> fromTrain = Loss(
                t_seq, icfs, totalEnergy, kineticEnergy, potentialEnergy, n, Np, d);
        torch::Tensor grads = std::get<0>(fromTrain);
        torch::Tensor loss = std::get<1>(fromTrain);
        if ((epoch) % 1000 == 0)
            cout << "loss in main epoch " << epoch << " : " << loss <<
                 std::get<2>(fromTrain) << endl << std::get<3>(fromTrain) << endl <<
                 std::get<4>(fromTrain) << endl << std::get<5>(fromTrain) << endl;

        torch::Tensor v1 = beta[0].item<double>() * velocities;
        torch::Tensor v2 = (1 - beta[0].item<double>()) * grads;
        velocities = v1 + v2;
        torch::Tensor velocities_t = velocities / (1 - tmp_beta[0]);
        S = beta[1] * S + (1 - beta[1]) * (torch::pow(grads, 2));
        torch::Tensor S_t = alpha / (torch::sqrt(S / (1 - tmp_beta[1])) + epsilon);
        params = params - (S_t * (beta[0] * velocities_t + (1 - beta[0]) / (1 - tmp_beta[0]) * grads));

        return std::make_tuple(params, velocities, S, loss);
    }

    std::pair<torch::Tensor, torch::Tensor> mainTrain(torch::Tensor params, torch::Tensor t_seq,
                                                      std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> icfs,
                                                      int num_epochs, torch::Tensor totalEnergy, torch::Tensor kineticEnergy, torch::Tensor potentialEnergy, int n, int Np, int d,
                                                      double learn_rate, double momentum) {
        this->params = params;
        torch::Tensor velocities = torch::zeros(this->params.sizes());
        torch::Tensor S = torch::zeros(this->params.sizes());
        torch::Tensor loss;

        auto start = std::chrono::high_resolution_clock::now();
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            auto values = UpdateParamsNADAM(t_seq, icfs, velocities, S, totalEnergy, kineticEnergy, potentialEnergy, epoch, n, Np, d, 0.001,
                                            pow(10, -7),
                                            torch::tensor({0.999, 0.999}));
            params = std::get<0>(values);
            velocities = std::get<1>(values);
            S = std::get<2>(values);
            loss = std::get<3>(values);
            if ((epoch + 1) % 1000 == 0) {
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed_seconds = end - start;
                cout << elapsed_seconds.count() << endl;

            }
        }

        return std::make_pair(params, loss);
    }


};

// Main method
int main(int argc, char **argv) {
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);
    // Create Subsystem based on input file
    SubSystem subsystem = SubSystem();
    SubSystem checkPointState;

    cout << "nglob = " << subsystem.nglob << endl;
    int Np = subsystem.atoms.size();

    ScratchPad pingu = ScratchPad();
    pingu.defineParams(Np);

    vector<float> t_seq_vect;
    vector<float> kinetic_energy_vect;
    vector<float> potential_energy_vect;

    torch::Tensor positionsAndVelocitiesOverTimeSteps;
    torch::Tensor totalEnergy;

    for (int stepCount = 1; stepCount <= subsystem.StepLimit + pingu.StepTrain; stepCount++) {
        vector<float> x_vect, y_vect, z_vect;
        vector<int> boundaryCrossingAtomIndices;
        if(stepCount == subsystem.StepLimit + pingu.StepTrain) {
            SubSystem finalState = subsystem;
            finalState.ShiftAtoms();
            for (auto it_atom = finalState.atoms.begin(); it_atom != finalState.atoms.end(); ++it_atom) {
                if (it_atom->isResident) {
                    x_vect.push_back(it_atom->x);
                    y_vect.push_back(it_atom->y);
                    z_vect.push_back(it_atom->z);
                }
            }
        } else {
            for (auto it_atom = subsystem.atoms.begin(); it_atom != subsystem.atoms.end(); ++it_atom) {
                if (it_atom->isResident) {
                    x_vect.push_back(it_atom->x);
                    y_vect.push_back(it_atom->y);
                    z_vect.push_back(it_atom->z);
                }
            }
        }

        (stepCount >= subsystem.StepLimit) ? SingleStep(subsystem, true) : SingleStep(subsystem, false);

        subsystem.EvalProps(stepCount);
        // use WriteXYZ(frame suffix) from SubSystem to print frames
        if (stepCount >= subsystem.StepLimit) subsystem.WriteXYZ(stepCount);

        t_seq_vect.push_back(stepCount * subsystem.DeltaT);
        kinetic_energy_vect.push_back(subsystem.kinEnergy);
        potential_energy_vect.push_back(subsystem.potEnergy);

        torch::Tensor positionsAndVelocitiesPerTimeStep;
        torch::Tensor velocitiesPerTimeStep;
        vector<float> vx_vect, vy_vect, vz_vect;
        for (auto it_atom = subsystem.atoms.begin(); it_atom != subsystem.atoms.end(); ++it_atom) {
            if (it_atom->isResident) {
                vx_vect.push_back(it_atom->vx);
                vy_vect.push_back(it_atom->vy);
                vz_vect.push_back(it_atom->vz);
            }
        }

        int numberOfAtoms = x_vect.size();
        positionsAndVelocitiesPerTimeStep = torch::from_blob(&x_vect[0], {1, numberOfAtoms});
        positionsAndVelocitiesPerTimeStep = torch::cat(
                {positionsAndVelocitiesPerTimeStep, torch::from_blob(&y_vect[0], {1, numberOfAtoms}),
                 torch::from_blob(&z_vect[0], {1, numberOfAtoms}),
                 torch::from_blob(&vx_vect[0], {1, numberOfAtoms}),
                 torch::from_blob(&vy_vect[0], {1, numberOfAtoms}),
                 torch::from_blob(&vz_vect[0], {1, numberOfAtoms})}, 1);

        if (stepCount == 1) {
            positionsAndVelocitiesOverTimeSteps = positionsAndVelocitiesPerTimeStep;
        } else {
            positionsAndVelocitiesOverTimeSteps = torch::cat(
                    {positionsAndVelocitiesOverTimeSteps, positionsAndVelocitiesPerTimeStep}, 0);
        }
        if (stepCount == subsystem.StepLimit) {
            float totalEnergyFloat = subsystem.totEnergy;
            totalEnergy = torch::tensor({totalEnergyFloat});
            checkPointState = subsystem;
        }

    }

    pingu.setCheckPoint(checkPointState);

    int numberOfTimeSequences = t_seq_vect.size();
    torch::Tensor t_seq_entire = torch::from_blob(t_seq_vect.data(), {numberOfTimeSequences, 1});

    // Store results from md only for the steps after warm-up steps for pre-train purpose
    torch::Tensor md_qt = positionsAndVelocitiesOverTimeSteps.narrow(0, subsystem.StepLimit, pingu.StepTrain);
    torch::Tensor t_seq = t_seq_entire.narrow(0, subsystem.StepLimit, pingu.StepTrain);
    cout << "obtained md positions and velocities matrix" << endl;
    cout << "time sequences to predict" << t_seq << endl;

    torch::Tensor kinetic_energy_entire = torch::from_blob(kinetic_energy_vect.data(), {numberOfTimeSequences, 1});
    torch::Tensor kineticEnergy = kinetic_energy_entire.narrow(0, subsystem.StepLimit, pingu.StepTrain);

    torch::Tensor potential_energy_entire = torch::from_blob(potential_energy_vect.data(), {numberOfTimeSequences, 1});
    torch::Tensor potentialEnergy = potential_energy_entire.narrow(0, subsystem.StepLimit, pingu.StepTrain);
    cout << "obtained md potential energy matrix" << endl;

    // Set the inital and final values
    torch::Tensor initialPositions = md_qt.narrow(1, 0, 3*Np)[1];
    cout << "obtained initial positions" << endl;
    torch::Tensor finalPositions = md_qt.narrow(1, 0, 3*Np)[pingu.StepTrain - 2];
    cout << "obtained final positions" << endl;
    torch::Tensor initialVelocities = md_qt.narrow(1, 3 * Np, 3*Np)[1];
    cout << "obtained initial velocities" << endl;
    torch::Tensor finalVelocities = md_qt.narrow(1, 3 * Np, 3*Np)[pingu.StepTrain - 2];
    cout << "obtained final velocities" << endl;

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> icfs =
            std::make_tuple(initialPositions, finalPositions, initialVelocities, finalVelocities);

    // Make call for pre-train step
    cout << "Pre-training in progress" << endl;
    std::pair<torch::Tensor, torch::Tensor> preTrainLossValues = pingu.PreTrain(t_seq, icfs, pingu.PreTrainEpochs,
                                                                                pingu.nodes, Np, 3);
    cout << "The mean loss from pre-train is " << preTrainLossValues.second << endl;

    // Small pause here to see the pre-train loss
    struct timespec tim, tim2;
    tim.tv_sec = 2;
    tim.tv_nsec = 0L;
    nanosleep(&tim, &tim2);

    // Store results from md only for the steps after pre-train steps for main-train purpose
    md_qt = positionsAndVelocitiesOverTimeSteps.narrow(0, subsystem.StepLimit, pingu.StepTrain);

    auto mainTrainLossValues = pingu.mainTrain(preTrainLossValues.first, t_seq, icfs, pingu.MainTrainEpochs,
                                               totalEnergy,kineticEnergy, potentialEnergy, pingu.nodes, Np, 3, 0.0001, 0.99);
    cout << "main train loss value: " << mainTrainLossValues.second << endl;
    // Get the new weights
    pingu.params = mainTrainLossValues.first;

    torch::Tensor w0 = pingu.params.narrow(0, 0, pingu.nodes);
    torch::Tensor b0 = pingu.params.narrow(0, pingu.nodes, pingu.nodes);
    torch::Tensor w1 = pingu.params.narrow(0, 2 * pingu.nodes, 2 * pingu.nodes * 3 * Np).reshape(
            {2 * 3 * Np, pingu.nodes});
    torch::Tensor b1 = pingu.params.narrow(0, 2 * pingu.nodes + (2 * pingu.nodes * 3 * Np), 2 * 3 * Np);
    torch::Tensor tmp = torch::ger(torch::squeeze(w0), torch::squeeze(t_seq)) + b0;
    torch::Tensor tmp1 = torch::sigmoid(tmp);
    torch::Tensor q = torch::matmul(w1, tmp1) + b1;
    torch::Tensor qt = q.transpose(0, 1);

    auto PEs = LJ3D_M(checkPointState, qt.narrow(1, 0, 3 * Np), Np); // Send the initial X,Y,Z
    torch::Tensor PE = std::get<0>(PEs).div(Np);
    torch::Tensor KE = 0.5 * (qt.narrow(1, 3 * Np, 3 * Np).pow(2).sum(1).div(Np).reshape_as(PE));
    torch::Tensor HM = PE + KE;
    cout << "Time sequences to predict\n" << t_seq << "\nKE from prediction\n" << KE << "\nPE from prediction\n" << PE
         << "\nHM from prediction\n" << HM << endl;
    cout << "Atom positions and velocities over time grids from Ground truth\n" << md_qt << endl;
    cout << "Atom positions and velocities over time grids from prediction\n" << qt << endl;

}

/*--------------------------------------------------------------------*/
void SingleStep(SubSystem &subsystem, bool shouldTrack) {
/*----------------------------------------------------------------------
r & rv are propagated by DeltaT using the velocity-Verlet scheme.
----------------------------------------------------------------------*/
    double DeltaTH = subsystem.DeltaT / 2.0;
    subsystem.Kick(DeltaTH); /* First half kick to obtain v(t+Dt/2) */
    subsystem.Update(subsystem.DeltaT);
    subsystem.AtomMove(shouldTrack);
    subsystem.AtomCopy();
    ComputeAccel(subsystem); /* Computes new accelerations, a(t+Dt) */
    subsystem.Kick(DeltaTH); /* Second half kick to obtain v(t+Dt) */
}

std::tuple<float, torch::Tensor> LJ3D(SubSystem &predictedSystem, torch::Tensor qt, int Np) {

    std::vector<torch::Tensor> positionsAlongAxis = torch::split(qt, Np, 0);
    int index_pt = 0;
    for (auto it_atom = predictedSystem.atoms.begin(); it_atom != predictedSystem.atoms.end(); ++it_atom, ++index_pt) {
        if (it_atom->isResident) {
            it_atom->x = positionsAlongAxis[0][index_pt].item<double>();
            it_atom->y = positionsAlongAxis[1][index_pt].item<double>();
            it_atom->z = positionsAlongAxis[2][index_pt].item<double>();
        }
    }
    predictedSystem.WrapAtoms();
    double DeltaTH = predictedSystem.DeltaT / 2.0;
    predictedSystem.Kick(DeltaTH); /* First half kick to obtain v(t+Dt/2) */
    predictedSystem.Update(predictedSystem.DeltaT);
    predictedSystem.AtomMove(false);
    predictedSystem.AtomCopy();
    auto lpeForces = ComputeAccelPredicted(predictedSystem);
    predictedSystem.Kick(DeltaTH); /* Second half kick to obtain v(t+Dt) */

    return lpeForces; // Create object before return.
}

std::tuple<torch::Tensor, torch::Tensor> LJ3D_M(SubSystem checkPointState, torch::Tensor qt, int Np) {
    SubSystem predictedSystem = checkPointState;
    std::vector<torch::Tensor> positionsAlongTime = torch::chunk(qt, qt.sizes()[0], 0);
    int count = 1;
    torch::Tensor potentials, forces;
    for (auto &position : positionsAlongTime) {
        auto fromLJ3D = LJ3D(predictedSystem, position.squeeze(), Np);
        if (count == 1) {
            potentials = torch::tensor(std::get<0>(fromLJ3D));
            potentials.resize_({1, 1});
            forces = std::get<1>(fromLJ3D);
            forces.resize_({1, 3 * Np});
        } else {
            potentials = torch::cat({potentials, torch::tensor(std::get<0>(fromLJ3D)).reshape({1, 1})}, 0);
            forces = torch::cat({forces, std::get<1>(fromLJ3D)}, 0);
        }
        count++;
    }

    return std::make_tuple(potentials, forces);
}

/**
 *  Given atomic coordinates, r[0:n+nb-1][], for the extended (i.e.,
 *  resident & copied) system, computes the acceleration, ra[0:n-1][], for
 *  the residents.
 *
 * @param subsystem
 * @return
 */
double ComputeAccel(SubSystem &subsystem) {
    /*----------------------------------------------------------------------

      ----------------------------------------------------------------------*/
    int i, j, a, lc2[3], lcyz2, lcxyz2, mc[3], c, mc1[3], c1;
    int bintra;
    double dr[3], rr, ri2, ri6, r1, rrCut, fcVal, f, vVal, lpe;

    double Uc, Duc;

    array<int, 3> lc{};
    array<double, 3> rc{};

    vector<int> head;
    //map<int, int> head;
    vector<int> lscl(subsystem.atoms.size());
    int EMPTY = -1;

    /* Compute the # of cells for linked cell lists */
    for (a = 0; a < 3; a++) {
        lc[a] = subsystem.al[a] / RCUT;
        rc[a] = subsystem.al[a] / lc[a];
    }

    /* Constants for potential truncation */
    rr = RCUT * RCUT;
    ri2 = 1.0 / rr;
    ri6 = ri2 * ri2 * ri2;
    r1 = sqrt(rr);
    Uc = 4.0 * ri6 * (ri6 - 1.0);
    Duc = -48.0 * ri6 * (ri6 - 0.5) / r1;

    /* Reset the potential & forces */
    lpe = 0.0;
    for (auto &atom : subsystem.atoms) {
        atom.ax = 0.0;
        atom.ay = 0.0;
        atom.az = 0.0;
    }

    /* Make a linked-cell list, lscl------------------------------------*/

    for (a = 0; a < 3; a++) lc2[a] = lc[a] + 2;
    lcyz2 = lc2[1] * lc2[2];
    lcxyz2 = lc2[0] * lcyz2;

    /* Reset the headers, head */
    for (c = 0; c < lcxyz2; c++) head.push_back(EMPTY);

    /* Scan atoms to construct headers, head, & linked lists, lscl */
    for (auto it_atom = subsystem.atoms.begin(); it_atom != subsystem.atoms.end(); ++it_atom) {
        mc[0] = (it_atom->x + rc[0]) / rc[0];
        mc[1] = (it_atom->y + rc[1]) / rc[1];
        mc[2] = (it_atom->z + rc[2]) / rc[2];
        /* Translate the vector cell index, mc, to a scalar cell index */
        c = mc[0] * lcyz2 + mc[1] * lc2[2] + mc[2];

        cout.precision(6);
        cout.setf(ios::fixed, ios::floatfield);
        lscl[distance(subsystem.atoms.begin(), it_atom)] = head[c];
        head[c] = distance(subsystem.atoms.begin(), it_atom);
    } /* Endfor atom i */

    /* Calculate pair interaction---------------------------------------*/
    rrCut = RCUT * RCUT;

    /* Scan inner cells */
    for (mc[0] = 1; mc[0] <= lc[0]; (mc[0])++)
        for (mc[1] = 1; mc[1] <= lc[1]; (mc[1])++)
            for (mc[2] = 1; mc[2] <= lc[2]; (mc[2])++) {
                /* Calculate a scalar cell index */
                c = mc[0] * lcyz2 + mc[1] * lc2[2] + mc[2];
                /* Skip this cell if empty */
                if (head[c] == EMPTY) continue;

                /* Scan the neighbor cells (including itself) of cell c */
                for (mc1[0] = mc[0] - 1; mc1[0] <= mc[0] + 1; (mc1[0])++)
                    for (mc1[1] = mc[1] - 1; mc1[1] <= mc[1] + 1; (mc1[1])++)
                        for (mc1[2] = mc[2] - 1; mc1[2] <= mc[2] + 1; (mc1[2])++) {

                            /* Calculate the scalar cell index of the neighbor cell */
                            c1 = mc1[0] * lcyz2 + mc1[1] * lc2[2] + mc1[2];
                            /* Skip this neighbor cell if empty */
                            if (head[c1] == EMPTY) continue;

                            /* Scan atom i in cell c */
                            i = head[c];
                            while (i != EMPTY) {

                                /* Scan atom j in cell c1 */
                                j = head[c1];
                                while (j != EMPTY) {
                                    /* No calculation with itself */
                                    if (j != i) {
                                        /* Logical flag: bintra(true)- or inter(false)-pair atom */
                                        bintra = (j < subsystem.n);

                                        /* Pair vector dr = r[i] - r[j] */

                                        dr[0] = subsystem.atoms[i].x - subsystem.atoms[j].x;
                                        dr[1] = subsystem.atoms[i].y - subsystem.atoms[j].y;
                                        dr[2] = subsystem.atoms[i].z - subsystem.atoms[j].z;
                                        for (rr = 0.0, a = 0; a < 3; a++)
                                            rr += dr[a] * dr[a];
                                        /* Calculate potential & forces for intranode pairs (i < j)
                                           & all the internode pairs if rij < RCUT; note that for
                                           any copied atom, i < j */
                                        if (i < j && rr < rrCut) {
                                            ri2 = 1.0 / rr;
                                            ri6 = ri2 * ri2 * ri2;
                                            r1 = sqrt(rr);
                                            fcVal = 48.0 * ri2 * ri6 * (ri6 - 0.5) + Duc / r1;
                                            vVal = 4.0 * ri6 * (ri6 - 1.0) - Uc - Duc * (r1 - RCUT);
                                            if (bintra) lpe += vVal; else lpe += 0.5 * vVal;

                                            f = fcVal * dr[0];
                                            subsystem.atoms[i].ax += f;
                                            if (bintra) subsystem.atoms[j].ax -= f;

                                            f = fcVal * dr[1];
                                            subsystem.atoms[i].ay += f;
                                            if (bintra) subsystem.atoms[j].ay -= f;

                                            f = fcVal * dr[2];
                                            subsystem.atoms[i].az += f;
                                            if (bintra) subsystem.atoms[j].az -= f;
                                        }
                                    } /* Endif not self */

                                    j = lscl[j];
                                } /* Endwhile j not empty */

                                i = lscl[i];
                            } /* Endwhile i not empty */

                        } /* Endfor neighbor cells, c1 */

            } /* Endfor central cell, c */
    /* Global potential energy */
    MPI_Allreduce(&lpe, &subsystem.potEnergy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return lpe;
}

/**
 * Given atomic coordinates, r[0:n+nb-1][], for the extended (i.e.,
 * resident & copied) system, computes the acceleration, ra[0:n-1][], for
 * the residents.
 * @param subsystem
 * @return
 */
std::tuple<float, torch::Tensor> ComputeAccelPredicted(SubSystem &subsystem) {

    int i, j, a, lc2[3], lcyz2, lcxyz2, mc[3], c, mc1[3], c1;
    int bintra;
    double dr[3], rr, ri2, ri6, r1, rrCut, fcVal, f, vVal, lpe;

    double Uc, Duc;

    array<int, 3> lc{};
    array<double, 3> rc{};

    vector<int> head;
    //map<int, int> head;
    vector<int> lscl(subsystem.atoms.size());
    int EMPTY = -1;

    /* Compute the # of cells for linked cell lists */
    for (a = 0; a < 3; a++) {
        lc[a] = subsystem.al[a] / RCUT;
        rc[a] = subsystem.al[a] / lc[a];
    }

    /* Constants for potential truncation */
    rr = RCUT * RCUT;
    ri2 = 1.0 / rr;
    ri6 = ri2 * ri2 * ri2;
    r1 = sqrt(rr);
    Uc = 4.0 * ri6 * (ri6 - 1.0);
    Duc = -48.0 * ri6 * (ri6 - 0.5) / r1;

    /* Reset the potential & forces */
    lpe = 0.0;
    for (auto &atom : subsystem.atoms) {
        atom.ax = 0.0;
        atom.ay = 0.0;
        atom.az = 0.0;
    }

    /* Make a linked-cell list, lscl------------------------------------*/

    for (a = 0; a < 3; a++) lc2[a] = lc[a] + 2;
    lcyz2 = lc2[1] * lc2[2];
    lcxyz2 = lc2[0] * lcyz2;

    /* Reset the headers, head */
    for (c = 0; c < lcxyz2; c++) head.push_back(EMPTY);

    /* Scan atoms to construct headers, head, & linked lists, lscl */
    for (auto it_atom = subsystem.atoms.begin(); it_atom != subsystem.atoms.end(); ++it_atom) {
        mc[0] = (it_atom->x + rc[0]) / rc[0];
        mc[1] = (it_atom->y + rc[1]) / rc[1];
        mc[2] = (it_atom->z + rc[2]) / rc[2];
        /* Translate the vector cell index, mc, to a scalar cell index */
        c = mc[0] * lcyz2 + mc[1] * lc2[2] + mc[2];

        cout.precision(6);
        cout.setf(ios::fixed, ios::floatfield);
        lscl[distance(subsystem.atoms.begin(), it_atom)] = head[c];
        head[c] = distance(subsystem.atoms.begin(), it_atom);
    } /* Endfor atom i */

    /* Calculate pair interaction---------------------------------------*/
    rrCut = RCUT * RCUT;
    /* Scan inner cells */
    for (mc[0] = 1; mc[0] <= lc[0]; (mc[0])++)
        for (mc[1] = 1; mc[1] <= lc[1]; (mc[1])++)
            for (mc[2] = 1; mc[2] <= lc[2]; (mc[2])++) {
                /* Calculate a scalar cell index */
                c = mc[0] * lcyz2 + mc[1] * lc2[2] + mc[2];
                /* Skip this cell if empty */
                if (head[c] == EMPTY) continue;

                /* Scan the neighbor cells (including itself) of cell c */
                for (mc1[0] = mc[0] - 1; mc1[0] <= mc[0] + 1; (mc1[0])++)
                    for (mc1[1] = mc[1] - 1; mc1[1] <= mc[1] + 1; (mc1[1])++)
                        for (mc1[2] = mc[2] - 1; mc1[2] <= mc[2] + 1; (mc1[2])++) {

                            /* Calculate the scalar cell index of the neighbor cell */
                            c1 = mc1[0] * lcyz2 + mc1[1] * lc2[2] + mc1[2];
                            /* Skip this neighbor cell if empty */
                            if (head[c1] == EMPTY) continue;

                            /* Scan atom i in cell c */
                            i = head[c];
                            while (i != EMPTY) {

                                /* Scan atom j in cell c1 */
                                j = head[c1];
                                while (j != EMPTY) {
                                    /* No calculation with itself */
                                    if (j != i) {
                                        /* Logical flag: bintra(true)- or inter(false)-pair atom */
                                        bintra = (j < subsystem.n);

                                        /* Pair vector dr = r[i] - r[j] */
                                        dr[0] = subsystem.atoms[i].x - subsystem.atoms[j].x;
                                        dr[1] = subsystem.atoms[i].y - subsystem.atoms[j].y;
                                        dr[2] = subsystem.atoms[i].z - subsystem.atoms[j].z;
                                        for (rr = 0.0, a = 0; a < 3; a++)
                                            rr += dr[a] * dr[a];
                                        /* Calculate potential & forces for intranode pairs (i < j)
                                           & all the internode pairs if rij < RCUT; note that for
                                           any copied atom, i < j */
                                        if (i < j && rr < rrCut) {
                                            if (rr < 0.5) break;
                                            ri2 = 1.0 / rr;
                                            ri6 = ri2 * ri2 * ri2;
                                            r1 = sqrt(rr);
                                            fcVal = 48.0 * ri2 * ri6 * (ri6 - 0.5) + Duc / r1;
                                            vVal = 4.0 * ri6 * (ri6 - 1.0) - Uc - Duc * (r1 - RCUT);
                                            if (bintra) lpe += vVal; else lpe += 0.5 * vVal;

                                            f = fcVal * dr[0];
                                            subsystem.atoms[i].ax += f;
                                            if (bintra) subsystem.atoms[j].ax -= f;

                                            f = fcVal * dr[1];
                                            subsystem.atoms[i].ay += f;
                                            if (bintra) subsystem.atoms[j].ay -= f;

                                            f = fcVal * dr[2];
                                            subsystem.atoms[i].az += f;
                                            if (bintra) subsystem.atoms[j].az -= f;
                                        }
                                    } /* Endif not self */

                                    j = lscl[j];
                                } /* Endwhile j not empty */

                                i = lscl[i];
                            } /* Endwhile i not empty */

                        } /* Endfor neighbor cells, c1 */

            } /* Endfor central cell, c */
    /* Global potential energy */
    MPI_Allreduce(&lpe, &subsystem.potEnergy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    vector<vector<float> > forcesAlongAxis(3);
    for (auto it_atom = subsystem.atoms.begin(); it_atom != subsystem.atoms.end(); ++it_atom) {
        if (it_atom->isResident == 1) {
            forcesAlongAxis[0].push_back((float) it_atom->ax);
            forcesAlongAxis[1].push_back((float) it_atom->ay);
            forcesAlongAxis[2].push_back((float) it_atom->az);
        }
    }
    int Np = forcesAlongAxis[0].size();
    torch::Tensor forces = torch::from_blob(&forcesAlongAxis[0][0], {1, Np});
    forces = torch::cat({forces, torch::from_blob(&forcesAlongAxis[1][0], {1, Np}),
                         torch::from_blob(&forcesAlongAxis[2][0], {1, Np})}, 1);
    auto test = std::make_tuple(lpe, forces);
    return std::make_tuple(lpe, forces);
}
