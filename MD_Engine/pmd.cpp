/*----------------------------------------------------------------------
Program pmd.cpp performs parallel molecular-dynamics for Lennard-Jones
systems using the Message Passing Interface (MPI) standard.
----------------------------------------------------------------------*/
#include <iostream>
#include <cmath>
#include <string>
#include <sstream>
#include "pmd.hpp"
#include <algorithm>
#include <fstream>

using namespace std;

const double RCUT = 2.5; // Potential cut-off length
const double MOVED_OUT = -1.0e10;

Atom::Atom()
        : type(0), isResident(true), x(0.0), y(0.0), z(0.0),
          ax(0.0), ay(0.0), az(0.0), vx(0.0), vy(0.0), vz(0.0), hasMovedIn(false), iv{}  {}

/* Create subsystem with parameters input parameters to calculate
   the number of atoms and give them random velocities */
SubSystem::SubSystem() : pid(0), n(0), nglob(0), comt(0.0), al{}, vid{}, myparity{}, nn{}, sv{}, vSum{}, gvSum{},
                         atoms{}, kinEnergy(0.0), potEnergy(0.0), totEnergy(0.0), temperature(0) {

    int pid; // Sequential processor ID
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);  // My processor ID

    // Open pmd.in file and read inputs
    ifstream ifs("pmd.in", ifstream::in);
    if (!ifs.is_open()) {
        cerr << "failed to open md input file" << endl;
        terminate();
    }

    ifs >> vproc[0] >> vproc[1] >> vproc[2];
    ifs >> InitUcell[0] >> InitUcell[1] >> InitUcell[2];
    ifs >> Density >> InitTemp >> DeltaT;
    ifs >> StepLimit >> StepAvg;

    ifs.close();

    /* Compute basic parameters */
    for (int i = 0; i < 3; i++) al[i] = InitUcell[i] / cbrt(Density / 4.0);
    // if (pid == 0) cout << "al = " << al[0] << " " << al[1] <<  " " << al[2] << endl;

    // Prepare the Neighbot-node table
    InitNeighborNode(vproc);

    // Initialize lattice positions and assign random velocities
    array<double, 3> c{}, gap{};
    int j, a, nX, nY, nZ;

    /* FCC atoms in the original unit cell */
    vector<vector<double> > origAtom = {{0.0, 0.0, 0.0},
                                        {0.0, 0.5, 0.5},
                                        {0.5, 0.0, 0.5},
                                        {0.5, 0.5, 0.0}};

    /* Set up a face-centered cubic (fcc) lattice */
    for (a = 0; a < 3; a++) gap[a] = al[a] / InitUcell[a];

    for (nZ = 0; nZ < InitUcell[2]; nZ++) {
        c[2] = nZ * gap[2];
        for (nY = 0; nY < InitUcell[1]; nY++) {
            c[1] = nY * gap[1];
            for (nX = 0; nX < InitUcell[0]; nX++) {
                c[0] = nX * gap[0];
                for (j = 0; j < 4; j++) {
                    Atom atom;

                    atom.x = c[0] + gap[0] * origAtom[j][0];

                    atom.y = c[1] + gap[1] * origAtom[j][1];

                    atom.z = c[2] + gap[2] * origAtom[j][2];
                    // if(pid == 0) cout << "atom coordinates - " << atom.x << " " << atom.y << " " << atom.z << endl;

                    atoms.push_back(atom);
                }
            }
        }
    }
    /* Total # of atoms summed over processors */
    n = atoms.size();
    MPI_Allreduce(&n, &nglob, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    /* Generate random velocities */
    double seed = 13597.0 + pid;
    double vMag = sqrt(3 * InitTemp);
    double e[3];
    for (a = 0; a < 3; a++) vSum[a] = 0.0;
    for (auto &atom : atoms) {
        RandVec3(e, &seed);
        atom.vx = vMag * e[0];
        vSum[0] += atom.vx;
        atom.vy = vMag * e[1];
        vSum[1] += atom.vy;
        atom.vz = vMag * e[2];
        vSum[2] += atom.vz;
    }
    MPI_Allreduce(&vSum[0], &gvSum[0], 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // Make the total momentum zero
    for (a = 0; a < 3; a++) gvSum[a] /= nglob;
    for (auto &atom : atoms) {
        atom.vx -= gvSum[0];
        atom.vy -= gvSum[1];
        atom.vz -= gvSum[2];
    }


}

void SubSystem::InitNeighborNode(array<int, 3> vproc) {
    // Prepare neighbor-node ID table for a subsystem
    vid[0] = pid / (vproc[1] * vproc[2]);
    vid[1] = (pid / vproc[2]) % vproc[1];
    vid[2] = pid % vproc[2];

    vector<vector<int> > iv = {
            {-1, 0,  0},
            {1,  0,  0},
            {0,  -1, 0},
            {0,  1,  0},
            {0,  0,  -1},
            {0,  0,  1}
    };

    int ku, a, k1[3];

    /* Set up neighbor tables, nn & sv */
    for (ku = 0; ku < 6; ku++) {
        /* Vector index of neighbor ku */
        for (a = 0; a < 3; a++)
            k1[a] = (vid[a] + iv[ku][a] + vproc[a]) % vproc[a];
        /* Scalar neighbor ID, nn */
        nn[ku] = k1[0] * vproc[1] * vproc[2] + k1[1] * vproc[2] + k1[2];
        /* Shift vector, sv */
        vector<double> svEntry;
        for (a = 0; a < 3; a++) svEntry.push_back(al[a] * iv[ku][a]);
        sv.push_back(svEntry);
    }

    // Set up node parity table
    for (a = 0; a < 3; a++) {
        if (vproc[a] == 1)
            myparity[a] = 2;
        else if (vid[a] % 2 == 0)
            myparity[a] = 0;
        else
            myparity[a] = 1;
    }
}

// Update Atomic co-ordinates to r(t+Dt)
void SubSystem::Update(double DeltaT) {
    for (auto &atom : atoms) {
        atom.x = atom.x + DeltaT * atom.vx;
        atom.y = atom.y + DeltaT * atom.vy;
        atom.z = atom.z + DeltaT * atom.vz;
        //if(pid == 0) cout << " : " << atom.vx << " " << atom.vy << atom.vz << endl;
    }
}

// Update the velocities after a time-step DeltaT
void SubSystem::Kick(double DeltaT) {
    for (auto &atom : atoms) {
        atom.vx = atom.vx + DeltaT * atom.ax;
        atom.vy = atom.vy + DeltaT * atom.ay;
        atom.vz = atom.vz + DeltaT * atom.az;
        //if(pid == 0) cout << "kick : " << atom.vx << " " << atom.vy << atom.vz << endl;
    }
}

// Exchange boundaty-atom co-ordinates among neighbor nodes
void SubSystem::AtomCopy() {
    int kd, kdd, i, ku, inode, nsd, nrc;
    double com1 = 0;
    vector<vector<int> > lsb(6);

    //atoms.erase(remove_if(atoms.begin(), atoms.end(), [](Atom atom) {return !atom.isResident;}), atoms.end());
    /* Main loop over x, y & z directions starts--------------------------*/
    for (kd = 0; kd < 3; kd++) {

        // Iterate through all atoms in this cell
        for (auto it_atom = atoms.begin(); it_atom != atoms.end(); ++it_atom) {
            for (kdd = 0; kdd < 2; kdd++) {
                ku = 2 * kd + kdd; /* Neighbor ID */
                /* Add an atom to the boundary-atom list, LSB, for neighbor ku
                   according to bit-condition function, bbd */
                i = distance(atoms.begin(), it_atom);
                if (bbd(*it_atom, ku)) {
                    lsb[ku].push_back(i);
                    //if(pid == 0) cout << "Tested positive for copy " << it_atom->x << " " << it_atom->z << " " << it_atom->y << endl;
                }
            }
        }
        //if(pid ==0) cout << "atoms identified for copy to " << (ku -1)<< " & " << ku << " : " << lsb[ku-1].size() << " " << lsb[ku].size() << endl;
        // if(pid == 0)cout << "atoms searched as far as " << i << endl;
        /* Message passing------------------------------------------------*/

        com1 = MPI_Wtime(); /* To calculate the communication time */

        /* Loop over the lower & higher directions */
        for (kdd = 0; kdd < 2; kdd++) {

            vector<double> sendBuf;
            vector<double> recvBuf;

            ku = 2 * kd + kdd;
            // if(pid ==0) cout << "ku = " << ku << endl;
            inode = nn[ku]; /* Neighbor node ID */
            // if(pid == 0) cout << "inode = " << inode << endl;

            nsd = lsb[ku].size();
            // cout << "copy number " << nsd << endl;

            /* Even node: send & recv */
            if (myparity[kd] == 0) {
                MPI_Send(&nsd, 1, MPI_INT, inode, 10, MPI_COMM_WORLD);
                MPI_Recv(&nrc, 1, MPI_INT, MPI_ANY_SOURCE, 10,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
                /* Odd node: recv & send */
            else if (myparity[kd] == 1) {
                MPI_Recv(&nrc, 1, MPI_INT, MPI_ANY_SOURCE, 10,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(&nsd, 1, MPI_INT, inode, 10, MPI_COMM_WORLD);
            }
                /* Single layer: Exchange information with myself */
            else
                nrc = nsd;

            // if(pid == 0) cout << "nrc = " << nrc /4<< endl;
            /* Now nrc is the # of atoms to be received */

            /* Send & receive information on boundary atoms-----------------*/

            /* Message buffering */
            for (auto it_index = lsb[ku].begin(); it_index != lsb[ku].end(); ++it_index) {
                sendBuf.push_back(atoms[*it_index].type);
                sendBuf.push_back(atoms[*it_index].x - sv[ku][0]);
                // if(pid == 0) cout << "copy - " << atoms[*it_index].x<< " ";
                sendBuf.push_back(atoms[*it_index].y - sv[ku][1]);
                // if(pid ==0) cout << atoms[*it_index].y<< " ";
                sendBuf.push_back(atoms[*it_index].z - sv[ku][2]);
                // if(pid ==0) cout << atoms[*it_index].z << " " << endl;
            }
            // resize the receive buffer for nrc
            recvBuf.resize(4 * nrc);

            /* Even node: send & recv */
            if (myparity[kd] == 0) {
                MPI_Send(&sendBuf[0], 4 * nsd, MPI_DOUBLE, inode, 20, MPI_COMM_WORLD);
                MPI_Recv(&recvBuf[0], 4 * nrc, MPI_DOUBLE, MPI_ANY_SOURCE, 20,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
                /* Odd node: recv & send */
            else if (myparity[kd] == 1) {
                MPI_Recv(&recvBuf[0], 4 * nrc, MPI_DOUBLE, MPI_ANY_SOURCE, 20,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(&sendBuf[0], 4 * nsd, MPI_DOUBLE, inode, 20, MPI_COMM_WORLD);
            }
                /* Single layer: Exchange information with myself */
            else
                sendBuf.swap(recvBuf);

            // Message storing
            for (i = 0; i < 4 * nrc; i++) {
                Atom rAtom;

                rAtom.type = recvBuf[i];
                ++i;
                rAtom.isResident = false;
                rAtom.x = recvBuf[i];
                ++i;
                rAtom.y = recvBuf[i];
                ++i;
                rAtom.z = recvBuf[i];
                // if(pid == 0) cout << " arriving in atom copy - x: " << rAtom.x;
                //   rAtom.x = *it_recv;
                //   ++it_recv;
                // if(pid == 0) cout << " y: " << rAtom.y;
                //   rAtom.y = *it_recv;
                //   ++it_recv;
                // if(pid == 0) cout << " z: " << rAtom.z << endl;
                //   rAtom.z = *it_recv;

                atoms.push_back(rAtom);
            }
            /* Internode synchronization */
            MPI_Barrier(MPI_COMM_WORLD);

        } /* Endfor lower & higher directions, kdd */

        comt += MPI_Wtime() - com1; /* Update communication time, COMT */
    }
}

// Send moved-out atoms to neighbor nodes and receive moved-in atoms
// from neighbor nodes
vector<int> SubSystem::AtomMove() {
    vector<int> boundaryCrossingAtomIndices;
    vector<vector<int> > mvque(6);
    int ku, kd, i, kdd, kul, kuh, inode, nsd, nrc;
    double com1 = 0;

    atoms.erase(remove_if(atoms.begin(), atoms.end(), [](Atom atom) { return !atom.isResident; }), atoms.end());
    //atoms.resize(n);

    /* Main loop over x, y & z directions starts------------------------*/

    for (kd = 0; kd < 3; kd++) {

        /* Make a moved-atom list, mvque----------------------------------*/

        /* Scan all the residents & immigrants to list moved-out atoms */
        for (auto it_atom = atoms.begin(); it_atom != atoms.end(); ++it_atom) {
            kul = 2 * kd; /* Neighbor ID */
            kuh = 2 * kd + 1;
            /* Register a to-be-copied atom in mvque[kul|kuh][] */

            i = distance(atoms.begin(), it_atom);
            if (it_atom->x > MOVED_OUT) {
                /* Move to the lower direction */
                if (bmv(*it_atom, kul)) {
                    mvque[kul].push_back(i);
                    // if(pid == 0) cout << "Tested positive for move " << it_atom->x << " " << it_atom->z << " " << it_atom->y << endl;
                }
                /* Move to the higher direction */
                else if (bmv(*it_atom, kuh)) {
                    mvque[kuh].push_back(i);
                    // if(pid == 0) cout << "Tested positive for move " << it_atom->x << " " << it_atom->z << " " << it_atom->y << endl;}
                }
            }

            // if(pid ==0) {cout << "atoms identified for move to " << kul << " & " << kuh << " : " << mvque[kul].size() << " " << mvque[kuh].size() << endl;
            // cout << "their indices are \n";
            // for(auto & val : mvque[kul])
            // cout << val << " ";
            // cout << "\n";
            // for(auto & val : mvque[kuh])
            // cout << val << " ";
            // cout << "\n";
        }

        /* Message passing------------------------------------------------*/

        com1 = MPI_Wtime(); /* To calculate the communication time */

        /* Loop over the lower & higher directions */
        for (kdd = 0; kdd < 2; kdd++) {
            // if(pid ==0) cout << "kdd " << kdd << endl;
            vector<double> sendBuf;
            vector<double> recvBuf;

            ku = 2 * kd + kdd;
            // if(pid ==0) cout << "ku = " << ku << endl;
            inode = nn[ku]; /* Neighbor node ID */
            // if(pid == 0) cout << "inode = " << inode << endl;

            nsd = mvque[ku].size();
            // cout << "copy number " << nsd << endl;

            /* Even node: send & recv */
            if (myparity[kd] == 0) {
                MPI_Send(&nsd, 1, MPI_INT, inode, 10, MPI_COMM_WORLD);
                MPI_Recv(&nrc, 1, MPI_INT, MPI_ANY_SOURCE, 10,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
                /* Odd node: recv & send */
            else if (myparity[kd] == 1) {
                MPI_Recv(&nrc, 1, MPI_INT, MPI_ANY_SOURCE, 10,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(&nsd, 1, MPI_INT, inode, 10, MPI_COMM_WORLD);
            }
                /* Single layer: Exchange information with myself */
            else
                nrc = nsd;

            /* Now nrc is the # of atoms to be received */

            /* Send & receive information on boundary atoms-----------------*/

            /* Message buffering */
            for (auto it_index = mvque[ku].begin(); it_index != mvque[ku].end(); ++it_index) {
                sendBuf.push_back(atoms[*it_index].type);

                sendBuf.push_back(atoms[*it_index].x - sv[ku][0]);
                sendBuf.push_back(atoms[*it_index].y - sv[ku][1]);
                sendBuf.push_back(atoms[*it_index].z - sv[ku][2]);
                // In AtomMove we will also be considering the velocities
                sendBuf.push_back(atoms[*it_index].vx);
                sendBuf.push_back(atoms[*it_index].vy);
                sendBuf.push_back(atoms[*it_index].vz);

                // TODO: Send the flag here specifying atom has moved. Also send the correct index vector
//                sendBuf.push_back(ku);
                // if(pid == 0) cout << "move - " << atoms[*it_index].x << " ";
                // if(pid ==0) cout << atoms[*it_index].y << " ";
                // if(pid ==0) cout << atoms[*it_index].z << " " << endl;
                //atoms[*it_index].isResident = false;
                // Mark the atom as moved out
                atoms[*it_index].x = MOVED_OUT;
            }

            // resize the receive buffer for nrc
            recvBuf.resize(7 * nrc);

            /* Even node: send & recv */
            if (myparity[kd] == 0) {
                MPI_Send(&sendBuf[0], 7 * nsd, MPI_DOUBLE, inode, 20, MPI_COMM_WORLD);
                MPI_Recv(&recvBuf[0], 7 * nrc, MPI_DOUBLE, MPI_ANY_SOURCE, 20,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
                /* Odd node: recv & send */
            else if (myparity[kd] == 1) {
                MPI_Recv(&recvBuf[0], 7 * nrc, MPI_DOUBLE, MPI_ANY_SOURCE, 20,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(&sendBuf[0], 7 * nsd, MPI_DOUBLE, inode, 20, MPI_COMM_WORLD);
            }
                /* Single layer: Exchange information with myself */
            else
                sendBuf.swap(recvBuf);

            // Message storing
            for (i = 0; i < 7 * nrc; i++) {
                Atom rAtom;

                rAtom.type = recvBuf[i];
                ++i;
                rAtom.isResident = true;
                rAtom.x = recvBuf[i];
                ++i;
                rAtom.y = recvBuf[i];
                ++i;
                rAtom.z = recvBuf[i];
                ++i;
                rAtom.vx = recvBuf[i];
                ++i;
                rAtom.vy = recvBuf[i];
                ++i;
                rAtom.vz = recvBuf[i];
                for (unsigned i = 0; i < atoms.size(); i++) {
                    if (atoms[i].x <= MOVED_OUT) {
                        atoms[i] = rAtom;
                        atoms[i].hasMovedIn = true;
                        boundaryCrossingAtomIndices.push_back(i);
                        break;
                    }
                }
//	atoms.push_back(rAtom);
            }

            /* Internode synchronization */
            MPI_Barrier(MPI_COMM_WORLD);
        } /* Endfor lower & higher directions, kdd */

        comt += MPI_Wtime() - com1; /* Update communication time, COMT */
    } /* Endfor x, y & z directions, kd */

    /* Main loop over x, y & z directions ends--------------------------*/

    /* Compress resident arrays including new immigrants */

    // if(pid == 0) cout << "atoms before AtomMove = " << n << endl;
//  atoms.erase(remove_if(atoms.begin(), atoms.end(),
//			[](Atom atom) { return atom.x <= MOVED_OUT; }), atoms.end());
    n = atoms.size();
    MPI_Allreduce(&n, &nglob, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    // if(pid == 0) cout << "atoms after AtomMove = " << n << endl;

    return boundaryCrossingAtomIndices;

}

// Return true if an Atom lies in the boundary to a neighbor ID
int SubSystem::bbd(Atom atom, int ku) {
    // if (atom.isResident == false) return false; // Do not consider atoms that have moved already 
    int kd, kdd;
    kd = ku / 2; /* x(0)|y(1)|z(2) direction */
    kdd = ku % 2; /* Lower(0)|higher(1) direction */
    if (kdd == 0) {
        if (kd == 0)
            return atom.x < RCUT;
        if (kd == 1)
            return atom.y < RCUT;
        if (kd == 2)
            return atom.z < RCUT;
    } else {
        if (kd == 0)
            return al[0] - RCUT < atom.x;
        if (kd == 1)
            return al[1] - RCUT < atom.y;
        if (kd == 2)
            return al[2] - RCUT < atom.z;
    }
    return 0; // default return
}

// Return true if an Atom lies in the boundary to a neighbor ID
int SubSystem::bmv(Atom atom, int ku) {
    int kd, kdd;
    kd = ku / 2; /* x(0)|y(1)|z(2) direction */
    kdd = ku % 2; /* Lower(0)|higher(1) direction */
    if (kdd == 0) {
        if (kd == 0)
            return atom.x < 0.0;
        if (kd == 1)
            return atom.y < 0.0;
        if (kd == 2)
            return atom.z < 0.0;
    } else {
        if (kd == 0)
            return al[0] < atom.x;
        if (kd == 1)
            return al[1] < atom.y;
        if (kd == 2)
            return al[2] < atom.z;
    }
    return 0;
}

// Evaluates physical properties: kinetic, potential & total energies.
void SubSystem::EvalProps(int stepCount) {
    double vv = 0, lke = 0;

    /* Total kinetic energy */
    for (auto &atom : atoms) {
        vv = (atom.vx * atom.vx + atom.vy * atom.vy + atom.vz * atom.vz);
        lke += vv;
    }
    lke *= 0.5;
    MPI_Allreduce(&lke, &kinEnergy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    /* Energy per atom */
    kinEnergy /= nglob;
    potEnergy /= nglob;
    totEnergy = kinEnergy + potEnergy;
    temperature = kinEnergy * 2.0 / 3.0;

    /* Print the computed properties */
    if (pid == 0) cout << stepCount * DeltaT << " " << kinEnergy << " " << potEnergy << " " << totEnergy << endl;
}

void SubSystem::WriteXYZ(int step) {
    MPI_File fh;

    stringstream entriesXYZBuf; // string buffer to load local XYZ co-ordinates into
    string entriesXYZ;
    string header;
    int entrySize, offset;

    string filename = "frame" + to_string(step) + ".xyz";

    entriesXYZBuf.precision(9);
    entriesXYZBuf.setf(ios::fixed, ios::floatfield); // Setting floatfield precision

    header = to_string(nglob) + "\n";

    int c = 0;
    for (auto &atom : atoms) {
        if (atom.isResident) {
            entriesXYZBuf << "\n" << ++c << " " << atom.x << " " << atom.y << " " << atom.z;
            //if(pid == 0 && step == 1)
            //  cout << c << " " << atom.x  << " " << atom.y  << " " << atom.z << endl;
        }
    }

    entriesXYZ = entriesXYZBuf.str();
    entrySize = entriesXYZ.size();
    if (pid == 0) entrySize += header.size();

    MPI_File_open(MPI_COMM_WORLD, filename.c_str(), MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &fh);
    //MPI_File_set_view(fh, sizeof(header), MPI_CHAR, MPI_CHAR, "", MPI_INFO_NULL);
    if (pid == 0) {
        MPI_File_write(fh, header.c_str(), sizeof(header), MPI_CHAR, MPI_STATUS_IGNORE);
    }
    MPI_Scan(&entrySize, &offset, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    //if(step == 1) cout << "pid :" << pid << " entrysize:" << entrySize << " offset :" << offset << endl;
    offset -= entriesXYZ.size();
    MPI_File_write_at_all(fh, offset, entriesXYZ.c_str(), entriesXYZ.size(), MPI_CHAR, MPI_STATUS_IGNORE);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_File_close(&fh);
}
