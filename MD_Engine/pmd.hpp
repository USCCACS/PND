#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include "mpi.h"

/* Constants for the random number generator */
#define D2P31M 2147483647.0
#define DMUL 16807.0

class Atom {
public:
    double type;      ///< identifier for atom type
    bool isResident; ///< Marker to identify if the atom is moving out of the system after the time step

    double x;        ///< position in x axis
    double y;        ///< position in y axis
    double z;        ///< position in y axis

    double ax;        ///< acceleration on x axis
    double ay;        ///< acceleration on y axis
    double az;        ///< acceleration on y axis

    double vx;        ///< velocity on x axis
    double vy;        ///< velocity on y axis
    double vz;        ///< velocity on y axis

    std::array<int, 6> shiftCount; ///< Track the cells that the atom has moved

    Atom(); ///< Default constructor
};

class SubSystem {
public:
    int pid; ///<sequential processor ID of this cell
    int n; ///< Number of resident atoms in this processor
    int nglob; ///< Total number of atoms summed over processors
    double comt; ///< elapsed wall clock time & Communication time in second
    std::array<double, 3> al; ///< Box length per processor
    std::array<int, 3> vid; ///< Vector index of this processor
    std::array<int, 3> myparity; ///< Parity of this processor
    std::array<int, 6> nn; ///< Neighbor node list of this processor
    std::vector<std::vector<double> > sv; ///< Shift vector to the 6 neighbors
    std::array<double, 3> vSum, gvSum;
    std::vector<Atom> atoms; ///< resident and moved in atoms

    std::array<int, 3> vproc{}, ///< Vector processor decomposition of subsystems arranged in a 3D array
    InitUcell{}; ///< Unit cell system properties
    double Density, ///< Density of fcc
    InitTemp, ///< Inital temperature of system
    DeltaT; ///< Time scale of single step in simulation
    int StepLimit, ///< Total number of steps to carry out the simulation
    StepAvg; ///< Average of simulation steps before reporting system information

    double kinEnergy, ///< Calculated kinetic energy while simulation
    potEnergy, ///< Calculated potential energy while simulation
    totEnergy, ///< Calculated total energy energy while simulation
    temperature; ///< Calculated temperature of the system at the end of the time-step

    /** Constructor - Creates subsystem with with the specified atoms in the input file and assigns random velocities */
    SubSystem();

    /**
     * Initiates the neighbor node table for the Subsystem decomposed in X,Y & Z as a 3D matrix as defined by vproc
     * @param vproc Vector processor decomposition of subsystems arranged in a 3D array
     */
    void InitNeighborNode(std::array<int, 3> vproc);

    /**
     * Update Atomic co-ordinates to r(t+Dt)
     * @param DeltaT duration for which the state must be updated
     */
    void Update(double DeltaT);

    /**
     * Update the velocities after a time-step DeltaT
     * @param DeltaT
     */
    void Kick(double DeltaT);


    /**
     * Exchange boundary-atom co-ordinates among neighbor nodes
     */
    void AtomCopy();

    /**
     * Send moved-out atoms to neighbor nodes and receive moved-in atoms
     * from neighbor nodes returns the indexes of the atoms that have
     * moved
     */
    std::vector<int> AtomMove(bool);

    /**
     * Obtain the final position of the atoms by shifting the atom co-ordinates of the moved in-atoms
     * to the position of the atoms they replace for the time step
     */
    void ShiftAtoms();

    /**
     * Take atom co-ordinated from global co-ordinates and translate them to wrapped box co-ordinates
     */
    void WrapAtoms();

    /**
     * Returns true if an Atom lies in them boundary to a neighbor ID
     *
     * @param atom atom id in the system
     * @param ku the neighboring direction w.r.t the subsystem
     * @return 1 true, 0 false
    */
    int bbd(Atom atom, int ku);

    /**
     * Return true if an Atom lies in them boundary to a neighbor ID
     *
     * @param atom atom id in the system
     * @param ku the neighboring direction w.r.t the subsystem
     * @return 1 true, 0 false
     */
    int bmv(Atom atom, int ku);

    /**
     * Evaluates physical properties: kinetic, potential & total energies
     * @param stepCount Time step corresponding to the which the evaluation takes place
     */
    void EvalProps(int stepCount);

    /**
     * Write out XYZ co-ordinates from each frame into separate files
     *
     * @param step
     */
    void WriteXYZ(int step);

    /**
     * Method to return modulus
     * @param a
     * @param b
     * @return
     */
    static double Dmod(double a, double b) {
        int n;
        n = (int) (a / b);
        return (a - b * n);
    }

    /**
     * Method to generat erandom double value from a seed
     * @param seed
     * @return
     */
    static double RandR(double *seed) {
        *seed = Dmod(*seed * DMUL, D2P31M);
        return (*seed / D2P31M);
    }

    /**
     * Fills up a double pointer of length 3 with random values
     * @param p
     * @param seed
     */
    static void RandVec3(double *p, double *seed) {
        double x = 0, y = 0, s = 2.0;
        while (s > 1.0) {
            x = 2.0 * RandR(seed) - 1.0;
            y = 2.0 * RandR(seed) - 1.0;
            s = x * x + y * y;
        }
        p[2] = 1.0 - 2.0 * s;
        s = 2.0 * sqrt(1.0 - s);
        p[0] = s * x;
        p[1] = s * y;
    }

};
