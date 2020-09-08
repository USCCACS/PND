# DSN-MD
Differential Solver Neuralnet MD

## Build Command
Use CMake to build the target `grad_lap`. This is binary for predicting energies using auto gradient using hamiltonian.
`cmake --build . --target pingu_example`

## Input Parameters
All system input parameters go into pmd.in
+ The first line is vproc[3]
+ The second line is InitUcell[3]
+ Density
+ Initial Temperature
+ DeltaT or time step
+ Number of warmup steps for MD - Results from these steps will not be considered for any trainig purpose. The purpose of this step to bring the system to active state
+ Step Average - This is the number of steps to perform MD for before printing the state of the system to console
+ Number of steps to perform pre-training 
+ Number of steps to perform main-training after pre-trainig
+ Epochs for pre-training
+ Epochs for main-trainig

The following is an example

```
1 1 1   # Cell size, i.e., dimensions of the parallel-piped system along (x,y,z)       
2 2 2   # Spatial decomposition, i.e., number of subsystems that will be assigned
          to MPI processes (l.b.h)
0.18    # Density or inter-atomic spacing
0.7     # Initial temperature
0.01    # The shortest time step for which trajectories must be computed
130     # Time at which the system reaches its relaxation state
1       # Interval to display output log 
25      # Time steps to predict 
50000   # Training epochs for pre-training
200000  # Training epochs 
```

## Build on USC HPC
To build on HPC you will need to load and 
execute following bash scripts into the current bash 
process <br/>
+ Openmpi - `/usr/usc/openmpi/default/setup.sh`
+ Gnu compiler collection above 5.3.0 - `/usr/usc/gnu/gcc/8.3.0/setup.sh`
+ Cmake - `/usr/usc/cmake/3.12.3/setup.sh`
+ `export CC=/usr/usc/gnu/gcc/8.3.0/bin/gcc`
+ `export CXX=/usr/usc/gnu/gcc/8.3.0/bin/g++`
+ Install pre-cxx11 ABI copy of libtorch from 
Pytorch's getting started [page](https://pytorch.org/get-started/locally/)
+ Add location of LibTorch library to cmake in prefix path like `cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch`. 
Make sure that the  C and CXX compiler identified are GNU 5.3.0 or above. We try to ensure
this by exporting CC and CXX flags in previous steps
+ Follow build command mentioned above from project directory.


# Build note on Intel devcloud
```
cmake .. -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=dpcpp -DCMAKE_PREFIX_PATH=$PWD/../libtorch/ 
```
and edit CMakeFiles/grad_lap.dir/flags.make
```
# compile CXX with /opt/intel/inteloneapi/compiler/latest/linux/bin/dpcpp
CXX_FLAGS =   -D_GLIBCXX_USE_CXX11_ABI=0   -D_GLIBCXX_USE_CXX11_ABI=0  -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers -Wno-write-strings -Wno-unknown-pragmas -Wno-missing-braces -openmp -std=gnu++14
```
