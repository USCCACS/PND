# PND: Physics-Informed Neural-Network Molecular Dynamics 
We have developed PND, a differential equation solver software based on physics-informed neural network (PINN) for molecular dynamics simulators. Based on automatic differentiation technique provided by PyTorch, our software allows users to flexibly implement equation of motion for atoms, initial and boundary conditions, and conservation laws as loss function to train the network. PND comes with a parallel molecular dynamic engine in order to examine and optimize loss function design, and different conservation laws and boundary conditions, and hyperparameters, thereby accelerating PINN-based development for molecular applications. 

![pnd scematic](/img/PND_Schematic_2.png)

## Documentation
For documentation please visit this [page](https://usccacs.github.io/PND/annotated.html)

## Build Command
Use CMake to build the target `pnd_example`. This is binary for predicting energies using auto gradient using hamiltonian.
`cmake --build . --config Release`


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

(The following steps are implemented in the `build_run_PND_Discovery.sh` file. Details provided here)

Create a compressed source coude file using the follwing command
`tar -cvzf PND.tgz PND/`

To build on HPC you will need to load the offered python module with the folling commands. 
```
module purge
module load usc
module load cuda/10.1.243 python/3.7.6 cmake/3.16.2 cudnn/8.0.2-10.1
```
The next step is to use CMake to build the target 

```
cds2 # change into your scratch directory

rm -rf run_PND

mkdir run_PND; cd run_PND

cp ~/PND.tgz ./

tar -xzvf PND.tgz 

cd PND/

mkdir build; cd build

python3 -c 'import torch ; print(torch.utils.cmake_prefix_path)'

CC=gcc CXX=g++ cmake -DCMAKE_PREFIX_PATH='/spack/apps/linux-centos7-x86_64/gcc-8.3.0/python-3.7.6-dd2am3dyvlpovhd4rizwfzc45wnsajxf/lib/python3.7/site-packages/torch/share/cmake;/usr/lib64' ../

cmake --build . --config Release

./pnd_example

```

copy the sbatch file `sbatch_PND.sh` to the directory level containing PND and run it.
This script requests for resources, compresses the source code and runs the build commands listed in 
`build_run_PND_Discovery.sh`.

# Build note on Intel devcloud
```
cmake .. -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=dpcpp -DCMAKE_PREFIX_PATH=$PWD/../libtorch/ 
```
and edit CMakeFiles/grad_lap.dir/flags.make

```
# compile CXX with /opt/intel/inteloneapi/compiler/latest/linux/bin/dpcpp
CXX_FLAGS =   -D_GLIBCXX_USE_CXX11_ABI=0   -D_GLIBCXX_USE_CXX11_ABI=0  -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers -Wno-write-strings -Wno-unknown-pragmas -Wno-missing-braces -openmp -std=gnu++14
```
