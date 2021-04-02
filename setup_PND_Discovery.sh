#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=30
#SBATCH --output="PND-sbatch-test-%A.out"

module purge
module load usc
module load cuda/10.1.243 python/3.7.6 cmake/3.16.2 cudnn/8.0.2-10.1

cds2

[ -d "run_PND" ] && { echo "Purging past source code from scratch"; rm -rf run_PND }

mkdir run_PND; cd run_PND

cp ~/PND.tgz ./

tar -xzvf PND.tgz 

cd PND/

mkdir build; cd build

python3 -c 'import torch ; print(torch.utils.cmake_prefix_path)'

CC=gcc CXX=g++ cmake -DCMAKE_PREFIX_PATH='/spack/apps/linux-centos7-x86_64/gcc-8.3.0/python-3.7.6-dd2am3dyvlpovhd4rizwfzc45wnsajxf/lib/python3.7/site-packages/torch/share/cmake;/usr/lib64' ../

cmake --build . --config Release

./pnd_example

