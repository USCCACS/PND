#!/bin/sh
#COBALT -n 1 -t 0:30:00
#COBALT -q debug-cache-quad --attrs mcdram=cache:numa=quad
#COBALT -A UltrafastMat
#COBALT -o Pytorch_CMake_Message.out

module load datascience/pytorch-0.5.0-mkldnn datascience/horovod-0.13.11

NPROC_PER_NODE=4
NPROC=$((NPROC_PER_NODE*COBALT_JOBSIZE))

mkdir build; cd build

python3 -c 'import horovod.torch as hvd; print(hvd.torch.utils.cmake_prefix_path)'

CC=gcc CXX=g++ cmake -DCMAKE_PREFIX_PATH='python3 -c 'import horovod.torch as hvd; print(hvd.torch.utils.cmake_prefix_path)'' ../

cmake --build . --config Release
