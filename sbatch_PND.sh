#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=30
#SBATCH --output="PND-sbatch-test-%A.out"

sh build_run_PND_Discovery.sh